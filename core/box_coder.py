from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import tensorflow as tf
from detection.utils import shape_utils
from detection.core import box_list

EPSILON = 1e-8


class BoxCoder(object):
  """Abstract base class for box coder."""
  __metaclass__ = ABCMeta

  @abstractproperty
  def code_size(self):
    """Return the size of each code, which is a constant integer specifying
    the length of code.

    To be implemented by subclasses.

    Returns:
      an int scalar.
    """
    pass

  def encode(self, boxlist, anchors):
    """Encode a box list relative to an anchor list. Calls and wraps `_encode`
    with a name scope.

    Args:
      boxlist: a BoxList holding `num_boxes` boxes to be encoded.
      anchors: a BoxList holding `num_boxes` anchors that `boxlist` are encoded 
        relative to.

    Returns:
      rel_codes: a tensor of shape [num_boxes, code_size] where each row holds 
        the anchor-encoded box coordinates.
    """
    with tf.name_scope('Encode'):
      return self._encode(boxlist, anchors)

  def decode(self, rel_codes, anchors):
    """Decode relative encoding of box coordinates back to coordinates.
    Calls and wraps `self._decode` with a name scope.

    Args:
      rel_codes: a tensor of shape [num_boxes, code_size] where each row holds 
        the anchor-encoded box coordinates.
      anchors: a BoxList holding `num_boxes` anchors that `rel_codes` are 
        decoded relative to.

    Returns:
      boxlist: a BoxList holding `num_boxes` boxes.
    """
    with tf.name_scope('Decode'):
      return self._decode(rel_codes, anchors)

  @abstractmethod
  def _encode(self, boxlist, anchors):
    """Encode a box list relative to an anchor list. 

    To be implemented by subclasses.

    Args:
      boxlist: a BoxList holding `num_boxes` boxes to be encoded.
      anchors: a BoxList holding `num_boxes` anchors that `boxlist` are encoded 
        relative to.

    Returns:
      rel_codes: a tensor of shape [num_boxes, code_size] where each row holds 
        the anchor-encoded box coordinates.
    """
    pass

  @abstractmethod
  def _decode(self, rel_codes, anchors):
    """Decode relative encoding of box coordinates back to coordinates.

    To be implemented by subclasses.

    Args:
      rel_codes: a tensor of shape [num_boxes, code_size] where each row holds 
        the anchor-encoded box coordinates.
      anchors: a BoxList holding `num_boxes` anchors that `rel_codes` are 
        decoded relative to.

    Returns:
      boxlist: a BoxList holding `num_boxes` boxes.
    """
    pass


class FasterRcnnBoxCoder(BoxCoder):
  """Faster RCNN Box Coder."""

  def __init__(self, scale_factors=None):
    """Constructor.

    Args:
      scale_factors: list of 4 positive scalar floats, storing the 
        factors by which the anchor-encoded prediction targets `ty`, `tx`, `th`
        and `tw` are scaled. If None, no scaling is performed.
    """
    if scale_factors:
      if len(scale_factors) != 4 or any(map(lambda f: f <= 0, scale_factors)):
        raise ValueError('scale_factors must be a list of 4 positive numbers.')
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 4

  def _encode(self, boxlist, anchors):
    """Encode a box list relative to an anchor list.

    Args:
      boxlist: a BoxList holding `num_boxes` boxes to be encoded.
      anchors: a BoxList holding `num_boxes` anchors that `boxlist` are encoded 
        relative to.

    Returns:
      rel_codes: a tensor of shape [num_boxes, 4] where each row holds the 
        anchor-encoded box coordinates in ty, tx, th, tw format.
    """
    ycenter, xcenter, h, w = boxlist.get_center_coordinates_and_sizes()
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    ty = (ycenter - ycenter_a) / ha
    tx = (xcenter - xcenter_a) / wa
    th = tf.log(h / ha)
    tw = tf.log(w / wa)

    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    rel_codes = tf.stack([ty, tx, th, tw], axis=1)
    return rel_codes

  def _decode(self, rel_codes, anchors):
    """Decode relative encoding of box coordinates back to coordinates.

    Args:
      rel_codes: a tensor of shape [num_boxes, 4] where each row holds the 
        anchor-encoded box coordinates in ty, tx, th, tw format.
      anchors: a BoxList holding `num_boxes` anchors that `rel_codes` are 
        decoded relative to.

    Returns:
      boxlist: a BoxList holding `num_boxes` boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(rel_codes, axis=1)
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    h = tf.exp(th) * ha
    w = tf.exp(tw) * wa
    # convert box coordinates back to ymin, xmin, ymax, xmax format.
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return box_list.BoxList(tf.stack([ymin, xmin, ymax, xmax], axis=1))


def batch_decode(batch_box_encodings, anchor_boxlist_list, box_coder):
  """Decode a batch of box encodings w.r.t. anchors to box coordinates.

  Args:
    batch_box_encodings: a float tensor of shape 
      [batch_size, num_anchors, num_classes, 4] holding box encoding 
      predictions. 
    anchors_boxlist_list: a list of BoxList instance holding float tensor
      of shape [num_anchors, 4] as anchor box coordinates. Lenght is equal
      to `batch_size`.
    box_coder: a BoxCoder instance to decode anchor-encoded location predictions
      into box coordinate predictions.

  Returns:
    decoded_boxes: a float tensor of shape 
        [batch_size, num_anchors, num_classes, 4].
  """
  shape = shape_utils.combined_static_and_dynamic_shape(batch_box_encodings)

  box_encodings_list = [tf.reshape(box_encoding, [-1, box_coder.code_size]) 
      for box_encoding in tf.unstack(batch_box_encodings, axis=0)]
  # tile anchors in the 1st dimension to `shape[2]`(i.e. num of classes)
  anchor_boxlist_list = [box_list.BoxList(
      tf.reshape(tf.tile(tf.expand_dims(anchor_boxlist.get(), 1), 
          [1, shape[2], 1]), [-1, box_coder.code_size])) 
      for anchor_boxlist in anchor_boxlist_list]

  decoded_boxes = []
  for box_encodings, anchor_boxlist in zip(
      box_encodings_list, anchor_boxlist_list):
    decoded_boxes.append(box_coder.decode(box_encodings, anchor_boxlist).get())

  decoded_boxes = tf.reshape(decoded_boxes, shape)
  return decoded_boxes

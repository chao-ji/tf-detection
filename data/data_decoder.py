import collections

import tensorflow as tf

from detection.core import standard_names as names


class DataDecoder(object):
  """Data decoder implements a `decode` method that converts a serialized 
  protobuf string into tensors of desired shape and type."""
  def __init__(self, keys_to_features=None, load_masks=False):
    """Constructor.

    Args:
      keys_to_features: dict or None, a dict mapping from tensor names to 
        feature parsers. If None, a default mapping will be built.
      load_masks: bool scalar, whether to load instance masks.
    """
    if keys_to_features is None:
      keys_to_features = {
          names.TfRecordFields.image_encoded:
              tf.FixedLenFeature((), tf.string, default_value=''),
          names.TfRecordFields.object_bbox_ymin:
              tf.VarLenFeature(tf.float32),
          names.TfRecordFields.object_bbox_xmin:
              tf.VarLenFeature(tf.float32),
          names.TfRecordFields.object_bbox_ymax:
              tf.VarLenFeature(tf.float32),
          names.TfRecordFields.object_bbox_xmax:
              tf.VarLenFeature(tf.float32),
          names.TfRecordFields.object_class_label:
              tf.VarLenFeature(tf.int64),
          names.TfRecordFields.object_mask:
              tf.VarLenFeature(tf.string)
      }
    self._keys_to_features = keys_to_features
    self._load_masks = load_masks

  def decode(self, protobuf_str, scope=None):
    """Decodes a protobuf string into a tensor dict.

    Args:
      protobuf_str: a scalar string tensor holding a serialized
        proto buffer corresponding to data for a single image.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
        { 'images': float tensor of shape [height, width, channels],
          'groundtruth_boxes': float tensor of shape [num_gt_boxes, 4],
          'groundtruth_labels': int tensor of shape [num_gt_boxes],
          'groundtruth_masks': (Optional) uint8 tensor of shape 
            [num_gt_boxes, image_height, image_width]}
    """
    with tf.name_scope(scope, 'Decode', [protobuf_str]):
      tfrecord_dict = tf.parse_single_example(
          protobuf_str, self._keys_to_features)

      encoded = tfrecord_dict[names.TfRecordFields.image_encoded] 
      ymin = tfrecord_dict[names.TfRecordFields.object_bbox_ymin].values
      xmin = tfrecord_dict[names.TfRecordFields.object_bbox_xmin].values
      ymax = tfrecord_dict[names.TfRecordFields.object_bbox_ymax].values
      xmax = tfrecord_dict[names.TfRecordFields.object_bbox_xmax].values
      labels = tfrecord_dict[names.TfRecordFields.object_class_label].values

      image = tf.to_float(
          tf.image.decode_jpeg(encoded, channels=3))
      groundtruth_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
      groundtruth_labels = tf.to_int32(labels)

      tensor_dict = collections.OrderedDict([
          (names.TensorDictFields.image, image),
          (names.TensorDictFields.groundtruth_boxes, groundtruth_boxes), 
          (names.TensorDictFields.groundtruth_labels, groundtruth_labels)
      ])

      if self._load_masks:
        if not names.TfRecordFields.object_mask in tfrecord_dict:
          raise ValueError('instance mask not found in tfrecord file.')
        masks = tfrecord_dict[names.TfRecordFields.object_mask].values
        image_height, image_width = tf.unstack(tf.shape(image)[:2])
        groundtruth_masks = _decode_png_masks(masks, image_height, image_width)
        tensor_dict[names.TensorDictFields.groundtruth_masks
            ] = groundtruth_masks

      return tensor_dict


def _decode_png_masks(png_encoded_masks, image_height, image_width):
  """Decode string tensors encoding png masks into 2-D tensors holding instance 
  masks.

  Args:
    png_encoded_masks: 1-D string tensor of shape [num_boxes], each holding the 
      png-encoded binary mask of a single instance.
    image_height: int scalar tensor, image height.
    image_width: int scalar tensor, image width.

  Returns:
    3-D tensor of shape [num_boxes, image_height, image_width] holding binary 
      instance masks.
  """
  # when there are > 0 instances
  def decode_nonempty(encoded_masks):
    mask = tf.squeeze(
        tf.image.decode_image(encoded_masks, channels=1), axis=2)
    mask.set_shape([None, None])
    mask = tf.to_float(mask > 0)
    return mask

  # when there is 0 instances
  def decode_empty(image_height, image_width):
    return tf.zeros([0, image_height, image_width])

  png_masks = tf.cond(
      tf.size(png_encoded_masks) > 0,
      lambda: tf.map_fn(decode_nonempty, png_encoded_masks, dtype=tf.float32),
      lambda: decode_empty(image_height, image_width))
  return tf.cast(png_masks, tf.uint8)

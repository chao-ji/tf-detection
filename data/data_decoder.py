import collections

import tensorflow as tf

from detection.core import standard_names as names


class DataDecoder(object):
  """Data decoder implements a `decode` method that converts a serialized 
  protobuf string into tensors of desired shape and type."""
  def __init__(self, keys_to_features=None):
    """Constructor.

    Args:
      keys_to_features: dict or None, a dict mapping from tensor names to 
        feature parsers. If None, a default mapping will be built.
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
      }
    self._keys_to_features = keys_to_features

  def decode(self, protobuf_str, scope=None):
    """Decodes a protobuf string into a tensor dict.

    Args:
      protobuf_str: a scalar string tensor holding a serialized
        proto buffer corresponding to data for a single image.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
        { 'images': float tensor of shape [height, width, channels],
          'groundtruth_boxes': float tensor of shape [num_gt_boxes, 4],
          'groundtruth_labels': int tensor of shape [num_gt_boxes] }
    """
    with tf.name_scope(scope, 'Decode', [protobuf_str]):
      tensor_dict = tf.parse_single_example(
          protobuf_str, self._keys_to_features)
   
      encoded = tensor_dict[names.TfRecordFields.image_encoded] 
      ymin = tensor_dict[names.TfRecordFields.object_bbox_ymin].values
      xmin = tensor_dict[names.TfRecordFields.object_bbox_xmin].values
      ymax = tensor_dict[names.TfRecordFields.object_bbox_ymax].values
      xmax = tensor_dict[names.TfRecordFields.object_bbox_xmax].values
      labels = tensor_dict[names.TfRecordFields.object_class_label].values

      image = tf.to_float(
          tf.image.decode_jpeg(encoded, channels=3))
      groundtruth_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
      groundtruth_labels = tf.to_int32(labels)

      tensor_dict = collections.OrderedDict([
          (names.TensorDictFields.image, image),
          (names.TensorDictFields.groundtruth_boxes, groundtruth_boxes), 
          (names.TensorDictFields.groundtruth_labels, groundtruth_labels)
      ])

      return tensor_dict


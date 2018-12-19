import collections

import tensorflow as tf

from detection.core import standard_names as names


def add_runtime_shapes(tensor_dict):
  """Add runtime shapes of tensors in a tensor dict.

  Args:
    tensor_dict: a dict mapping from tensor names to tensors.

  Returns:
    tensor_dict_with_shapes: a tensor dict containing original tensors in
      `tensor_dict` and their runtime shapes.
  """
  tensor_dict_with_shapes = collections.OrderedDict()
  for key, value in tensor_dict.items():
    tensor_dict_with_shapes[key] = value
    tensor_dict_with_shapes[
        key + names.TensorDictFields.runtime_shape_str] = tf.shape(value)

  return tensor_dict_with_shapes


def unbatch_padded_tensors(tensor_dict, static_shapes):
  """Unbatch and unpad a batch of padded tensors.

  This function first unbatch a tensor with an outer batch dimension into a list
  of unbatched tensors (with padding), and then unpad each tensor in the list by
  slicing out the portion containing non-padded values.

  For example, given input tensor_dict
   {'image': tensor of shape [batch_size, height, width, 3],
    'image_shape': tensor of shape [batch_size, 3],
    'gt_boxes': tensor of shape [batch_size, num_boxes, 4],
    'gt_boxes_shape': tensor of shape [batch_size, 2],
    'gt_labels': tensor of shape [batch_size, num_boxes],
    'gt_labels_shape': tensor of shape [batch_size, 1]}

  output tensor_dict would be 
   {'image': a list of `batch_size` tensors of shape [height_i, width_i, 3],
    'gt_boxes': a list of `batch_size` tensors of shape [num_boxes_i, 4],
    'gt_labels': a list of `batch_size` tensors of shape [num_boxes_i]}

  Args:
    tensor_dict: a dict mapping from tensor names to tensors. The tensors 
      contain both the original tensors and their runtime shapes.
    static_shapes: a dict mapping from tensor names to tf.TensorShape instances.
      Only contains original tensors. Used to set shapes for unpadded tensors.

  Returns:
    sliced_tensor_dict: a dict with the same number of entries as `tensor_dict`,
      where each value of the dict is a list (with length batch_size) containing
      properly unpadded tensors as opposed to a single tensor in `tensor_dict`.
  """
  tensors = collections.OrderedDict()
  shapes = collections.OrderedDict()

  for key, batched_tensor in tensor_dict.items():
    unbatched_tensor_list = tf.unstack(batched_tensor)
    if names.TensorDictFields.runtime_shape_str in key:
      shapes[key] = unbatched_tensor_list
    else:
      tensors[key] = unbatched_tensor_list

  sliced_tensor_dict = collections.OrderedDict()
  for key in tensors.keys():
    unbatched_tensor_list = tensors[key]
    unbatched_shape_list = shapes[
        key + names.TensorDictFields.runtime_shape_str]

    sliced_tensor_list = []
    for unbatched_tensor, unbatched_shape in zip(
        unbatched_tensor_list, unbatched_shape_list):
      sliced_tensor = tf.slice(unbatched_tensor,
                               tf.zeros_like(unbatched_shape),
                               unbatched_shape)
      sliced_tensor.set_shape(static_shapes[key])
      sliced_tensor_list.append(sliced_tensor)

    sliced_tensor_dict[key] = sliced_tensor_list

  return sliced_tensor_dict


def sparse_to_one_hot_labels(tensor_dict, 
                             num_classes, 
                             label_id_offset=1, 
                             add_background_class=True):
  """Convert groundtruth labels in sparse representation to one-hot 
  representation.

  For example, given the groundtruth labels `[5, 1, 3, 2]` and `num_classes` 6,
  the one-hot encoded groundtruth labels would be
     [[0, 0, 0, 0, 1, 0],
      [1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0, 0]]
  It's assumed that the background class has index `0`, so the offset `1` is 
  subtracted off the real object class indices.

  Note that when an image has no groundtruth labels, the output one-hot 
  representation would be a vector of `num_classes` 0's.

  If `add_background_class` is True, a column of zeros is added to the left of
  the one-hot encoded matrix.

  Args:
    tensor_dict: a dict mapping from tensor names to tensors. Must contain the 
      field 'groundtruth_labels' pointing to a list (with length `batch_size`) 
      of tensors of shape [num_boxes], holding the groundtruth labels of 
      `num_boxes` boxes for each image in a mini-batch.
    num_classes: int scalar, num of classes.
    label_id_offset: int scalar, num of dummy classes preceding the real 
      object classes, typically 1 for the background class.
    add_background_class: bool scalar, whether to pad the resulting one-hot 
      encoded labels with background class.

  Returns:
    a tensor dict where the groundtruth labels are now in one-hot
      represetnation (of shape [num_boxes, num_classes]).
  """
  labels_list = []

  def one_hot_labels(labels):
    """Convert labels into one hot representation.

    Args:
      labels: a tensor of shape [num_boxes] holding label indices.
    
    Returns:
      a tensor of shape [num_boxes, num_classes].
    """
    labels -= label_id_offset
    labels = tf.one_hot(
      labels, depth=num_classes, on_value=1., off_value=0., dtype=tf.float32)
    return labels

  for labels in tensor_dict[names.TensorDictFields.groundtruth_labels]:
    labels = tf.cond(tf.greater(tf.size(labels), 0),
                     lambda: one_hot_labels(labels),
                     lambda: tf.zeros((0, num_classes)))
    if add_background_class:
      labels = tf.pad(labels, [[0, 0], [1, 0]], mode='CONSTANT')
    labels_list.append(labels)
  tensor_dict[names.TensorDictFields.groundtruth_labels] = labels_list
  return tensor_dict

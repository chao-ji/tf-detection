""""""
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


def sparse_to_one_hot_labels(tensor_dict, depth, label_id_offset=1):
  """Convert sparse groundtruth labels to one-hot representation.

  Args:
    tensor_dict: a dict mapping from tensor names to tensors.
    depth: int scalar, num of classes.
    label_id_offset: int scalar, num of dummy classes preceding the real 
      object classes, typically 1 for the background class.

  Returns: a tensor dict where the groundtruth labels are now in one-hot
    represetnation.
  """
  labels_list = []

  def one_hot_labels(labels):
    """Convert labels into one hot representation.

    Args:
      labels: a rank-1 tensor containing label indices.
    
    Returns:
      a rank-2 tensor with shape [num_boxes, num_classes].
    """
    labels -= label_id_offset
    labels = tf.one_hot(
      labels, depth=depth, on_value=1., off_value=0., dtype=tf.float32)
    return labels

  for labels in tensor_dict[names.TensorDictFields.groundtruth_labels]:
    labels_list.append(
        # handle the case where the image has no groundtruth boxes
        tf.cond(tf.greater(tf.size(labels), 0),
            lambda: one_hot_labels(labels),
            lambda: tf.zeros((0, depth))
        )
    )
  tensor_dict[names.TensorDictFields.groundtruth_labels] = labels_list
  return tensor_dict


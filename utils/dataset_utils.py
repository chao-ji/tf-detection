import collections

import tensorflow as tf

from detection.data.preprocessor import IMAGENET_MEAN
from detection.core.standard_names import TensorDictFields
from detection.core import box_list
from detection.core import box_list_ops


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
        key + TensorDictFields.runtime_shape_str] = tf.shape(value)

  return tensor_dict_with_shapes


def unbatch_padded_tensors(tensor_dict, static_shapes, keep_padded_list):
  """Unbatch and unpad a batch of padded tensors.

  This function first unbatch a tensor with an outer batch dimension into a list
  of unbatched tensors (with padding), and then unpad each tensor in the list by
  slicing out the portion containing non-padded values. You can optionally 
  choose a subset of tensors (specifying their keys in `keep_padded_list`) so 
  that these tensors will stay in padded form (e.g. 'image').

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
    keep_padded_list: a list or tuple of strings, holding the keys to the 
      tensor_dict for which the padded tensor will stay in padded form.

  Returns:
    sliced_tensor_dict: a dict with the same number of entries as `tensor_dict`,
      where each value of the dict is a list (with length batch_size) containing
      properly unpadded tensors as opposed to a single tensor in `tensor_dict`.
  """
  tensors = collections.OrderedDict()
  shapes = collections.OrderedDict()

  for key, batched_tensor in tensor_dict.items():
    unbatched_tensor_list = tf.unstack(batched_tensor)
    if TensorDictFields.runtime_shape_str in key:
      shapes[key] = unbatched_tensor_list
    else:
      tensors[key] = unbatched_tensor_list

  sliced_tensor_dict = collections.OrderedDict()
  for key in tensors.keys():
    unbatched_tensor_list = tensors[key]
    unbatched_shape_list = shapes[
        key + TensorDictFields.runtime_shape_str]

    sliced_tensor_list = []
    for unbatched_tensor, unbatched_shape in zip(
        unbatched_tensor_list, unbatched_shape_list):

      if key not in keep_padded_list:
        sliced_tensor = tf.slice(unbatched_tensor,
                                 tf.zeros_like(unbatched_shape),
                                 unbatched_shape)
      else:
        sliced_tensor = unbatched_tensor

      sliced_tensor.set_shape(static_shapes[key])
      sliced_tensor_list.append(sliced_tensor)

    sliced_tensor_dict[key] = sliced_tensor_list

  # We need to adjust the groundtruth boxes to the new dimensions for padded
  # images (when `batch_size` > 1). Convert to absolute coordinates using
  # original dimensions, and convert back to normalized coordinates using
  # padded dimensions.
  batch_size = len(sliced_tensor_dict[TensorDictFields.groundtruth_boxes]) 
  if batch_size > 1:
    print('asdfadsfasd')
    for i in range(batch_size):
      boxlist = box_list.BoxList(
          sliced_tensor_dict[TensorDictFields.groundtruth_boxes][i])
      # original dimensions
      height, width = tf.unstack(shapes[TensorDictFields.image + 
          TensorDictFields.runtime_shape_str][i][:-1])
      boxlist = box_list_ops.to_absolute_coordinates(boxlist, height, width)
      # padded dimensions
      new_height, new_width = tf.unstack(tf.shape(
          sliced_tensor_dict[TensorDictFields.image][i])[:-1])
      boxlist = box_list_ops.to_normalized_coordinates(
          boxlist, new_height, new_width)

      sliced_tensor_dict[TensorDictFields.groundtruth_boxes][i] = boxlist.get()

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

  for labels in tensor_dict[TensorDictFields.groundtruth_labels]:
    labels = tf.cond(tf.greater(tf.size(labels), 0),
                     lambda: one_hot_labels(labels),
                     lambda: tf.zeros((0, num_classes)))
    if add_background_class:
      labels = tf.pad(labels, [[0, 0], [1, 0]], mode='CONSTANT')
    labels_list.append(labels)
  tensor_dict[TensorDictFields.groundtruth_labels] = labels_list
  return tensor_dict


def image_size_bucketed_batching(tensor_dict_dataset,
                                 batch_size,
                                 height_boundaries,
                                 width_boundaries,
                                 padded_pixel=IMAGENET_MEAN):
  """Given a tensor dict holding tensors corresponding to a single image, place
  them into buckets based on image spatial size (i.e. heigth & width). Then we 
  pad tensors to the same shape, and batch `batch_size` tensors into a single 
  tensor. The goal is to group similarly sized images into the same bucket to 
  save the amount of padding.

  Example:

  Suppose `height_boundaries` = `width_boundaries` = [100, 200, 300], then the
  height and width dimension will be divided into four inteversl:
  [0, 100), [100, 200), [200, 300), [300, inf)

  A 250x450 image will be placed into interval 2 and 3 in the height and width
  dimension, respecitvely, so its key = 3 + 2 * (3 + 1) = 11
  

  Note: The image tensor is padded at the right (width dimension) and at the 
  bottom (height dimension) to the maximum image width and height within the 
  same bucket with the pixel value `padded_pixel`, while the other tensors 
  are simply padded with zeros.

  Args:
    tensor_dict_dataset: A tf.data.Dataset instance holding the image tensor of
      shape [height, width, 3], the groundtruth_boxes tensor of shape 
      [num_boxes, 4], the groundtruth_labels tensor of shape [num_boxes]. 
    batch_size: int scalar, batch size.
    height_boundaries: list or tuple of increasing int scalars, bucket 
      boundaries of heights.  
    width_boundaries: list or tuple of increasing int scalars, bucket
      boundaries of widths.
    padded_pixel: list or tuple of 3 int scalars, the RGB value of the pixel 
      that the image tensor will be padded with.

  Returns:
    tensor_dict_dataset: A tf.data.Dataset instance holding the image tensor of
      shape [batch_size, height, width, 3], the groundtruth_boxes tensor of 
      shape [batch_size, num_boxes, 4], the groundtruth_labels tensor of shape 
      [batch_size, num_boxes]. And optionally the groundtruth_masks tensor of
      shape [num_boxes, height, width].
  """
  height_boundaries = tf.convert_to_tensor(height_boundaries, dtype=tf.int32)
  width_boundaries = tf.convert_to_tensor(width_boundaries, dtype=tf.int32)

  def _adjust_image_mean(tensor_dict, offset):
    tensor_dict[TensorDictFields.image] += offset
    return tensor_dict

  def _get_bucket_index(val, boundaries):
    less = tf.concat([val < boundaries, [True]], axis=0)
    greater_equal = tf.concat([[True], val >= boundaries], axis=0)
    return tf.to_int32(tf.where(tf.logical_and(less, greater_equal))[0][0])

  def _key_fn(tensor_dict):
    image = tensor_dict[TensorDictFields.image]
    height, width = tf.unstack(tf.shape(image)[:2])

    height_index = _get_bucket_index(height, height_boundaries)
    width_index = _get_bucket_index(width, width_boundaries)

    return tf.cast(width_index + height_index * (
        tf.size(width_boundaries) + 1), tf.int64)

  def _reduce_fn(unused_key, dataset):
    return dataset.padded_batch(batch_size,
                                padded_shapes=dataset.output_shapes,
                                drop_remainder=True)

  imagenet_mean = tf.to_float(padded_pixel)
  tensor_dict_dataset = tensor_dict_dataset.map(
      lambda tensor_dict: _adjust_image_mean(tensor_dict, -imagenet_mean))
  tensor_dict_dataset = tensor_dict_dataset.apply(
      tf.contrib.data.group_by_window(
          key_func=_key_fn,
          reduce_func=_reduce_fn,
          window_size=batch_size))

  tensor_dict_dataset = tensor_dict_dataset.map(
      lambda tensor_dict: _adjust_image_mean(tensor_dict, imagenet_mean))

  return tensor_dict_dataset

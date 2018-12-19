import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.utils import shape_utils

slim = tf.contrib.slim

#I
def split_separable_conv2d(inputs,
                           num_outputs,
                           kernel_size,
                           depth_multiplier,
                           stride=1,
                           padding='SAME',
                           rate=1,
                           depthwise_biases_initializer=None,
                           scope=None,
                           **kwargs):
  """Separable 2D convolution split into a depthwise conv op followed by a 1x1 
  pointwise conv op. This allows one to separately configure the batch-norm, 
  biases, and the activation function of the depthwise conv op.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels]. 
    num_outputs: The number of pointwise convolution output filters. If is
      None, then we skip the pointwise convolution stage. 
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same. 
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      depthwise convolution stride. Can be an int if both strides are the same.
    padding: One of 'VALID' or 'SAME'.
    rate: A list of length 2: [rate_height, rate_width], specifying the dilation
      rates for atrous convolution. Can be an int if both rates are the same.
    depthwise_biases_initializer: The biases initializer for the depthwise conv.
      If any value is larger than one, then both stride values need to be one.
    scope: Optional scope for variable_scope.
    **kwargs: additional keyword arguments to pass to slim.conv2d or
      slim.separable_conv2d
  """
  depthwise = slim.separable_conv2d(
      inputs,
      None,
      kernel_size,
      depth_multiplier,
      stride=stride,
      padding=padding,
      rate=rate,
      biases_initializer=depthwise_biases_initializer,
      scope=scope + '_depthwise',
      **kwargs)
  pointwise = slim.conv2d(depthwise,
                          num_outputs,
                          kernel_size=1,
                          stride=1,
                          padding='SAME',
                          scope=scope + '_pointwise')
  return pointwise


def normalized_to_absolute_coordinates(boxes_normalized, image_shape):
  def _to_absolute_coordinates(normalized_boxes):
    normalized_boxes = normalized_boxes[0]
    return box_list_ops.to_absolute_coordinates(
        box_list.BoxList(normalized_boxes),
        image_shape[1], image_shape[2]).get(),

  absolute_boxes = shape_utils.static_map_fn(_to_absolute_coordinates, elems=(boxes_normalized,))
  return absolute_boxes



def get_unit_square(batch_size=None):
  """Returns the ymin, xmin ymax, xmax coordinate of a unit square box.

  Args:
    batch_size: int scalar or int scalar tensor, batch_size.

  Returns:
    clip_window: float tensor of shape [4] or [batch_size, 4], holding the 
      coordinates of the unit square box (optionally tiled to have batch size
      `batch_size`).
  """
  if batch_size is None:
    return tf.convert_to_tensor([0., 0., 1., 1.])
  else:
    return tf.convert_to_tensor([[0., 0., 1., 1.] for _ in range(batch_size)])



def random_sample(tensor, sample_size, seed=None):
  """Randomly samples `sample_size` elements from `tensor` along the first 
  dimension. Or returns `tensor` as is if `sample_size` is greater than or
  equal to `tf.shape(tensor)[0]`.

  Args:
    tensor: any tensor with rank >= 1.
    sample_size: int scalar or int scalar tensor, sample size.

  Returns:
    sampled_tensor: tensor of shape 
      [tf.minimum(sample_size, tf.shape(tensor)[0]), ...], subtensor sampled 
      from `tensor`.
  """
  shape = shape_utils.combined_static_and_dynamic_shape(tensor)
  sampled_tensor = tf.cond(tf.greater(shape[0], sample_size), 
      lambda: tf.random_shuffle(tensor, seed=seed)[:sample_size], 
      lambda: tensor)
  return sampled_tensor


def balanced_subsample(indicator, sample_size, labels, pos_frac=0.5, seed=None):
  """Sample from a set of elements with binary labels such that the fraction of 
  positives is at most `pos_frac`. Example:

  Given `indicator = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]`,
           `labels = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1]`,
         `pos_frac = 0.5`, and `sample_size = 5` the
    output might be [0, 0, 1, 0, 1, 0, 0, 1, 1, 1], where 2, 4, 8 are negatives 
  and 7, 9 are positives. so positive fraction = 2 / 5 <= 0.5
  

  Args:
    indicator: bool tensor of shape [batch_size] where only True elements 
      are to be sampled from.
    sample_size: int scalar, num of samples to be drawn from `indicator`.
    labels: bool tensor of shape [batch_size], holding binary class labels.

  Returns:
    sampled_indicator: bool tensor of shape [batch_size] holding the subset
      sampled from the input.
  """
  seed = 0
  neg_indicator = tf.logical_not(labels)
  pos_indicator = tf.logical_and(labels, indicator)
  neg_indicator = tf.logical_and(neg_indicator, indicator)
  pos_indices = tf.reshape(tf.where(pos_indicator), [-1])
  neg_indices = tf.reshape(tf.where(neg_indicator), [-1])

  num_pos = int(pos_frac * sample_size)
  sampled_pos_indices = random_sample(pos_indices, num_pos, seed=seed)
  num_neg = sample_size - tf.size(sampled_pos_indices)
  sampled_neg_indices = random_sample(neg_indices, num_neg, seed=seed)

  shape = shape_utils.combined_static_and_dynamic_shape(indicator)

  sampled_indicator = tf.cast(tf.one_hot(
      tf.concat([sampled_pos_indices, sampled_neg_indices], axis=0),
      depth=shape[0]), tf.bool)
  sampled_indicator = tf.reduce_any(sampled_indicator, axis=0)

  return sampled_indicator


def create_gradient_update_op(optimizer, 
                              loss, 
                              global_step, 
                              gradient_clipping_by_norm=0.):
  """Creates gradient update op by computing the gradients and optionally
  applying postprocessing on the computed grads and vars.

  Args:
    optimizer: a tensorflow Optimizer instance.
    loss: float scalar tensor, the loss to optimizer over. Includes all
      localization loss, classification loss, and regularization loss (if any).
    global_step: float scalar tensor, global step.
    gradient_clipping_by_norm: float tensor, if > 0, clip the gradient tensors
      such that their norms <= `gradient_clipping_by_norm`.

  Returns:
    grouped_update_op: an instance of tf.Operation, the grouped gradient 
      update op to be executed in a tf.Session.
  """
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  grads_and_vars = optimizer.compute_gradients(loss)

  if gradient_clipping_by_norm > 0:
      with tf.name_scope('clip_grads'):
        grads_and_vars = slim.learning.clip_gradient_norms(
            grads_and_vars, gradient_clipping_by_norm)
 
  grad_update_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)

  update_ops.append(grad_update_op)

  grouped_update_op = tf.group(*update_ops, name='update_barrier')

  return grouped_update_op

#I
def reduce_sum_trailing_dimensions(tensor, ndims):
  """Computes sum across all dimensions following first `ndims` dimensions."""
  return tf.reduce_sum(tensor, axis=tuple(range(ndims, tensor.shape.ndims)))




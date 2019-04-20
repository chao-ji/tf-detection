import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.utils import shape_utils

slim = tf.contrib.slim


def split_separable_conv2d(inputs,
                           num_outputs,
                           kernel_size,
                           depth_multiplier=1,
                           stride=1,
                           rate=1,
                           scope=None):
  """Separable 2D convolution split into depthwise and pointwise stages. One can 
  separately configure batch-norm, activation function, weight initializer and
  regularizer, etc., for the pointwise and the depthwise conv op.

  Args:
    inputs: 4-D float tensor of shape [batch_size, height, width, channels].
    num_outputs: int scalar, the depth of the output tensor.
    kernel_size: int scalar or a 2-tuple of ints, kernel size.
    depth_multiplier: int scalar, num of depthwise conv output channels for each 
      input channel. The total num of depthwise conv output channels equals 
      `depth_in * depth_multiplier`.
    stride: int scalar or a 2-tuple of ints, stride for the depthwise conv.
    rate: int scalar or a 2-tuple of ints, atrous rate for the depthwise conv.
    scope: string scalar, scope name.

  Returns:
    4-D float tensor of shape [batch_size, height_out, width_out, channesl_out].
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size,
      depth_multiplier,
      stride=stride,
      rate=rate,
      scope=scope + '_depthwise')
  return slim.conv2d(outputs,
                     num_outputs,
                     kernel_size=1,
                     stride=1,
                     padding='SAME',
                     scope=scope + '_pointwise')


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
  """Randomly samples `sample_size` elements from `tensor` along the 0th 
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
         `pos_frac = 0.5`, and `sample_size = 5` 
  
  `indicator` indicates that elements 1, 2, 4, 7, 8, 9 are candidates. One 
  output might be [0, 0, 1, 0, 1, 0, 0, 1, 1, 1], where 2, 4, 8 are negatives 
  and 7, 9 are positives. so positive fraction = 2 / 5 <= 0.5
  
  Args:
    indicator: bool tensor of shape [batch_size] where only True elements 
      are to be sampled.
    sample_size: int scalar, num of samples to be drawn from `indicator`.
    labels: bool tensor of shape [batch_size], holding binary class labels.
    pos_frac: float scalar, fraction of positives of the entire sample.
    seed: int scalar, random seed.

  Returns:
    sampled_indicator: bool tensor of shape [batch_size] holding the subset
      sampled from the input.
  """
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


def to_image_size_masks(masks, boxes, image_height, image_width):
  """Transform box-size instance masks to image-size instance masks.
  
  As shown below, the small box represent box-size instance mask, i.e. 
  `masks[i, :, :]`, while the large box represent image-size instance mask.


    A---------------------+
    |                     |
    |     a---+           |
    |     |   |           |
    |     |   |           |
    |     |   |           |
    |     +---b           |
    +---------------------B

  The coordinates of `a`, `b`, `A`, `B` are 

  a: ymin, xmin
  b: ymax, xmax
  A: YMIN, XMIN
  B: YMAX, XMAX

  `ymin`, `xmin`, `ymax`, `xmax` are rows of `boxes`, and they vary in [0, 1],
  and are relative to the full-image box [0, 0, 1, 1].

  We need to scale the y and x coordinates such that 
  1. the full-image box [0, 0, 1, 1] becomes `YMIN`, `XMIN`, `YMAX`, `XMAX`
  2. `ymin`, `xmin`, `ymax`, `xmax` becomes [0, 0, 1, 1]
  3. the large box has absolute size `[image_height, image_width]`

  We can use the built-in function `tf.image.crop_and_resize` in which the 
  rows of `boxes` argument contain `[YMIN, XMIN, YMAX, XMAX]` that we need to 
  compute.

              small box                   large box
  original    ymin  xmin  ymax  xmax      0     0     1     1
  new         0     0     1     1         YMIN  XMIN  YMAX  XMAX

  The original-to-new transformation scales the y and x coordinates, with ratio
  `k_y = 1/(ymax - ymin)`, `k_x = 1/(xmax - xmin)` 

  we have

  (ymin - 0) k_y = 0 - YMIN           (xmin - 0) k_x = 0 - XMIN
  (ymax - 1) k_y = 1 - YMAX           (xmax - 1) k_x = 1 - XMAX

  So

  YMIN = - ymin / (ymax - ymin)
  XMIN = - xmin / (xmax - xmin)
  YMAX = (1 - ymin) / (ymax - ymin)
  XMAX = (1 - xmin) / (xmax - xmin) 


  Args:
    masks: a float tensor of shape [max_num_proposals, mask_height, mask_width],
      holding box-size instance masks for a single image.
    boxes: a float tensor of shape [max_num_proposals, 4], holding refined 
      proposal boxes.
    image_height: int scalar tensor, image height.
    image_width: int scalar tensor, image width.

  Returns:
    masks: a float tensor of shape [num_boxes, image_height, image_width, 1], 
      holding image-size instance masks.
  """
  num_boxes = tf.shape(masks)[0]

  def non_empty_masks(masks):
    masks = tf.expand_dims(masks, axis=3)
    
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=-1) 
    height = ymax - ymin
    width = xmax - xmin

    reframed_boxes = (tf.stack([-ymin, -xmin, 1 - ymin, 1 - xmin ], axis=1) / 
        tf.stack([height, width, height, width], axis=1))

    return tf.image.crop_and_resize(
        image=masks, 
        boxes=reframed_boxes, 
        box_ind=tf.range(num_boxes), 
        crop_size=[image_height, image_width], 
        extrapolation_value=0.0)

  def empty_masks():
    return tf.zeros([0, image_height, image_width, 1], dtype=tf.float32) 

  masks = tf.squeeze(tf.cond(num_boxes > 0, 
      lambda: non_empty_masks(masks), empty_masks), axis=3)
  return masks

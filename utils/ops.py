# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A module for helper tensorflow ops."""

import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.utils import shape_utils

slim = tf.contrib.slim


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


def expanded_shape(orig_shape, start_dim, num_dims):
  """Inserts multiple ones into a shape vector.

  Inserts an all-1 vector of length num_dims at position start_dim into a shape.
  Can be combined with tf.reshape to generalize tf.expand_dims.

  Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
  Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
  """
  with tf.name_scope('ExpandedShape'):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


def normalized_to_image_coordinates(normalized_boxes, image_shape,
                                    parallel_iterations=32):
  """Converts a batch of boxes from normal to image coordinates.

  Args:
    normalized_boxes: a float32 tensor of shape [None, num_boxes, 4] in
      normalized coordinates.
    image_shape: a float32 tensor of shape [4] containing the image shape.
    parallel_iterations: parallelism for the map_fn op.

  Returns:
    absolute_boxes: a float32 tensor of shape [None, num_boxes, 4] containg the
      boxes in image coordinates.
  """
  def _to_absolute_coordinates(normalized_boxes):
    return box_list_ops.to_absolute_coordinates(
        box_list.BoxList(normalized_boxes),
        image_shape[1], image_shape[2], check_range=False).get()

  absolute_boxes = shape_utils.static_or_dynamic_map_fn(
      _to_absolute_coordinates,
      elems=(normalized_boxes),
      dtype=tf.float32,
      parallel_iterations=parallel_iterations,
      back_prop=True)
  return absolute_boxes


def meshgrid(x, y):
  """Tiles the contents of x and y into a pair of grids.

  Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
  are vectors. Generally, this will give:

  xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
  ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)

  Keep in mind that the order of the arguments and outputs is reverse relative
  to the order of the indices they go into, done for compatibility with numpy.
  The output tensors have the same shapes.  Specifically:

  xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

  Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
  Returns:
    A tuple of tensors (xgrid, ygrid).
  """
  with tf.name_scope('Meshgrid'):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid


def padded_one_hot_encoding(indices, depth, left_pad):
  """Returns a zero padded one-hot tensor.

  This function converts a sparse representation of indices (e.g., [4]) to a
  zero padded one-hot representation (e.g., [0, 0, 0, 0, 1] with depth = 4 and
  left_pad = 1). If `indices` is empty, the result will simply be a tensor of
  shape (0, depth + left_pad). If depth = 0, then this function just returns
  `None`.

  Args:
    indices: an integer tensor of shape [num_indices].
    depth: depth for the one-hot tensor (integer).
    left_pad: number of zeros to left pad the one-hot tensor with (integer).

  Returns:
    padded_onehot: a tensor with shape (num_indices, depth + left_pad). Returns
      `None` if the depth is zero.

  Raises:
    ValueError: if `indices` does not have rank 1 or if `left_pad` or `depth are
      either negative or non-integers.

  TODO(rathodv): add runtime checks for depth and indices.
  """
  if depth < 0 or not isinstance(depth, six.integer_types):
    raise ValueError('`depth` must be a non-negative integer.')
  if left_pad < 0 or not isinstance(left_pad, six.integer_types):
    raise ValueError('`left_pad` must be a non-negative integer.')
  if depth == 0:
    return None

  rank = len(indices.get_shape().as_list())
  if rank != 1:
    raise ValueError('`indices` must have rank 1, but has rank=%s' % rank)

  def one_hot_and_pad():
    one_hot = tf.cast(tf.one_hot(tf.cast(indices, tf.int64), depth,
                                 on_value=1, off_value=0), tf.float32)
    return tf.pad(one_hot, [[0, 0], [left_pad, 0]], mode='CONSTANT')
  result = tf.cond(tf.greater(tf.size(indices), 0), one_hot_and_pad,
                   lambda: tf.zeros((depth + left_pad, 0)))
  return tf.reshape(result, [-1, depth + left_pad])


def dense_to_sparse_boxes(dense_locations, dense_num_boxes, num_classes):
  """Converts bounding boxes from dense to sparse form.

  Args:
    dense_locations:  a [max_num_boxes, 4] tensor in which only the first k rows
      are valid bounding box location coordinates, where k is the sum of
      elements in dense_num_boxes.
    dense_num_boxes: a [max_num_classes] tensor indicating the counts of
       various bounding box classes e.g. [1, 0, 0, 2] means that the first
       bounding box is of class 0 and the second and third bounding boxes are
       of class 3. The sum of elements in this tensor is the number of valid
       bounding boxes.
    num_classes: number of classes

  Returns:
    box_locations: a [num_boxes, 4] tensor containing only valid bounding
       boxes (i.e. the first num_boxes rows of dense_locations)
    box_classes: a [num_boxes] tensor containing the classes of each bounding
       box (e.g. dense_num_boxes = [1, 0, 0, 2] => box_classes = [0, 3, 3]
  """

  num_valid_boxes = tf.reduce_sum(dense_num_boxes)
  box_locations = tf.slice(dense_locations,
                           tf.constant([0, 0]), tf.stack([num_valid_boxes, 4]))
  tiled_classes = [tf.tile([i], tf.expand_dims(dense_num_boxes[i], 0))
                   for i in range(num_classes)]
  box_classes = tf.concat(tiled_classes, 0)
  box_locations.set_shape([None, 4])
  return box_locations, box_classes


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])


def reduce_sum_trailing_dimensions(tensor, ndims):
  """Computes sum across all dimensions following first `ndims` dimensions."""
  return tf.reduce_sum(tensor, axis=tuple(range(ndims, tensor.shape.ndims)))


def replace_nan_groundtruth_label_scores_with_ones(label_scores):
  """Replaces nan label scores with 1.0.

  Args:
    label_scores: a tensor containing object annoation label scores.

  Returns:
    a tensor where NaN label scores have been replaced by ones.
  """
  return tf.where(
      tf.is_nan(label_scores), tf.ones(tf.shape(label_scores)), label_scores)


def position_sensitive_crop_regions(image,
                                    boxes,
                                    box_ind,
                                    crop_size,
                                    num_spatial_bins,
                                    global_pool,
                                    extrapolation_value=None):
  """Position-sensitive crop and pool rectangular regions from a feature grid.

  The output crops are split into `spatial_bins_y` vertical bins
  and `spatial_bins_x` horizontal bins. For each intersection of a vertical
  and a horizontal bin the output values are gathered by performing
  `tf.image.crop_and_resize` (bilinear resampling) on a a separate subset of
  channels of the image. This reduces `depth` by a factor of
  `(spatial_bins_y * spatial_bins_x)`.

  When global_pool is True, this function implements a differentiable version
  of position-sensitive RoI pooling used in
  [R-FCN detection system](https://arxiv.org/abs/1605.06409).

  When global_pool is False, this function implements a differentiable version
  of position-sensitive assembling operation used in
  [instance FCN](https://arxiv.org/abs/1603.08678).

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    box_ind:  A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers
      to.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    num_spatial_bins: A list of two integers `[spatial_bins_y, spatial_bins_x]`.
      Represents the number of position-sensitive bins in y and x directions.
      Both values should be >= 1. `crop_height` should be divisible by
      `spatial_bins_y`, and similarly for width.
      The number of image channels should be divisible by
      (spatial_bins_y * spatial_bins_x).
      Suggested value from R-FCN paper: [3, 3].
    global_pool: A boolean variable.
      If True, we perform average global pooling on the features assembled from
        the position-sensitive score maps.
      If False, we keep the position-pooled features without global pooling
        over the spatial coordinates.
      Note that using global_pool=True is equivalent to but more efficient than
        running the function with global_pool=False and then performing global
        average pooling.
    extrapolation_value: An optional `float`. Defaults to `0`.
      Value used for extrapolation, when applicable.
  Returns:
    position_sensitive_features: A 4-D tensor of shape
      `[num_boxes, K, K, crop_channels]`,
      where `crop_channels = depth / (spatial_bins_y * spatial_bins_x)`,
      where K = 1 when global_pool is True (Average-pooled cropped regions),
      and K = crop_size when global_pool is False.
  Raises:
    ValueError: Raised in four situations:
      `num_spatial_bins` is not >= 1;
      `num_spatial_bins` does not divide `crop_size`;
      `(spatial_bins_y*spatial_bins_x)` does not divide `depth`;
      `bin_crop_size` is not square when global_pool=False due to the
        constraint in function space_to_depth.
  """
  total_bins = 1
  bin_crop_size = []

  for (num_bins, crop_dim) in zip(num_spatial_bins, crop_size):
    if num_bins < 1:
      raise ValueError('num_spatial_bins should be >= 1')

    if crop_dim % num_bins != 0:
      raise ValueError('crop_size should be divisible by num_spatial_bins')

    total_bins *= num_bins
    bin_crop_size.append(crop_dim // num_bins)

  if not global_pool and bin_crop_size[0] != bin_crop_size[1]:
    raise ValueError('Only support square bin crop size for now.')

  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
  spatial_bins_y, spatial_bins_x = num_spatial_bins

  # Split each box into spatial_bins_y * spatial_bins_x bins.
  position_sensitive_boxes = []
  for bin_y in range(spatial_bins_y):
    step_y = (ymax - ymin) / spatial_bins_y
    for bin_x in range(spatial_bins_x):
      step_x = (xmax - xmin) / spatial_bins_x
      box_coordinates = [ymin + bin_y * step_y,
                         xmin + bin_x * step_x,
                         ymin + (bin_y + 1) * step_y,
                         xmin + (bin_x + 1) * step_x,
                        ]
      position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

  image_splits = tf.split(value=image, num_or_size_splits=total_bins, axis=3)

  image_crops = []
  for (split, box) in zip(image_splits, position_sensitive_boxes):
    crop = tf.image.crop_and_resize(split, box, box_ind, bin_crop_size,
                                    extrapolation_value=extrapolation_value)
    image_crops.append(crop)

  if global_pool:
    # Average over all bins.
    position_sensitive_features = tf.add_n(image_crops) / len(image_crops)
    # Then average over spatial positions within the bins.
    position_sensitive_features = tf.reduce_mean(
        position_sensitive_features, [1, 2], keep_dims=True)
  else:
    # Reorder height/width to depth channel.
    block_size = bin_crop_size[0]
    if block_size >= 2:
      image_crops = [tf.space_to_depth(
          crop, block_size=block_size) for crop in image_crops]

    # Pack image_crops so that first dimension is for position-senstive boxes.
    position_sensitive_features = tf.stack(image_crops, axis=0)

    # Unroll the position-sensitive boxes to spatial positions.
    position_sensitive_features = tf.squeeze(
        tf.batch_to_space_nd(position_sensitive_features,
                             block_shape=[1] + num_spatial_bins,
                             crops=tf.zeros((3, 2), dtype=tf.int32)),
        squeeze_dims=[0])

    # Reorder back the depth channel.
    if block_size >= 2:
      position_sensitive_features = tf.depth_to_space(
          position_sensitive_features, block_size=block_size)

  return position_sensitive_features


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.

  TODO(rathodv, jonathanhuang): enable sparse matmul option.

  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  with tf.name_scope(scope, 'MatMulGather'):
    params_shape = shape_utils.combined_static_and_dynamic_shape(params)
    indices_shape = shape_utils.combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))


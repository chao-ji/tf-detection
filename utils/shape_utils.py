
import tensorflow as tf


def _is_tensor(t):
  """Returns a boolean indicating whether the input is a tensor.

  Args:
    t: the input to be tested.

  Returns:
    a boolean that indicates whether t is a tensor.
  """
  return isinstance(t, (tf.Tensor, tf.SparseTensor, tf.Variable))


def _set_dim_0(t, d0):
  """Sets the 0-th dimension of the input tensor.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    d0: an integer indicating the 0-th dimension of the input tensor.

  Returns:
    the tensor t with the 0-th dimension set.
  """
  t_shape = t.get_shape().as_list()
  t_shape[0] = d0
  t.set_shape(t_shape)
  return t


def pad_tensor(tensor, size):
  """Pad input tensor with zeors along the 0th dimension up to the `size`.
    
  `size` must be `>= shape(tensor)[0]`.

  Args:
    tensor: input tensor with rank >= 1.
    size: int scalar or int scalar tensor, the desired size of 0th dimension
      of the padded tensor.

  Returns:
    padded: tensor of shape [size, ...], the padded tensor.  
  """
  shape = tf.shape(tensor)
  pad_shape_dim0 = tf.expand_dims(size - shape[0], 0)
  pad_shape = tf.cond(tf.greater(tf.rank(tensor), 1), 
      lambda: tf.concat([pad_shape_dim0, shape[1:]], 0),
      lambda: pad_shape_dim0)
  padded = tf.concat([tensor, tf.zeros(pad_shape, dtype=tensor.dtype)], 0)
  if not _is_tensor(size):
    padded = _set_dim_0(padded, size)
  return padded


def clip_tensor(t, length):
  """Clips the input tensor along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after clipping, assuming length <= t.shape[0].

  Returns:
    clipped_t: the clipped tensor, whose first dimension is length. If the
      length is an integer, the first dimension of clipped_t is set to length
      statically.
  """
  clipped_t = tf.gather(t, tf.range(length))
  if not _is_tensor(length):
    clipped_t = _set_dim_0(clipped_t, length)
  return clipped_t


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after processing.

  Returns:
    processed_t: the processed tensor, whose first dimension is length. If the
      length is an integer, the first dimension of the processed tensor is set
      to length statically.
  """
  processed_t = tf.cond(
      tf.greater(tf.shape(t)[0], length),
      lambda: clip_tensor(t, length),
      lambda: pad_tensor(t, length))
  if not _is_tensor(length):
    processed_t = _set_dim_0(processed_t, length)
  return processed_t


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def static_map_fn(fn, elems):
  """Runs map_fn as a (static) for loop when possible.
  """
  elems = [tf.unstack(elem) for elem in elems]
  arg_tuples = list(zip(*elems))
  outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  stacked_outputs = [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  return stacked_outputs if len(stacked_outputs) > 1 else stacked_outputs[0]


def get_feature_map_spatial_dims(feature_map_list):
  """Get feature map spatial dimensions of a list of feature map tensors.

  Args: 
    feature_map_list: a list of feature map tensors of shape 
      [batch_size, height, width, channels]

  Returns:
    a list of int 2-tuples containing (height, width).
  """
  feature_map_shape_list = [tuple(combined_static_and_dynamic_shape(
      feature_map)[1:3]) for feature_map in feature_map_list]
  return feature_map_shape_list

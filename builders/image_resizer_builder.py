import functools

import tensorflow as tf

from detection.data import preprocessor
from detection.protos import image_resizer_pb2


RESIZE_METHOD = {
    image_resizer_pb2.BILINEAR:
        tf.image.ResizeMethod.BILINEAR,
    image_resizer_pb2.NEAREST_NEIGHBOR:
        tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    image_resizer_pb2.BICUBIC:
        tf.image.ResizeMethod.BICUBIC,
    image_resizer_pb2.AREA:
        tf.image.ResizeMethod.AREA
}


def build(config):
  """Factory function that builds the image resizer function.

  Args:
    config: a protobuf message storing ImageResizer configurations.

  Returns:
    image_resizer_fn: a callable that takes as input a 3-D float tensor `image`
      and returns 3-D float tensor holding the resized image.
  """
  if not isinstance(config, image_resizer_pb2.ImageResizer):
    raise ValueError('config must be an instance of ImageResizer message.')

  image_resizer_type = config.WhichOneof('image_resizer_oneof') 
  if image_resizer_type == 'fixed_shape_resizer':
    config = config.fixed_shape_resizer
    if config.resize_method not in RESIZE_METHOD:
      raise ValueError('Unknown resize_method: {}'.format(config.resize_method)) 
    method = RESIZE_METHOD[config.resize_method]
    image_resizer_fn = functools.partial(
        preprocessor.resize_image,
        new_height=config.height,
        new_width=config.width,
        method=method)
    return image_resizer_fn
  elif image_resizer_type == 'keep_aspect_ratio_resizer':
    config = config.keep_aspect_ratio_resizer
    if config.resize_method not in RESIZE_METHOD:
      raise ValueError('Unknown resize_method: {}'.format(config.resize_method))
    method = RESIZE_METHOD[config.resize_method]
    if config.min_dimension > config.max_dimension:
      raise ValueError(
          '`min_dimension` must not be greather than `max_dimension`')
    image_resizer_fn = functools.partial(
      preprocessor.rescale_image,
      min_dimension=config.min_dimension,
      max_dimension=config.max_dimension,
      method=method)
    return image_resizer_fn
  raise ValueError('Unknown image resizer: {}'.format(image_resizer_type))


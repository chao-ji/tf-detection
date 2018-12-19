"""Builds data preprocessor for doing data augmentation.

A data preprocessor is built by chaining a sequence of preprocessor functions
in a pipeline, where each function is specified by a 2-tuple containing a 
callable a and dict holding keyword-to-argument mapping for the callable.

Each preprocessing function has a separate builder (`def _build_*`) that returns 
the 2-tuple.
"""
import tensorflow as tf

from detection.data import preprocessor
from detection.protos import preprocessor_pb2

RESIZE_METHOD_MAP = {
    preprocessor_pb2.ResizeImage.AREA: tf.image.ResizeMethod.AREA,
    preprocessor_pb2.ResizeImage.BICUBIC: tf.image.ResizeMethod.BICUBIC,
    preprocessor_pb2.ResizeImage.BILINEAR: tf.image.ResizeMethod.BILINEAR,
    preprocessor_pb2.ResizeImage.NEAREST_NEIGHBOR: (
        tf.image.ResizeMethod.NEAREST_NEIGHBOR),
}


def build(config):
  """Builds data preprocessor.

  Args:
    config: a protobuf message storing Preprocessor configurations.

  Returns:
    an instance of DataPreprocessor or None.
  """
  if not isinstance(config, preprocessor_pb2.Preprocessor):
    raise ValueError('config must be an instance of Preprocessor message.')

  kwargs_map_list = []
  for option in config.data_augmentation_options:
    step_type = option.WhichOneof('preprocessing_step_oneof')

    if step_type == 'random_horizontal_flip':
      kwargs_map_list.append(
          _build_random_horizontal_flip(option.random_horizontal_flip))
    elif step_type == 'random_crop_image':
      kwargs_map_list.append(
          _build_random_crop_image(option.random_crop_image))
    elif step_type == 'ssd_random_crop':
      kwargs_map_list.append(
          _build_ssd_random_crop(option.ssd_random_crop))
    elif step_type == 'resize_image':
      kwargs_map_list.append(
          _build_resize_image(option.resize_image))
    else:
      raise ValueError('Unknown option: {}'.format(step_type))

  if kwargs_map_list:
    return preprocessor.DataPreprocessor(kwargs_map_list)
  else:
    return None


def _build_random_horizontal_flip(config):
  """Returns a 2-tuple containing a callable performing random horizontal flip
  and a dummy dict holding the empty keyword-to-argument mapping.
  """
  kwargs_map = {}
  return preprocessor.random_horizontal_flip, kwargs_map


def _build_random_crop_image(config):
  """Builds random crop image function.

  Args:
    config: a protobuf message storing RandomCropImage configurations.

  Returns:
    a 2-tuple containing a callable performing random image crop and a dict 
      holding the keyword-to-argument mapping.
  """
  if not isinstance(config, preprocessor_pb2.RandomCropImage):
    raise ValueError('config must be an instance of RandomCropImage message.')

  kwargs_map = {'min_object_covered': config.min_object_covered,
                'aspect_ratio_range': (config.min_aspect_ratio,
                                       config.max_aspect_ratio),
                'area_range': (config.min_area,
                               config.max_area),
                'overlap_thresh': config.overlap_thresh,
                'prob_to_keep_original': config.prob_to_keep_original}
  return preprocessor.random_crop_image, kwargs_map


def _build_ssd_random_crop(config):
  """Builds ssd random image crop function.

  Args:
    config: a protobuf message storing SSDRandomCrop configurations.

  Returns:
    a 2-tuple containing a callable performing ssd random image crop and a dict
      holding keyword-to-argument mapping.
  """
  if not isinstance(config, preprocessor_pb2.SSDRandomCrop):
    raise ValueError('config must be an instance of SSDRandomCrop message.')

  kwargs_map = {}
  if config.operations:
    min_object_covered = [op.min_object_covered for op in config.operations]
    aspect_ratio_range = [(op.min_aspect_ratio, op.max_aspect_ratio)
                          for op in config.operations]
    area_range = [(op.min_area, op.max_area) for op in config.operations]
    overlap_thresh = [op.overlap_thresh for op in config.operations]
    prob_to_keep_original = [op.prob_to_keep_original
                             for op in config.operations]

    kwargs_map = {'min_object_covered': min_object_covered,
                  'aspect_ratio_range': aspect_ratio_range,
                  'area_range': area_range,
                  'overlap_thresh': overlap_thresh,
                  'prob_to_keep_original': prob_to_keep_original}
  return preprocessor.ssd_random_crop, kwargs_map


def _build_resize_image(config):
  """Builds image resize function.

  Args:
    config: a protobuf message storing ResizeImage configurations.

  Returns:
    a 2-tuple containing a callable performing image resizing and a dict holding
      keyword-to-argument mapping.
  """
  if not isinstance(config, preprocessor_pb2.ResizeImage):
    raise ValueError('config')

  kwargs_map = {'new_height': config.new_height,
                'new_width': config.new_width,
                'method': RESIZE_METHOD_MAP[config.method]}
  return preprocessor.resize_image, kwargs_map


def build_normalizer_fn(config):
  """Factory function that builds image normalizer function.
 
  Args:
    config: a protobuf message storing NormalizerRange configurations. 

  Returns: 
    normalizer_fn: a callable that takes in a float tensor `image` and returns
      float tensor of same shape and type in which values are mapped to 
      [low, high].
  """
  if not isinstance(config, preprocessor_pb2.Normalizer):
    raise ValueError('config must be an instance of Normalizer message.')

  if config.type == preprocessor_pb2.Normalizer.RANGE:
    high, low = config.high, config.low
    normalizer_fn = lambda image: ((high - low) / 255.0) * image + low
    return normalizer_fn
  elif config.type == preprocessor_pb2.Normalizer.SUBTRACT_MEAN:
    r_mean, g_mean, b_mean = config.r_mean, config.g_mean, config.b_mean
    normalizer_fn = lambda image: image - (r_mean, g_mean, b_mean)
  else:
    raise ValueError('Unknown normalizer type.')

  return normalizer_fn

import tensorflow as tf

from detection.preprocess import preprocessor
from detection.protos import preprocessor_pb2

RESIZE_METHOD_MAP = {
    preprocessor_pb2.ResizeImage.AREA: tf.image.ResizeMethod.AREA,
    preprocessor_pb2.ResizeImage.BICUBIC: tf.image.ResizeMethod.BICUBIC,
    preprocessor_pb2.ResizeImage.BILINEAR: tf.image.ResizeMethod.BILINEAR,
    preprocessor_pb2.ResizeImage.NEAREST_NEIGHBOR: (
        tf.image.ResizeMethod.NEAREST_NEIGHBOR),
}


def build(config):
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
  kwargs_map = {}
  return preprocessor.random_horizontal_flip, kwargs_map

def _build_random_crop_image(config):
  kwargs_map = {'min_object_covered': config.min_object_covered,
                'aspect_ratio_range': (config.min_aspect_ratio,
                                       config.max_aspect_ratio),
                'area_range': (config.min_area,
                               config.max_area),
                'overlap_thresh': config.overlap_thresh,
                'prob_to_keep_original': config.prob_to_keep_original}
  return preprocessor.random_crop_image, kwargs_map

def _build_ssd_random_crop(config):
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
  kwargs_map = {'new_height': config.new_height,
                'new_width': config.new_width,
                'method': RESIZE_METHOD_MAP[config.method]}
  return preprocessor.resize_image, kwargs_map

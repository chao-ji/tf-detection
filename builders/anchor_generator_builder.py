from detection.core.anchor_generators import multiple_grid_anchor_generator
from detection.core.anchor_generators import grid_anchor_generator
from detection.protos import anchor_generator_pb2


def build(config):
  """Builds an AnchorGenerator instance.

  Args:
    config: a protobuf message storing AnchorGenerator configurations.  

  Returns:
    an AnchorGenerator instance. 
  """
  if not isinstance(config, anchor_generator_pb2.AnchorGenerator):
    raise ValueError('config must be an instance of AnchorGenerator message.')

  if config.WhichOneof(
      'anchor_generator_oneof') == 'ssd_anchor_generator':
    config = config.ssd_anchor_generator
    anchor_strides = None
    if config.height_stride:
      anchor_strides = zip(config.height_stride, config.width_stride)
    anchor_offsets = None
    if config.height_offset:
      anchor_offsets = zip(config.height_offset, config.width_offset)

    return multiple_grid_anchor_generator.create_ssd_anchors(
        num_layers=config.num_layers,
        min_scale=config.min_scale,
        max_scale=config.max_scale,
        lowest_scale=config.lowest_scale,
        scales=[float(scale) for scale in config.scales],
        aspect_ratios=config.aspect_ratios,
        interpolated_scale_aspect_ratio=config.interpolated_scale_aspect_ratio,
        reduce_boxes_in_lowest_layer=config.reduce_boxes_in_lowest_layer)

  elif config.WhichOneof(
      'anchor_generator_oneof') == 'grid_anchor_generator':
    config = config.grid_anchor_generator 
    return grid_anchor_generator.GridAnchorGenerator(
        scale_list=tuple(config.scale_list),
        aspect_ratio_list=tuple(config.aspect_ratio_list),
        base_anchor_size=(config.height, config.width),
        anchor_stride=(config.height_stride, config.width_stride),
        anchor_offset=(config.height_offset, config.width_offset))

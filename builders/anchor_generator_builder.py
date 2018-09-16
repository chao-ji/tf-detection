
from detection.anchor_generators import multiple_grid_anchor_generator
from detection.protos import anchor_generator_pb2


def build(config):

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
        scales=[float(scale) for scale in config.scales],
        aspect_ratios=config.aspect_ratios,
        interpolated_scale_aspect_ratio=config.interpolated_scale_aspect_ratio,
        base_anchor_size=(config.base_anchor_height, config.base_anchor_width),
        anchor_strides=anchor_strides,
        anchor_offsets=anchor_offsets,
        reduce_boxes_in_lowest_layer=config.reduce_boxes_in_lowest_layer)


from detection.box_coders import faster_rcnn_box_coder
from detection.protos import box_coder_pb2

def build(config):
  if config.WhichOneof('box_coder_oneof') == 'faster_rcnn_box_coder':
    config = config.faster_rcnn_box_coder
    return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[
        config.y_scale, config.x_scale, config.height_scale, config.width_scale
    ])

  raise ValueError('Unknown box coder')


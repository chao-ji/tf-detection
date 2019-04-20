from detection.protos import mask_predictor_pb2
from detection.core import mask_predictors
from detection.builders import conv_hyperparams_builder


def build(config, conv_hyperparams_fn=None):
  if not isinstance(config, mask_predictor_pb2.MaskPredictor):
    raise ValueError('config must be an instance of MaskPredictor message.')

  if config.WhichOneof('mask_predictor_oneof') == 'rcnn_mask_predictor':
    return _build_rcnn_mask_predictor(
        config.rcnn_mask_predictor, conv_hyperparams_fn)

  raise ValueError('Unknown mask predictor.')


def _build_rcnn_mask_predictor(config, conv_hyperparams_fn=None):
  if not isinstance(config, mask_predictor_pb2.RcnnMaskPredictor):
    raise ValueError('config must be an instance of RcnnMaskPredictor message.')

  if conv_hyperparams_fn is None:
    conv_hyperparams_fn = conv_hyperparams_builder.build(
        config.conv_hyperparams)
    
  predictor = mask_predictors.RcnnMaskPredictor(
      conv_hyperparams_fn=conv_hyperparams_fn,
      num_masks=config.num_masks,
      mask_height=config.mask_height,
      mask_width=config.mask_width,
      num_conv_layers=config.num_conv_layers,
      depths=config.depths)
  return predictor


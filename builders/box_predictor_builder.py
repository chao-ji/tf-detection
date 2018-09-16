from detection.protos import box_predictor_pb2
from detection.core import box_predictors
from detection.builders import conv_hyperparams_builder 

def build(config, num_classes, anchor_generator=None, conv_hyperparams_fn=None):
  if config.WhichOneof('box_predictor_oneof') == 'convolutional_box_predictor':
    return _build_convolutional_box_predictor(
        config.convolutional_box_predictor,
        num_classes,
        anchor_generator,
        conv_hyperparams_fn)

  raise ValueError('Unknown box predictor.')


def _build_convolutional_box_predictor(config,
                                       num_classes,
                                       anchor_generator=None,
                                       conv_hyperparams_fn=None):

  if conv_hyperparams_fn is None:
    conv_hyperparams_fn = conv_hyperparams_builder.build(
        config.conv_hyperparams)

  if anchor_generator is None:
    num_predictions_list = [i for i in config.num_predictions]
  else:
    num_predictions_list = anchor_generator.num_anchors_per_location()

  predictor = box_predictors.ConvolutionalBoxPredictor(
      num_classes=num_classes,
      num_predictions_list=num_predictions_list,
      conv_hyperparams_fn=conv_hyperparams_fn,
      kernel_size=config.kernel_size,
      box_code_size=config.box_code_size,
      use_depthwise=config.use_depthwise
  )

  return predictor

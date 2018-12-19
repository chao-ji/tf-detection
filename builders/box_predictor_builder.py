from detection.protos import box_predictor_pb2
from detection.core import box_predictors
from detection.builders import conv_hyperparams_builder 


def build(config, num_classes, anchor_generator=None, conv_hyperparams_fn=None):
  """Builds box predictor. You can optionally pass in the global 
  `anchor_generator` and `conv_hyperparams_fn` used in the feature extractor. 
  Otherwise, those specific to box predictor will be built.

  Args:
    config: an instance of BoxPredictor message.
    num_classes: int scalar, num of classes.
    anchor_generator: an instance of AnchorGenerator or None.
    conv_hyperparams_fn: a callable that returns arg_scope for building feature 
      extractor or None. 

  Returns:
    an instance of BoxPredictor.
  """
  if not isinstance(config, box_predictor_pb2.BoxPredictor):
    raise ValueError('config must be an instance of BoxPredictor message.')

  if config.WhichOneof('box_predictor_oneof') == 'convolutional_box_predictor':
    return _build_convolutional_box_predictor(
        config.convolutional_box_predictor,
        num_classes,
        anchor_generator,
        conv_hyperparams_fn)
  if config.WhichOneof('box_predictor_oneof') == 'rcnn_box_predictor':
    return _build_rcnn_box_predictor(config.rcnn_box_predictor, num_classes)

  raise ValueError('Unknown box predictor.')


def _build_convolutional_box_predictor(config,
                                       num_classes,
                                       anchor_generator=None,
                                       conv_hyperparams_fn=None):
  """Builds convolutional box predictor.

  Args:
    config: an instance of ConvolutionalBoxPredictor message.
    num_classes: int scalar, num of classes.
    anchor_generator: an instance of AnchorGenerator or None.
    conv_hyperparams_fn: a callable that returns arg_scope for building feature 
      extractor or None.

  Returns:
    predictor: an instance of ConvolutionalBoxPredictor.
  """
  if not isinstance(config, box_predictor_pb2.ConvolutionalBoxPredictor):
    raise ValueError('config must be an instance of ConvolutionalBoxPredictor '
        'message.')

  # Use conv hyperparam settings specific to box predictor
  if conv_hyperparams_fn is None:
    conv_hyperparams_fn = conv_hyperparams_builder.build(
        config.conv_hyperparams)

  # Use anchor generator specific to box predictor
  if anchor_generator is None:
    num_predictions_list = [i for i in config.num_predictions]
  else:
    num_predictions_list = anchor_generator.num_anchors_per_location

  predictor = box_predictors.ConvolutionalBoxPredictor(
      num_classes=num_classes,
      num_predictions_list=num_predictions_list,
      conv_hyperparams_fn=conv_hyperparams_fn,
      kernel_size=config.kernel_size,
      box_code_size=config.box_code_size,
      use_depthwise=config.use_depthwise
  )

  return predictor


def _build_rcnn_box_predictor(config, num_classes):
  """Builds RCNN box predictor.

  Args:
    config: an instance of RcnnBoxPredictor message.
    num_classes: int scalar, num of classes.

  Returns:
    predictor: an instance of RcnnBoxPredictor.
  """
  if not isinstance(config, box_predictor_pb2.RcnnBoxPredictor):
    raise ValueError('config must be an instance of RcnnBoxPredictor message.')

  fc_hyperparams_fn = conv_hyperparams_builder.build(config.fc_hyperparams)
  predictor = box_predictors.RcnnBoxPredictor(
      num_classes=num_classes,
      box_code_size=config.box_code_size,
      fc_hyperparams_fn=fc_hyperparams_fn)
  return predictor
  

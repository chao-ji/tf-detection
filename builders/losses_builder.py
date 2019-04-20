from detection.core import losses
from detection.protos import losses_pb2


def build(config):
  """Builds localization loss, classification loss and optionally 
  hard example miner.

  Args:
    config: a protobuf message storing Loss configurations.

  Returns:
    localization_loss_fn: a callable that computes localization loss.
    classification_loss_fn: a callable that computes classification loss.
    mask_loss_fn: a callable that computes mask loss.
    localization_loss_weight: float scalar, scales the contribution of 
        localization loss relative to classification loss.
    classification_loss_weight: float scalar, scales the contribution of
        classification loss relative to localization loss.
    mask_loss_weight: float scalar, scales the contribution of mask loss
        relative to localization loss and classification loss.
    hard_example_miner: a callable that performs hard example mining such
        that gradient is backpropagated to high-loss anchorwise predictions.
  """
  if not isinstance(config, losses_pb2.Loss):
    raise ValueError('config must be an instance of Loss message.')

  localization_loss_fn = _build_localization_loss(
      config.localization_loss)
  classification_loss_fn = _build_classification_loss(
      config.classification_loss)
  mask_loss_fn = None
  if config.HasField('mask_loss'):
    mask_loss_fn = _build_classification_loss(
        config.mask_loss)

  localization_loss_weight = config.localization_weight
  classification_loss_weight = config.classification_weight
  mask_loss_weight = None
  if config.HasField('mask_weight'):
    mask_loss_weight = config.mask_weight

  hard_example_miner = None
  if config.HasField('hard_example_miner'):
    hard_example_miner = build_hard_example_miner(
        config.hard_example_miner,
        classification_loss_weight,
        localization_loss_weight)

  return (localization_loss_fn, classification_loss_fn, mask_loss_fn, 
      localization_loss_weight, classification_loss_weight, mask_loss_weight, 
      hard_example_miner)


def _build_classification_loss(config):
  """Builds classification loss.

  Args:
    config: a protobuf message storing ClassificationLoss configurations.

  Returns:
    an instance of ClassificationLoss.
  """
  if not isinstance(config, losses_pb2.ClassificationLoss):
    raise ValueError('config must be an instance of ClassificationLoss '
        'message.')
  loss_type = config.WhichOneof('classification_loss_oneof')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()

  if loss_type == 'weighted_softmax':
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.weighted_softmax.logit_scale)

  return ValueError('Unknown classification loss')


def _build_localization_loss(config):
  """Builds localization loss.

  Args:
    config: a protobuf message storing LocalizationLoss configurations.

  Returns:
    an instance of LocalizationLoss.
  """
  if not isinstance(config, losses_pb2.LocalizationLoss):
    raise ValueError('config must be an instance of LocalizationLoss message.')
  loss_type = config.WhichOneof('localization_loss_oneof')

  if loss_type == 'weighted_l2':
    return losses.WeightedL2LocalizationLoss()

  if loss_type == 'weighted_smooth_l1':
    return losses.WeightedSmoothL1LocalizationLoss(
        config.weighted_smooth_l1.delta)

  return ValueError('Unknown localization loss')


def build_hard_example_miner(config,
                             localization_weight,
                             classification_weight):
  """Builds hard example miner.

  Args:
    config: a protobuf message storing HardExampleMiner configurations.
    localization_weight: float scalar, localization loss weight.
    classification_weight: float scalar, classification loss weight.

  Returns:
    an instance of HardExampleMiner.
  """
  if not isinstance(config, losses_pb2.HardExampleMiner):
    raise ValueError('config must be an instance of HardExampleMinder message.')

  loss_type = None
  if config.loss_type == losses_pb2.HardExampleMiner.BOTH:
    loss_type = 'both'
  if config.loss_type == losses_pb2.HardExampleMiner.CLASSIFICATION:
    loss_type = 'cls'
  if config.loss_type == losses_pb2.HardExampleMiner.LOCALIZATION:
    loss_type = 'loc'

  max_negatives_per_positive = None
  num_hard_examples = None
  if config.max_negatives_per_positive > 0:
    max_negatives_per_positive = config.max_negatives_per_positive
  if config.num_hard_examples > 0:
    num_hard_examples = config.num_hard_examples
  hard_example_miner = losses.HardExampleMiner(
      num_hard_examples=num_hard_examples,
      iou_threshold=config.iou_threshold,
      loss_type=loss_type,
      cls_loss_weight=classification_weight,
      loc_loss_weight=localization_weight,
      max_negatives_per_positive=max_negatives_per_positive,
      min_negatives_per_image=config.min_negatives_per_image)
  return hard_example_miner


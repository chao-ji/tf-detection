from detection.core import losses
from detection.protos import losses_pb2


def build(config):
  classification_loss = _build_classification_loss(
      config.classification_loss)
  localization_loss = _build_localization_loss(
      config.localization_loss)

  classification_weight = config.classification_weight
  localization_weight = config.localization_weight

  hard_example_miner = None
  if config.HasField('hard_example_miner'):
    hard_example_miner = build_hard_example_miner(
        config.hard_example_miner,
        classification_weight,
        localization_weight)

  return (classification_loss, localization_loss, classification_weight,
       localization_weight, hard_example_miner)


def _build_classification_loss(config):
  loss_type = config.WhichOneof('classification_loss_oneof')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()

  if loss_type == 'weighted_softmax':
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.weighted_softmax.logit_scale)

  return ValueError('Unknown classification loss')


def _build_localization_loss(config):
  loss_type = config.WhichOneof('localization_loss_oneof')

  if loss_type == 'weighted_l2':
    return losses.WeightedL2LocalizationLoss()

  if loss_type == 'weighted_smooth_l1':
    return losses.WeightedSmoothL1LocalizationLoss(
        config.weighted_smooth_l1.delta)

  return ValueError('Unknown localization loss')


def build_hard_example_miner(config,
                             classification_weight,
                             localization_weight):
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

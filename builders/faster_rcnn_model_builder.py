import functools

from detection.builders import image_resizer_builder
from detection.builders import conv_hyperparams_builder
from detection.builders import anchor_generator_builder
from detection.builders import box_predictor_builder
from detection.builders import mask_predictor_builder
from detection.builders import box_coder_builder
from detection.builders import target_assigner_builder
from detection.builders import losses_builder
from detection.builders import optimizer_builder
from detection.builders import postprocessing_builder
from detection.builders import dataset_builder
from detection.builders.preprocessor_builder import build_normalizer_fn
from detection.builders.conv_hyperparams_builder import build_batch_norm_params

from detection.feature_extractors.faster_rcnn_inception_v2 import (
    FasterRcnnInceptionV2FeatureExtractor)
from detection.feature_extractors.faster_rcnn_resnet_v1 import (
    FasterRcnnResnet101V1FeatureExtractor)

from detection.faster_rcnn import trainer
from detection.faster_rcnn import evaluator
from detection.faster_rcnn import inferencer

from detection.core.standard_names import ModeKeys
from detection.utils import ops

from detection.protos import faster_rcnn_model_pb2
from detection.protos import dataset_pb2
from detection.protos import train_config_pb2


def build_faster_rcnn_train_session(model_config,
                                    dataset_config,
                                    train_config,
                                    num_classes):
  """Builds faster rcnn model, dataset, and optimizer for training.

  Args:
    model_config: a protobuf message storing FasterRcnnModel configuration.
    dataset_config: a protobuf message storing Dataset configuration.
    train_config: a protobuf message storing TrainConfig configuration.
    num_classes: int scalar, num of classes.

  Returns:
    model_trainer: an instance of FasterRcnnModelTrainer.
    dataset: an instance of TrainerDataset.
    optimizer_builder_fn: a callabel that (when called with no arguments)
      returns a 2-tuple containing an instance of optimizer and a learning rate
      tensor.
  """
  if not isinstance(model_config, faster_rcnn_model_pb2.FasterRcnnModel):
    raise ValueError('model_config must be an instance of FasterRcnnModel '
        'message.')
  if not isinstance(dataset_config, dataset_pb2.Dataset):
    raise ValueError('dataset_config must be an instance of Dataset message.')
  if not isinstance(train_config, train_config_pb2.TrainConfig):
    raise ValueError('train_config must be an instance of TrainConfig message.')

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
  normalizer_fn = build_normalizer_fn(model_config.normalizer)
  feature_extractor = build_feature_extractor(model_config.feature_extractor)
  box_coder = box_coder_builder.build(model_config.box_coder)
  rpn_anchor_generator = anchor_generator_builder.build(
      model_config.rpn_anchor_generator)
  rpn_box_predictor = box_predictor_builder.build(
      model_config.rpn_box_predictor,
      num_classes=1,
      anchor_generator=rpn_anchor_generator)
  frcnn_box_predictor = box_predictor_builder.build(
      model_config.frcnn_box_predictor,
      num_classes=num_classes)
  frcnn_mask_predictor = None
  if model_config.HasField('frcnn_mask_predictor'):
    frcnn_mask_predictor = mask_predictor_builder.build(
        model_config.frcnn_mask_predictor)
   
  rpn_target_assigner = target_assigner_builder.build(
      model_config.rpn_target_assigner,
      box_coder=box_coder)
  frcnn_target_assigner = target_assigner_builder.build(
      model_config.frcnn_target_assigner,
      box_coder=box_coder)
  rpn_minibatch_sampler_fn = functools.partial(
      ops.balanced_subsample,
      pos_frac=model_config.rpn_minibatch_positive_fraction)
  frcnn_minibatch_sampler_fn = functools.partial(
      ops.balanced_subsample,
      pos_frac=model_config.frcnn_minibatch_positive_fraction) 

  rpn_nms_fn, rpn_score_conversion_fn = postprocessing_builder.build(
      model_config.rpn_post_processing)

  ( rpn_localization_loss_fn,
    rpn_classification_loss_fn,
    _,
    rpn_localization_loss_weight,
    rpn_classification_loss_weight,
    _,
    _) = losses_builder.build(model_config.rpn_loss)

  ( frcnn_localization_loss_fn,
    frcnn_classification_loss_fn,
    frcnn_mask_loss_fn,
    frcnn_localization_loss_weight,
    frcnn_classification_loss_weight,
    frcnn_mask_loss_weight, 
    _) = losses_builder.build(model_config.frcnn_loss)

  proposal_crop_size=model_config.proposal_crop_size
  rpn_minibatch_size = model_config.rpn_minibatch_size
  frcnn_minibatch_size = model_config.frcnn_minibatch_size
  rpn_box_predictor_depth = model_config.rpn_box_predictor_depth
  first_stage_atrous_rate = model_config.first_stage_atrous_rate
  freeze_batch_norm = model_config.freeze_batch_norm
  gradient_clipping_by_norm = train_config.gradient_clipping_by_norm

  # faster rcnn model
  model_trainer = trainer.FasterRcnnModelTrainer(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      box_coder=box_coder,
      rpn_anchor_generator=rpn_anchor_generator,
      rpn_box_predictor=rpn_box_predictor,
      frcnn_box_predictor=frcnn_box_predictor,
      frcnn_mask_predictor=frcnn_mask_predictor,

      rpn_target_assigner=rpn_target_assigner,
      rpn_minibatch_sampler_fn=rpn_minibatch_sampler_fn,
      frcnn_target_assigner=frcnn_target_assigner,
      frcnn_minibatch_sampler_fn=frcnn_minibatch_sampler_fn,

      rpn_localization_loss_fn=rpn_localization_loss_fn,
      rpn_classification_loss_fn=rpn_classification_loss_fn,
      frcnn_localization_loss_fn=frcnn_localization_loss_fn,
      frcnn_classification_loss_fn=frcnn_classification_loss_fn,
      frcnn_mask_loss_fn=frcnn_mask_loss_fn,

      rpn_nms_fn=rpn_nms_fn,
      rpn_score_conversion_fn=rpn_score_conversion_fn,

      rpn_localization_loss_weight=rpn_localization_loss_weight,
      rpn_classification_loss_weight=rpn_classification_loss_weight,
      frcnn_localization_loss_weight=frcnn_localization_loss_weight,
      frcnn_classification_loss_weight=frcnn_classification_loss_weight,
      frcnn_mask_loss_weight=frcnn_mask_loss_weight,

      proposal_crop_size=proposal_crop_size,
      rpn_minibatch_size=rpn_minibatch_size,
      frcnn_minibatch_size=frcnn_minibatch_size,
      rpn_box_predictor_depth=rpn_box_predictor_depth,
      first_stage_atrous_rate=first_stage_atrous_rate,
      freeze_batch_norm=freeze_batch_norm, 
      gradient_clipping_by_norm=gradient_clipping_by_norm)
  # dataset
  dataset = dataset_builder.build(dataset_config, ModeKeys.train, num_classes)
  # optimizer
  optimizer_builder_fn = optimizer_builder.build(train_config.optimizer)

  return model_trainer, dataset, optimizer_builder_fn


def build_faster_rcnn_evaluate_session(model_config,
                                       dataset_config,
                                       num_classes):
  """Builds faster rcnn model and dataset for evaluation.

  Args:
    model_config: a protobuf message storing FasterRcnnModel configuration.
    dataset_config: a protobuf message storing Dataset configuration.
    num_classes: int scalar, num of classes.

  Returns:
    model_evaluator: an instance of FasterRcnnModelEvaluator.
    dataset: an instance of EvaluatorDataset.
  """
  if not isinstance(model_config, faster_rcnn_model_pb2.FasterRcnnModel):
    raise ValueError('model_config must be an instance of FasterRcnnModel '
        'message.')
  if not isinstance(dataset_config, dataset_pb2.Dataset):
    raise ValueError('dataset_config must be an instance of Dataset message.')

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
  normalizer_fn = build_normalizer_fn(model_config.normalizer)
  feature_extractor = build_feature_extractor(model_config.feature_extractor)
  box_coder = box_coder_builder.build(model_config.box_coder)
  rpn_anchor_generator = anchor_generator_builder.build(
      model_config.rpn_anchor_generator)
  rpn_box_predictor = box_predictor_builder.build(
      model_config.rpn_box_predictor,
      num_classes=1,
      anchor_generator=rpn_anchor_generator)
  frcnn_box_predictor = box_predictor_builder.build(
      model_config.frcnn_box_predictor,
      num_classes=num_classes)
  frcnn_mask_predictor = None
  if model_config.HasField('frcnn_mask_predictor'):
    frcnn_mask_predictor = mask_predictor_builder.build(
        model_config.frcnn_mask_predictor)

  rpn_target_assigner = target_assigner_builder.build(
      model_config.rpn_target_assigner,
      box_coder=box_coder)
  frcnn_target_assigner = target_assigner_builder.build(
      model_config.frcnn_target_assigner,
      box_coder=box_coder)
  frcnn_nms_fn, frcnn_score_conversion_fn = postprocessing_builder.build(
      model_config.frcnn_post_processing)
  rpn_minibatch_sampler_fn = functools.partial(
      ops.balanced_subsample,
      pos_frac=model_config.rpn_minibatch_positive_fraction)

  rpn_nms_fn, rpn_score_conversion_fn = postprocessing_builder.build(
      model_config.rpn_post_processing)

  ( rpn_localization_loss_fn,  
    rpn_classification_loss_fn,
    _,
    rpn_localization_loss_weight,  
    rpn_classification_loss_weight,
    _,
    _) = losses_builder.build(model_config.rpn_loss)

  ( frcnn_localization_loss_fn,  
    frcnn_classification_loss_fn,
    frcnn_mask_loss_fn,
    frcnn_localization_loss_weight,  
    frcnn_classification_loss_weight,
    frcnn_mask_loss_weight,
    _) = losses_builder.build(model_config.frcnn_loss)

  proposal_crop_size=model_config.proposal_crop_size  
  rpn_minibatch_size = model_config.rpn_minibatch_size
  rpn_box_predictor_depth = model_config.rpn_box_predictor_depth
  first_stage_atrous_rate = model_config.first_stage_atrous_rate

  # faster rcnn model
  model_evaluator = evaluator.FasterRcnnModelEvaluator(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      box_coder=box_coder,
      rpn_anchor_generator=rpn_anchor_generator,
      rpn_box_predictor=rpn_box_predictor,
      frcnn_box_predictor=frcnn_box_predictor,
      frcnn_mask_predictor=frcnn_mask_predictor,

      rpn_target_assigner=rpn_target_assigner,
      rpn_minibatch_sampler_fn=rpn_minibatch_sampler_fn,
      frcnn_target_assigner=frcnn_target_assigner,
      frcnn_score_conversion_fn=frcnn_score_conversion_fn,
      frcnn_nms_fn=frcnn_nms_fn,

      rpn_localization_loss_fn=rpn_localization_loss_fn,
      rpn_classification_loss_fn=rpn_classification_loss_fn,
      frcnn_localization_loss_fn=frcnn_localization_loss_fn,
      frcnn_classification_loss_fn=frcnn_classification_loss_fn,
      frcnn_mask_loss_fn=frcnn_mask_loss_fn,

      rpn_nms_fn=rpn_nms_fn,
      rpn_score_conversion_fn=rpn_score_conversion_fn,

      rpn_localization_loss_weight=rpn_localization_loss_weight,
      rpn_classification_loss_weight=rpn_classification_loss_weight,
      frcnn_localization_loss_weight=frcnn_localization_loss_weight,
      frcnn_classification_loss_weight=frcnn_classification_loss_weight,
      frcnn_mask_loss_weight=frcnn_mask_loss_weight,

      proposal_crop_size=proposal_crop_size,
      rpn_minibatch_size=rpn_minibatch_size,
      rpn_box_predictor_depth=rpn_box_predictor_depth,
      first_stage_atrous_rate=first_stage_atrous_rate)
  # dataset
  dataset = dataset_builder.build(dataset_config, ModeKeys.eval, num_classes)

  return model_evaluator, dataset


def build_faster_rcnn_inference_session(model_config,
                                        dataset_config,
                                        num_classes):
  """Builds faster rcnn model and dataset for inference.

  Args:
    model_config: a protobuf message storing FasterRcnnModel configuration.
    dataset_config: a protobuf message storing Dataset configuration.
    num_classes: int scalar, num of classes.

  Returns:
    model_inferencer: an instance of FasterRcnnModelInferencer.
    dataset: an instance of InferencerDataset.
  """
  if not isinstance(model_config, faster_rcnn_model_pb2.FasterRcnnModel):
    raise ValueError('model_config must be an instance of FasterRcnnModel '
        'message.')
  if not isinstance(dataset_config, dataset_pb2.Dataset):
    raise ValueError('dataset_config must be an instance of Dataset message.')

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
  normalizer_fn = build_normalizer_fn(model_config.normalizer)
  feature_extractor = build_feature_extractor(model_config.feature_extractor)
  box_coder = box_coder_builder.build(model_config.box_coder)
  rpn_anchor_generator = anchor_generator_builder.build(
      model_config.rpn_anchor_generator)
  rpn_box_predictor = box_predictor_builder.build(
      model_config.rpn_box_predictor,
      num_classes=1,
      anchor_generator=rpn_anchor_generator)
  frcnn_box_predictor = box_predictor_builder.build(
      model_config.frcnn_box_predictor,
      num_classes=num_classes)
  frcnn_mask_predictor = None
  if model_config.HasField('frcnn_mask_predictor'):
    frcnn_mask_predictor = mask_predictor_builder.build(
        model_config.frcnn_mask_predictor)

  frcnn_nms_fn, frcnn_score_conversion_fn = postprocessing_builder.build(
      model_config.frcnn_post_processing)
  rpn_nms_fn, rpn_score_conversion_fn = postprocessing_builder.build(
      model_config.rpn_post_processing)

  proposal_crop_size=model_config.proposal_crop_size
  rpn_box_predictor_depth = model_config.rpn_box_predictor_depth
  first_stage_atrous_rate = model_config.first_stage_atrous_rate

  # faster rcnn model
  model_inferencer = inferencer.FasterRcnnModelInferencer(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      box_coder=box_coder,
      rpn_anchor_generator=rpn_anchor_generator,
      rpn_box_predictor=rpn_box_predictor,
      frcnn_box_predictor=frcnn_box_predictor,
      frcnn_mask_predictor=frcnn_mask_predictor,

      frcnn_nms_fn=frcnn_nms_fn,
      frcnn_score_conversion_fn=frcnn_score_conversion_fn,
      rpn_nms_fn=rpn_nms_fn,
      rpn_score_conversion_fn=rpn_score_conversion_fn,

      proposal_crop_size=model_config.proposal_crop_size,
      rpn_box_predictor_depth=rpn_box_predictor_depth,
      first_stage_atrous_rate=first_stage_atrous_rate)
  # dataset
  dataset = dataset_builder.build(dataset_config, ModeKeys.infer, num_classes)
  
  return model_inferencer, dataset


def build_feature_extractor(config):
  """Builds faster rcnn feature extractor.

  Args:
    config: a protobuf message storing Feature Extractor configurations.

  Returns:
    feature_extractor: an instance of FeatureExtractor.
  """
  if not isinstance(config, faster_rcnn_model_pb2.FasterRcnnFeatureExtractor):
    raise ValueError('config must be an instance of '
        'FasterRcnnFeatureExtractor message.')

  reuse_weights = (config.reuse_weights 
      if config.HasField('reuse_weights') else None)

  if config.type == 'faster_rcnn_inception_v2':
    batch_norm_params = build_batch_norm_params(config.batch_norm_params)
    depth_multiplier = config.depth_multiplier
    feature_extractor = FasterRcnnInceptionV2FeatureExtractor(
        batch_norm_params=batch_norm_params, 
        depth_multiplier=depth_multiplier, 
        reuse_weights=reuse_weights)
    return feature_extractor
  elif config.type == 'faster_rcnn_resnet_v1':
    if config.resnet_num_layers not in [50, 101]:
      raise ValueError('Num of layers of ResNet must be in [50, 101].')
    output_stride = config.output_stride
    weight_decay = config.weight_decay
    feature_extractor = FasterRcnnResnet101V1FeatureExtractor(
      output_stride=output_stride,
      weight_decay=weight_decay,
      reuse_weights=reuse_weights)
    return feature_extractor
  else:
    pass

  raise ValueError('Unknown feature extractor.')

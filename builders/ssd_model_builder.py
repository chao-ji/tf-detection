from detection.builders import image_resizer_builder
from detection.builders import conv_hyperparams_builder
from detection.builders import anchor_generator_builder
from detection.builders import box_predictor_builder
from detection.builders import box_coder_builder
from detection.builders import target_assigner_builder
from detection.builders import losses_builder
from detection.builders import optimizer_builder
from detection.builders import postprocessing_builder
from detection.builders import dataset_builder
from detection.builders.preprocessor_builder import build_normalizer_fn

from detection.feature_extractors.ssd_inception_v2 import (
    SsdInceptionV2FeatureExtractor)
from detection.feature_extractors.ssd_mobilenet_v2 import (
    SsdMobileNetV2FeatureExtractor)

from detection.ssd import trainer
from detection.ssd import evaluator
from detection.ssd import inferencer

from detection.core.standard_names import ModeKeys 

from detection.protos import ssd_model_pb2
from detection.protos import dataset_pb2
from detection.protos import train_config_pb2


def build_ssd_train_session(model_config,
                            dataset_config,
                            train_config,
                            num_classes):
  """Builds ssd model, dataset, and optimizer for training.

  Args:
    model_config: a protobuf message storing SsdModel configuration.
    dataset_config: a protobuf message storing Dataset configuration.
    train_config: a protubuf message storing TrainConfig configuration.
    num_classes: int scalar, num of classes.

  Returns:
    model_trainer: an instance of SsdModelTrainer.
    dataset: an instance of TrainerDataset.
    optimizer_builder_fn: a callable that (when called with no arguments) 
      returns a 2-tuple containing an instance of optimizer and a learning rate 
      tensor.
  """
  if not isinstance(model_config, ssd_model_pb2.SsdModel):
    raise ValueError('model_config must be an instance of SsdModel message.')
  if not isinstance(dataset_config, dataset_pb2.Dataset):
    raise ValueError('dataset_config must be an instance of Dataset message.')
  if not isinstance(train_config, train_config_pb2.TrainConfig):
    raise ValueError('train_config must be an instance of TrainConfig message.')

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
  normalizer_fn = build_normalizer_fn(model_config.normalizer)
  conv_hyperparams_fn = conv_hyperparams_builder.build(
      model_config.conv_hyperparams)
  feature_extractor = build_feature_extractor(model_config.feature_extractor, 
                                              conv_hyperparams_fn)
  anchor_generator = anchor_generator_builder.build(
      model_config.anchor_generator)
  box_predictor = box_predictor_builder.build(model_config.box_predictor,
                                              num_classes,
                                              anchor_generator,
                                              conv_hyperparams_fn)

  box_coder = box_coder_builder.build(model_config.box_coder)
  target_assigner = target_assigner_builder.build(model_config.target_assigner,
                                                  box_coder=box_coder)

  (localization_loss_fn, classification_loss_fn,
   localization_loss_weight, classification_loss_weight,
   hard_example_miner) = losses_builder.build(model_config.loss)

#  normalize_loss_by_num_matches = model_config.normalize_loss_by_num_matches
#  normalize_loc_loss_by_code_size = model_config.normalize_loc_loss_by_code_size
  freeze_batch_norm = model_config.freeze_batch_norm
  add_background_class = model_config.add_background_class
  gradient_clipping_by_norm = train_config.gradient_clipping_by_norm

  # ssd model
  model_trainer = trainer.SsdModelTrainer(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      anchor_generator=anchor_generator,
      box_predictor=box_predictor,

      box_coder=box_coder,
      target_assigner=target_assigner,

      localization_loss_fn=localization_loss_fn,
      classification_loss_fn=classification_loss_fn,
      hard_example_miner=hard_example_miner,

      localization_loss_weight=localization_loss_weight,
      classification_loss_weight=classification_loss_weight,
#      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
#      normalize_loc_loss_by_codesize=normalize_loc_loss_by_code_size,
      freeze_batch_norm=freeze_batch_norm,
      add_background_class=add_background_class,
      gradient_clipping_by_norm=gradient_clipping_by_norm)
  # dataset
  dataset = dataset_builder.build(dataset_config, ModeKeys.train, num_classes)
  # optimizer
  optimizer_builder_fn = optimizer_builder.build(train_config.optimizer)

  return model_trainer, dataset, optimizer_builder_fn
      

def build_ssd_evaluate_session(model_config,
                               dataset_config,
                               num_classes):
  """Builds ssd model and dataset for evaluation. 
  
  Args:
    model_config: a protobuf message storing SsdModel configuration.
    dataset_config: a protobuf message storing Dataset configuration.
    num_classes: int scalar, num of classes.

  Returns:
    model_evaluator: an instance of SsdModelEvaluator.
    dataset: an instance of EvaluatorDataset.
  """
  if not isinstance(model_config, ssd_model_pb2.SsdModel):
    raise ValueError('model_config must be an instance of SsdModel message.')
  if not isinstance(dataset_config, dataset_pb2.Dataset):
    raise ValueError('dataset_config must be an instance of Dataset message.')

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
  normalizer_fn = build_normalizer_fn(model_config.normalizer)
  conv_hyperparams_fn = conv_hyperparams_builder.build(
      model_config.conv_hyperparams)
  feature_extractor = build_feature_extractor(model_config.feature_extractor,
                                              conv_hyperparams_fn)
  anchor_generator = anchor_generator_builder.build(
      model_config.anchor_generator)
  box_predictor = box_predictor_builder.build(model_config.box_predictor,
                                              num_classes,
                                              anchor_generator,
                                              conv_hyperparams_fn)

  box_coder = box_coder_builder.build(model_config.box_coder)
  target_assigner = target_assigner_builder.build(model_config.target_assigner,
                                                  box_coder=box_coder)

  (localization_loss_fn, classification_loss_fn,
   localization_loss_weight, classification_loss_weight,
   hard_example_miner) = losses_builder.build(model_config.loss)
  non_max_suppression_fn, score_converter_fn = postprocessing_builder.build(
      model_config.post_processing)

#  normalize_loss_by_num_matches = model_config.normalize_loss_by_num_matches
#  normalize_loc_loss_by_code_size = model_config.normalize_loc_loss_by_code_size
  add_background_class = model_config.add_background_class

  # ssd model
  model_evaluator = evaluator.SsdModelEvaluator(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      anchor_generator=anchor_generator,
      box_predictor=box_predictor,

      box_coder=box_coder,
      target_assigner=target_assigner,

      localization_loss_fn=localization_loss_fn,
      classification_loss_fn=classification_loss_fn,
      hard_example_miner=hard_example_miner,
      score_converter_fn=score_converter_fn,
      non_max_suppression_fn=non_max_suppression_fn,

      localization_loss_weight=localization_loss_weight,
      classification_loss_weight=classification_loss_weight,
#      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
#      normalize_loc_loss_by_codesize=normalize_loc_loss_by_code_size,
      add_background_class=add_background_class)
  # dataset
  dataset = dataset_builder.build(dataset_config, ModeKeys.eval, num_classes)

  return model_evaluator, dataset


def build_ssd_inference_session(model_config,
                                dataset_config,
                                num_classes):
  """Builds ssd model and dataset for inference. 

  Args:
    model_config: a protobuf message storing SsdModel configuration.
    dataset_config: a protobuf message storing Dataset configuration.
    num_classes: int scalar, num of classes.

  Returns:
    model_inferencer: an instance of SsdModelInferencer.
    dataset: an instance of InferencerDataset.
  """
  if not isinstance(model_config, ssd_model_pb2.SsdModel):
    raise ValueError('model_config must be an instance of SsdModel message.')
  if not isinstance(dataset_config, dataset_pb2.Dataset):
    raise ValueError('dataset_config must be an instance of Dataset message.')

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)
  normalizer_fn = build_normalizer_fn(model_config.normalizer)
  conv_hyperparams_fn = conv_hyperparams_builder.build(
      model_config.conv_hyperparams)
  feature_extractor = build_feature_extractor(model_config.feature_extractor,
                                              conv_hyperparams_fn)
  anchor_generator = anchor_generator_builder.build(
      model_config.anchor_generator)
  box_predictor = box_predictor_builder.build(model_config.box_predictor,
                                              num_classes,
                                              anchor_generator,
                                              conv_hyperparams_fn)

  box_coder = box_coder_builder.build(model_config.box_coder)
  non_max_suppression_fn, score_converter_fn = postprocessing_builder.build(
      model_config.post_processing)

  # ssd model
  model_inferencer = inferencer.SsdModelInferencer(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      anchor_generator=anchor_generator,
      box_predictor=box_predictor,

      box_coder=box_coder,

      score_converter_fn=score_converter_fn,
      non_max_suppression_fn=non_max_suppression_fn)
  # dataset
  dataset = dataset_builder.build(dataset_config, ModeKeys.infer, num_classes)

  return model_inferencer, dataset
 

def build_feature_extractor(config, conv_hyperparams_fn=None):
  """Builds ssd feature extractor.

  Args:
    config: a protobuf message storing Feature Extractor configurations.
    conv_hyperparams_fn: a callable that returns arg_scope for building feature 
      extractor.

  Returns:
    feature_extractor: an instance of FeatureExtractor. 
  """
  if not isinstance(config, ssd_model_pb2.SsdFeatureExtractor):
    raise ValueError('config must be an instance of SsdFeatureExtractor '
        'message.')

  if conv_hyperparams_fn is None:
    conv_hyperparams_fn = conv_hyperparams_builder.build(
        config.conv_hyperparams)
  if config.type == 'ssd_inception_v2':  
    feature_extractor = SsdInceptionV2FeatureExtractor(
        conv_hyperparams_fn=conv_hyperparams_fn,
        depth_multiplier=config.depth_multiplier,
        reuse_weights=None,
        use_depthwise=config.use_depthwise)
    return feature_extractor
  elif config.type == 'ssd_mobilenet_v2':
    feature_extractor = SsdMobileNetV2FeatureExtractor(
        conv_hyperparams_fn=conv_hyperparams_fn,
        depth_multiplier=config.depth_multiplier,
        reuse_weights=None,
        use_depthwise=config.use_depthwise)
    return feature_extractor
  else:    
    pass
  
  raise ValueError('Unknown feature extractor.')

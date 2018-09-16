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

from detection.feature_extractors.ssd_inception_v2 import (
    SsdInceptionV2FeatureExtractor)

from detection.ssd_meta_arch import trainer
from detection.ssd_meta_arch import evaluator
from detection.ssd_meta_arch import inferencer

from detection.core.standard_names import DatasetDictFields


def build_ssd_train_session(model_config,
                            dataset_config,
                            num_classes):

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)

  normalizer_fn = _build_normalizer_fn(model_config.normalizer_range)

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

  (classification_loss_fn, localization_loss_fn,
   classification_loss_weight, localization_loss_weight,
   hard_example_miner) = losses_builder.build(model_config.loss)

  optimizer_builder_fn = optimizer_builder.build(
      model_config.optimizer)

  dataset_dict = dataset_builder.build(dataset_config, num_classes)

  normalize_loss_by_num_matches = model_config.normalize_loss_by_num_matches

  normalize_loc_loss_by_code_size = model_config.normalize_loc_loss_by_code_size

  freeze_batch_norm = model_config.freeze_batch_norm

  add_background_class = model_config.add_background_class

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
      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
      normalize_loc_loss_by_codesize=normalize_loc_loss_by_code_size,
      freeze_batch_norm=freeze_batch_norm,
      add_background_class=add_background_class)

  return (model_trainer, dataset_dict[DatasetDictFields.trainer_dataset],
      optimizer_builder_fn)
      

def build_ssd_evaluate_session(model_config,
                               dataset_config,
                               num_classes):

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)

  normalizer_fn = _build_normalizer_fn(model_config.normalizer_range)

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

  (classification_loss_fn, localization_loss_fn,
   classification_loss_weight, localization_loss_weight,
   hard_example_miner) = losses_builder.build(model_config.loss)

  non_max_suppression_fn, score_converter_fn = postprocessing_builder.build(
      model_config.post_processing)

  dataset_dict = dataset_builder.build(dataset_config, num_classes)

  normalize_loss_by_num_matches = model_config.normalize_loss_by_num_matches

  normalize_loc_loss_by_code_size = model_config.normalize_loc_loss_by_code_size

  add_background_class = model_config.add_background_class

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
      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
      normalize_loc_loss_by_codesize=normalize_loc_loss_by_code_size,
      add_background_class=add_background_class)
  
  return model_evaluator, dataset_dict[DatasetDictFields.evaluator_dataset]


def build_ssd_inference_session(model_config,
                                dataset_config,
                                num_classes):

  image_resizer_fn = image_resizer_builder.build(model_config.image_resizer)

  normalizer_fn = _build_normalizer_fn(model_config.normalizer_range)

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

  dataset_dict = dataset_builder.build(dataset_config, num_classes)

  model_inferencer = inferencer.SsdModelInferencer(
      image_resizer_fn=image_resizer_fn,
      normalizer_fn=normalizer_fn,
      feature_extractor=feature_extractor,
      anchor_generator=anchor_generator,
      box_predictor=box_predictor,

      box_coder=box_coder,

      score_converter_fn=score_converter_fn,
      non_max_suppression_fn=non_max_suppression_fn)

  return model_inferencer, dataset_dict[DatasetDictFields.inferencer_dataset]
 

def build_feature_extractor(config, conv_hyperparams_fn=None):

  if conv_hyperparams_fn is None:
    conv_hyperparams_fn = conv_hyperparams_builder.build(
        config.conv_hyperparams)
  
  if config.type == 'ssd_inception_v2':  
    feature_extractor = SsdInceptionV2FeatureExtractor(
        conv_hyperparams_fn=conv_hyperparams_fn,
        depth_multiplier=config.depth_multiplier,
        reuse_weights=None,
        use_depthwise=config.use_depthwise,
        override_base_feature_extractor_hyperparams=
            config.override_base_feature_extractor_hyperparams)
 
    return feature_extractor

  raise ValueError('Unknown feature extractor.')


def _build_normalizer_fn(config):
  high, low = config.high, config.low
  normalizer_fn = lambda image: ((high - low) / 255.0) * image + low

  return normalizer_fn


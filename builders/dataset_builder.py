
from detection.preprocess import dataset
from detection.protos import dataset_pb2
from detection.builders import decoder_builder
from detection.builders import preprocessor_builder
from detection.core.standard_names import DatasetDictFields

def build(config, num_classes):
  decoder = decoder_builder.build(config.data_decoder)

  trainer_dataset = _build_trainer_dataset(config.trainer_dataset,
                                           num_classes,
                                           decoder)

  evaluator_dataset = _build_evaluator_dataset(config.evaluator_dataset,
                                               num_classes,
                                               decoder)

  inferencer_dataset = _build_inferencer_dataset(config.inferencer_dataset)

  dataset_dict = {DatasetDictFields.trainer_dataset: trainer_dataset,
                  DatasetDictFields.evaluator_dataset: evaluator_dataset,
                  DatasetDictFields.inferencer_dataset: inferencer_dataset}

  return dataset_dict 

def _build_trainer_dataset(config,
                           num_classes,
                           decoder,
                           preprocessor=None):

  if preprocessor is None:
    preprocessor = preprocessor_builder.build(config.preprocessor)

  trainer_dataset = dataset.TrainerDataset(
      batch_size=config.batch_size,
      num_epochs=config.num_epochs,
      decoder=decoder,
      preprocessor=preprocessor,
      num_classes=num_classes,
      shuffle=config.shuffle,
      filename_dataset_shuffle_buffer_size=(
          config.filename_dataset_shuffle_buffer_size),
      record_dataset_shuffle_buffer_size=(
          config.record_dataset_shuffle_buffer_size),
      random_seed=config.random_seed,
      num_readers=config.num_readers,
      read_block_length=config.read_block_length)

  return trainer_dataset


def _build_evaluator_dataset(config,
                             num_classes,
                             decoder,
                             preprocessor=None):

  if preprocessor is None:
    preprocessor = preprocessor_builder.build(config.preprocessor)

  evaluator_dataset = dataset.EvaluatorDataset(
      decoder=decoder,
      preprocessor=preprocessor,
      num_classes=num_classes)

  return evaluator_dataset


def _build_inferencer_dataset(config):
  return dataset.InferencerDataset()


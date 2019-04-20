from detection.data import dataset
from detection.protos import dataset_pb2
from detection.builders import decoder_builder
from detection.builders import preprocessor_builder
from detection.core.standard_names import ModeKeys


def build(config, mode, num_classes=None):
  """Builds the datasets for trainer, evaluator and inferencer.

  Args:
    config: a protobuf message 
    mode: string scalar, mode of the dataset.
    num_classes: int scalar, num of classes.

  Returns:
    dataset_dict: a dict holding Trainer, Evaluator and Inferencer Dataset
      instances.
  """
  if not isinstance(config, dataset_pb2.Dataset):
    raise ValueError('config must be an instance of Dataset message.')

  decoder = decoder_builder.build(config.data_decoder)

  if mode == ModeKeys.train:
    dataset = _build_trainer_dataset(config.trainer_dataset,
                                     num_classes,
                                     decoder)
  elif mode == ModeKeys.eval:
    dataset = _build_evaluator_dataset(config.evaluator_dataset,
                                       num_classes,
                                       decoder)
  elif mode == ModeKeys.infer:
    dataset = _build_inferencer_dataset(config.inferencer_dataset)
  else:
    raise ValueError('Unsupported mode: {}'.format(mode))

  return dataset


def _build_trainer_dataset(config,
                           num_classes,
                           decoder,
                           preprocessor=None):
  """Builds trainer dataset. You can optionally pass in a `preprocessor`. 
  Otherwise, one with configurations specified in `config.preprocessor` will
  be built.
 
  Args:
    config: a protobuf message storing TrainerDataset configurations. 
    num_classes: int scalar, num of classes.
    decoder: an instance of DataDecoder.
    preprocessor: an instance of Preprocessor or None.

  Returns:
    trainer_dataset: an instance of TrainerDataset.
  """
  if not isinstance(config, dataset_pb2.TrainerDataset):
    raise ValueError('config must be an instance of TrainerDataset message.')

  if preprocessor is None:
    preprocessor = preprocessor_builder.build(config.preprocessor)

  trainer_dataset = dataset.TrainerDataset(
      batch_size=config.batch_size,
      num_epochs=config.num_epochs,
      decoder=decoder,
      preprocessor=preprocessor,
      num_classes=num_classes,
      shuffle=config.shuffle,
      reader_buffer_size=config.reader_buffer_size,
      shuffle_buffer_size=config.shuffle_buffer_size,
      random_seed=config.random_seed,
      num_readers=config.num_readers,
      bucketed_batching=config.bucketed_batching,
      height_boundaries=list(config.height_boundaries),
      width_boundaries=list(config.width_boundaries))

  return trainer_dataset


def _build_evaluator_dataset(config,
                             num_classes,
                             decoder,
                             preprocessor=None):
  """Builds evaluator dataset. You can optionally pass in a `preprocessor`. 
  Otherwise, one with configurations specified in `config.preprocessor` will
  be built.

  Args:
    config: a protobuf message storing EvaluatorDataset configurations.
    num_classes: int scalar, num of classes.
    decoder: an instance of DataDecoder.
    preprocessor: an instance of Preprocessor or None.

  Returns:
    evaluator_dataset: an instance of EvaluatorDataset.
  """
  if not isinstance(config, dataset_pb2.EvaluatorDataset):
    raise ValueError('config must be an instance of EvaluatorDataset message.')

  if preprocessor is None:
    preprocessor = preprocessor_builder.build(config.preprocessor)

  evaluator_dataset = dataset.EvaluatorDataset(
      decoder=decoder,
      preprocessor=preprocessor,
      num_classes=num_classes)

  return evaluator_dataset


def _build_inferencer_dataset(config):
  """Returns an instance of InferencerDataset."""
  return dataset.InferencerDataset()


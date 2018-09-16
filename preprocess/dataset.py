""""""
import functools
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.preprocess import utils
from detection.core.standard_names import ModeKeys


class DetectionDataset(object):
  """Abstract base class for detection dataset."""

  __metaclass__ = ABCMeta

  @abstractmethod
  def mode(self):
    """Indicating whether a dataset is `TrainerDataset`, 
    `EvaluatorDataset` or `InferencerDataset`.
  
    To be implemented by subclasses. 
  
    Returns:
      a scalar string. 
    """
    pass

  def get_tensor_dict(self, filename_list, scope=None):
    """Generates a tensor dict from a list of input filenames.

    Args:
      filename_list: a list of strings, containing full input filenames.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    with tf.name_scope(scope, 'GetTensorDict', [filename_list]):
      self._tensor_dict = self._get_tensor_dict(filename_list)
      return self._tensor_dict

  @abstractmethod
  def _get_tensor_dict(self, filename_list):
    """Generates a tensor dict from a list of input filenames.

    To be implemented by subclasses.

    Args:
      filename_list: a list of strings, containing full input filenames.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    pass

  def _train_eval_get_tensor_dict_from_record_dataset(self,
                                                      record_dataset,
                                                      batch_size):
    """Generates a tensor dict from a record dataset.

    It transforms a dataset containing TFRecords into one containing
    tensor dicts, which will be then unbatched and unpadded.
    Can be called ONLY by `TrainerDataset` or `EvaluatorDataset`.

    Args:
      record_dataset: a tf.data.Dataset containing scalar string tensors.
      batch_size: int scalar, batch size. 

    Returns:
      tensor_dict: a dict mapping from tensor names to list of tensors:
        {'images' -> list of tensors with shape [height, width, channels],
        'groundtruth_boxes' -> list of tensors with shape [num_gt_boxes, 4],
        'groundtruth_labels' -> list of tensors with shape [num_gt_boxes] or
        [num_gt_boxes, num_classes]}
        Length of list is equal to batch size.
    """
    if self.mode != ModeKeys.train and self.mode != ModeKeys.eval:
      raise ValueError('Must be call by `TrainerDataset` or `EvaluatorDataset`')

#    print('\n', record_dataset, '\n')

    tensor_dict_dataset = record_dataset.map(
        lambda protobuf_str: self._decoder.decode(protobuf_str))

#    print('\n', tensor_dict_dataset, '\n')

    tensor_dict_dataset = tensor_dict_dataset.map(
        lambda tensor_dict: (self._preprocessor.preprocess(tensor_dict)
            if self._preprocessor is not None else tensor_dict))

#    print('\n', tensor_dict_dataset, '\n')

    static_shapes = tensor_dict_dataset.output_shapes

#    print('\n', static_shapes, '\n')

    tensor_dict_dataset = tensor_dict_dataset.map(
        lambda tensor_dict: utils.add_runtime_shapes(tensor_dict))

#    print('\n', tensor_dict_dataset, '\n')

    tensor_dict_dataset = tensor_dict_dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size, padded_shapes=tensor_dict_dataset.output_shapes))

#    print('\n', tensor_dict_dataset, '\n')

    iterator = tensor_dict_dataset.make_one_shot_iterator()
    tensor_dict = iterator.get_next()

#    print('\n', tensor_dict, '\n')

    tensor_dict = utils.unbatch_padded_tensors(
        tensor_dict, static_shapes)

#    print('\n', tensor_dict, '\n')

    if self._num_classes is not None:
      tensor_dict = utils.sparse_to_one_hot_labels(
          tensor_dict, self._num_classes)

#    print('\n', tensor_dict, '\n')

    return tensor_dict


class TrainerDataset(DetectionDataset):
  """Dataset for training detection model."""
  def __init__(self,
               batch_size,
               num_epochs,
               decoder,
               preprocessor,
               num_classes=None,
               shuffle=False,
               filename_dataset_shuffle_buffer_size=100,
               record_dataset_shuffle_buffer_size=2048,
               random_seed=0,
               num_readers=10,
               read_block_length=32):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      num_epochs: int scalar, num of times the original dataset is repeated.
        If None, the dataset is repeated infinitely.
      decoder: a detection.core.DataDecoder instance. 
      preprocessor: a detection.preprocess.DataPreprocessor instance.
      num_classes: int scalar, num of object categories excluding background.
        If not None, the groundtruth class labels in tensor_dict will be in 
        one-hot representation.
      shuffle: bool scalar, whether input data is to be shuffled.
      filename_dataset_shuffle_buffer_size: int scalar, shuffle buffer size
        for filename dataset. 
      record_dataset_shuffle_buffer_size: int scalar, shuffle buffer size
        for TFRecord dataset.
      random_seed: int scalar, random seed.
      num_readers: int scalar, num of readers.
      read_block_length: int scalar, read block length.
    """
    self._batch_size = batch_size
    self._num_epochs = num_epochs
    self._decoder = decoder
    self._preprocessor = preprocessor
    self._num_classes = num_classes
    self._shuffle = shuffle
    self._filename_dataset_shuffle_buffer_size = (
        filename_dataset_shuffle_buffer_size)
    self._record_dataset_shuffle_buffer_size = (
        record_dataset_shuffle_buffer_size)
    self._random_seed = random_seed
    self._num_readers = num_readers
    self._read_block_length = read_block_length

    self._file_read_fn = functools.partial(
        tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000)

  @property
  def mode(self):
    return ModeKeys.train

  def _get_tensor_dict(self, filename_list):
    """Generates a tensor dict from a list of input filenames.

    Args:
      filename_list: a list of strings, containing full names for 
        tfrecord files.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    filename_dataset = tf.data.Dataset.from_tensor_slices(filename_list)
    if self._shuffle:
      filename_dataset = filename_dataset.shuffle(
          self._filename_dataset_shuffle_buffer_size, seed=self._random_seed)
    filename_dataset = filename_dataset.repeat(self._num_epochs or None)
    record_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            self._file_read_fn,
            cycle_length=self._num_readers,
            block_length=self._read_block_length,
            sloppy=self._shuffle))
    if self._shuffle:
      record_dataset = record_dataset.shuffle(
          self._record_dataset_shuffle_buffer_size, seed=self._random_seed)

    tensor_dict = self._train_eval_get_tensor_dict_from_record_dataset(
        record_dataset, self._batch_size)

    return tensor_dict


class EvaluatorDataset(DetectionDataset):
  """Dataset for evaluating detection model."""
  def __init__(self,
               decoder,
               preprocessor,
               num_classes=None):
    """Constructor.

    Args:
      decoder: a detection.core.DataDecoder instance. 
      preprocessor: a detection.preprocess.DataPreprocessor instance.
      num_classes: int scalar, num of object categories excluding background.
        If not None, the groundtruth class labels in tensor_dict will have 
        shape [num_gt_boxes, num_classes] (one hot) rather than [num_gt_boxes].
    """
    self._decoder = decoder
    self._preprocessor = preprocessor
    self._num_classes = num_classes

  @property
  def mode(self):
    return ModeKeys.eval

  def _get_tensor_dict(self, filename_list):
    """Generates a tensor dict from a list of input filenames.

    Args:
      filename_list: a list of strings, containing full names for 
        TFRecord files.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    record_dataset = tf.data.TFRecordDataset(filename_list)

    tensor_dict = self._train_eval_get_tensor_dict_from_record_dataset(
        record_dataset, 1)

    return tensor_dict
    

class InferencerDataset(DetectionDataset):
  """Dataset for making inferences by detection model."""

  @property
  def mode(self):
    return ModeKeys.infer

  def _get_tensor_dict(self, filename_list):
    """Generates a tensor dict from a list of input filenames.

    Args:
      filename_list: a list of strings, containing full names for 
        unprocessed raw image files (e.g. jpg, png).

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    filename_dataset = tf.data.Dataset.from_tensor_slices(filename_list)
    dataset = filename_dataset.map(
        lambda filename: [
            tf.image.decode_image(tf.read_file(filename), channels=3)])
    iterator = dataset.make_one_shot_iterator()
    image_list = iterator.get_next()
    for image in image_list:
      image.set_shape([None, None, 3])
    tensor_dict = {'image': image_list}

    return tensor_dict


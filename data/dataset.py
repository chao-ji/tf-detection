import functools
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import tensorflow as tf

from detection.utils import dataset_utils as utils
from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields
 

class DetectionDataset(object):
  """Abstract base class for detection dataset.

  Subclass must implement abstractproperty `mode` and abstractmethod 
  `_get_tensor_dict`. 
  """
  __metaclass__ = ABCMeta

  @abstractproperty
  def mode(self):
    """Indicating whether a dataset is `TrainerDataset`, `EvaluatorDataset` or 
    `InferencerDataset`.
  
    To be implemented by subclasses. 
  
    Returns:
      a scalar string. 
    """
    pass

  def get_tensor_dict(self, filenames, scope=None):
    """Generates a tensor dict holding input tensors that are ready to be 
    processed by model runners. Calls and wraps `self._get_tensor_dict()`
    with a name scope.

    Args:
      filenames: a list of strings, holding input filenames.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
        { 'images': list of tensors of shape [height, width, channels],
          'groundtruth_boxes' (Optional): list of tensors of shape 
            [num_gt_boxes, 4],
          'groundtruth_labels'(Optional): list of tensors of shape 
            [num_gt_boxes] or [num_gt_boxes, num_classes] }
        Length of list is equal to batch size.
    """
    with tf.name_scope(scope, 'GetTensorDict', [filenames]):
      self._tensor_dict = self._get_tensor_dict(filenames)
      return self._tensor_dict

  @abstractmethod
  def _get_tensor_dict(self, filenames):
    """Generates a tensor dict holding input tensors that are ready to be
    processed by model runners.

    To be implemented by subclasses.

    Args:
      filenames: a list of strings, holding input filenames.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    pass

  def _train_eval_get_tensor_dict_from_record_dataset(self,
                                                      record_dataset,
                                                      batch_size,
                                                      bucketed_batching=False,
                                                      height_boundaries=(),
                                                      width_boundaries=()):
    """Generates a tensor dict from a record dataset.

    It transforms a dataset holding TFRecords into one holding
    tensor dicts, which will be then unbatched and unpadded.
    For `TrainerDataset` and `EvaluatorDataset` **ONLY**.

    Args:
      record_dataset: tf.data.Dataset holding tfrecords (scalar string tensors).
      batch_size: int scalar, batch size.
      bucketed_batching: bool scalar, whether to group tensors into buckets
         based on image spatial dimensions before batching. Defaults to False.
      height_boundaries: list or tuple of increasing int scalars, bucket 
        boundaries of heights. Ignored if `bucketed_batching` is False.
      width_boundaries: list or tuple of increasing int scalars, bucket
        boundaries of widths.Ignored if `bucketed_batching` is False.

    Returns:
      tensor_dict: a dict mapping from tensor names to list of tensors:
        { 'image': list of tensors of shape [height, width, channels],
          'groundtruth_boxes': list of tensors of shape [num_gt_boxes, 4],
          'groundtruth_labels': list of tensors of shape [num_gt_boxes] or
          [num_gt_boxes, num_classes],
          'groundtruth_masks': (Optional) list of tensors of shape 
            [num_gt_boxes, height, width] }
        Length of list is equal to batch size.
    """
    if self.mode != ModeKeys.train and self.mode != ModeKeys.eval:
      raise ValueError('For `TrainerDataset` and `EvaluatorDataset` ONLY.')

    tensor_dict_dataset = record_dataset.map(
        lambda protobuf_str: self._decoder.decode(protobuf_str))

    tensor_dict_dataset = tensor_dict_dataset.map(
        lambda tensor_dict: (self._preprocessor.preprocess(tensor_dict)
            if self._preprocessor is not None else tensor_dict))

    static_shapes = tensor_dict_dataset.output_shapes

    tensor_dict_dataset = tensor_dict_dataset.map(
        lambda tensor_dict: utils.add_runtime_shapes(tensor_dict))

    if bucketed_batching:
      tensor_dict_dataset = utils.image_size_bucketed_batching(
          tensor_dict_dataset,
          batch_size,
          height_boundaries,
          width_boundaries)
    else:
      tensor_dict_dataset = tensor_dict_dataset.padded_batch(batch_size, 
          padded_shapes=tensor_dict_dataset.output_shapes, drop_remainder=True)

    iterator = tensor_dict_dataset.make_one_shot_iterator()
    tensor_dict = iterator.get_next()

    # We will unpad all tensors EXCEPT the image tensors and mask tensors (if
    # not None), because they will later be stacked into 
    # [batch, image_height, image_width, channels] tensor for image (or 
    # [batch, num_boxes, image_height, image_width] tensor for masks) to be fed 
    # to the network:
    tensor_dict = utils.unbatch_padded_tensors(
        tensor_dict, 
        static_shapes, 
        keep_padded_list=[TensorDictFields.image, 
                          TensorDictFields.groundtruth_masks])

    if self._num_classes is not None:
      tensor_dict = utils.sparse_to_one_hot_labels(
          tensor_dict, self._num_classes)

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
               reader_buffer_size=100,
               shuffle_buffer_size=2048,
               random_seed=0,
               num_readers=10,
               bucketed_batching=False,
               height_boundaries=(),
               width_boundaries=()):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      num_epochs: int scalar, num of times the original dataset is repeated.
        If None or <= 0, the dataset is repeated infinitely.
      decoder: a detection.core.DataDecoder instance.
      preprocessor: a detection.preprocess.DataPreprocessor instance.
      num_classes: int scalar, num of object categories excluding background.
        If not None, the groundtruth class labels in tensor_dict will have 
        shape [num_gt_boxes, num_classes] (one hot) rather than [num_gt_boxes].
      shuffle: bool scalar, whether input data is to be shuffled.
      reader_buffer_size: int scalar, buffer size for `tf.data.TFRecordDataset`.
      shuffle_buffer_size: int scalar, shuffle buffer size for TFRecord dataset.
      random_seed: int scalar, random seed.
      num_readers: int scalar, num of readers.
      bucketed_batching: bool scalar, whether to group tensors into buckets
         based on image spatial dimensions before batching. Defaults to False.
      height_boundaries: list or tuple of increasing int scalars, bucket 
        boundaries of heights. Ignored if `bucketed_batching` is False.
      width_boundaries: list or tuple of increasing int scalars, bucket
        boundaries of widths.Ignored if `bucketed_batching` is False.
    """
    self._batch_size = batch_size
    self._num_epochs = num_epochs
    self._decoder = decoder
    self._preprocessor = preprocessor
    self._num_classes = num_classes
    self._shuffle = shuffle
    self._reader_buffer_size = reader_buffer_size
    self._shuffle_buffer_size = shuffle_buffer_size
    self._random_seed = random_seed
    self._num_readers = num_readers
    self._bucketed_batching = bucketed_batching
    self._height_boundaries = height_boundaries
    self._width_boundaries = width_boundaries

    self._file_read_fn = functools.partial(
        tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000)

  @property
  def mode(self):
    return ModeKeys.train

  def _get_tensor_dict(self, filenames):
    """Generates a tensor dict holding input tensors that are ready to be
    processed by model trainer.

    Args:
      filenames: a list of strings, holding filenames for tfrecord files.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    record_dataset = tf.data.TFRecordDataset(
        filenames, 
        buffer_size=self._reader_buffer_size,
        num_parallel_reads=self._num_readers)

    record_dataset = record_dataset.repeat(
        self._num_epochs if self._num_epochs > 0 else None)
    if self._shuffle:
      record_dataset = record_dataset.shuffle(
          self._shuffle_buffer_size, seed=self._random_seed)

    tensor_dict = self._train_eval_get_tensor_dict_from_record_dataset(
        record_dataset, 
        self._batch_size, 
        self._bucketed_batching, 
        self._height_boundaries, 
        self._width_boundaries)

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

  def _get_tensor_dict(self, filenames):
    """Generates a tensor dict holding input tensors that are ready to be
    processed by model evaluator. 

    Args:
      filenames: a list of strings, holding filenames for tfrecord files.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    record_dataset = tf.data.TFRecordDataset(filenames)
    tensor_dict = self._train_eval_get_tensor_dict_from_record_dataset(
        record_dataset, 1)

    return tensor_dict
    

class InferencerDataset(DetectionDataset):
  """Dataset for making inferences by detection model."""

  @property
  def mode(self):
    return ModeKeys.infer

  def _get_tensor_dict(self, filenames):
    """Generates a tensor dict holding input tensors that are ready to be
    processed by model inferencer. 

    Args:
      filenames: a list of strings, holding filenames for 
        unprocessed raw image files (e.g. jpg, png).

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors.
    """
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filename_dataset.map(lambda filename: [
        tf.image.decode_image(tf.read_file(filename), channels=3)])
    image_list = dataset.make_one_shot_iterator().get_next()
    for image in image_list:
      image.set_shape([None, None, 3])
    tensor_dict = {'image': image_list}

    return tensor_dict


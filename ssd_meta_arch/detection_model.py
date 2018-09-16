""""""
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import tensorflow as tf

from detection.core import box_list_ops
from detection.utils import shape_utils
from detection.core.standard_names import PredTensorDictFields as DictFields

slim = tf.contrib.slim


class DetectionModel(object):
  """Abstract base class for detection model.

  This base class is to be subclassed by ModelTrainer, ModelEvaluator and 
  ModelInferencer to perform training, evaluating, and inference making, 
  respectively. Since all three subclasses share the same workflow of 
  preprocessing input images and running through the forward pass of the conv 
  net, the method `preprocess` and `predict` are implemented in the base 
  class.

  NOTE: The `is_training` argument for batch norm is set dynamically in the
  `predict` method depending on the `self.is_training` property of the subclass.
  """

  __metaclass__ = ABCMeta

  def __init__(self,
               image_resizer_fn,
               normalizer_fn,
               feature_extractor,
               anchor_generator,
               box_predictor):
    """Constructor.

    Args:
      image_resizer_fn: a callable, that wraps one of the tensorflow image 
        resizing functions. See `tf.image.ResizeMethod`.
      normalizer_fn: a callable, that normalizes input image pixel values into 
        a range.
      feature_extractor: an object that extracts features from input images.
      anchor_generator: an object that generates a fixed number of anchors
        given a single input image.
      box_predictor: an object that generates box encoding predictions from
        a list of feature map tensors. 
    """
    self._image_resizer_fn = image_resizer_fn
    self._normalizer_fn = normalizer_fn
    self._feature_extractor = feature_extractor 
    self._anchor_generator = anchor_generator
    self._box_predictor = box_predictor
  
    self._extract_features_scope = 'FeatureExtractor'
    self._anchors = None
    self._feature_maps = None

  @abstractproperty
  def is_training(self):
    """Returns a bool scalar indicating if model is in training mode."""
    pass

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating mode of model (train, eval or infer).
    """
    pass

  @property
  def anchors(self):
    """Returns the anchors as a box_list.BoxList instance."""
    return self._anchors

  @abstractmethod
  def create_restore_saver(self, *args, **kwargs):
    """Creates a tf.train.Saver instance for restoring model.

    Depending on the `mode` of subclass, return a Saver instance initialized 
    with either all global variables or a subset of global variables.

    To be implemented by subclasses.

    Returns:
      a tf.train.Saver instance.
    """
    pass

  def check_dataset_mode(self, dataset):
    """Checks if mode (train, eval, or infer) of dataset and model match.

    Args:
      dataset: a dataset.DetectionDataset instance.

    Raises:
      ValueError if mode of `dataset` and `self` do not match.
    """
    if dataset.mode != self.mode:
      raise ValueError('mode of dataset({}) and model({}) do not match.'
          .format(dataset.mode, self.mode))

  def predict(self, inputs):
    """Runs the input images through the forward pass to get predicted
    tensors, which are to be fed to loss and postprocess functions. 

    NOTE: Anchor boxes are generated as a side effect.

    Args:
      inputs: a rank-4 tensor with shape [batch, height, with, channels]
        containing the input images.

    Returns:
      prediction_tensor_dict: a dict mapping from tensor names to predicted
        tensors:
        {
          'box_encoding_predictions': [batch_size, num_anchors, 4],
          'class_score_predictions': [batch_size, num_anchors, num_classes + 1]
        }
        `num_anchors` is the total num of anchors summed over all feature maps.
    """
    batchnorm_updates_collections = tf.GraphKeys.UPDATE_OPS

    with slim.arg_scope([slim.batch_norm],
                        is_training=(self.is_training and
                            not self._freeze_batch_norm),
                        updates_collections=batchnorm_updates_collections):
      with tf.variable_scope(None, self._extract_features_scope, [inputs]):
        # [[1, 19, 19, d1], [1, 10, 10, d2], [1, 5, 5, d3],
        #  [1, 3, 3, d4], [1, 2, 2, d5], [1, 1, 1, d6]]
        feature_map_list = self._feature_extractor.extract_features(inputs)
      # [[19, 19], [10, 10], [5, 5], [3, 3], [2, 2], [1, 1]]
      feature_map_spatial_dims = self._get_feature_map_spatial_dims(
          feature_map_list)
      # [4]: 1, 300, 300, 3
      image_shape = shape_utils.combined_static_and_dynamic_shape(inputs)

      anchors = box_list_ops.concatenate(
          self._anchor_generator.generate(
              feature_map_spatial_dims,
              im_height=image_shape[1],
              im_width=image_shape[2]))

      (box_encoding_predictions_list, class_score_predictions_list
          ) = self._box_predictor.predict(feature_map_list)

      box_encoding_predictions = tf.concat(
          box_encoding_predictions_list, axis=1)
      if (box_encoding_predictions.shape.ndims == 4 and 
          box_encoding_predictions.shape[2] == 1):
        box_encoding_predictions = tf.squeeze(box_encoding_predictions, axis=2)

      class_score_predictions = tf.concat(class_score_predictions_list, axis=1)

      self._anchors = anchors
      self._feature_map_list = feature_map_list

      prediction_tensor_dict = {
          DictFields.box_encoding_predictions:box_encoding_predictions,
          DictFields.class_score_predictions:class_score_predictions}

      return prediction_tensor_dict

  def _get_feature_map_spatial_dims(self, feature_map_list):
    """Get feature map spatial dimensions of a list of feature map tensors.

    Args: feature_map_list a list of feature map tensors with shape 
      [batch_size, height, width, channels]

    Returns:
      a list of int 2-tuples containing (height, width).
    """
    feature_map_shape_list = [shape_utils.combined_static_and_dynamic_shape(
        feature_map) for feature_map in feature_map_list]
    return [(shape[1], shape[2]) for shape in feature_map_shape_list]

  def preprocess(self, image_list):
    """Reshapes the input images and optionally normalizes pixel values as the 
    input to the feature extractor.

    Args:
      image_list: a list of rank-3 tensors with shape [height, width, channels].

    Returns:
      images: a float tensor with shape [batch, height, width, channels] to be
        fed to `DetectionModel.predict`.
      true_image_shapes: a int tensors with shape [num_batches, 3] containing 
        the height, width, and channels of true image shapes in a batch.
    """
    def _preprocess_single_image(image):
      resized_image = self._image_resizer_fn(image)
      if self._normalizer_fn is not None:
        resized_image = self._normalizer_fn(resized_image)
      return resized_image

    image_list = [_preprocess_single_image(image) for image in image_list]
    true_image_shapes = tf.stack([tf.shape(image) for image in image_list])
    images = tf.stack(image_list)

    return images, true_image_shapes


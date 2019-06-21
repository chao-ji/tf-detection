"""DetectionModel implements the methods needed to run the forward pass of
SSD (thus shared by all three model runners -- Trainer, Evaluator, Inferencer).
"""
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

  This base class is to be subclassed by Trainer, Evaluator and Inferencer to 
  perform training, evaluating, and inference making, respectively.

  Implements methods `preprocess`, `predict`, which are shared in the workflows 
  of Trainer, Evaluator, and Inferencer. The `is_training` and `mode` properties
  are set accordingly in the subclasses.
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
      image_resizer_fn: a callable that resizes an input image (3-D tensor) into
        one with desired property. 
      normalizer_fn: a callable that normalizes input image pixel values. 
      feature_extractor: an instance of SsdFeatureExtractor. 
      anchor_generator: an instance of AnchorGenerator. 
      box_predictor: an instance of BoxPredictor. Generates box encoding 
        predictions and class predictions.
    """
    self._image_resizer_fn = image_resizer_fn
    self._normalizer_fn = normalizer_fn
    self._feature_extractor = feature_extractor 
    self._anchor_generator = anchor_generator
    self._box_predictor = box_predictor

  @abstractproperty
  def is_training(self):
    """Returns a bool scalar indicating if model is in training mode."""
    pass

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating mode of model (train, eval or infer).
    """
    pass

  @abstractmethod
  def create_restore_saver(self, *args, **kwargs):
    """Creates a tf.train.Saver instance for restoring model.

    Depending on the `mode` of subclass, return a Saver instance initialized 
    with either all global variables (eval, infer) or a subset of global 
    variables (train).

    To be implemented by subclasses.

    Returns:
      a tf.train.Saver instance.
    """
    pass

  @property
  def extract_features_scope(self):
    return 'FeatureExtractor'

  def predict(self, inputs):
    """Runs the batched input images through the forward pass, generating the
    box encoding prediction tensor and class prediction tensor.

    Args:
      inputs: a float tensor of shape [batch_size, height, with, channels]
        holding batched input images in a minibatch.

    Returns:
      predictionr_dict: a dict mapping from strings to tensors, holding the
        following entries: 
        {
          'box_encoding_predictions': [batch_size, num_anchors, 1, 4],
          'class_predictions': [batch_size, num_anchors, num_classes + 1],
          'anchor_boxlist_list': a list of BoxList instance, each holding
            `num_anchors_i` anchor boxes. Length is equal to `batch_size`.    
        }
        Note: `num_anchors` is the total num of anchors summed over all feature 
        maps.
    """
    with slim.arg_scope([slim.batch_norm],
                        is_training=(self.is_training and
                            not self._freeze_batch_norm),
                        updates_collections=tf.GraphKeys.UPDATE_OPS):
      with tf.variable_scope(None, self.extract_features_scope, [inputs]):
        # generates feature map list
        feature_map_list = self._feature_extractor.extract_features(inputs)
      # generates box encoding and class predictions (like output layer)
      (box_encoding_predictions_list, class_predictions_list
          ) = self._box_predictor.predict(feature_map_list)
      box_encoding_predictions = tf.concat(
          box_encoding_predictions_list, axis=1)
      class_predictions = tf.concat(class_predictions_list, axis=1)

      feature_map_spatial_dims = shape_utils.get_feature_map_spatial_dims(
          feature_map_list)
      anchors = box_list_ops.concatenate(
          self._anchor_generator.generate(feature_map_spatial_dims))

      anchors = [anchors] * box_encoding_predictions.shape[0].value
      prediction_dict = {
          DictFields.box_encoding_predictions:box_encoding_predictions,
          DictFields.class_predictions:class_predictions,
          'anchor_boxlist_list': anchors}
      return prediction_dict

  def preprocess(self, image_list):
    """Preprocesses input images.

    The input images (a list of tensors of shape [height, width, channels]) 
    with possibly variable spatial dimensions, will be resized to have the same
    height and width, with pixel values optionally normalized. Then the
    preprocessed image list will be batched into a 4-D tensor.

    Args:
      image_list: a list of `batch_size` float tensors of shape 
        [height_i, width_i, channels].

    Returns:
      images: a float tensor of shape [batch_size, height, width, channels].
    """
    def _preprocess_single_image(image):
      resized_image, _ = self._image_resizer_fn(image)
      if self._normalizer_fn is not None:
        resized_image = self._normalizer_fn(resized_image)
      return resized_image

    images = tf.stack([_preprocess_single_image(image) for image in image_list])
    return images

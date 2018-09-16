""""""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.utils import context_manager
from detection.utils import ops
from detection.utils import shape_utils

slim = tf.contrib.slim


class BoxPredictor(object):
  """Abstract base class for box predictor.

  A box predictor takes as input a list of feature map tensors and generates
  a tensor containing box location predictions and a tensor containing box
  class score predictions.
  """

  __metaclass__ = ABCMeta

  def __init__(self, num_classes):
    """Constructor.

    Args:
      num_classes: int scalar, num of classes.
    """
    self._num_classes = num_classes

  def predict(self, feature_map_tensor_list, scope=None):
    """Generates the box location predictions and box class score predictions.
    
    Args:
      feature_map_tensor_list: A list of feature map tensors with shape 
        [batch, height, width, channels].

    Returns:
      a list of groundtruth box coordinates predictions.
      a list of box class score predictions.
    """
    variable_scope = (context_manager.IdentityContextManager()
        if scope is None else tf.variable_scope(scope))
    with variable_scope:
      return self._predict(feature_map_tensor_list)

  @abstractmethod
  def _predict(self, feature_map_tensor_list):
    """Generates the box location predictions and box class score predictions.   

    Args:
      feature_map_tensor_list: A list of feature map tensors with shape 
        [batch, height, width, channels].

    Returns:
      a list of groundtruth box coordinates predictions.
      a list of box class score predictions.
    """
    pass


class ConvolutionalBoxPredictor(BoxPredictor):
  """Convolutional box predictor.

  Note that this box predictor predicts ONE set of box locations shared by
  ALL object classes as opposed to making predictions separately for EACH 
  object class.
  """
  def __init__(self,
               num_classes,
               num_predictions_list,
               conv_hyperparams_fn,
               kernel_size,
               box_code_size=4,
               use_depthwise=False):
    """Constructor.

    Args:
      num_classes: int scalar, num of classes.
      num_predictions_list: a list of ints, num of anchor boxes per feature 
        map cell. 
      conv_hyperparams_fn: a function that creates argument to 
        `slim.arg_scope`.
      kernel_size: int scalar or int 2-tuple, kernel size used for the conv op. 
      box_code_size: int scalar, box code size. Default is 4.
      use_depthwise: bool scalar, whether to use separable_conv2d instead of
        conv2d. 
    """
    super(ConvolutionalBoxPredictor, self).__init__(num_classes)
    self._num_predictions_list = num_predictions_list
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._kernel_size = kernel_size
    self._box_code_size = box_code_size
    self._use_depthwise = use_depthwise 

  def _predict(self, feature_map_tensor_list):
    """Generates the box location predictions and box class score predictions.

    Args:
      feature_map_tensor_list: A list of feature map tensors with shape 
        [batch, height, width, channels].

    Returns:
      box_encoding_predictions_list: a list of rank-4 tensors with shape 
        [batch, num_anchors_i, 1, 4] containing the anchors-encoded coordinate
        predictions (i.e. [t_y, t_x, t_height, t_width]) of the matched
        groundtruth boxes, where `num_anchors_i` is the num of anchors
        in the ith feature map. 
      class_score_predictions_list: a list of rank-3 tensors with shape 
        [batch, num_anchors_i, num_classes + 1], where `num_anchors_i` is the 
        num of anchors in the ith feature map.
    """
    box_encoding_predictions_list = []
    class_score_predictions_list = []
    num_class_slots = self._num_classes + 1

    box_predictor_scopes = [context_manager.IdentityContextManager()]
    if len(feature_map_tensor_list) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(feature_map_tensor_list))]

    with slim.arg_scope(self._conv_hyperparams_fn()):
      # The following overrides the settings in self._conv_hyperparams_fn to
      # make sure that the output of slim.conv2d be simply an affine 
      # transformation (i.e. Only biases are added to the output of conv op).
      with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
          activation_fn=None, normalizer_fn=None, normalizer_params=None):
        for tensor, num_predictions, box_predictor_scope in zip(
            feature_map_tensor_list,
            self._num_predictions_list,
            box_predictor_scopes):
          with box_predictor_scope:
            # box encoding predictions branching out of `tensor`
            output_size = num_predictions * self._box_code_size
            if self._use_depthwise:
              box_encoding_predictions = ops.split_separable_conv2d(
                  tensor,
                  output_size,
                  self._kernel_size,
                  depth_multiplier=1,
                  stride=1,
                  padding='SAME',
                  scope='BoxEncodingPredictor')
            else:
              box_encoding_predictions = slim.conv2d(
                  tensor,
                  output_size,
                  self._kernel_size,
                  scope='BoxEncodingPredictor')
            # class score predictions branching out of `tensor`
            output_size = num_predictions * num_class_slots
            if self._use_depthwise:
              class_score_predictions = (
                  ops.split_separable_conv2d(
                      tensor,
                      output_size,
                      self._kernel_size,
                      depth_multiplier=1,
                      stride=1,
                      padding='SAME',
                      scope='ClassPredictor'))
            else:
              class_score_predictions = slim.conv2d(
                  tensor,
                  output_size,
                  self._kernel_size,
                  scope='ClassPredictor')
                  
            batch, height, width, _ = (
                shape_utils.combined_static_and_dynamic_shape(tensor))

            box_encoding_predictions = tf.reshape(
                box_encoding_predictions, 
                tf.stack([batch, 
                          height * width * num_predictions,
                          1,
                          self._box_code_size])) 
            box_encoding_predictions_list.append(box_encoding_predictions)

            class_score_predictions = tf.reshape(
                class_score_predictions,
                tf.stack([batch,
                          height * width * num_predictions,
                          num_class_slots]))
            class_score_predictions_list.append(class_score_predictions)

    return box_encoding_predictions_list, class_score_predictions_list


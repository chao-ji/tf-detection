from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.utils import misc_utils 
from detection.utils import ops
from detection.utils import shape_utils

slim = tf.contrib.slim


class BoxPredictor(object):
  """Abstract base class for box predictor.

  A box predictor is like the output layer in the classification network that 
  generates predicted logits through linear transformation (i.e. Y = XW + B). 
  It takes a (or a list of) feature map tensor(s) as input and outputs 
    -- a (or a list of) tensor(s) holding box location encoding predictions.
    -- a (or a list of) tensor(s) holding class score predictions.
  """
  __metaclass__ = ABCMeta

  def __init__(self, num_classes, box_code_size=4):
    """Constructor.

    Args:
      num_classes: int scalar, num of classes.
      box_code_size: int scalar, box code size. Defaults to 4.
    """
    self._num_classes = num_classes
    self._box_code_size = box_code_size

  def predict(self, feature_map_tensor_list, scope=None):
    """Generates the box location encoding predictions and box class score 
    predictions. Each tensor in the output list `box_encoding_predictions_list`
    and `class_score_predictions_list` corresponds to a tensor in the input
    `feature_map_tensor_list`.

    Calls and wraps `self._predict()` with a name scope.
    
    Args:
      feature_map_tensor_list: a list of float tensors of shape 
        [batch_size, height_i, width_i, channels_i].
      scope: string scalar, name scope.

    Returns:
      box_encoding_predictions_list: a list of float tensors of shape 
        [batch_size, num_anchors_i, q, code_size], where `q` is 1 or 
        `num_classes`.
      class_score_predictions_list: a list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1].
    """
    variable_scope = (misc_utils.IdentityContextManager()
        if scope is None else tf.variable_scope(scope))
    with variable_scope:
      (box_encoding_predictions_list, class_score_predictions_list
          ) = self._predict(feature_map_tensor_list)
      return box_encoding_predictions_list, class_score_predictions_list

  @abstractmethod
  def _predict(self, feature_map_tensor_list):
    """Generates the box location encoding predictions and box class score 
    predictions. Each tensor in the output list `box_encoding_predictions_list`
    and `class_score_predictions_list` corresponds to a tensor in the input
    `feature_map_tensor_list`.

    Args:
      feature_map_tensor_list: a list of float tensors of shape 
        [batch_size, height_i, width_i, channels_i].

    Returns:
      box_encoding_predictions_list: a list of float tensors of shape 
        [batch_size, num_anchors_i, q, code_size], where `q` is 1 or
        `num_classes`.
      class_score_predictions_list: a list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1].
    """
    pass


class ConvolutionalBoxPredictor(BoxPredictor):
  """Convolutional box predictor.

  Note that this subclass of BoxPredictor predicts **ONE** set of box location 
  encodings shared by **ALL** object classes as opposed to making predictions 
  separately for **EACH** object class.
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
      conv_hyperparams_fn: a callable that, when called, creates a dict holding 
        arguments to `slim.arg_scope`.
      kernel_size: int scalar or int 2-tuple, kernel size used for the conv op. 
      box_code_size: int scalar, box code size. Defaults to 4.
      use_depthwise: bool scalar, whether to use separable_conv2d instead of
        conv2d. 
    """
    super(ConvolutionalBoxPredictor, self).__init__(num_classes, box_code_size)
    self._num_predictions_list = num_predictions_list
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise 

  def _predict(self, feature_map_tensor_list):
    """Generates the box location encoding predictions and box class score 
    predictions. Each tensor in the output list `box_encoding_predictions_list`
    and `class_score_predictions_list` corresponds to a tensor in the input
    `feature_map_tensor_list`, and the num of anchors generated for `i`th 
    feature map, `num_anchors_i = height_i * width_i * num_predictions_list[i]`.

    For example, given input feature map list of shapes
       [[1, 19, 19, channels_1],
        [1, 10, 10, channels_2],
        [1, 5,  5,  channels_3],
        [1, 3,  3,  channels_4],
        [1, 2,  2,  channels_5],
        [1, 1,  1,  channels_6]]
    and
    `num_predictions_list` = [3, 6, 6, 6, 6, 6],

    the output tensor lists have `num_anchors_i` = [1083, 600, 150, 54, 24, 6].

    Args:
      feature_map_tensor_list: a list of float tensors of shape 
        [batch_size, height_i, width_i, channels_i].

    Returns:
      box_encoding_predictions_list: a list of float tensors of shape 
        [batch_size, num_anchors_i, 1, 4], holding anchor-encoded box 
        coordinate predictions (i.e. t_y, t_x, t_h, t_w).
      class_score_predictions_list: a list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1], holding one-hot
        encoded box class score predictions.
    """
    box_encoding_predictions_list = []
    class_score_predictions_list = []
    num_class_slots = self._num_classes + 1
    box_code_size = self._box_code_size

    box_predictor_scopes = [misc_utils.IdentityContextManager()]
    if len(feature_map_tensor_list) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(feature_map_tensor_list))]

    with slim.arg_scope(self._conv_hyperparams_fn()):
      # the following inner arg_scope overrides the settings in outer scope 
      # self._conv_hyperparams_fn to make sure that the conv ops only perform 
      # linear projections (i.e. like the output layer in the classification
      # network).
      with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
          activation_fn=None, normalizer_fn=None, normalizer_params=None):
        for tensor, num_predictions, box_predictor_scope in zip(
            feature_map_tensor_list,
            self._num_predictions_list,
            box_predictor_scopes):
          with box_predictor_scope:
            # box encoding predictions branching out of `tensor`
            output_size = num_predictions * box_code_size
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
              class_score_predictions = ops.split_separable_conv2d(
                  tensor,
                  output_size,
                  self._kernel_size,
                  depth_multiplier=1,
                  stride=1,
                  padding='SAME',
                  scope='ClassPredictor')
            else:
              class_score_predictions = slim.conv2d(
                  tensor,
                  output_size,
                  self._kernel_size,
                  scope='ClassPredictor')
                  
            batch, height, width, _ = (
                shape_utils.combined_static_and_dynamic_shape(tensor))

            box_encoding_predictions = tf.reshape(box_encoding_predictions, 
                [batch, height * width * num_predictions, 1, box_code_size]) 
            box_encoding_predictions_list.append(box_encoding_predictions)

            class_score_predictions = tf.reshape(class_score_predictions,
                [batch, height * width * num_predictions, num_class_slots])
            class_score_predictions_list.append(class_score_predictions)
    return box_encoding_predictions_list, class_score_predictions_list


class RcnnBoxPredictor(BoxPredictor):
  """RCNN Box Predictor. 

  Generates box location encoding predictions and class score predictions for 
  the Fast RCNN branch in a Faster RCNN network. It takes as input a feature
  map of shape [batch_num_proposals, height, width, channels], where slice 
  [i, :, :, :] holds the features of the ith proposal from the RPN, and outputs 
  box encoding predictions tensor of shape 
  [batch_num_proposals, 1, num_classes, 4], and class score predictions tensor 
  of shape 
  [batch_num_proposals, 1, num_classes + 1].

  NOTE: the `batch_num_proposals` in the shape of input and output tensors is 
  equal to `batch_size * max_num_proposals`, as the proposals from different 
  images in the same batch are arranged in the 0th dimension.
  """
  def __init__(self,
               num_classes,
               fc_hyperparams_fn,
               box_code_size=4):
    """Constructor.

    Args:
      num_classes: int scalar, num of classes.
      fc_hyperparams_fn: a callable that, when called, creates a dict holding 
        arguments to `slim.arg_scope`.
      box_coder_size: int scalar, box code size. Defaults to 4.
    """
    super(RcnnBoxPredictor, self).__init__(num_classes, box_code_size)
    self._fc_hyperparams_fn = fc_hyperparams_fn

  def _predict_boxes_and_classes(self, feature_map):
    """Generates the box location encoding predictions and box class score 
    predictions. 

    Args:
      feature_map: a tensor of shape 
        [batch_num_proposals, height, width, channels].

    Returns:
      box_encoding_predictions_list: a tensor of shape 
        [batch_num_proposals, 1, num_classes, 4], holding anchor-encoded box 
        coordinate predictions (i.e. t_y, t_x, t_h, t_w).
      class_score_predictions_list: a tensor of shape
        [batch_num_proposals, 1, num_classes + 1], holding one-hot
        encoded box class score predictions.
    """
    spatial_averaged_feature_map = tf.reduce_mean(
        feature_map, [1, 2], keepdims=True, name='AvgPool')

    flattened_feature_map = tf.squeeze(spatial_averaged_feature_map)
    with slim.arg_scope(self._fc_hyperparams_fn()):
      box_encodings = slim.fully_connected(
          flattened_feature_map,
          self._num_classes * self._box_code_size,
          activation_fn=None,
          scope='BoxEncodingPredictor')
      class_predictions = slim.fully_connected(
          flattened_feature_map,
          self._num_classes + 1,
          activation_fn=None,
          scope='ClassPredictor')

    box_encodings = tf.reshape(
        box_encodings, [-1, 1, self._num_classes, self._box_code_size])
    class_predictions = tf.reshape(
        class_predictions, [-1, 1, self._num_classes + 1])

    return box_encodings, class_predictions

  def _predict(self, feature_map_tensor_list):
    """Generates the box location encoding predictions and box class score 
    predictions. 

    Args:
      feature_map_tensor_list: a list (length 1) of float tensors of shape 
        [batch_num_proposals, height, width, channels].

    Returns:
      box_encoding_predictions_list: a list (length 1) of float tensors of shape
        [batch_num_proposals, 1, num_classes, 4], holding anchor-encoded box 
        coordinate predictions (i.e. t_y, t_x, t_h, t_w).
      class_score_predictions_list: a list (length 1) of float tensors of shape
        [batch_num_proposals, 1, num_classes + 1], holding one-hot
        encoded box class score predictions.      
    """
    # [1 * 300, 4, 4, 1024]
    box_encodings, class_predictions = self._predict_boxes_and_classes(
        feature_map_tensor_list[0]) 
    box_encoding_predictions_list = [box_encodings]
    class_score_predictions_list = [class_predictions]
    return box_encoding_predictions_list, class_score_predictions_list


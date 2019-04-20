from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.utils import misc_utils

slim = tf.contrib.slim


class MaskPredictor(object):
  """Abstract base class for mask predictor.

  A mask predictor generates instance mask predictions. It takes a list of 
  feature map tensors as input and outputs a list of predicted mask tensors.
  """
  __metaclass__ = ABCMeta

  def predict(self, feature_map_tensor_list, scope=None):
    """Generates mask predictions. Each tensor in the output list corresponds
    to a tensor in the input `feature_map_tensor_list`. 

    Args:
      feature_map_tensor_list: a list of float tensors of shape
        [batch_size, height_i, width_i, channels_i].
      scope: string scalar, name scope.

    Returns:
      mask_predictions_list: a list of float tensors of shape 
        [batch_size, num_classes, mask_height, mask_width]. 
    """
    variable_scope = (misc_utils.IdentityContextManager()
        if scope is None else tf.variable_scope(scope))
    with variable_scope:
      mask_predictions_list = self._predict(feature_map_tensor_list)
      return mask_predictions_list

  @abstractmethod
  def _predict(self, feature_map_tensor_list):
    """Generates mask predictions. Each tensor in the output list corresponds
    to a tensor in the input `feature_map_tensor_list`.

    Args:
      feature_map_tensor_list: a list of float tensors of shape
        [batch_size, height_i, width_i, channels_i].

    Returns:
      mask_predictions_list: a list of float tensors of shape
        [batch_size, num_classes, mask_height, mask_width]. 
    """
    pass


class RcnnMaskPredictor(MaskPredictor):
  """RCNN Mask Predictor.

  Generates mask predictions for the mask branch in a Mask RCNN network. It 
  takes as input a feature map of shape 
  [batch_num_proposals, height, width, channels], where the slice [i, :, :, :] 
  holds the features of the ith proposal from the RPN, and outputs mask 
  predictions tensor of shape 
  [batch_num_proposals, num_classes, mask_height, mask_width].

  NOTE: the `batch_num_proposals` in the shape of input and output tensors is 
  equal to `batch_size * max_num_proposals`, as the proposals from different 
  images in the same batch are arranged in the 0th dimension.
  """
  def __init__(self, 
               conv_hyperparams_fn, 
               num_masks, 
               mask_height, 
               mask_width, 
               num_conv_layers=2, 
               depths=256):
    """Constructor.

    Args:
      conv_hyperparams_fn: a callable that, when called, creates a dict holding
        arguments to `slim.arg_scope`.
      num_masks: int scalar, num of masks to be predicted per feature map. 
        Typically set to `num_classes` or 1.
      mask_height: int scalar, mask height.
      mask_width: int scalar, mask width.
      num_conv_layers: int scalar, num of conv layers between input feature map 
        and predicted mask tensor.
      depths: int scalar, depth of all but the final convolution operation (
        final conv op has depth `num_masks`).
    """
    self._num_masks = num_masks
    self._mask_height = mask_height
    self._mask_width = mask_width
    self._num_conv_layers = num_conv_layers
    self._depths = depths
    self._conv_hyperparams_fn = conv_hyperparams_fn

  def _predict(self, feature_map_tensor_list):
    """Generates mask predictions.

    Args:
      feature_map_tensor_list: a list (length 1) of float tensors of shape
        [batch_num_proposals, height, width, channels].

    Returns:
      mask_predictions_list: a list (length 1) of float tensors of shape 
        [batch_num_proposals, num_classes, mask_height, mask_width].
    """
    with slim.arg_scope(self._conv_hyperparams_fn()):
      # [batch_num_proposals, mask_height, mask_width, channels] 
      # e.g. 64, 33, 33, 2048
      upsampled_features = tf.image.resize_bilinear(
          feature_map_tensor_list[0],
          (self._mask_height, self._mask_width),
          align_corners=True)

      for _ in range(self._num_conv_layers - 1):
        upsampled_features = slim.conv2d(
            upsampled_features, num_outputs=self._depths, kernel_size=3)
      # [batch_num_proposals, mask_height, mask_width, num_masks] 
      # e.g. 64, 33, 33, 90
      mask_predictions = slim.conv2d(upsampled_features, 
                                     num_outputs=self._num_masks, 
                                     activation_fn=None, 
                                     kernel_size=3)
      # e.g. [64, 90, 33, 33]
      mask_predictions = tf.transpose(mask_predictions, perm=[0, 3, 1, 2])
      mask_predictions_list = [mask_predictions]

    return mask_predictions_list

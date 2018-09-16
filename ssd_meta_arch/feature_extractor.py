""""""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

slim = tf.contrib.slim


class SsdFeatureExtractor(object):
  """Abstract base class for feature extractor for SSD model."""
  __metaclass__ = ABCMeta

  def __init__(self,
               conv_hyperparams_fn,
               depth_multiplier,
               reuse_weights=None,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      conv_hyperparams_fn: a function that creates argument to 
        `slim.arg_scope`.
      depth_multiplier: float scalar, float multiplier for the depth (number
        of channels) for all convolution ops. The value must be greater than
        zero. Typical usage will be to set this value in (0, 1) to reduce the
        number of parameters or computational cost of the model.
      reuse_weights: bool scalar or None, whether to reuse variables in
        tf.variable_scope.
      use_depthwise: bool scalar, whether to use separable_conv2d instead of
        conv2d.
      override_base_feature_extractor_hyperparams: bool scalar, whether to 
        override hyperparameters of the base feature extractor with the one
        from `conv_hyperparams_fn`.
    """
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._depth_multiplier = depth_multiplier
    self._reuse_weights = reuse_weights
    self._use_depthwise = use_depthwise
    self._override_base_feature_extractor_hyperparams = (
        override_base_feature_extractor_hyperparams)

  @abstractmethod
  def extract_features(self, inputs):
    """Extracts features from inputs.

    To be implemented by subclasses.

    Args:
      inputs: a rank-4 tensor with shape [batch, height, with, channels]
        containing the input images. 

    Returns: a list of rank-4 ([batch, height, width, channels]) feature 
      map tensors to be fed to box predictor.
    """
    pass


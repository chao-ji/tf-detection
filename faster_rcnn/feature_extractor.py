from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

slim = tf.contrib.slim


class FasterRcnnFeatureExtractor(object):
  """Abstract base class for feature extractor for Faster RCNN model."""
  __metaclass__ = ABCMeta

  def __init__(self, reuse_weights=None):
    """Constructor.

    Args:
      reuse_weights: bool scalar, whether to reuse variables in
        `tf.variable_scope`.
    """
    self._reuse_weights = reuse_weights

    self._first_stage_feature_extractor_scope = 'FirstStageFeatureExtractor'
    self._second_stage_feature_extractor_scope = 'SecondStageFeatureExtractor'

  def extract_first_stage_features(self, inputs, scope=None):
    """Extracts first stage features for RPN proposal prediction and
    for ROI pooling.

    Calls and wraps `_extract_first_stage_features` with a name scope.

    Args:
      inputs: float tensor of shape [batch_size, height, width, depth].
      scope: string scalar, scope name.

    Returns:
      shared_feature_map: float tensor of shape 
        [batch_size, height_out, width_out, depth_out].
    """
    with tf.variable_scope(
        scope, self._first_stage_feature_extractor_scope, [inputs]):
      return self._extract_first_stage_features(inputs)

  @abstractmethod
  def _extract_first_stage_features(self):
    """Extracts first stage features for RPN proposal prediction and
    for ROI pooling. To be implemented by subclasses.
    """
    pass

  def extract_second_stage_features(self, proposal_feature_maps, scope=None):
    """Extracts second stage features for final box encoding and class 
    prediction.

    Calls and wraps `_extract_first_stage_features` with a name scope.

    Args:
      proposal_feature_maps: float tensor of shape 
        [batch_size * num_proposals, height_in, width_in, depth_in].
      scope: string scalar, scope name.

    Returns: 
      proposal_classifier_features: float tensor of shape
        [batch_size * num_proposals, height_out, width_out, depth_out].
    """
    with tf.variable_scope(scope, 
      self._second_stage_feature_extractor_scope, [proposal_feature_maps]):
      return self._extract_second_stage_features(proposal_feature_maps)

  @abstractmethod
  def _extract_second_stage_features(self):
    """Extracts second stage features for final box encoding and class 
    prediction. To be implemented by subclasses.
    """
    pass

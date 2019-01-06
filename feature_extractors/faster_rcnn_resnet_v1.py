import tensorflow as tf

from detection.faster_rcnn import feature_extractor
from nets import resnet_utils
from nets import resnet_v1

slim = tf.contrib.slim


class FasterRcnnResnetV1FeatureExtractor(
    feature_extractor.FasterRcnnFeatureExtractor):
  """ResNet feature extractor for Faster RCNN model."""

  def __init__(self, 
               resnet_name, 
               resnet_fn, 
               output_stride, 
               weight_decay=0.0,
               reuse_weights=None):
    """Constructor.

    Args:
      resnet_name: string scalar, ResNet model name (e.g. 'resnet_v1_101')
      resnet_fn: a callable that takes as input the input feature maps and
        generates the output feature map.
      output_stride: int scalar, output stride (e.g. 16, 32). 
      weight_decay: float scalar, weight decay.
      reuse_weights: bool scalar, whether to reuse variables in
        `tf.variable_scope`.
    """
    super(FasterRcnnResnetV1FeatureExtractor, self).__init__(
        reuse_weights=reuse_weights)
    self._resnet_name = resnet_name
    self._resnet_fn = resnet_fn
    self._output_stride = output_stride
    self._weight_decay = weight_decay

  def _extract_first_stage_features(self, inputs):
    """Extracts first stage features for RPN proposal prediction and
    for ROI pooling.

    Args:
      inputs: float tensor of shape [batch_size, height, width, depth].

    Returns:
      shared_feature_map: float tensor of shape 
        [batch_size, height_out, width_out, depth_out].
    """
    with slim.arg_scope(
        resnet_utils.resnet_arg_scope(
            batch_norm_epsilon=1e-5,
            batch_norm_scale=True,
            weight_decay=self._weight_decay)):
      with tf.variable_scope(
          self._resnet_name, reuse=self._reuse_weights) as scope:
        _, end_points = self._resnet_fn(
            inputs,
            num_classes=None,
            is_training=None, # is_training is speciefied in outer scope
            global_pool=False,
            output_stride=self._output_stride,
            spatial_squeeze=False,
            scope=scope)
    return end_points[self._first_stage_feature_extractor_scope + '/%s/block3' 
        % self._resnet_name]

  def _extract_second_stage_features(self, proposal_feature_maps):
    """Extracts second stage features for final box encoding and class 
    prediction.

    Args:
      proposal_feature_maps: float tensor of shape 
        [batch_size * num_proposals, height_in, width_in, depth_in]

    Returns:
      proposal_classifier_features: float tensor of shape
        [batch_size * num_proposals, height_out, width_out, depth_out]

    Note that the conv layers in the funtion combined will have the same
    effect of a conv layer with 'stride = 2', so 
    `height_out = (height_in + 1) / 2` and `width_out = (width_in + 1) / 2`.
    """
    with tf.variable_scope(self._resnet_name, reuse=self._reuse_weights):
      with slim.arg_scope(
          resnet_utils.resnet_arg_scope(
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              weight_decay=self._weight_decay)):
        blocks = [resnet_utils.Block('block4', resnet_v1.bottleneck, 
            [{'depth': 2048, 'depth_bottleneck': 512, 'stride': 1}] * 3)]
        proposal_classifier_features = resnet_utils.stack_blocks_dense(
            proposal_feature_maps, blocks)
    return proposal_classifier_features


class FasterRcnnResnet101V1FeatureExtractor(FasterRcnnResnetV1FeatureExtractor):
  """ResNetV1-101 feature extractor for Faster RCNN model."""
  def __init__(self,
               output_stride,
               weight_decay=0.0,
               reuse_weights=None):
    """Constructor.

    Args:
      output_stride: int scalar, output stride (e.g. 16, 32).
      weight_decay: float scalar, weight decay.
      reuse_weights: bool scalar, whether to reuse variables in
        `tf.variable_scope`.
    """
    super(FasterRcnnResnet101V1FeatureExtractor, self).__init__('resnet_v1_101',
        resnet_v1.resnet_v1_101, output_stride, weight_decay, reuse_weights)

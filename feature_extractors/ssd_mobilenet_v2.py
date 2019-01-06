import tensorflow as tf

from detection.ssd import feature_extractor
from detection.feature_extractors import feature_map_generators
from nets.mobilenet import mobilenet_v2

slim = tf.contrib.slim


class SsdMobileNetV2FeatureExtractor(feature_extractor.SsdFeatureExtractor):
  """Mobilenet v2 feature extractor for SSD model."""
  def __init__(self, 
               conv_hyperparams_fn, 
               depth_multiplier, 
               reuse_weights=None, 
               use_depthwise=False):
    """Constructor.

    Args:
      conv_hyperparams_fn: a callable that, when called, creates a dict holding 
        arguments to `slim.arg_scope`.
      depth_multiplier: float scalar, float multiplier for the depth (num
        of channels) for all convolution ops. The value must be greater than
        zero. Typical usage will be to set this value in (0, 1) to reduce the
        number of parameters or computational cost of the model.
      reuse_weights: bool scalar, whether to reuse variables in
        `tf.variable_scope`.
      use_depthwise: bool scalar, whether to use separable_conv2d instead of
        conv2d.
    """
    super(SsdMobileNetV2FeatureExtractor, self).__init__(
        conv_hyperparams_fn,
        depth_multiplier,
        reuse_weights,
        use_depthwise)
 
  def extract_features(self, inputs):
    """Extracts features from inputs.

    This function adds 4 additional feature maps on top of 
    'layer_15/expansion_output' and 'layer_19' in the base Mobilenet v2 network.

    Args:
      inputs: a tensor of shape [batch_size, height, with, channels],
        holding the input images.

    Returns: 
      a list of 6 float tensors of shape [batch_size, height, width, channels],
        holding feature map tensors to be fed to box predictor.
    """
    feature_map_specs_dict = {
        'layer_name': ['layer_15/expansion_output', 'layer_19', 
            None, None, None, None],
        'layer_depth': [None, None, 512, 256, 256, 128]}

    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)):
        _, end_points = mobilenet_v2.mobilenet_base(
            inputs, 
            final_endpoint='layer_19', 
            depth_multiplier=self._depth_multiplier, 
            scope=scope)

      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.ssd_feature_maps(
            feature_map_tensor_dict=end_points,
            feature_map_specs_dict=feature_map_specs_dict,
            depth_multiplier=1,
            use_depthwise=self._use_depthwise,
            insert_1x1_conv=True)
        feature_map_list = list(feature_maps.values())
        return feature_map_list

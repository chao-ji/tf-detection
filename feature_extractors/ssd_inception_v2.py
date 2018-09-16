""""""
import tensorflow as tf

from detection.ssd_meta_arch import feature_extractor 
from detection.feature_extractors import feature_map_generators
from nets import inception_v2

slim = tf.contrib.slim


class SsdInceptionV2FeatureExtractor(feature_extractor.SsdFeatureExtractor):
  """Inception v2 feature extractor for SSD model."""
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
    super(SsdInceptionV2FeatureExtractor, self).__init__(
        conv_hyperparams_fn,
        depth_multiplier,
        reuse_weights,
        use_depthwise,
        override_base_feature_extractor_hyperparams)
    if not self._override_base_feature_extractor_hyperparams:
      raise ValueError('SSD Inception V2 feature extractor always uses'
                       'scope returned by `conv_hyperparams_fn` for both the '
                       'base feature extractor and the additional layers '
                       'added since there is no arg_scope defined for the base '
                       'feature extractor.')

  def extract_features(self, inputs):
    """Extracts features from inputs.

    This function adds four additional feature maps on top of 'Mixed_4c' and
    'Mixed_5c' in the base Inception v2 network.

    Args:
      inputs: a rank-4 tensor with shape [batch, height, with, channels]
        containing the input images. 

    Returns: a list of six rank-4 ([batch, height, width, channels]) feature 
      map tensors to be fed to box predictor.
    """
    feature_map_specs_dict = {
        'layer_name': ['Mixed_4c', 'Mixed_5c', None, None, None, None],
        'layer_depth': [None, None, 512, 256, 256, 128]
    }

    with slim.arg_scope(self._conv_hyperparams_fn()):
      with tf.variable_scope('InceptionV2', reuse=self._reuse_weights) as scope:
        _, feature_map_tensor_dict = inception_v2.inception_v2_base(
          inputs,
          final_endpoint='Mixed_5c',
          min_depth=16,
          depth_multiplier=self._depth_multiplier,
          scope=scope)
        feature_maps = feature_map_generators.ssd_feature_maps(
            feature_map_tensor_dict,
            feature_map_specs_dict,
            self._depth_multiplier,
            self._use_depthwise, 
            insert_1x1_conv=True)
        feature_map_list = list(feature_maps.values())
        return feature_map_list


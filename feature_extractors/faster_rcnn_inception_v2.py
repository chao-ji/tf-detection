import tensorflow as tf

from detection.faster_rcnn import feature_extractor
from nets import inception_v2

slim = tf.contrib.slim


class FasterRcnnInceptionV2FeatureExtractor(
    feature_extractor.FasterRcnnFeatureExtractor):
  """Inception v2 feature extractor for Faster RCNN model."""

  def __init__(self, 
               batch_norm_params,
               depth_multiplier=1.0, 
               reuse_weights=None):
    """Constructor.

    Args:
      batch_norm_params:
      depth_multiplier: float scalar, float multiplier for the depth (num
        of channels) for all convolution ops. The value must be greater than
        zero. Typical usage will be to set this value in (0, 1) to reduce the
        number of parameters or computational cost of the model. 
      reuse_weights: bool scalar, whether to reuse variables in
        `tf.variable_scope`.
    """
    super(FasterRcnnInceptionV2FeatureExtractor, self).__init__(
        reuse_weights=reuse_weights)
    self._depth_multiplier=depth_multiplier
    self._batch_norm_params = batch_norm_params

  def _extract_first_stage_features(self, inputs):
    """Extracts first stage features for RPN proposal prediction and
    for ROI pooling.

    Args:
      inputs: float tensor of shape [batch_size, height, width, depth].

    Returns:
      shared_feature_map: float tensor of shape 
        [batch_size, height_out, width_out, depth_out].
    """
    with tf.variable_scope('InceptionV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
          # is_training
          normalizer_fn=slim.batch_norm,
          normalizer_params=self._batch_norm_params):
        _, end_points = inception_v2.inception_v2_base(
            inputs,
            final_endpoint='Mixed_4e',
            min_depth=16,
            depth_multiplier=self._depth_multiplier,
            scope=scope)
        return end_points['Mixed_4e']

  def _extract_second_stage_features(self, proposal_feature_maps):
    """Extracts second stage features for final box encoding and class 
    prediction.

    Args:
      proposal_feature_maps: float tensor of shape 
        [batch_size * num_proposals, height_in, width_in, depth_in].

    Returns:
      proposal_classifier_features: float tensor of shape
        [batch_size * num_proposals, height_out, width_out, depth_out].

    Note that the conv layers in the funtion combined will have the same
    effect of a conv layer with 'stride = 2', so 
    `height_out = (height_in + 1) / 2` and `width_out = (width_in + 1) / 2`.
    """
    depth = lambda d: max(int(d * self._depth_multiplier), 16)
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
    concat_dim = 3

    net = proposal_feature_maps
    with tf.variable_scope('InceptionV2', reuse=self._reuse_weights):
      with slim.arg_scope(
          [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
          stride=1,
          padding='SAME',
          data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
            normalizer_fn=slim.batch_norm,
            normalizer_params=self._batch_norm_params):
          with tf.variable_scope('Mixed_5a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(
                  net, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                     scope='Conv2d_0b_3x3')
              branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                         scope='MaxPool_1a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], concat_dim)

          with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(352), [1, 1],
                                     scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                     scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(
                  net, depth(160), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(
                  branch_3, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.1),
                  scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_dim)

          with tf.variable_scope('Mixed_5c'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(352), [1, 1],
                                     scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                     scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
              branch_3 = slim.conv2d(
                  branch_3, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.1),
                  scope='Conv2d_0b_1x1')
            proposal_classifier_features = tf.concat(
                [branch_0, branch_1, branch_2, branch_3], concat_dim)
    return proposal_classifier_features

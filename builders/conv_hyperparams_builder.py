import tensorflow as tf

from detection.protos import conv_hyperparams_pb2

slim = tf.contrib.slim


def build(config):
  """Factory function that builds a function that returns arg_scope for building
  feature extractor.

  Args:
    config: a protobuf message storing ConvHyperparams configurations.

  Returns:
    arg_scope_fn: a callable that returns arg_scope for building feature 
      extractor. 
  """
  if not isinstance(config, conv_hyperparams_pb2.ConvHyperparams):
    raise ValueError('config must be an instance of ConvHyperparams message.')

  affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
  if (config.HasField('type') and 
      config.type == conv_hyperparams_pb2.ConvHyperparams.FC):
    affected_ops = [slim.fully_connected]

  regularizer = _build_regularizer(config.regularizer)
  initializer = _build_initializer(config.initializer)
  activation = _build_activation_fn(config.activation)

  batch_norm = None
  batch_norm_params = None
  if config.HasField('batch_norm'):
    batch_norm = slim.batch_norm
    batch_norm_params = build_batch_norm_params(config.batch_norm)


  # parameters specific to depthwise convolution
  depthwise_regularizer = None
  if (config.HasField('regularize_depthwise') and 
      config.regularize_depthwise is True):
    depthwise_regularizer = regularizer

  depthwise_activation = None
  if (config.HasField('activate_depthwise') and
      config.activate_depthwise is True):
    depthwise_activation = activation

  depthwise_batch_norm = None
  depthwise_batch_norm_params = None
  if (config.HasField('batch_norm_depthwise') and 
      config.batch_norm_depthwise is True):
    depthwise_batch_norm = batch_norm
    depthwise_batch_norm_params = batch_norm_params


  def arg_scope_fn():
    with slim.arg_scope(
        affected_ops,
        weights_regularizer=regularizer,
        weights_initializer=initializer,
        activation_fn=activation,
        normalizer_fn=batch_norm,
        normalizer_params=batch_norm_params):
      # potentially overriding the settings of separable_conv2d in the outer
      # arg_scope
      with slim.arg_scope(
          [slim.separable_conv2d],
          weights_regularizer=depthwise_regularizer,
          activation_fn=depthwise_activation,
          normalizer_fn=depthwise_batch_norm,
          normalizer_params=depthwise_batch_norm_params) as sc:
        return sc

  return arg_scope_fn


def _build_activation_fn(config):
  """Builds activation function.

  Args:
    config: an int scalar representing the enum value of 
      ConvHyperparams.Activation. 

  Returns:
    tf.nn.relu or tf.nn.relu6 or None.
  """
  if config == conv_hyperparams_pb2.ConvHyperparams.NONE:
    return None
  if config == conv_hyperparams_pb2.ConvHyperparams.RELU:
    return tf.nn.relu
  if config == conv_hyperparams_pb2.ConvHyperparams.RELU_6:
    return tf.nn.relu6
  raise ValueError('Unknown activation function.') 


def _build_regularizer(config):
  """Builds regularizer.

  Args:
    config: a protobuf message storing Regularizer configuration.

  Returns:
    slim.l1_regularizer or slim.l2_regularizer. 
  """
  if not isinstance(config, conv_hyperparams_pb2.Regularizer):
    raise ValueError('config must be an instance of Regularizer message.')

  regularizer_oneof = config.WhichOneof('regularizer_oneof')
  if regularizer_oneof == 'l1_regularizer':
    return slim.l1_regularizer(scale=float(config.l1_regularizer.weight))
  if regularizer_oneof == 'l2_regularizer':
    return slim.l2_regularizer(scale=float(config.l2_regularizer.weight))
  raise ValueError('Unknown regularizer')


def _build_initializer(config):
  """Builds initializer.

  Args:
    config: a protobuf message storing Initializer configuration.

  Returns:
    tf.truncated_normal_initializer or tf.random_normal_initializer.
  """
  if not isinstance(config, conv_hyperparams_pb2.Initializer):
    raise ValueError('config must be an instance of Initializer message.')

  initializer_oneof = config.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.truncated_normal_initializer(
        mean=config.truncated_normal_initializer.mean,
        stddev=config.truncated_normal_initializer.stddev)
  if initializer_oneof == 'random_normal_initializer':
    return tf.random_normal_initializer(
        mean=config.random_normal_initializer.mean,
        stddev=config.random_normal_initializer.stddev)
  if initializer_oneof == 'variance_scaling_initializer':
    return slim.variance_scaling_initializer(
        config.variance_scaling_initializer.factor,
        config.variance_scaling_initializer.mode,
        config.variance_scaling_initializer.uniform)

  raise ValueError('Unknown initializer')


def build_batch_norm_params(config):
  """Builds a dict holing batch norm parameters.

  Args:
    config: a protobuf message storing BatchNorm configuration.

  Returns:
    batch_norm_params: dict pamming arg name to arg value. 
  """
  if not isinstance(config, conv_hyperparams_pb2.BatchNorm):
    raise ValueError('config must be an instance of BatchNorm message.')

  batch_norm_params = {
      'decay': config.decay,
      'epsilon': config.epsilon,
      'center': config.center,
      'scale': config.scale
  }
  return batch_norm_params
  

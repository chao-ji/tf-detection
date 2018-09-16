import tensorflow as tf

from detection.protos import conv_hyperparams_pb2

slim = tf.contrib.slim


def build(config):
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
    batch_norm_params = _build_batch_norm_params(config.batch_norm)


  depthwise_regularizer = None
  if (config.HasField('regularize_depthwise') and 
      config.regularize_depthwise is True):
    depthwise_regularizer = regularizer

  depthwise_batch_norm = None
  depthwise_batch_norm_params = None
  if (config.HasField('batch_norm_depthwise') and 
      config.batch_norm_depthwise is True):
    depthwise_batch_norm = batch_norm
    depthwise_batch_norm_params = batch_norm_params

  depthwise_activation = None
  if (config.HasField('activate_depthwise') and
      config.activate_depthwise is True):
    depthwise_activation = activation

  def arg_scope_fn():
    with slim.arg_scope(
        affected_ops,
        weights_regularizer=regularizer,
        weights_initializer=initializer,
        activation_fn=activation,
        normalizer_fn=batch_norm,
        normalizer_params=batch_norm_params):
      # potentially overriding the settings of separable_conv2d
      with slim.arg_scope(
          [slim.separable_conv2d],
          weights_regularizer=depthwise_regularizer,
          activation_fn=depthwise_activation,
          normalizer_fn=depthwise_batch_norm,
          normalizer_params=depthwise_batch_norm_params) as sc:
        return sc

  return arg_scope_fn


def _build_activation_fn(config):
  if config == conv_hyperparams_pb2.ConvHyperparams.NONE:
    return None
  if config == conv_hyperparams_pb2.ConvHyperparams.RELU:
    return tf.nn.relu
  if config == conv_hyperparams_pb2.ConvHyperparams.RELU_6:
    return tf.nn.relu6
  raise ValueError('Unknown activation function') 


def _build_regularizer(config):
  regularizer_oneof = config.WhichOneof('regularizer_oneof')
  if regularizer_oneof == 'l1_regularizer':
    return slim.l1_regularizer(scale=float(config.l1_regularizer.weight))
  if regularizer_oneof == 'l2_regularizer':
    return slim.l2_regularizer(scale=float(config.l2_regularizer.weight))
  raise ValueError('Unknown regularizer')


def _build_initializer(config):
  initializer_oneof = config.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.truncated_normal_initializer(
        mean=config.truncated_normal_initializer.mean,
        stddev=config.truncated_normal_initializer.stddev)
  if initializer_oneof == 'random_normal_initializer':
    return tf.random_normal_initializer(
        mean=config.random_normal_initializer.mean,
        stddev=config.random_normal_initializer.stddev)
  raise ValueError('Unknown initializer')


def _build_batch_norm_params(config):
  batch_norm_params = {
      'decay': config.decay,
      'epsilon': config.epsilon,
      'center': config.center,
      'scale': config.scale
  }
  return batch_norm_params
  

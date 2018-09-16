import tensorflow as tf


def build(config):
  optimizer_type = config.WhichOneof('optimizer_oneof')
  use_moving_average = config.use_moving_average
  moving_average_decay = config.moving_average_decay

  if optimizer_type == 'rms_prop_optimizer':
    config = config.rms_prop_optimizer
    def rms_prop_optimizer_builder_fn():
      learning_rate = _build_learning_rate(config.learning_rate)
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          decay=config.decay,
          momentum=config.momentum_optimizer_value,
          epsilon=config.epsilon)
      if use_moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=moving_average_decay)
      return optimizer, learning_rate
    return rms_prop_optimizer_builder_fn

  if optimizer_type == 'momentum_optimizer':
    config = config.momentum_optimizer
    def momentum_optimizer_builder_fn():
      learning_rate = _build_learning_rate(config.learning_rate)
      optimizer = tf.train.MomentumOptimizer(
          learning_rate,
          momentum=config.momentum_optimizer_value)
      if use_moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=moving_average_decay)
      return optimizer, learning_rate
    return momentum_optimizer_builder_fn

  if optimizer_type == 'adam_optimizer':
    config = config.adam_optimizer
    def adam_optimizer_builder_fn():
      learning_rate = _build_learning_rate(config.learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      if use_moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=moving_average_decay)
      return optimizer, learning_rate
    return adam_optimizer_builder_fn  

  raise ValueError('Unknown optimizer.')


def _build_learning_rate(config):
  learning_rate_type = config.WhichOneof('learning_rate_oneof')

  if learning_rate_type == 'constant_learning_rate':
    config = config.constant_learning_rate
    learning_rate = tf.constant(config.learning_rate,
                                dtype=tf.float32,
                                name='learning_rate')
    return learning_rate

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = config.exponential_decay_learning_rate
    learning_rate = tf.train.exponential_decay(
        config.initial_learning_rate,
        tf.train.get_or_create_global_step(),
        config.decay_steps,
        config.decay_factor,
        staircase=config.staircase,
        name='learning_rate')
    return learning_rate

  raise ValueError('Unknown learning rate.')


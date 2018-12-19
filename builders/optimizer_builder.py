import tensorflow as tf
from detection.protos import optimizer_pb2


def build(config):
  """Builds optimizer and learning rate.

  Args:
    config: a protobuf message storing Optimizer configurations.

  Returns:
    a callable that returns a 2-tuple containing an instance of optimizer and
      a learning rate tensor.
  """
  if not isinstance(config, optimizer_pb2.Optimizer):
    raise ValueError('config must be an instance of Optimizer message.')

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
  """Builds learning rate.

  Args:
    config: a protobuf message storing LearningRate configurations.

  Returns:
    learning_rate: a float scalar tensor storing learning rate.
  """
  if not isinstance(config, optimizer_pb2.LearningRate):
    raise ValueError('config must be an instance of LearningRate message.')

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

  if learning_rate_type == 'manual_step_learning_rate':
    config = config.manual_step_learning_rate
    
    boundaries = [s.step for s in config.schedule]
    rates = [config.initial_learning_rate] + [s.learning_rate for s in config.schedule]
    learning_rate = manual_stepping(tf.train.get_or_create_global_step(), boundaries, rates, config.warmup)
    return learning_rate  

  raise ValueError('Unknown learning rate.')


def manual_stepping(global_step, boundaries, rates, warmup=False):
  """Generates manually stepped learning rate schedule.

  Example:
  Given boundaries = [10, 20], and rates [.1, .01, .001], the learning rate is
  returned as a scalar tensor, which equals
    .1 for global steps in interval [0, 10);
    .01 for global steps in interval [10, 20);
    .001 for global steps in interval [20, inf).
  If `warmup` is True, then the learning rate is linearly interpolated between
  .1 and .01 for global_steps in interval [0, 10).

  Note `boundaries` must be an increasing list of ints starting from a positive 
  integer, and has length `len(rates) - 1`. 

  Args:
    global_step: int scalar tensor, the global step starting from 0.
    boundaries: a list of increasing ints starting from a positive int, the
      steps at which learning rate is changed.
    rates: a list of floats of length `len(boundaries) + 1`, the learning rates
      in the intervals defined by the integers in `boundaries`.
    warmup: bool scalar, whether to linearly interpolate learning rates from 
      `rates[0]` to `rates[1]` for global steps within the interval 
      `[0, boundaries[0])`.

  Returns
    learning_rate: float scalar tensor, the learning rate at global step 
      `global_step`.
  """
  if len(rates) != len(boundaries) + 1:
    raise ValueError('`len(rates)` must be equal to `len(boundaries) + 1`.')

  if warmup:
    slope = float(rates[1] - rates[0]) / boundaries[0]
    warmup_steps = list(range(boundaries[0]))
    warmup_rates = [rates[0] + slope * step for step in warmup_steps]
    boundaries = warmup_steps[1:] + boundaries
    rates = warmup_rates + rates[1:]
     
  lower_cond = tf.concat([tf.less(global_step, boundaries), [True]], 0)
  upper_cond = tf.concat([[True], tf.greater_equal(global_step, boundaries)], 0)
  indicator = tf.to_float(tf.logical_and(lower_cond, upper_cond))
  learning_rate = tf.reduce_sum(rates * indicator)
  return learning_rate

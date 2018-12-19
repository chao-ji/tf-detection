import functools

import tensorflow as tf

from detection.core import post_processing
from detection.protos import postprocessing_pb2


def build(config):
  """Builds postprocessor (non-max suppressor and score converter).

  Args:
    config: a protobuf message storing PostProcessing configurations.

  Returns:
    non_max_suppressor_fn: a callable that performs non-max suppression.
    score_converter_fn: a callable that converts logits to scores.
  """
  if not isinstance(config, postprocessing_pb2.PostProcessing):
    raise ValueError('config must be an instance of PostProcessing message.')

  non_max_suppressor_fn = _build_non_max_suppressor(
      config.batch_non_max_suppression)
  score_converter_fn = _build_score_converter(
      config.score_converter,
      config.logit_scale)
  return non_max_suppressor_fn, score_converter_fn


def _build_non_max_suppressor(config):
  """Builds non-max suppressor.

  Args:
    config: a protobuf message storing BatchNonMaxSuppression configurations.

  Returns:
    non_max_suppressor_fn: a callable that performs non-max suppression.
  """
  if not isinstance(config, postprocessing_pb2.BatchNonMaxSuppression):
    raise ValueError('config must be an instance of BatchNonMaxSuppression.')

  non_max_suppressor_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=config.score_threshold,
      iou_thresh=config.iou_threshold,
      max_size_per_class=config.max_detections_per_class,
      max_total_size=config.max_total_detections)
  return non_max_suppressor_fn


def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
  """Builds score converter function where logits are scaled.

  Args:
    tf_score_converter_fn: a callable that converts logits tensor to score
      tensor: tf.identity, tf.sigmoid, tf.nn.softmax.
    logit_scale: a float scalar by which the raw logits are divided before
      going through the score converter function. 

  Returns:
    a callable that converts logits to scores where logits are scaled by
      `logit_scale`.
  """
  def score_converter_fn(logits):
    scaled_logits = tf.divide(logits, logit_scale, name='scale_logits')
    return tf_score_converter_fn(scaled_logits, name='convert_scores')
  score_converter_fn.__name__ = '{}_with_logit_scale'.format(
      tf_score_converter_fn.__name__)
  return score_converter_fn


def _build_score_converter(config, logit_scale):
  """Builds score converter function.

  Args:
    config: an int scalar representing the score converter ID (0-IDENTITY, 
      1-SIGMOID, 2-SOFTMAX).
    logit_scale: a float scalar by which the raw logits are divided before
      going through the score converter function.

  Returns:
    a callable that converts logits to scores where logits are optionally scaled
      by `logit_scale`.
  """
  if config == postprocessing_pb2.PostProcessing.IDENTITY:
    return _score_converter_fn_with_logit_scale(tf.identity, logit_scale)
  if config == postprocessing_pb2.PostProcessing.SIGMOID:
    return _score_converter_fn_with_logit_scale(tf.sigmoid, logit_scale)
  if config == postprocessing_pb2.PostProcessing.SOFTMAX:
    return _score_converter_fn_with_logit_scale(tf.nn.softmax, logit_scale)
  raise ValueError('Unknown score converter.')


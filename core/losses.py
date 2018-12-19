from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.utils import shape_utils

slim = tf.contrib.slim


class Loss(object):
  """Abstract base class for loss functions."""
  __metaclass__ = ABCMeta

  def __call__(self,
               predictions,
               targets,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Calls and wraps the `_compute_loss` method in a name scope.

    Args:
      predictions: a numeric tensor of shape [batch_size, num_anchors, ...]
        holding predicted values.
      targets: a numeric tensor of shape [batch_size, num_anchors, ...]
        holding localization or classification targets.
      ignore_nan_targets: bool scalar, whether to ignore NaN targets in the loss
        computation.
      scope: string scalar, name scope.
      params: dict mapping from string to objects, holding additional keyword
        arguments for specific implementations of the loss.

    Returns:
      loss: a tensor holding the computed loss.
    """
    with tf.name_scope(scope, 'Loss', [predictions, targets, params]):
      if ignore_nan_targets:
        targets = tf.where(tf.is_nan(targets), predictions, targets)
      return self._compute_loss(predictions, targets, **params)

  @abstractmethod
  def _compute_loss(self, predictions, targets, **params):
    """Specific implementation of loss function.

    To be implemented by subclasses.

    Args:
      predictions: a numeric tensor of shape [batch_size, num_anchors, ...]
        holding predicted values.
      targets: a numeric tensor of shape [batch_size, num_anchors, ...]
        holding localization or classification targets.
      params: dict mapping from string to objects, holding additional keyword
        arguments for specific implementations of the loss. 

    Returns:
      loss: a tensor holding the computed loss.
    """
    pass


class WeightedL2LocalizationLoss(Loss):
  """L2 localization loss with anchorwise prediction weighting.

  loss[i,j] = .5 * ||weights[i,j] * (prediction[i,j,:] - target[i,j,:])||^2
  where 0 <= i < batch_size, and 0 <= j < num_anchors. 
  """

  def _compute_loss(self, predictions, targets, weights):
    """compute loss.

    Args:
      predictions: a float tensor of shape [batch_size, num_anchors, code_size] 
        holding the encoded prediction of box coordinates.
      targets: a float tensor of shape [batch_size, num_anchors, code_size] 
        holding regression targets.
      weights: a float tensor of shape [batch_size, num_anchors], holding
        anchorwise weights.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors], holding the
        anchorwise loss.
    """
    l2_loss = 0.5 * tf.square(
        (predictions - targets) * tf.expand_dims(weights, 2))
    return tf.reduce_sum(l2_loss, 2)


class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function (Huber Loss) with anchorwise
  prediction weighting.
  """

  def __init__(self, delta=1.0):
    """Constructor.

    Args:
      delta: float scalar, delta for smooth L1 loss.
    """
    self._delta = delta

  def _compute_loss(self, predictions, targets, weights):
    """Compute loss.

    Args:
      predictions: a float tensor of shape [batch_size, num_anchors, code_size]
        holding the encoded prediction of box coordinates.
      targets: a float tensor of shape [batch_size, num_anchors, code_size] 
        holding regression targets.
      weights: a float tensor of shape [batch_size, num_anchors], holding 
        anchorwise weights.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors], holding the
        anchorwise loss.
    """
    smoothed_l1_loss = tf.losses.huber_loss(
        targets,
        predictions,
        delta=self._delta,
        weights=tf.expand_dims(weights, axis=2),
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE)
    return tf.reduce_sum(smoothed_l1_loss, axis=2)


class WeightedSigmoidClassificationLoss(Loss):
  """Sigmoid cross entropy classification loss function."""

  def _compute_loss(self, predictions, targets, weights, reduce_last_dim=False):
    """Compute loss.

    Args:
      predictions: float tensor of shape [batch_size, num_anchors, num_classes] 
        holding predicted logits for each class.
      targets: float tensor of shape [batch_size, num_anchors, num_classes] 
        holding one-hot encoded classification targets.
      weights: float tensor of shape [batch_size, num_anchors], holding 
        anchorwise weights.
      reduce_last_dim: bool scalar, whether to reduce-sum the last dimension of
        the loss tensor. Defaults to False.

    Returns:
      float tensor of shape [batch_size, num_anchors, num_classes] or 
        [batch_size, num_anchors] (`reduce_last_dim == True`), holding the
        anchorwise (and classwise) loss.
    """
    weights = tf.expand_dims(weights, 2)
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=predictions)
    sigmoid_loss *= weights
    if reduce_last_dim:
      sigmoid_loss = tf.reduce_sum(sigmoid_loss, axis=2)
    return sigmoid_loss

class WeightedSoftmaxClassificationLoss(Loss):
  """Softmax cross entropy classification loss function.
  
  Note: the anchorwise softmax loss is a scalar, as opposed to a vector (
  with `num_classes`) in the case of anchorwise sigmoid loss (
  WeightedSigmoidClassificationLoss).
  """

  def __init__(self, logit_scale=1.0):
    """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
    self._logit_scale = logit_scale

  def _compute_loss(self, predictions, targets, weights):
    """Compute loss.

    Args:
      predictions: float tensor of shape [batch_size, num_anchors, num_classes] 
        holding predicted logits for each class.
      targets: float tensor of shape [batch_size, num_anchors, num_classes] 
        holding one-hot encoded classification targets.
      weights: float tensor of shape [batch_size, num_anchors], holding 
        anchorwise weights.

    Returns:
      float tensor of shape [batch_size, num_anchors], holding the anchorwise 
        loss.
    """
    num_classes = shape_utils.combined_static_and_dynamic_shape(predictions)[-1]
    predictions = tf.divide(
        predictions, self._logit_scale, name='scale_logit')
    softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.reshape(targets, [-1, num_classes]),
        logits=tf.reshape(predictions, [-1, num_classes]))
    return tf.reshape(softmax_loss, tf.shape(weights)) * weights


class HardExampleMiner(object):
  """Hard example mining for regions in a list of images.
  """

  def __init__(self,
               num_hard_examples=64,
               iou_threshold=0.7,
               loss_type='both',
               cls_loss_weight=0.05,
               loc_loss_weight=0.06,
               max_negatives_per_positive=None,
               min_negatives_per_image=0):
    """Constructor.

    Args:
      num_hard_examples: int scalar, max num of hard examples to be
        selected per image used in NMS.
      iou_threshold: float scalar, min IOU for a box to be considered as being 
        overlapped with a previously selected box during NMS.
      loss_type: string scalar 'cls', 'loc', 'both', hard-mining only 
        classification loss, localization loss or both. If 'both', 
        `cls_loss_weight` and `loc_loss_weight` are used to compute a weighted
        sum of the two losses.
      cls_loss_weight: float scalar, weight for classification loss.
      loc_loss_weight: float scalar, weight for localization loss.
      max_negatives_per_positive: int or float scalar, the max num of negative 
        anchors to be retained for each positive anchor. If None, no constraint 
        is enforced on the positive-to-negative ratio.
      min_negatives_per_image: int scalar, the min num of negative anchors to
        sample for a given image. If positive, negative anchors will be sampled
        for an image without any positive anchors -- it is OK to have images
        without any positive detections.
    """
    self._num_hard_examples = num_hard_examples
    self._iou_threshold = iou_threshold
    self._loss_type = loss_type
    self._cls_loss_weight = cls_loss_weight
    self._loc_loss_weight = loc_loss_weight
    self._max_negatives_per_positive = (float(max_negatives_per_positive) 
        if max_negatives_per_positive is not None 
        else max_negatives_per_positive)
    self._min_negatives_per_image = min_negatives_per_image

    self._num_positives_list = None
    self._num_negatives_list = None

  def __call__(self,
               loc_losses,
               cls_losses,
               decoded_boxlist_list,
               match_list=None):
    """Computes localization and classification losses after hard mining.

    Args:
      loc_losses: a float tensor of shape [batch_size, num_anchors],
        holding anchorwise localization losses before mining.
      cls_losses: a float tensor of shape [batch_size, num_anchors],
        holding anchorwise classification losses before mining.
      decoded_boxlist_list: a list of BoxList instances of length `batch_size`, 
        holding decoded location predictions (i.e. float tensor of shape 
        [num_anchors, 4]).
      match_list: a list of Match instances of length `batch_size`, holding
        matching results between anchors and groundtruth boxes, which is a
        1-D vector of shape [num_anchors] storing indices of 1) matched 
        groundtruth box (>= 0); 2) unmatched (-1); ignored (-2). This is used
        to, if `max_negatives_per_positive` is not None, enforce a desired 
        positive-to-negative ratio.

    Returns:
      mined_loc_loss: a float scalar with sum (over the minibatch) of 
        localization losses from selected hard examples.
      mined_cls_loss: a float scalar with sum (over the minibatch) of 
        classification losses from selected hard examples.
    """
    mined_loc_losses = []
    mined_cls_losses = []
    loc_losses = tf.unstack(loc_losses)
    cls_losses = tf.unstack(cls_losses)
    batch_size = len(decoded_boxlist_list)

    if not match_list:
      match_list = batch_size * [None]
    if not (len(loc_losses) == len(decoded_boxlist_list) == len(cls_losses)):
      raise ValueError('loc_losses, cls_losses and decoded_boxlist_list '
                       'do not have compatible shapes.')
    if not isinstance(match_list, list):
      raise ValueError('match_list must be a list.')
    if len(match_list) != len(decoded_boxlist_list):
      raise ValueError('match_list must either be None or have '
                       'length=len(decoded_boxlist_list).')

    num_positives_list = []
    num_negatives_list = []
    

    for i, detection_boxlist in enumerate(decoded_boxlist_list):

      match = match_list[i]

      anchor_losses = cls_losses[i]
      if self._loss_type == 'loc':
        anchor_losses = loc_losses[i]
      elif self._loss_type == 'both':
        anchor_losses *= self._cls_loss_weight
        anchor_losses += loc_losses[i] * self._loc_loss_weight

      num_hard_examples = (self._num_hard_examples or 
                           detection_boxlist.num_boxes())

      selected_indices = tf.image.non_max_suppression(detection_boxlist.get(), 
          anchor_losses, num_hard_examples, self._iou_threshold)

      if self._max_negatives_per_positive is not None and match:
        (selected_indices, num_positives, num_negatives
            ) = self._subsample_selection_to_desired_neg_pos_ratio(
            selected_indices, match, self._max_negatives_per_positive,
            self._min_negatives_per_image)
        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)

      mined_loc_losses.append(
          tf.reduce_sum(tf.gather(loc_losses[i], selected_indices)))
      mined_cls_losses.append(
          tf.reduce_sum(tf.gather(cls_losses[i], selected_indices)))

    loc_loss = tf.reduce_sum(tf.stack(mined_loc_losses))
    cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses))

    if match and self._max_negatives_per_positive:
      self._num_positives_list = num_positives_list
      self._num_negatives_list = num_negatives_list
    return loc_loss, cls_loss

  def _subsample_selection_to_desired_neg_pos_ratio(self,
                                                    indices,
                                                    match,
                                                    max_negatives_per_positive,
                                                    min_negatives_per_image=0):
    """Subsample a collection of selected indices to a desired 
    positive-to-negative ratio.

    For example,

    matched_column_indicator = [T, F, F, T, F, F, F, T, F, F]
    unmatched_column_indicator = [F, T, T, F, T, T, T, F, T, T]
    indices = [0, 2, 4, 5, 8, 9]
    min_negatives_per_image = 0
    max_negatives_per_positive = 3.0

    then,

    positive_indicator = [T, F, F, F, F, F]
    negative_indicator = [F, T, T, T, T, T]        
    topk_negatives_indicator = [T, T, T, T, F, F]
    subsampled_selection_indices = [[0], [1], [2], [3]]
    selected_indices = [0, 2, 4, 5]

    Args:
      indices: an int tensor of shape [m_in], holding the indices of anchors
        selected by NMS (i.e. losses of anchors are in decreasing order).
      match: a Match instance, holding matching results between anchors and 
        groundtruth boxes, which is a 1-D vector of shape [num_anchors] storing
        indices of 1) matched groundtruth box (>= 0); 2) unmatched (-1); ignored 
        (-2). This is used to enforce a desired positive-to-negative ratio.
      max_negatives_per_positive: int or float scalar, the max num of negative 
        anchors to be retained for each positive anchor.
      min_negatives_per_image: int scalar, the min num of negative anchors to
        sample for a given image. If positive, negative anchors will be sampled
        for an image without any positive anchors -- it is OK to have images
        without any positive detections.

    Returns:
      selected_indices: int tensor of shape [m_out] (`<= m_in`), holding the 
        indices of selected anchor. This can be used to gather the selected 
        losses, e.g. `tf.gather(loc_losses, selected_indices)`.
      num_positives: int scalar tensor, the num of positive examples in 
        selected set of indices.
      num_negatives: int scalar tensor, the num of negative examples in 
        selected set of indices.
    """
    positives_indicator = tf.gather(match.matched_column_indicator(), indices)
    negatives_indicator = tf.gather(match.unmatched_column_indicator(), indices)
    num_positives = tf.reduce_sum(tf.to_int32(positives_indicator))
    max_negatives = tf.maximum(min_negatives_per_image,
                               tf.to_int32(max_negatives_per_positive *
                                           tf.to_float(num_positives)))
    topk_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator)), max_negatives)
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, topk_negatives_indicator))
    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    selected_indices = tf.reshape(
        tf.gather(indices, subsampled_selection_indices), [-1])
    return selected_indices, num_positives, num_negatives

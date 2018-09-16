""""""
import tensorflow as tf

from detection.utils import shape_utils
from detection.core import box_list


class TargetAssigner(object):
  """Target Assigner creates regression and classification targets for each
  anchor in an anchor boxlist.

  It implements a method called `assign` that takes as input a 
  box_list.BoxList of anchors (i.e. initial random guesses of the bounding box 
  locations) as well as a box_list.BoxList of groundtruth detections and class
  labels, to perform the following steps:

  1. Computes the pairwise similarity between anchors and groundtruth boxes.
  2. For each anchor box determines if there is a match, or there is no match,
    or simply ignores it.
  3. Assigns regression and classification targets.
    a. For those anchors where there is a match, assigns a regression and 
      classification target according to the matched groundtruth box.
    b. For those anchors where there is no match, assigns a dummy regression
      target, and assigns the background class as the classification target.
    c. For those that are ignored, assigns a dummy regression and dummy
      classification targets.
    d. Assigns a weight of 1.0 to those anchors with a non-dummy target, and 
      weight of 0.0 to whose with a dummy target, so that they don't contribute
      to the loss.
  """
  def __init__(self,
               region_similarity_calculator,
               matcher,
               box_coder,
               negative_class_weight=1.0):
    """Constructor.

    Args:
      region_similarity_calculator: a (region_similarity_calculator
        .RegionSimilarityCalculator) instance to compute pairwise similarity
        between anchors and groundtruth boxes. 
      matcher: a matcher.Matcher instance to perform matching for each anchor
        box.
      box_coder: a box_coder.BoxCoder instance to compute the anchor-encoded 
        groundtruth box coordinates as regression targets.
      negative_class_weight: float scalar, weight for negative class.
    """
    self._region_similarity_calculator = region_similarity_calculator
    self._matcher = matcher
    self._box_coder = box_coder
    self._negative_class_weight = negative_class_weight

  def assign(self,
             anchors_boxlist,
             gt_boxlist,
             gt_labels,
             gt_weights=None):
    """Performs target assignment for a single image.

    Args:
      anchors_boxlist: a box_list.BoxList instance containing the anchor boxes
        with shape [num_anchors, 4] for a single image. 
      gt_boxlist: a box_list.BoxList instance with shape [num_gt_boxes, 4]
        containing groundtruth anchor boxes for a single image.
      gt_labels: a tensor with shape [num_gt_boxes, num_classes + 1] containing
        one-hot encoded groundtruth box labels for a single image.
      gt_weights: None, or a float tensor with shape [num_gt_boxes] containing
        weights for groundtruth boxes in a single simage.

    Returns:
      cls_targets: a float tensor with shape [num_anchors, num_classes + 1].
      cls_weights: a float tensor with shape [num_anchors].
      reg_targets: a float tensor with shape [num_anchors, 4].
      reg_weights: a float tensor with shape [num_anchors].
      match: a matcher.Match instance.     
    """
    if gt_weights is None:
      num_gt_boxes = gt_boxlist.num_boxes_static()
      if not num_gt_boxes:
        num_gt_boxes = gt_boxlist.num_boxes()
      gt_weights = tf.ones([num_gt_boxes], dtype=tf.float32)

    match_quality_matrix = self._region_similarity_calculator.compare(
        gt_boxlist, anchors_boxlist)

    match = self._matcher.match(match_quality_matrix)

    cls_targets = self._create_classification_targets(
        gt_labels, match)

    cls_weights = self._create_classification_weights(
        gt_weights, match)

    reg_targets = self._create_regression_targets(
        anchors_boxlist, gt_boxlist, match)

    reg_weights = self._create_regression_weights(
        gt_weights, match)

    return cls_targets, cls_weights, reg_targets, reg_weights, match

  def _unmatched_classification_target(self, num_class_slots):
    """Creates class target for unmatched anchor boxes.

    Args: int scalar tensor, num of class slots.

    Returns:
      a tensor with shape [num_class_slots] containing a value of one followed
        by `num_class_slots` - 1 zeros.
    """
    return tf.concat([tf.constant([1.]), tf.zeros(num_class_slots - 1)], axis=0)

  def _create_classification_targets(self, gt_labels, match):
    """Creates classification targets for a single image.

    Args:
      gt_labels: a tensor with shape [num_gt_boxes, num_classes + 1] containing
        one-hot encoded groundtruth box labels for a single image.
      match: a matcher.Match instance.

    Returns:
      cls_targets: a float tensor with shape [num_anchors, num_classes + 1].
    """
    shape = shape_utils.combined_static_and_dynamic_shape(gt_labels)
    unmatched_class_target = self._unmatched_classification_target(shape[1])

    cls_targets = match.gather_based_on_match(gt_labels,
        unmatched_value=unmatched_class_target,
        ignored_value=unmatched_class_target)
    return cls_targets

  def _create_classification_weights(self, gt_weights, match):
    """Creates classification weights for a single image.

    Args:
      gt_weights: a float tensor with shape [num_gt_boxes] containing weights
        for groundtruth boxes in a single simage.
      match: a matcher.Match instance.

    Returns: 
      cls_weights: a float tensor with shape [num_anchors].
    """
    cls_weights = match.gather_based_on_match(
        gt_weights,
        ignored_value=0.,
        unmatched_value=self._negative_class_weight)
    return cls_weights

  def _create_regression_targets(self, anchors_boxlist, gt_boxlist, match):
    """Creates regression targets for a single image.

    Args:
      anchors_boxlist: a box_list.BoxList instance containing the anchor boxes
        with shape [num_anchors, 4] for a single image.
      gt_boxlist: a box_list.BoxList instance with shape [num_gt_boxes, 4]
        containing groundtruth anchor boxes for a single image.
      match: a matcher.Match instance. 

    Returns:
      reg_targets: a float tensor with shape [num_anchors, 4].
    """
    unmatched_reg_value = ignored_reg_value = self._dummy_regression_target()

    reg_targets_boxes = match.gather_based_on_match(
        gt_boxlist.get(),
        unmatched_value=unmatched_reg_value,
        ignored_value=ignored_reg_value)

    reg_targets_boxlist = box_list.BoxList(reg_targets_boxes)
    # BoxLists `reg_targets_boxlist` and `anchors_boxlist` have one-to-one
    # correspondonse
    reg_targets = self._box_coder.encode(reg_targets_boxlist, anchors_boxlist)
    return reg_targets

  def _create_regression_weights(self, gt_weights, match):
    """Creates regression weights for a single image.

    Args:
      gt_weights: a float tensor with shape [num_gt_boxes] containing weights
        for groundtruth boxes in a single simage.
      match: a matcher.Match instance.

    Returns:
      reg_weights: a float tensor with shape [num_anchors].
    """
    reg_weights = match.gather_based_on_match(
        gt_weights, ignored_value=0., unmatched_value=0.)
    return reg_weights    

  def _dummy_regression_target(self):
    """Returns a zero-valued tensor with shape [code_size] as dummy regression
    target."""
    return tf.constant(self._box_coder.code_size*[0], tf.float32)


def batch_assign_targets(target_assigner,
                         anchors_boxlist_list,
                         gt_boxlist_list,
                         gt_labels_list,
                         gt_weights_list):
  """Assign targets to anchorwise predictions for a batch of images associated
  with groundtruth boxes.

  Args:
    target_assigner: a detection.core.target_assigner.TargetAssigner instance.
    anchors_boxlist_list: a list of of box_list.BoxList instances with shape 
      [num_anchors, 4] containing anchor boxes in a batch. Length of list is 
      equal to `batch_size`.
    gt_boxlist_list: a list of box_list.BoxList instances with shape
      [num_gt_boxes, 4] containing groundtruth boxes in a batch. Length
      of list is equal to `batch_size`.
    gt_labels_list: a list of tensors with shape [num_gt_boxes, num_classes + 1]
      containing one-hot encoded groundtruth box labels in a batch. Length of 
      list is equal to `batch_size`.
    gt_weights_list: a list of None or float tensors with shape [num_gt_boxes]
      containing weights for gt boxes. Length of list is equal to `batch_size`.

  Returns:
    batch_cls_targets: a float tensor with shape 
      [batch_size, num_anchors, num_classes + 1] containing anchorwise
      classification targets. 
    batch_cls_weights: a float tensor with shape [batch_size, num_anchors]
      containing anchorwise classification weights.
    batch_reg_targets: a float tensor with shape [batch_size, num_anchors, 4]
      containing anchorwise regression targets.
    batch_reg_weights: a float tensor with shape [batch_size, num_anchors]
      containing anchorwise regression weights. 
    match_list: a list of matcher.Match instances containing the anchorwise
      match info. Length of list is equal to `batch_size`.
  """
  cls_targets_list = []
  cls_weights_list = []
  reg_targets_list = []
  reg_weights_list = []
  match_list = []

  for anchors_boxlist, gt_boxlist, gt_labels, gt_weights in zip(
      anchors_boxlist_list, gt_boxlist_list, gt_labels_list, gt_weights_list):
    (cls_targets,
     cls_weights,
     reg_targets,
     reg_weights,
     match) = target_assigner.assign(
        anchors_boxlist, gt_boxlist, gt_labels, gt_weights)

    cls_targets_list.append(cls_targets)
    cls_weights_list.append(cls_weights)
    reg_targets_list.append(reg_targets)
    reg_weights_list.append(reg_weights)
    match_list.append(match)

  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)
  batch_reg_targets = tf.stack(reg_targets_list)
  batch_reg_weights = tf.stack(reg_weights_list)

  return (batch_cls_targets, batch_cls_weights, batch_reg_targets, 
          batch_reg_weights, match_list)


""""""
import tensorflow as tf

from detection.utils import shape_utils
from detection.core import box_list


class TargetAssigner(object):
  """Target Assigner creates classification targets, localization targets (i.e. 
  regression on the transformation of y_center, x_center, height, width that an
  anchor box must undergo to become the final predicted bounding box), and 
  optionally mask targets for each anchor in an anchor boxlist.

  It implements method `assign` that takes as input a BoxList of anchors as well
  as a BoxList of groundtruth box coordinates and class labels (and optionally 
  masks), where the following steps are carried out:

  1. Computes the pairwise similarity between anchors and groundtruth boxes.
  2. For each anchor box determines if there is a match (Positive), or there is 
    no match (Negative), or simply ignores it (Ignored).
  3. Assigns classification and localization (and optionally mask) targets.
    a. For Positive anchors, assigns a classification and localization (and 
      optionally mask) target according to the matched groundtruth box.
    b. For Negative anchors, assigns the background class as the classification 
      target, and assigns a dummy localization (and optionally mask) target.
    c. For Ignored anchors, assigns a dummy classification and 
      localization (and optionally mask) target.
    d. Assigns a weight of 1.0 to those anchors with a non-dummy target, and 
      weight of 0.0 to those with a dummy target, so that they don't contribute
      to the loss.

  In summary:
                  targets               |           weights
      cls         loc         msk       |   cls     loc     msk
  --------------------------------------|----------------------
  P   non-dummy   non-dummy   non-dummy |   1       1       1 
  N   non-dummy   dummy       dummy     |   1       0       0                                  
  I   dummy       dummy       dummy     |   0       0       0
  """
  def __init__(self,
               region_similarity_calculator,
               matcher,
               box_coder,
               negative_class_weight=1.0):
    """Constructor.

    Args:
      region_similarity_calculator: a RegionSimilarityCalculator instance to 
        compute pairwise similarity between anchors and groundtruth boxes. 
      matcher: a Matcher instance to perform matching for each anchor box.
      box_coder: a BoxCoder instance to compute the anchor-encoded groundtruth 
        box coordinates as localization targets.
      negative_class_weight: float scalar, weight for negative class.
    """
    self._region_similarity_calculator = region_similarity_calculator
    self._matcher = matcher
    self._box_coder = box_coder
    self._negative_class_weight = negative_class_weight

  def assign(self,
             anchors_boxlist,
             gt_boxlist,
             gt_weights=None):
    """Performs target assignment of anchors from a single image.

    Args:
      anchors_boxlist: a BoxList instance, holding float tensor of shape 
        [num_anchors, 4] as the anchor boxes coordinates for a single image. 
      gt_boxlist: a BoxList instance, holding float tensor of shape 
        [num_gt_boxes, 4] as the groundtruth boxes coordinates for a single 
        image. Must also have the field 'labels', holding float tensor of shape
        [num_gt_boxes, num_class_slots] as the one-hot encoded groundtruth box 
        labels for a single image. It may optionally have the field 'mask', 
        holding float tensor of shape [num_gt_boxes, height, width].
      gt_weights: a float tensor of shape [num_gt_boxes] holding weights 
        for groundtruth boxes in a single simage.

    Returns:
      loc_targets: a float tensor of shape [num_anchors, 4].
      loc_weights: a float tensor of shape [num_anchors].
      cls_targets: a float tensor of shape [num_anchors, num_class_slots].
      cls_weights: a float tensor of shape [num_anchors].
      msk_targets: (Optional) a float tensor of shape [num_anchors, height, 
        width].
      match: a Match instance.     
    """
    if not gt_boxlist.has_field('labels'):
      raise ValueError('`gt_boxlist` must have the labels field.')

    if gt_weights is None:
      num_gt_boxes = gt_boxlist.num_boxes()
      gt_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
    gt_labels = gt_boxlist.get_field('labels')

    match_quality_matrix = self._region_similarity_calculator.compare(
        gt_boxlist, anchors_boxlist)

    match = self._matcher.match(match_quality_matrix)

    loc_targets = self._create_localization_targets(
        anchors_boxlist, gt_boxlist, match)

    loc_weights = self._create_localization_weights(
        gt_weights, match)

    cls_targets = self._create_classification_targets(
        gt_labels, match)

    cls_weights = self._create_classification_weights(
        gt_weights, match)

    if not gt_boxlist.has_field('masks'):
      msk_targets = None
    else:
      gt_masks = gt_boxlist.get_field('masks')
      msk_targets = self._create_classification_targets(
          gt_masks, match, tensor_type='masks')

    return (loc_targets, loc_weights, cls_targets, 
            cls_weights, msk_targets, match)

  def _create_classification_targets(
      self, gt_tensor, match, tensor_type='labels'):
    """Creates classification targets for a single image.

    Args:
      gt_tensor: a tensor of shape [num_gt_boxes, num_class_slots] holding 
        one-hot encoded groundtruth box labels if `tensor_type` == 'labels',
        Or a tensor of shape [num_gt_boxes, height, width] holding binary 
        instance mask if `tensor_type` == 'masks'.
      match: a Match instance.
      tensor_type: string scalar, type of groundtruth tensor.

    Returns:
      cls_targets: a float tensor of shape [num_anchors, num_class_slots] if
        `tensor_type` == 'labels', Or a tensor of shape [num_anchors, height, 
        width] if `tensor_type` == 'masks'.
    """
    shape = shape_utils.combined_static_and_dynamic_shape(gt_tensor)
    if tensor_type == 'labels': 
      unmatched_cls_target = self._unmatched_classification_target(shape[1])
    elif tensor_type == 'masks':
      unmatched_cls_target = self._dummy_mask_target(shape[1], shape[2])
    else:
      raise ValueError('Unsupported tensor type %s' % tensor_type)
    ignored_cls_target = unmatched_cls_target
    cls_targets = match.gather_based_on_match(gt_tensor,
        unmatched_value=unmatched_cls_target,
        ignored_value=ignored_cls_target)
    return cls_targets

  def _create_classification_weights(self, gt_weights, match):
    """Creates classification weights for a single image.

    Args:
      gt_weights: a float tensor of shape [num_gt_boxes] holding weights
        for groundtruth boxes in a single simage.
      match: a Match instance.

    Returns: 
      cls_weights: a float tensor of shape [num_anchors].
    """
    cls_weights = match.gather_based_on_match(
        gt_weights,
        ignored_value=0.,
        unmatched_value=self._negative_class_weight)
    return cls_weights

  def _create_localization_targets(self, anchors_boxlist, gt_boxlist, match):
    """Creates localization targets for a single image.

    Args:
      anchors_boxlist: a BoxList instance, holding float tensor of shape
        [num_anchors, 4] as the anchor boxes coordinates for a single image.
      gt_boxlist: a BoxList instance, holding float tensor of shape 
        [num_gt_boxes, 4] as the groundtruth boxes coordinates for a single 
        image. 
      match: a Match instance. 

    Returns:
      loc_targets: a float tensor of shape [num_anchors, 4].
    """
    unmatched_loc_target = self._dummy_localization_target()
    ignored_loc_target = unmatched_loc_target

    loc_targets = match.gather_based_on_match(
        gt_boxlist.get(),
        unmatched_value=unmatched_loc_target,
        ignored_value=ignored_loc_target)

    loc_targets_boxlist = box_list.BoxList(loc_targets)
    # BoxLists `loc_targets_boxlist` and `anchors_boxlist` have one-to-one
    # correspondonse
    loc_targets = self._box_coder.encode(loc_targets_boxlist, anchors_boxlist)

    return loc_targets

  def _create_localization_weights(self, gt_weights, match):
    """Creates localization weights for a single image.

    Args:
      gt_weights: a float tensor of shape [num_gt_boxes] holding weights
        for groundtruth boxes in a single simage.
      match: a Match instance.

    Returns:
      loc_weights: a float tensor of shape [num_anchors].
    """
    loc_weights = match.gather_based_on_match(
        gt_weights, ignored_value=0., unmatched_value=0.)
    return loc_weights    

  def _unmatched_classification_target(self, num_class_slots):
    """Creates class target for unmatched anchor boxes.

    Args: 
      num_class_slots: int scalar tensor, typically equals `num_classes + 1`.

    Returns:
      a tensor of shape [num_class_slots] containing a value of one followed
        by `num_class_slots - 1` zeros.
    """
    return tf.concat([tf.constant([1.]), tf.zeros(num_class_slots - 1)], axis=0)

  def _dummy_localization_target(self):
    """Returns a zero-valued tensor of shape [code_size] as dummy localization
    target (i.e. unmatched and ignored).
    """
    return tf.zeros([self._box_coder.code_size], dtype=tf.float32)

  def _dummy_mask_target(self, height, width):
    """Returns a zero-valued tensor of shape [height, width] as dummy mask
    target (i.e. unmatched and ignored).
    """
    return tf.zeros([height, width], dtype=tf.uint8)


def batch_assign_targets(target_assigner,
                         anchors_boxlist_list,
                         gt_boxlist_list):
  """Assign targets to anchorwise predictions (e.g. class scores, box encodings,
  masks) for a batch of images that come with groundtruth boxes.

  Args:
    target_assigner: a TargetAssigner instance.
    anchors_boxlist_list: a list of BoxList instances, each holding float
      tensor of shape [num_anchors, 4] as the anchor boxes coordinates of a 
      single image in a batch. Length of list is equal to `batch_size`.
    gt_boxlist_list: a list of BoxList instances, each holding float tensor of 
      shape [num_gt_boxes, 4] as the groundtruth boxes coordinates of a single 
      image in a batch. Length of list is equal to `batch_size`. Each BoxList
      must also has the field 'labels', holding float tensor of shape
      [num_gt_boxes, num_class_slots].

  Returns:
    batch_loc_targets: a float tensor of shape [batch_size, num_anchors, 4]
      containing anchorwise localization targets.
    batch_loc_weights: a float tensor of shape [batch_size, num_anchors]
      containing anchorwise localization weights. 
    batch_cls_targets: a float tensor of shape 
      [batch_size, num_anchors, num_class_slots] containing anchorwise
      classification targets. 
    batch_cls_weights: a float tensor of shape [batch_size, num_anchors]
      containing anchorwise classification weights.
    batch_msk_targets: (Optional) a float tensor of shape [batch_size, 
      num_anchors, image_height, image_width] containing instance masks.
    match_list: a list of Match instances containing the anchorwise
      match info. Length of list is equal to `batch_size`.
  """
  loc_targets_list = []
  loc_weights_list = []
  cls_targets_list = []
  cls_weights_list = []
  msk_targets_list = []
  match_list = []

  for anchors_boxlist, gt_boxlist in zip(anchors_boxlist_list, gt_boxlist_list):
    (loc_targets, loc_weights, cls_targets, cls_weights, msk_targets, match
        ) = target_assigner.assign(anchors_boxlist, gt_boxlist)

    loc_targets_list.append(loc_targets)
    loc_weights_list.append(loc_weights)
    cls_targets_list.append(cls_targets)
    cls_weights_list.append(cls_weights)
    msk_targets_list.append(msk_targets)
    match_list.append(match)

  batch_loc_targets = tf.stack(loc_targets_list)
  batch_loc_weights = tf.stack(loc_weights_list)
  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)

  if msk_targets_list and msk_targets_list[0] is None:
    batch_msk_targets = None
  else:
    batch_msk_targets = tf.stack(msk_targets_list)

  return (batch_loc_targets, batch_loc_weights, batch_cls_targets,
          batch_cls_weights, batch_msk_targets, match_list)

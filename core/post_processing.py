"""
"""
import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.core.standard_names import BoxListFields
from detection.utils import shape_utils


def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   scope=None):
  """Performs multiclass version of non maximum suppression on a single image. 

  The multiclass NMS is performed in two stages:
  1. NMS is performed on boxes independently for each class, where boxes are 
  filtered by score, clipped to a window, before going through NMS. Note that
  NMS will cap the total number of nms'ed boxes to a given size.

  2. Then the nms'ed boxes over all classes are merged, and sorted in descending
  order by their class-specific scores, and only the top scoring boxes are 
  retained.

  Note it is required that `boxes` and `scores` have matched `num_classes` -- 
  `shape(boxes)[1] == shape(scores)[1]`. If different classes (> 1) share the 
  same set of box encodings (e.g. SSD, in which case shape(boxes)[1] == 1), 
  the caller of this function needs to tile `boxes` to have size `num_classes` 
  in the 1st dimension.

  Args:
    boxes: float tensor of shape [num_boxes, num_classes, 4], holding box 
      coordinates for each of the `num_classes` classes.
    scores: float tensor of shape [num_boxes, num_classes], holding box scores
      for each of the `num_classes` classes.
    score_thresh: float scalar, boxes with score < `score_thresh` are removed.
    iou_thresh: float scalar, IOU threshold for non-max suppression. Must be in
      [0.0, 1.0]. 
    max_size_per_class: int scalar, max num of retained boxes per class after 
      NMS.
    max_total_size: int scalar, max num of boxes retained over all classes. 
    clip_window: float tensor of shape [4], holding ymin, xmin, ymax, xmax of
      a clip window.
    scope: string scalar, name scope.

  Returns:
    sorted_boxlist: a BoxList instance holding up to `max_total_size` boxes, 
      with extra fields 'scores' (float tensor of shape [num_boxes]), 'classes'
      (int tensor of shape [num_boxes]), where `num_boxes` <= `max_total_size`.
      Note this BoxList contains boxes from all classes and they are sorted in
      descending order of their class-specific score.
  """
  if boxes.shape[1].value is None or scores.shape[1].value is None:
    raise ValueError('`shape(boxes)[1]` and `shape(scores)[1]` must be '
        'statically defined.')
  if boxes.shape[1].value != scores.shape[1].value:
    raise ValueError('`shape(boxes)[1]` must be equal to `shape(scores)[1]`. ')

  with tf.name_scope(scope, 'MultiClassNonMaxSuppression'):
    num_classes = boxes.shape[1].value
    selected_boxlist_list = []
    per_class_boxes_list = tf.unstack(boxes, axis=1)
    per_class_scores_list = tf.unstack(scores, axis=1)

    # stage 1: class-wise non-max suppression
    for class_index, per_class_boxes, per_class_scores in zip(
        range(num_classes), per_class_boxes_list, per_class_scores_list):
      per_class_boxlist = box_list.BoxList(per_class_boxes)
      per_class_boxlist.set_field(BoxListFields.scores, per_class_scores)

      # filter out boxes with score < `score_thresh`
      boxlist_filtered = box_list_ops.filter_by_score(
          per_class_boxlist, score_thresh)
      # optionally clip boxes to clip_window
      if clip_window is not None:
        boxlist_filtered = box_list_ops.clip_to_window(
            boxlist_filtered, clip_window)

      max_selection_size = tf.minimum(max_size_per_class,
                                      boxlist_filtered.num_boxes())
      # len(selected_indices) <= max_selection_size
      selected_indices = tf.image.non_max_suppression(
          boxlist_filtered.get(),
          boxlist_filtered.get_field(BoxListFields.scores),
          max_selection_size,
          iou_threshold=iou_thresh)
      nmsed_boxlist = box_list_ops.gather(boxlist_filtered, selected_indices)
      nmsed_boxlist.set_field(
          BoxListFields.classes, tf.zeros_like(
              nmsed_boxlist.get_field(BoxListFields.scores)) + class_index + 1)

      selected_boxlist_list.append(nmsed_boxlist)
    # stage 2: merge nms'ed boxes from all classes
    selected_boxlist = box_list_ops.concatenate(selected_boxlist_list)
    sorted_boxlist = box_list_ops.sort_by_field(selected_boxlist,
                                                BoxListFields.scores)
    if max_total_size:
      max_total_size = tf.minimum(max_total_size,
                                  sorted_boxlist.num_boxes())
      sorted_boxlist = box_list_ops.gather(sorted_boxlist,
                                           tf.range(max_total_size))
    return sorted_boxlist


def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         num_valid_boxes=None,
                                         scope=None):
  """Performs multiclass non maximum suppression on a batch of images.

  Args:
    boxes: float tensor of shape [batch_size, num_boxes, num_classes, 4], 
      holding decoded box coordinates for each of the `num_classes` classes for
      each of `batch_size` images.
    scores: float tensor of shape [batch_size, num_boxes, num_classes], holding 
      box scores for each of the `num_classes` classes for each of `batch_size` 
      images. 
    score_thresh: float scalar, boxes with score < `score_thresh` are removed.
    iou_thresh: float scalar, IOU threshold for non-max suppression. Must be in
      [0.0, 1.0]. 
    max_size_per_class: int scalar, max num of retained boxes per class after 
      NMS.
    max_total_size: int scalar, max num of boxes retained over all classes. 
    clip_window: float tensor of shape [batch_size, 4], holding ymin, xmin, 
      ymax, xmax of a window to clip boxes to before NMS.
    num_valid_boxes: int tensor of shape [batch_size], holding the num of valid
      boxes (not zero-padded) to be considered for each image in a batch. If 
      None, all boxes in `boxes` are considered valid.
    scope: string scalar, scope name.

  Returns:
    batch_nmsed_boxes: float tensor of shape [batch_size, max_total_size, 4].
    batch_nmsed_scores: float tensor of shape [batch_size, max_total_size].
    batch_nmsed_classes: int tensor of shape [batch_size, max_total_size].
    batch_num_valid_boxes: int tensor of shape [batch_size], holding num of 
      valid (not zero-padded) boxes per image in a batch. 
  """
  with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
    batch_size, num_boxes = shape_utils.combined_static_and_dynamic_shape(
        boxes)[:2]
    if num_valid_boxes is None:
      num_valid_boxes = tf.ones([batch_size], dtype=tf.int32) * num_boxes

    def _single_image_nms_fn(args):
      per_image_boxes = args[0]
      per_image_scores = args[1]
      per_image_clip_window = args[2]
      per_image_num_valid_boxes = args[-1]

      per_image_boxes = per_image_boxes[:per_image_num_valid_boxes]
      per_image_scores = per_image_scores[:per_image_num_valid_boxes]

      nmsed_boxlist = multiclass_non_max_suppression(
          per_image_boxes,
          per_image_scores,
          score_thresh,
          iou_thresh,
          max_size_per_class,
          max_total_size,
          clip_window=per_image_clip_window)
      padded_boxlist = box_list_ops.pad_or_clip_box_list(nmsed_boxlist,
                                                         max_total_size)
      num_boxes = nmsed_boxlist.num_boxes()
      nmsed_boxes = padded_boxlist.get()
      nmsed_scores = padded_boxlist.get_field(BoxListFields.scores)
      nmsed_classes = padded_boxlist.get_field(BoxListFields.classes)
      return nmsed_boxes, nmsed_scores, nmsed_classes, num_boxes

    batch_outputs = shape_utils.static_map_fn(
        _single_image_nms_fn,
        elems=[boxes, scores, clip_window, num_valid_boxes])

    batch_nmsed_boxes = batch_outputs[0]
    batch_nmsed_scores = batch_outputs[1]
    batch_nmsed_classes = batch_outputs[2]
    batch_num_valid_boxes = batch_outputs[-1]

    return (batch_nmsed_boxes, batch_nmsed_scores, 
            batch_nmsed_classes, batch_num_valid_boxes)

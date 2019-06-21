"""Implements functions shared by two out of the three model runners (Trainer,
Evaluator, Inferencer) to carry out their logic.
"""
import tensorflow as tf

from detection.utils import ops
from detection.utils import shape_utils
from detection.core import box_list
from detection.core import target_assigner
from detection.core import box_coder

from detection.core.standard_names import ModeKeys 
from detection.core.standard_names import LossTensorDictFields
from detection.core.standard_names import DetTensorDictFields
from detection.core.standard_names import PredTensorDictFields


def compute_losses(model, prediction_dict, gt_boxlist_list):
  """Creates localization and classification loss. 

  Args:
    model: an instance of DetectionModel . 
    prediction_dict: a dict mapping from strings to tensors/BoxList.
      Must hold the following entries:
      { 'box_encoding_predictions': float tensor of shape 
          [batch_size, num_anchors, 1, 4],
        'class_predictions': float tensor of shape
          [batch_size, num_anchors, num_classes + 1]}
    gt_boxlist_list: a list of BoxList instances, each holding `num_gt_boxes`
      groundtruth_boxes, with extra field 'labels' holding float tensor of shape
      [num_gt_boxes, num_classes + 1] (groundtruth boxes labels). Length of 
        list is equal to `batch_size`.

  Returns:
    losses_dict: a dict mapping from strings to tensors, holding the following
      entries:
      { 'loc_loss': float scalar tensor,
        'cls_loss': float scalar tensor}
  """
  box_encoding_predictions = prediction_dict[PredTensorDictFields.box_encoding_predictions]
  class_predictions = prediction_dict[PredTensorDictFields.class_predictions]
  anchors_boxlist_list = prediction_dict['anchor_boxlist_list']

  with tf.name_scope('Loss'):
    (batch_loc_targets, 
     batch_loc_weights, 
     batch_cls_targets, 
     batch_cls_weights,
     _, 
     match_list) = target_assigner.batch_assign_targets(
        model.target_assigner, anchors_boxlist_list, gt_boxlist_list)

    loc_losses = model.localization_loss_fn( 
        tf.squeeze(box_encoding_predictions, axis=2), 
        batch_loc_targets,
        ignore_nan_targets=True,
        weights=batch_loc_weights)
    cls_losses = model.classification_loss_fn(
        class_predictions, 
        batch_cls_targets,
        weights=batch_cls_weights)


    # scalar tensors: `loclization_loss`, `classification_loss`
    if model.hard_example_miner:
      decoded_boxes = box_coder.batch_decode(
          box_encoding_predictions, 
          anchors_boxlist_list,
          model.box_coder)

      decoded_boxes_list = tf.unstack(tf.squeeze(decoded_boxes, axis=2))
      decoded_boxlist_list = [box_list.BoxList(decoded_boxes) 
          for decoded_boxes in decoded_boxes_list]
      mined_indicator = model.hard_example_miner(
          loc_losses=loc_losses,
          cls_losses=cls_losses,
          decoded_boxlist_list=decoded_boxlist_list,
          match_list=match_list)

      loc_losses = tf.multiply(loc_losses, mined_indicator)
      cls_losses = tf.multiply(cls_losses, mined_indicator)
#  sample_sizes = tf.maximum(tf.reduce_sum(mined_indicator, axis=1), 1)
#  sample_sizes = tf.maximum(tf.reduce_sum(batch_loc_weights, axis=1), 1)
  sample_sizes = tf.to_float(tf.maximum(tf.reduce_sum(batch_loc_weights), 1))
#  loc_loss = tf.reduce_mean(tf.reduce_sum(loc_losses, axis=1) / sample_sizes)
#  cls_loss = tf.reduce_mean(tf.reduce_sum(cls_losses, axis=1) / sample_sizes)
  
  loc_loss = tf.reduce_sum(loc_losses) / sample_sizes
  cls_loss = tf.reduce_sum(cls_losses) / sample_sizes
  loc_loss = tf.multiply(loc_loss, model._localization_loss_weight, name='loc_loss')
  cls_loss = tf.multiply(cls_loss, model._classification_loss_weight, name='cls_loss')

  losses_dict = {
      LossTensorDictFields.localization_loss: loc_loss,
      LossTensorDictFields.classification_loss: cls_loss}

  return losses_dict


def postprocess(model, prediction_dict):
  """Postprocess output tensors.

  Box encoding predictions are decoded w.r.t the anchor boxes, and the go 
  through the multiclass version of NMS. 

  Note: the output tensors are potentially zero-padded. Use the `num_detections`
  for unpadding.

  Args:
    model: an instance of DetectionModel. 
    prediction_dict: a dict mapping from strings to tensors, holding the
      following entries: 
      {
        'box_encoding_predictions': [batch_size, num_anchors, 1, 4],
        'class_predictions': [batch_size, num_anchors, num_classes + 1],
        'anchor_boxlist_list': a list of BoxList instance, each holding
          `num_anchors_i` anchor boxes. Length is equal to `batch_size`.    
      }
      Note: `num_anchors` is the total num of anchors summed over all feature 
      maps.  
   
  Returns:
    detection_dict: a dict mapping from strings to tensors, holding the 
      following entries:
      { 'boxes': float tensor of shape [batch_size, max_num_boxes, 4].
        'scores': float tensor of shape [batch_size, max_num_boxes].
        'classes': int tensor of shape [batch_size, max_num_boxes].
        'num_detections': int tensor of shape [batch_size], holding num of
          valid (not zero-padded) detections in each of the above tensors.}
  """
  box_encoding_predictions = prediction_dict[
      PredTensorDictFields.box_encoding_predictions]
  class_predictions = prediction_dict[PredTensorDictFields.class_predictions]
  anchors_boxlist_list = prediction_dict['anchor_boxlist_list']
  batch_size = len(anchors_boxlist_list)
  num_classes = model._box_predictor._num_classes 

  with tf.name_scope('Postprocessing'):
    detection_boxes = box_coder.batch_decode(
        box_encoding_predictions, anchors_boxlist_list, model.box_coder)

    # detection_boxes: [batch_size, 1917, 1, 4]
    detection_boxes = tf.tile(detection_boxes, [1, 1, num_classes, 1])
    # remove scores of background class
    # detection_scores: [batch_size, 1917, num_classes]
    detection_scores = model.score_converter_fn(class_predictions)
    detection_scores = detection_scores[:, :, 1:]

    (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections
        ) = model.non_max_suppression_fn(
            detection_boxes,
            detection_scores,
            clip_window=ops.get_unit_square(batch_size))

    detection_dict = {
        'boxes': nmsed_boxes,
        'scores': nmsed_scores,
        'classes': nmsed_classes,
        'num_detections': num_detections} 
    return detection_dict

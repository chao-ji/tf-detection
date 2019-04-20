"""Implements functions shared by two out of the three model runners (Trainer, 
Evaluator, Inferencer) to carry out their logic.
"""
import tensorflow as tf

from detection.utils import shape_utils
from detection.core import box_list_ops
from detection.core import box_list
from detection.core import target_assigner 
from detection.core import box_coder
from detection.utils import ops


def postprocess_frcnn(model,
                      frcnn_prediction_dict,
                      rpn_detection_dict):
  """Postprocess output tensors from Fast RCNN.

  Box encoding predictions are decoded w.r.t. the proposal boxes predicted
  from RPN, and then go through the multiclass version of NMS.

  Args:
    model: an instance of DetectionModel.
    frcnn_prediction_dict: a dict mapping from strings to tensors,
      holding the following entries:
      { 'box_encoding_predictions': float tensor of shape 
          [batch_size, max_num_boxes, num_classes, 4], 
        'class_predictions': float tensor of shape
          [batch_size, max_num_boxes, num_class + 1].}
    rpn_detection_dict: a dict mapping from strings to tensors/BoxLists,
      holding the following entries:
      { 'proposal_boxlist_list': a list of BoxList instances, each holding 
          `max_num_boxes` proposal boxes (coordinates normalized). The fields
          are potentially zero-padded up to `max_num_boxes`. Length of list
          is equal to `batch_size`. 
        'num_proposals': int tensor of shape [batch_size], holding the num of
          valid boxes (not zero-padded) in each BoxList of 
          `proposal_boxlist_list`.}

  Returns:
    frcnn_detection_dict: a dict mapping from strings to tensors,
      holding the following entries:
      { 'boxes': float tensor of shape [batch_size, max_num_boxes, 4].
        'scores': float tensor of shape [batch_size, max_num_boxes].
        'classes': int tensor of shape [batch_size, max_num_boxes].
        'num_detections': int tensor of shape [batch_size], holding num of 
          valid (not zero-padded) detections in each of the above tensors.}
  """
  box_encoding_predictions = frcnn_prediction_dict['box_encoding_predictions']
  class_predictions = frcnn_prediction_dict['class_predictions']
  proposal_boxlist_list = rpn_detection_dict['proposal_boxlist_list']
  num_proposals = rpn_detection_dict['num_proposals'] 
  batch_size = len(proposal_boxlist_list)

  with tf.name_scope('SecondStagePostprocessor'):
    box_coordinate_predictions = box_coder.batch_decode(
        box_encoding_predictions, 
        proposal_boxlist_list, 
        model._box_coder)

    class_predictions = model._frcnn_score_conversion_fn(
        class_predictions)[:, :, 1:]

    (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes, 
        batch_num_detections) = model._frcnn_nms_fn(
            box_coordinate_predictions,
            class_predictions,
            clip_window=ops.get_unit_square(batch_size),
            num_valid_boxes=num_proposals)

    return {'boxes': batch_nmsed_boxes, 
            'scores': batch_nmsed_scores, 
            'classes': batch_nmsed_classes, 
            'num_detections': batch_num_detections}


def compute_rpn_loss(model, rpn_prediction_dict, gt_boxlist_list):
  """Compute the localization and classification (objectness) loss of RPN.

  Args:
    model: an instance of DetectionModel.
    rpn_prediction_dict: a dict mapping from strings to tensors/BoxList.
      Must hold the following entries:
      { 'box_encoding_predictions': float tensor of shape 
          [batch_size, num_anchors, 1, 4],
        'objectness_predictions': float tensor of shape 
          [batch_size, num_anchors, 2],
        'anchor_boxlist_list': a list of BoxList instance, each holding 
          `num_anchors` anchor boxes. Length is equal to `batch_size`.} 
    gt_boxlist_list: a list of BoxList instances, each holding `num_gt_boxes`
      groundtruth boxes. No extra field holding groundtruth class labels
      are needed, as they will be generated for RPN. Length of list is equal to 
      `batch_size`.

  Returns:
    rpn_losses_dict: a tensor dict mapping from strings to tensors, holding
      the following entries,
      { 'loc_loss': float scalar tensor, 
          proposal box localization loss.
        'cls_loss': float scalar tensor, 
          proopsal box objectness (classification) loss}
  """
  rpn_box_encoding_predictions = tf.squeeze(rpn_prediction_dict[
      'box_encoding_predictions'], axis=2)
  rpn_objectness_predictions = rpn_prediction_dict[
      'objectness_predictions']
  anchors_boxlist_list = rpn_prediction_dict['anchor_boxlist_list']

  with tf.name_scope('RPNLoss'):
    batch_size = len(gt_boxlist_list) 
    rpn_gt_boxlist_list = []
    # generate objectness labels for rpn_gt_boxlist
    for gt_boxlist in gt_boxlist_list:
      gt_boxlist = box_list.BoxList(gt_boxlist.get())
      gt_boxlist.set_field('labels', 
          tf.tile([[0., 1.]], [gt_boxlist.num_boxes(), 1]))
      rpn_gt_boxlist_list.append(gt_boxlist)

    (batch_loc_targets, batch_loc_weights, batch_cls_targets, batch_cls_weights,
        _, _) = target_assigner.batch_assign_targets(
        model._rpn_target_assigner, anchors_boxlist_list, rpn_gt_boxlist_list)

    def rpn_minibatch_subsample_fn(args):
      cls_targets, cls_weights = args
      cls_targets = cls_targets[:, -1]
      return [model._rpn_minibatch_sampler_fn(
          tf.cast(cls_weights, tf.bool), 
          model._rpn_minibatch_size, 
          tf.cast(cls_targets, tf.bool))]

    # indicator of shape [batch_size, num_anchors], where each row sum to 
    # `rpn_minibatch_size`.
    batch_sampled_indicator = tf.to_float(shape_utils.static_map_fn(
        rpn_minibatch_subsample_fn, [batch_cls_targets, batch_cls_weights]))
    # indicator of shape [batch_size, num_anchors], where each row sum to
    # value < `rpn_minibatch_size`.
    sampled_loc_indicator = batch_sampled_indicator * batch_loc_weights
    # [batch_size]
    sample_sizes = tf.reduce_sum(batch_sampled_indicator, axis=1)

    # [batch_size, num_anchors]
    loc_losses = model._rpn_localization_loss_fn(rpn_box_encoding_predictions, 
        batch_loc_targets, weights=sampled_loc_indicator)
    # [batch_size, num_anchors]
    cls_losses = model._rpn_classification_loss_fn(rpn_objectness_predictions, 
        batch_cls_targets, weights=batch_sampled_indicator)

    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_losses, axis=1) / sample_sizes)
    cls_loss = tf.reduce_mean(tf.reduce_sum(cls_losses, axis=1) / sample_sizes)

    loc_loss = tf.multiply(loc_loss,
        model._rpn_localization_loss_weight, name='rpn_loc_loss')
    cls_loss = tf.multiply(cls_loss,
        model._rpn_classification_loss_weight, name='rpn_cls_loss')

    return {'loc_loss': loc_loss, 'cls_loss': cls_loss}


def compute_frcnn_loss(model, 
                       frcnn_prediction_dict,
                       rpn_detection_dict,
                       gt_boxlist_list):
  """Compute the localization and classification loss of Fast RCNN.

  Args:
    model: an instance of DetectionModel.
    frcnn_prediction_dict: a dict mapping from strings to tensors,
      holding the following entries:
      { 'box_encoding_predictions': float tensor of shape 
          [batch_size, max_num_boxes, num_classes, 4], 
        'class_predictions': float tensor of shape
          [batch_size, max_num_boxes, num_classes + 1]}
    rpn_detection_dict: a dict mapping from strings to tensors/BoxLists,
      holding the following entries:
      { 'proposal_boxlist_list':  a list of BoxList instances, each holding 
          `max_num_boxes` proposal boxes (coordinates normalized). The fields
          are potentially zero-padded up to `max_num_boxes`. Length of list
          is equal to `batch_size`. 
        'num_proposals': int tensor of shape [batch_size], holding the num of
          valid boxes (not zero-padded) in each BoxList of 
          `proposal_boxlist_list`.}
    gt_boxlist_list: a list of BoxList instances, each holding `num_gt_boxes`
      groundtruth boxes, with extra 'labels' field holding float tensor of shape
      [num_gt_boxes, num_classes + 1] (groundtruth boxes labels). Length of 
      list is equal to `batch_size`.

  Returns:
    frcnn_losses_dict: a tensor dict mapping from strings to tensors, holding
      the following entries,
      { 'loc_loss': float scalar tensor, 
          final box localization loss.
        'cls_loss': float scalar tensor, 
          final box classification loss}
  """
  box_encoding_predictions = frcnn_prediction_dict['box_encoding_predictions']
  class_predictions = frcnn_prediction_dict['class_predictions']
  proposal_boxlist_list = rpn_detection_dict['proposal_boxlist_list']
  num_proposals = rpn_detection_dict['num_proposals']

  with tf.name_scope('FastRcnnLoss'):
    if model.is_training:
      def _get_batch_results(field_name):
        if not proposal_boxlist_list[0].has_field(field_name):
          return None
        batch_results = tf.stack([proposal_boxlist.get_field(field_name) 
            for proposal_boxlist in proposal_boxlist_list])
        return batch_results

      batch_loc_targets = _get_batch_results('loc_targets')
      batch_loc_weights = _get_batch_results('loc_weights')
      batch_cls_targets = _get_batch_results('cls_targets')
      batch_cls_weights = _get_batch_results('cls_weights')
      batch_msk_targets = _get_batch_results('msk_targets')
    else:
      (batch_loc_targets, 
       batch_loc_weights, 
       batch_cls_targets, 
       batch_cls_weights,
       batch_msk_targets,
       _) = target_assigner.batch_assign_targets(model._frcnn_target_assigner, 
                                                 proposal_boxlist_list, 
                                                 gt_boxlist_list)

    box_encoding_predictions = tf.pad(
        box_encoding_predictions, [[0, 0], [0, 0], [1, 0], [0, 0]])

    box_encoding_predictions = tf.squeeze(
        tf.batch_gather(box_encoding_predictions, 
            tf.expand_dims(
                tf.argmax(batch_cls_targets, axis=2, output_type=tf.int32), 
            axis=2)
        ), axis=2)

    # [batch_size, max_num_proposals] 
    padding_indicator = tf.sequence_mask(num_proposals, 
        model.max_num_proposals, dtype=tf.float32)
    # true num of proposals (excluding padded ones)
    # make sure `sample_sizes` >= 1 elementwise
    sample_sizes = tf.to_float(tf.maximum(num_proposals, 1))

    # [batch_size, max_num_proposals]
    loc_losses = model._frcnn_localization_loss_fn(box_encoding_predictions,
        batch_loc_targets, weights=batch_loc_weights * padding_indicator)
    # [batch_size, max_num_proposals]
    cls_losses = model._frcnn_classification_loss_fn(class_predictions,
        batch_cls_targets, weights=batch_cls_weights * padding_indicator)

    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_losses, axis=1) / sample_sizes)
    cls_loss = tf.reduce_mean(tf.reduce_sum(cls_losses, axis=1) / sample_sizes)

    loc_loss = tf.multiply(loc_loss, 
        model._frcnn_localization_loss_weight, name='frcnn_loc_loss')
    cls_loss = tf.multiply(cls_loss, 
        model._frcnn_classification_loss_weight, name='frcnn_cls_loss')

    frcnn_losses_dict = {'loc_loss': loc_loss, 'cls_loss': cls_loss}

    if 'mask_predictions' in frcnn_prediction_dict:
      msk_loss = _compute_mask_loss(model,
          frcnn_prediction_dict['mask_predictions'],
          batch_cls_targets,
          batch_msk_targets,
          batch_loc_weights,
          padding_indicator,
          rpn_detection_dict['proposal_boxlist_list'])

      frcnn_losses_dict['msk_loss'] = msk_loss 

    return frcnn_losses_dict

def prune_outlying_anchors_and_predictions(box_encoding_predictions, 
                                           objectness_predictions, 
                                           anchor_boxlist,
                                           clip_window):
  """Remove anchors that are not completely bounded within the image window, as
  well as the predictions associated with these outlying anchors.

  Args:
    box_encoding_predictions: float tensor of shape 
      [batch_size, in_num_anchors, 1, 4].
    objectness_predictions: float tensor of shape 
      [batch_size, in_num_anchors, 2].
    anchor_boxlist: a list of BoxList instance, each holding 
      `in_num_anchors` anchor boxes. Length is equal to `batch_size`. 
    clip_window: float tensor of shape [4], holding the ymin, xmin, ymax, xmax
      coordinates of a unit square (i.e. 0, 0, 1, 1).

  Returns:
    pruned_box_encoding_predictions: float tensor of shape 
      [batch_size, out_num_anchors, 1, 4].
    pruned_objectness_preidctions: float tensor of shape 
      [batch_size, out_num_anchors, 2].
    pruned_anchor_boxlist: a list of BoxList instance, each holding 
      `out_num_anchors` anchor boxes. Length is equal to `batch_size`. 
  """
  pruned_anchor_boxlist, indices = box_list_ops.prune_outside_window(
      anchor_boxlist, clip_window)
  gather_fn = lambda args: (tf.gather(args[0], indices=indices),)
  pruned_box_encoding_predictions = shape_utils.static_map_fn(
      gather_fn, elems=[box_encoding_predictions])
  pruned_objectness_predictions = shape_utils.static_map_fn(
      gather_fn, elems=[objectness_predictions])

  return (pruned_box_encoding_predictions, pruned_objectness_predictions, 
      pruned_anchor_boxlist)


def sample_frcnn_minibatch(model,
                           batch_proposal_boxes,
                           batch_num_proposals,
                           gt_boxlist_list):
  """Sample a minibatch of proposal boxes to send to Fast RCNN at training time.

  The decoded, nms'ed, and clipped proposal boxes from RPN are further sampled 
  to an even smaller set to be used for extracting ROI feature maps for Fast 
  RCNN. 

  Args:
    model: an instance of DetectionModel. 
    batch_proposal_boxes: float tensor of shape [batch_size, num_boxes, 4].
    batch_num_proposals: int tensor of shape [batch_size].
    gt_boxlist_list: a list of BoxList instances, each holding `num_gt_boxes`
      groundtruth boxes, with extra field 'labels' holding float tensor of shape
      [num_gt_boxes, num_class + 1] (groundtruth boxes labels). Length of 
      list is equal to `batch_size`.

  Returns: 
    proposal_boxlist_list: a list of BoxList instances, each holding 
      `max_num_boxes` proposal boxes (coordinates normalized). The fields
      are potentially zero-padded up to `max_num_boxes`. Length of list
      is equal to `batch_size`.
    batch_num_proposals: int tensor of shape [batch_size], holding num of 
      sampled proposals in `proposal_boxlist_list`.
  """
  proposal_boxlist_list = []
  num_proposals_list = []

  for proposal_boxes, num_proposals, gt_boxlist in zip(
      tf.unstack(batch_proposal_boxes), 
      tf.unstack(batch_num_proposals), 
      gt_boxlist_list):

    # unpadded proposal BoxList 
    proposal_boxlist = box_list.BoxList(proposal_boxes[:num_proposals])

    sampled_proposal_boxlist = _sample_frcnn_minibatch_per_image(
        model, proposal_boxlist, gt_boxlist)
    padded_proposal_boxlist = box_list_ops.pad_or_clip_box_list(
        sampled_proposal_boxlist, size=model._frcnn_minibatch_size)

    proposal_boxlist_list.append(padded_proposal_boxlist)
    num_proposals_list.append(tf.minimum(sampled_proposal_boxlist.num_boxes(), 
                                         model._frcnn_minibatch_size))

  return proposal_boxlist_list, tf.stack(num_proposals_list)


def _sample_frcnn_minibatch_per_image(model,
                                      proposal_boxlist,
                                      gt_boxlist):
  """Sample a minibatch of proposal boxes for a single image.

  A total of `model._frcnn_minibatch_size` boxes are sampled according to a 
  desired positive percentage rate. If there are not enough positive boxes to 
  achieve this rate, negative boxes are padded to the sampled set of boxes. 

  Args:
    model: a DetectionModel instance.
    proposal_boxlist: a BoxList instance holding `in_num_boxes` proposal boxes
      of a single image. Note that zero-padded proposal boxes have been removed.
    gt_boxlist: a BoxList instance holding `num_gt_boxes` groundtruth boxes 
      labels of a single image.

  Returns:
    sampled_proposal_boxlist: a BoxList instance holding `out_num_boxes` 
      proposal boxes sampled from `proposal_boxlist`, where `in_num_boxes` <=
      `out_num_boxes`.
  """
  (loc_targets, loc_weights, cls_targets, cls_weights, msk_targets, _
      ) = model._frcnn_target_assigner.assign(proposal_boxlist, gt_boxlist)

  cls_weights += tf.to_float(tf.equal(tf.reduce_sum(cls_weights), 0))
  positive_indicator = tf.greater(tf.argmax(cls_targets, axis=1), 0)

  sampled_indicator = model._frcnn_minibatch_sampler_fn(
      tf.cast(cls_weights, tf.bool),
      model._frcnn_minibatch_size,
      positive_indicator)

  sampled_indices = tf.reshape(tf.where(sampled_indicator), [-1])

  proposal_boxlist.set_field('cls_targets', cls_targets)
  proposal_boxlist.set_field('cls_weights', cls_weights)
  proposal_boxlist.set_field('loc_targets', loc_targets)
  proposal_boxlist.set_field('loc_weights', loc_weights)
  if msk_targets is not None:
    proposal_boxlist.set_field('msk_targets', msk_targets)

  return box_list_ops.gather(proposal_boxlist, sampled_indices)


def _compute_mask_loss(model,
                       mask_predictions,
                       batch_cls_targets,
                       batch_msk_targets,
                       batch_msk_weights,
                       padding_indicator,
                       proposal_boxlist_list):
  """Compute mask loss.
  
  Each proposal (out of `max_num_proposals`) predictes `num_classes` masks of 
  shape [mask_height, mask_width]. However, only the one corresponding to the
  groundtruth class label `k` will be "selected" and contribute to the loss.
 
  Note: `batch_num_proposals` = `batch_size` * `max_num_proposals`,
  e.g. 64 = 1 * 64

  Args:
    mask_predictions: a float tensor of shape 
      [batch_num_proposals, num_classes, mask_height, mask_width], holding mask
      predictions.
    batch_cls_targets: a float tensor of shape 
      [batch_size, max_num_proposals, num_classes + 1], containing anchorwise
      classification targets. 
    batch_msk_targets: a float tensor of shape 
      [bathc_size, max_num_proposals, image_height, image_width], containing 
      instance mask targets. 
    batch_msk_weights a float tensor of shape [batch_size, max_num_proposals], 
      containing anchorwise localization weights.
    padding_indicator: a float tensor of shape [batch_size, max_num_proposals],
      holding indicator of padded proposals. 
    proposal_boxlist_list: a list of BoxList instances, each holding 
      `max_num_proposals` proposal boxes (coordinates normalized). The fields
      are potentially zero-padded up to `max_num_proposals`. Length of list
      is equal to `batch_size`.

  Returns:
    msk_loss: float scalar tensor, mask loss.
  """
  (batch_num_proposals, num_classes, mask_height, mask_width
      ) = shape_utils.combined_static_and_dynamic_shape(mask_predictions)
  batch_size = len(proposal_boxlist_list)

  # [batch_size * max_num_proposals, 4] e.g. 64, 4
  proposal_boxes = tf.reshape(tf.stack([proposal_boxlist.get() 
      for proposal_boxlist in proposal_boxlist_list], axis=0), 
      [batch_num_proposals, -1])
  # [batch_size * max_num_proposals, nums_classes + 1, mask_height, mask_width] 
  # e.g. 64, 91, 33, 33 
  mask_predictions = tf.pad(mask_predictions, [[0, 0], [1, 0], [0, 0], [0, 0]])

  # Only compute mask loss for the `k`th mask prediction, where `k` is the 
  # groundtruth
  # e.g. using class indices [64, 1] to gather from [64, 91, 33, 33], we get
  # tensor [64, 1, 33, 33]
  # [batch_size * max_num_proposals, 1, mask_height, mask_width]
  mask_predictions = tf.batch_gather(
      mask_predictions, 
      tf.to_int32(tf.expand_dims(tf.argmax(tf.reshape(batch_cls_targets, 
          [batch_num_proposals, -1]), axis=1), axis=-1)))

  # [batch_size, max_num_proposals, mask_height * mask_width]
  # e.g. 1, 64, 33 * 33 
  mask_predictions = tf.reshape(mask_predictions, 
                                [batch_size, -1, mask_height * mask_width])

  image_height, image_width = shape_utils.combined_static_and_dynamic_shape(
      batch_msk_targets)[2:]
 

  # [batch_size * max_num_proposals, image_height, image_width]
  batch_msk_targets = tf.reshape(batch_msk_targets, 
                                 [-1, image_height, image_width])

  # `batch_msk_targets` contains groundtruth instance masks as FULL SIZE 
  # images. Now we need to crop patches from it based on predicted region
  # proposals (i.e. `proposal_boxes`), and resize them to 
  # [mask_height, mask_width] to match the size of `mask_predictions`.
  #
  # [batch_size * max_num_proposals, mask_height, mask_weight, 1]
  # e.g. 64, 33, 33, 1
  batch_msk_targets = tf.image.crop_and_resize(
      tf.expand_dims(batch_msk_targets, -1), 
      proposal_boxes, 
      tf.range(batch_num_proposals),
      [mask_height, mask_width])

  # [batch_size, max_num_proposals, mask_height * mask_width]
  # e.g. 1, 64, 33 * 33
  batch_msk_targets = tf.reshape(batch_msk_targets, 
                                 [batch_size, -1, mask_height * mask_width])

  # [batch_size, max_num_proposals]
  msk_losses = model._frcnn_mask_loss_fn(
      mask_predictions, 
      batch_msk_targets, 
      weights=batch_msk_weights*padding_indicator)

  # normalize by
  # 1) mask size (`mask_height` * mask_width)
  # 2) num of pos proposals (only pos proposals' mask prediction matters)
  msk_losses = msk_losses / (mask_height * mask_width * tf.maximum(
      tf.reduce_sum(batch_msk_weights, axis=1, keep_dims=True), 
      tf.ones((batch_size, 1))))
  msk_loss = tf.reduce_sum(msk_losses)

  msk_loss = tf.multiply(msk_loss,
        model._frcnn_mask_loss_weight, name='frcnn_msk_loss')

  return msk_loss


def postprocess_masks(mask_predictions, frcnn_detection_dict):
  """Postprocess_masks to generate mask predictions.

  Each proposal (out of `max_num_proposals`) predictes `num_classes` masks of 
  shape [mask_height, mask_width]. However, only the one corresponding to the
  predicted class label `k` will be "selected" and generate a mask detection.

  Args:
    mask_predictions: a float tensor of shape 
      [batch_num_proposals, num_classes, mask_height, mask_width] 
    frcnn_detection_dict: a dict mapping from strings to tensors,
      holding the following entries:
      { 'boxes': float tensor of shape [batch_size, max_num_boxes, 4].
        'scores': float tensor of shape [batch_size, max_num_boxes].
        'classes': int tensor of shape [batch_size, max_num_boxes].
        'num_detections': int tensor of shape [batch_size], holding num of 
          valid (not zero-padded) detections in each of the above tensors.}

  Returns:
    mask_detections: a float tensor of shape 
      [batch_size, max_detection, mask_height, mask_width], holding the detected
      instance masks.
  """
  # [batch_size, max_num_proposals] e.g. 1, 300
  detection_classes = frcnn_detection_dict['classes']
  # e.g 300, 90, 33, 33 
  _, num_classes, mask_height, mask_width = mask_predictions.shape.as_list()
  batch_size, max_detection = detection_classes.get_shape().as_list()

  if num_classes > 1:
    # e.g. 300, 1, 33, 33
    mask_detections = tf.batch_gather(mask_predictions,
        tf.reshape(tf.to_int32(detection_classes) - 1, [-1, 1]))

  # e.g. 1, 300, 33, 33
  mask_detections = tf.reshape(tf.squeeze(mask_detections, axis=1),
      [batch_size, max_detection, mask_height, mask_width])

  mask_detections = tf.nn.sigmoid(mask_detections)
  return mask_detections

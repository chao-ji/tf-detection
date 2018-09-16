import tensorflow as tf

from detection.utils import shape_utils
from detection.core import box_list
from detection.core import target_assigner
from detection.utils import ops

from detection.core.standard_names import ModeKeys 
from detection.core.standard_names import LossTensorDictFields
from detection.core.standard_names import DetTensorDictFields


def batch_decode(box_encodings, anchors_boxlist, box_coder):
  """Decode a batch of box encodings w.r.t. anchors.

  Args:
    box_encodings: a float tensor containing box encoding predictions with 
      shape [batch_size, num_anchors, box_code_size]. 
    anchors_boxlist: a box_list.BoxList instance containing the anchor boxes
      with shape [num_anchors, box_code_size] for a single image. 
    box_coder: a detectin.core.box_coder.BoxCoder instance to decode anchor-
      encoded location predictions into absolute box coordinate predictions.

  Returns:
    decoded_boxes: a float tensor with shape 
      [batch_size, num_anchors, box_code_size].
  """
  batch_size, num_anchors, _ = shape_utils.combined_static_and_dynamic_shape(
      box_encodings)
  tiled_anchor_boxes = tf.tile(
      tf.expand_dims(anchors_boxlist.get(), 0), [batch_size, 1, 1])
  tiled_anchors_boxlist = box_list.BoxList(
      tf.reshape(tiled_anchor_boxes, [-1, box_coder.code_size]))
  decoded_boxes = box_coder.decode(
      tf.reshape(box_encodings, [-1, box_coder.code_size]),
      tiled_anchors_boxlist)
  decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
      [batch_size,
       num_anchors,
       box_coder.code_size]))
  return decoded_boxes


def assign_targets(anchors_boxlist,
                   assigner,
                   groundtruth_boxes_list,
                   groundtruth_labels_list,
                   groundtruth_weights_list=None,
                   add_background_class=True):
  """Assign targets to be predicted by each anchor.

  Args:
    anchors_boxlist: a box_list.BoxList instance containing the anchor boxes
      with shape [num_anchors, 4] for a single image.
    assigner: a detection.core.target_assigner.TargetAssigner instance.
    groundtruth_boxes_list: a list of float tensors with shape 
      [num_gt_boxes, 4] containing the groundtruth box coordinates. Length of 
      list is equal to `batch_size`. 
    groundtruth_labels_list: a list of float tensors with shape 
      [num_gt_boxes, num_classes] containing the groundtruth box labels in 
      one-hot representation. Length of list is equal to `batch_size`.
    groundtruth_weights_list: None, or a list of float tensors with shape 
      [num_gt_boxes] containing weights for groundtruth boxes. Length of list
      is equal to `batch_size`.
    add_background_class: bool scalar, whether to add background class to the
      one-hot encoding of the groundtruth labels. 

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
  groundtruth_boxlist_list = [
    box_list.BoxList(boxes) for boxes in groundtruth_boxes_list]
  if add_background_class:
    groundtruth_labels_list = [
        tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
        for one_hot_encoding in groundtruth_labels_list]

  batch_size = len(groundtruth_boxes_list)

  if groundtruth_weights_list is None:
    groundtruth_weights_list = [None] * batch_size 

  anchors_boxlist_list = [anchors_boxlist] * batch_size 

  return target_assigner.batch_assign_targets(
      assigner, anchors_boxlist_list, groundtruth_boxlist_list,
      groundtruth_labels_list, groundtruth_weights_list)


def create_losses(model,
                  pred_box_encodings,
                  pred_class_scores,
                  groundtruth_boxes_list,
                  groundtruth_labels_list,
                  groundtruth_weights_list,
                  scope=None):
  """Creates losses.

  Args:
    model: a trainer.SsdModelTrainer or evaluator.SsdModelEvaluator instance. 
    pred_box_encodings: a float tensor containing box encoding predictions
      with shape [batch_size, num_anchors, 4].
    pred_class_scores: a float tensor containing class score predictions
      with shape [batch_size, num_anchors, num_classes + 1].
    groundtruth_boxes_list: a list of float tensors with shape 
      [num_gt_boxes, 4] containing the groundtruth box coordinates. Length of 
      list is equal to `batch_size`. 
    groundtruth_labels_list: a list of float tensors with shape 
      [num_gt_boxes, num_classes] containing the groundtruth box labels in 
      one-hot representation. Length of list is equal to `batch_size`.
    groundtruth_weights_list: None, or a list of float tensors with shape 
      [num_gt_boxes] containing weights for groundtruth boxes. Length of list
      is equal to `batch_size`.
    scope: scalar string, the name scope.

  Returns:
    loss_tensor_dict: a dict mapping from tensor names to loss tensors
      {
        'localization_loss': float scalar tensor,
        'classification_loss': float scalar tensor
      }
  """
  if not (model.mode == ModeKeys.train or
          model.mode == ModeKeys.eval):
    raise ValueError('model must be in either train or eval mode to ',
        'create losses.')
  with tf.name_scope(scope, 'Loss', [pred_box_encodings,
                                     pred_class_scores,
                                     groundtruth_boxes_list,
                                     groundtruth_labels_list,
                                     groundtruth_weights_list]):
    (batch_cls_targets,
     batch_cls_weights,
     batch_reg_targets,
     batch_reg_weights,
     match_list) = assign_targets(model.anchors,
                                  model.target_assigner,
                                  groundtruth_boxes_list,
                                  groundtruth_labels_list,
                                  groundtruth_weights_list,
                                  model._add_background_class)

    loc_losses = model.localization_loss_fn(pred_box_encodings,
                                            batch_reg_targets,
                                            ignore_nan_targets=True,
                                            weights=batch_reg_weights)
    cls_losses = model.classification_loss_fn(pred_class_scores,
                                              batch_cls_targets,
                                              weights=batch_cls_weights)
    cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)

    # scalar tensors: `localization_loss`, `classification_loss`
    if model.hard_example_miner:
      decoded_boxes = batch_decode(pred_box_encodings,
                                   model.anchors,
                                   model.box_coder)

      decoded_boxes_list = tf.unstack(decoded_boxes)
      decoded_boxlist_list = [box_list.BoxList(decoded_boxes) 
          for decoded_boxes in decoded_boxes_list]
      (localization_loss, classification_loss
          ) = model.hard_example_miner(
              location_losses=loc_losses,
              cls_losses=cls_losses,
              decoded_boxlist_list=decoded_boxlist_list,
              match_list=match_list)
    else:
      localization_loss = tf.reduce_sum(loc_losses)
      classification_loss = tf.reduce_sum(cls_losses)

    # optionally normalizes localization and/or classification loss
    cls_loss_normalizer = tf.constant(1.0, dtype=tf.float32)

    if model._normalize_loss_by_num_matches:
      num_matches = tf.to_float(tf.reduce_sum(batch_reg_weights))
      cls_loss_normalizer = tf.maximum(num_matches, cls_loss_normalizer)

    loc_loss_normalizer = cls_loss_normalizer
    if model._normalize_loc_loss_by_codesize:
      loc_loss_normalizer *= model.box_coder.code_size

    localization_loss = tf.multiply(
        model._localization_loss_weight / loc_loss_normalizer,
        localization_loss,
        name='localization_loss')

    classification_loss = tf.multiply(
        model._classification_loss_weight / cls_loss_normalizer,
        classification_loss,
        name='classification_loss')

  loss_tensor_dict = {
      LossTensorDictFields.localization_loss: localization_loss,
      LossTensorDictFields.classification_loss: classification_loss
  }

  return loss_tensor_dict


def postprocess(model,
                images,
                pred_box_encodings,
                pred_class_scores,
                true_image_shapes):
  """Performs postprocessing.

  The raw detections from SSD include one set of box location and class label
  predictions from each anchor, most of which would be false positives. This
  function removes false positives by apply non-maximum suppression.

  Args:
    model: a evaluator.SsdModelEvaluator or inferencer.SsdModelInferencer
      instance.
    images: a rank-4 tensor with shape [batch, height, with, channels]
        containing the input images.
    pred_box_encodings: a float tensor containing box encoding predictions
      with shape [batch_size, num_anchors, 4].
    pred_class_scores: a float tensor containing class score predictions
      with shape [batch_size, num_anchors, num_classes + 1].
    true_image_shapes: a int tensor with shape [3] containing the height,
      depth, and channels of true image shapes.
     
  Returns:
    detection_tensor_dict: a dict mapping from tensor names to detection tensors
      {
        'boxes': [batch_size, MAX_DETECTIONS, 4]
        'scores': [batch_size, MAX_DETECTIONS]
        'classes': [batch_size, MAX_DETECTIONS]
        'num_detections': [batch_size]
      }
      `MAX_DETECTIONS` is the max num of detections after applying
      non-maximum suppression.
  """
  if not (model.mode == ModeKeys.eval or
          model.mode == ModeKeys.infer):
    raise ValueError('model must be in either eval or infer mode to ',
        'to perform postprocessing.')

  with tf.name_scope('Postprocessing'):
    detection_boxes = batch_decode(pred_box_encodings,
                                   model.anchors,
                                   model.box_coder)

    # detection_boxes: [batch_size, 1917, 1, 4]
    detection_boxes = tf.expand_dims(detection_boxes, axis=2)

    detection_scores = model.score_converter_fn(pred_class_scores)
    # detection_scores: [batch_size, 1917, num_classes]
    detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])

    (nmsed_boxes, nmsed_scores, nmsed_classes, _, _,
        num_detections) = model.non_max_suppression_fn(
            detection_boxes,
            detection_scores,
            clip_window=_compute_clip_window(images, true_image_shapes))

    detection_tensor_dict = {
        DetTensorDictFields.detection_boxes: nmsed_boxes,
        DetTensorDictFields.detection_scores: nmsed_scores,
        DetTensorDictFields.detection_classes: nmsed_classes,
        DetTensorDictFields.num_detections: num_detections} 

    return detection_tensor_dict


def _compute_clip_window(images, true_image_shapes):
  """Computes clip windows of the form [[ymin, xmin, ymax, xmax]] specifying
  the patch of each image in `images` that corresponds to actual image.

  Args:
    images: a float tensor with shape [batch_size, height, width, channels] 
      containing the images returned by the `model.preprocess` that may have
      been resized/padded. 
    true_image_shape: a int tensor with shape [batch_size, 3] where each row
      is of the form [height, width, channels] indicating the shapes of true
      images in the resized/padded images, Or None if the clip window should
      cover the full image.

  Returns:
    clip_window: a float tensor with shape [batch_size, 4] containing the clip
      window for each image in a batch in normalized coordinates (values varying
      in [0, 1]), where each row is of the form [ymin, xmin, ymax, xmax], Or a 
      default constant tensor of [0, 0, 1, 1].
  """
  if true_image_shapes is None:
    return tf.constant([0, 0, 1, 1], dtype=tf.float32)

  inputs_shape = shape_utils.combined_static_and_dynamic_shape(
      images)
  true_heights, true_widths, _ = tf.unstack(
      tf.to_float(true_image_shapes), axis=1)
  padded_height = tf.to_float(inputs_shape[1])
  padded_width = tf.to_float(inputs_shape[2])
  clip_window = tf.stack(
      [
          tf.zeros_like(true_heights),
          tf.zeros_like(true_widths),
          true_heights / padded_height,
          true_widths / padded_width
      ],
      axis=1)
  return clip_window

import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.faster_rcnn import commons

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields
from detection.utils import misc_utils
from detection.faster_rcnn import detection_model


class FasterRcnnModelEvaluator(detection_model.DetectionModel):
  """Evaluates a trained Faster RCNN model for object detection."""
  def __init__(self,
               image_resizer_fn,
               normalizer_fn,
               feature_extractor,
               box_coder,
               rpn_anchor_generator,
               rpn_box_predictor,
               frcnn_box_predictor,
               frcnn_mask_predictor,

               rpn_target_assigner,
               rpn_minibatch_sampler_fn,
               frcnn_target_assigner,
               frcnn_score_conversion_fn,
               frcnn_nms_fn, 

               rpn_localization_loss_fn,
               rpn_classification_loss_fn,
               frcnn_localization_loss_fn,
               frcnn_classification_loss_fn,
               frcnn_mask_loss_fn,

               rpn_nms_fn,
               rpn_score_conversion_fn,

               rpn_localization_loss_weight,
               rpn_classification_loss_weight,
               frcnn_localization_loss_weight,
               frcnn_classification_loss_weight,
               frcnn_mask_loss_weight,

               proposal_crop_size,
               rpn_minibatch_size,
               rpn_box_predictor_depth,
               first_stage_atrous_rate):
    """Constructor.

    Args:
      image_resizer_fn: a callable that resizes an input image (3-D tensor) into
        one with desired property.
      normalizer_fn: a callable that normalizes input image pixel values.
      feature_extractor: an instance of FasterRcnnFeatureExtractor. Extracts 
        features for RPN (first stage) and Fast RCNN (second stage).
      box_coder: an instance of BoxCoder. Transform box coordinates to and from 
        their encodings w.r.t. anchors.
      rpn_anchor_generator: an instance of AnchorGenerator. Generates anchors 
        for RPN.
      rpn_box_predictor: an instance of BoxPredictor. Generates box encoding
        predictions and class score predictions for RPN. 
      frcnn_box_predictor: an instance of BoxPredictor. Generates box encoding
        predictions and class score predictions for Fast RCNN.
      frcnn_mask_predictor: an instance of MaskPredictor. Generates mask 
        predictions for Fast RCNN.

      rpn_target_assigner: an instance of TargetAssigner that assigns 
        localization and classification targets to eacn anchorwise prediction
        for RPN.
      rpn_minibatch_sampler_fn: a callable that samples a subset of anchors to
        compute losses for.
      frcnn_target_assigner: an instance of TargetAssigner that assigns
        localization and classification targets to each proposal prediction for
        Fast RCNN.
      frcnn_score_conversion_fn: a callable that converts raw predicted class 
        logits into probability scores.
      frcnn_nms_fn: a callable that performs NMS on the box coordinate 
        predictions from Fast RCNN.
     
      rpn_localization_loss_fn: a callable that computes RPN's localization 
        loss.
      rpn_classification_loss_fn: a callable that computes RPN's classification 
        (objectness) loss.
      frcnn_localization_loss_fn: a callable that computes Fast RCNN's 
        localization loss.
      frcnn_classification_loss_fn: a callable that computes Fast RCNN's
        classification loss.
      frcnn_mask_loss_fn: a callable that computes Fast RCNN's mask loss.

      rpn_nms_fn: a callable that performs NMS on the proposal coordinate 
        predictions from RPN.
      rpn_score_conversion_fn: a callable that converts raw predicted class 
        logits into probability scores. 

      rpn_localization_loss_weight: float scalar, scales the contribution of 
        localization loss relative to total loss. 
      rpn_classification_loss_weight: float scalar, scales the contribution of
        classification loss relative to total loss.
      frcnn_localization_loss_weight: float scalar, scales the contribution of
        localization loss relative to total loss.
      frcnn_classification_loss_weight: float scalar, scales the contribution
        of classification loss relative to total loss.
      frcnn_mask_loss_weight: float scalar, scales the contribution of mask loss
        relative to total loss.

      proposal_crop_size: int scalar, ROI Align will be applied on the feature 
        map shared by RPN and Fast RCNN, which produces ROI feature maps of 
        spatial dimensions [proposal_crop_size, proposal_crop_size].
      rpn_minibatch_size: int scalar, a subset of `rpn_minibatch_size` anchors
        are sampled from the collection of all anchors in RPN for which losses 
        are computed and backpropogated. 
      rpn_box_predictor_depth: int scalar, the depth of feature map preceding
        rpn box predictor.
      first_stage_atrous_rate: int scalar, the atrous rate of the Conv2d 
        operation preceding rpn box predictor.
    """
    super(FasterRcnnModelEvaluator, self).__init__(
        image_resizer_fn=image_resizer_fn,
        normalizer_fn=normalizer_fn,
        feature_extractor=feature_extractor,
        box_coder=box_coder,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_box_predictor=rpn_box_predictor,
        frcnn_box_predictor=frcnn_box_predictor,
        frcnn_mask_predictor=frcnn_mask_predictor,

        rpn_nms_fn=rpn_nms_fn,
        rpn_score_conversion_fn=rpn_score_conversion_fn,

        proposal_crop_size=proposal_crop_size,
        rpn_box_predictor_depth=rpn_box_predictor_depth,
        first_stage_atrous_rate=first_stage_atrous_rate)

    self._rpn_target_assigner = rpn_target_assigner
    self._rpn_minibatch_sampler_fn = rpn_minibatch_sampler_fn
    self._frcnn_target_assigner = frcnn_target_assigner
    self._frcnn_score_conversion_fn = frcnn_score_conversion_fn
    self._frcnn_nms_fn = frcnn_nms_fn

    self._rpn_localization_loss_fn = rpn_localization_loss_fn
    self._rpn_classification_loss_fn = rpn_classification_loss_fn
    self._frcnn_localization_loss_fn = frcnn_localization_loss_fn
    self._frcnn_classification_loss_fn = frcnn_classification_loss_fn
    self._frcnn_mask_loss_fn = frcnn_mask_loss_fn

    self._frcnn_localization_loss_weight = frcnn_localization_loss_weight
    self._frcnn_classification_loss_weight = frcnn_classification_loss_weight
    self._rpn_localization_loss_weight = rpn_localization_loss_weight
    self._rpn_classification_loss_weight = rpn_classification_loss_weight
    self._frcnn_mask_loss_weight = frcnn_mask_loss_weight

    self._rpn_minibatch_size=rpn_minibatch_size

  @property
  def is_training(self):
    return False

  @property
  def mode(self):
    return ModeKeys.eval

  def evaluate(self, filename_list, dataset):
    """Adds evaluation related operations to the graph.

    Args:
      filename_list: a list of filenames of TFRecord files containing
        evaluation examples.
      dataset: an instance of DetectionDataset.

    Returns:
      to_be_run_dict: a dict mapping from strings to tensors, holding the
        following entries:
        { 'image': uint8 tensor of shape [height, width, depth], holding the 
            original image.
          'boxes': float tensor of shape [num_val_detections, 4], holding 
            coordinates of predicted boxes.
          'scores': float tensor of shape [num_val_detections], holding 
            predicted confidence scores.
          'classes': int tensor of shape [num_val_detections], holding predicted
            class indices.
          'gt_boxes': float tensor of shape [num_gt_boxes, 4], holding 
            coordinates of groundtruth boxes.
          'gt_labels': int tensor of shape [num_gt_boxes], holding groundtruth 
            box class indices.}
      losses_dict: a dict mapping from strings to tensors, holding the following
        entries:
        { 'loc_loss': float scalar tensor,
          'cls_loss': float scalar tensor}
    """
    misc_utils.check_dataset_mode(self, dataset)

    tensor_dict = dataset.get_tensor_dict(filename_list)

    image_list = tensor_dict[TensorDictFields.image]

    inputs = self.preprocess(image_list)

    gt_boxlist_list = misc_utils.preprocess_groundtruth(tensor_dict)

    rpn_prediction_dict = self.predict_rpn(inputs)

    rpn_losses_dict = commons.compute_rpn_loss(
        self, rpn_prediction_dict, gt_boxlist_list)

    rpn_detection_dict = self.postprocess_rpn(rpn_prediction_dict)

    frcnn_prediction_dict = self.predict_frcnn(
        rpn_detection_dict['proposal_boxlist_list'], 
        rpn_prediction_dict['shared_feature_map'])

    frcnn_detection_dict = commons.postprocess_frcnn(
        self, frcnn_prediction_dict, rpn_detection_dict)


    if self._frcnn_mask_predictor is not None:
      mask_predictions = self.predict_masks(
          frcnn_prediction_dict,
          rpn_detection_dict,
          rpn_prediction_dict['shared_feature_map'])
      frcnn_prediction_dict['mask_predictions'] = mask_predictions

      mask_detections = commons.postprocess_masks(mask_predictions, frcnn_detection_dict)
      frcnn_detection_dict['masks'] = mask_detections

    frcnn_losses_dict = commons.compute_frcnn_loss(
        self, frcnn_prediction_dict, rpn_detection_dict, gt_boxlist_list)

    losses_dict = { 'rpn_loc_loss': rpn_losses_dict['loc_loss'],
                    'rpn_cls_loss': rpn_losses_dict['cls_loss'],
                    'frcnn_loc_loss': frcnn_losses_dict['loc_loss'],
                    'frcnn_cls_loss': frcnn_losses_dict['cls_loss']}

    if 'msk_loss' in frcnn_losses_dict:
      losses_dict['frcnn_msk_loss'] = frcnn_losses_dict['msk_loss']

    to_be_run_dict = misc_utils.process_per_image_detection(
        image_list, frcnn_detection_dict, gt_boxlist_list)

    return to_be_run_dict, losses_dict

  def create_restore_saver(self):
    """Creates restore saver for persisting variables to a checkpoint.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    restore_saver = tf.train.Saver()
    return restore_saver

import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.faster_rcnn import commons

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields

from detection.faster_rcnn import detection_model
from detection.utils import misc_utils


class FasterRcnnModelInferencer(detection_model.DetectionModel):
  """Makes inferences using a trained Faster RCNN model for object detection.
  """
  def __init__(self,
               image_resizer_fn,
               normalizer_fn,
               feature_extractor,
               box_coder,
               rpn_anchor_generator,
               rpn_box_predictor,
               frcnn_box_predictor,
               frcnn_mask_predictor,

               frcnn_nms_fn, 
               frcnn_score_conversion_fn,
               rpn_nms_fn,
               rpn_score_conversion_fn,

               proposal_crop_size,
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

      frcnn_nms_fn: a callable that performs NMS on the box coordinate 
        predictions from Fast RCNN.
      frcnn_score_conversion_fn: a callable that converts raw predicted class 
        logits into probability scores.
      rpn_nms_fn: a callable that performs NMS on the proposal coordinate 
        predictions from RPN. 
      rpn_score_conversion_fn: a callable that converts raw predicted class 
        logits into probability scores.

      proposal_crop_size: int scalar, the height and width dimension of ROIs
        extracted from the feature map shared by RPN and Fast RCNN.
      rpn_box_predictor_depth: int scalar, the depth of feature map preceding
        rpn box predictor.
    """
    super(FasterRcnnModelInferencer, self).__init__(
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

    self._frcnn_nms_fn = frcnn_nms_fn
    self._frcnn_score_conversion_fn = frcnn_score_conversion_fn

  @property
  def is_training(self):
    return False

  @property
  def mode(self):
    return ModeKeys.infer

  def infer(self, filename_list, dataset):
    """Adds inference related operations to the graph.

    Args:
      filename_list: a list of filenames of raw image files on which
        detection is to be performed.
      dataset: an instance of DetectionDataset.

    Returns:
      to_be_run_tensor_dict: a dict mapping from strings to tensors, holding the
        following entries:
        { 'image': uint8 tensor of shape [height, width, depth], holding the 
            original image.
          'boxes': float tensor of shape [num_val_detections, 4], holding 
            coordinates of predicted boxes.
          'scores': float tensor of shape [num_val_detections], holding 
            predicted confidence scores.
          'classes': int tensor of shape [num_val_detections], holding predicted
            class indices.}
    """
    misc_utils.check_dataset_mode(self, dataset)

    tensor_dict = dataset.get_tensor_dict(filename_list)

    image_list = tensor_dict[TensorDictFields.image]

    inputs = self.preprocess(image_list)

    rpn_prediction_dict = self.predict_rpn(inputs)

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
      frcnn_detection_dict['masks_predictions'] = mask_predictions

      mask_detections = commons.postprocess_masks(mask_predictions, frcnn_detection_dict)
      frcnn_detection_dict['masks'] = mask_detections

    to_be_run_tensor_dict = misc_utils.process_per_image_detection(
        image_list, frcnn_detection_dict)

    return to_be_run_tensor_dict

  def create_restore_saver(self):
    """Creates restore saver for restoring variables from a checkpoint.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    restore_saver = tf.train.Saver()
    return restore_saver

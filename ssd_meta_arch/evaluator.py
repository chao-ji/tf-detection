
import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops


from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields 
from detection.core.standard_names import PredTensorDictFields
from detection.core.standard_names import LossTensorDictFields
from detection.core.standard_names import DetTensorDictFields

from detection.ssd_meta_arch import detection_model
from detection.ssd_meta_arch import core


class SsdModelEvaluator(detection_model.DetectionModel):
  """Evaluates a trained SSD detection model.
  """
  def __init__(self,
               image_resizer_fn,
               normalizer_fn,
               feature_extractor,
               anchor_generator,
               box_predictor,

               box_coder,
               target_assigner,
           
               localization_loss_fn,
               classification_loss_fn,
               hard_example_miner,
               score_converter_fn,
               non_max_suppression_fn,

               localization_loss_weight,
               classification_loss_weight,
               normalize_loss_by_num_matches,
               normalize_loc_loss_by_codesize,
               add_background_class):

    """Constructor.

    Args:
      image_resizer_fn: a callable, that wraps one of the tensorflow image 
        resizing functions. See `tf.image.ResizeMethod`.
      normalizer_fn: a callable, that normalizes input image pixel values into 
        a range.
      feature_extractor: an object that extracts features from input images.
      anchor_generator: an object that generates a fixed number of anchors
        given a single input image.
      box_predictor: an object that generates box encoding predictions from
        a list of feature map tensors. 
      box_coder: a box_coder.BoxCoder instance that converts between absolute 
        box coordinates and their relative encodings (w.r.t anchors).
      target_assigner: a target_assigner.TargetAssigner instance that assigns
        localization and classification targets to each anchorwise prediction.
      localization_loss_fn: a function that generates localization loss given
        anchorwise box encoding predictions and its assigned localization
        targets.
      classification_loss_fn: a function that generates classification loss 
        given anchorwise box class scores and its assigned classification 
        targets.
      hard_example_miner: a function that performs hard example mining such
        that gradient is backpropagated to high-loss anchorwise predictions.
      score_converter_fn: a function that converts raw box class logits into
        scores.
      non_max_suppression_fn: a function that performs non-maximum suppression
        independently for each class.
      localization_loss_weigh: a float scalar that scales the contribution of 
        localization loss to total loss.
      classification_loss_weight: a float scalar that scales the contribution
        of classification loss to total loss.
      normalize_loss_by_num_matches: a bool scalar, whether to normalize both
        types of losses by num of matches.    
      normalize_loc_loss_by_codesize: a bool scalar, whether to normalize 
        localization loss by box code size (e.g. 4 for the default box coder.)
      add_background_class: a bool scalar, whether to add background class. 
        Should be False if the examples already contains background class.
    """
    super(SsdModelEvaluator, self).__init__(
        image_resizer_fn=image_resizer_fn,
        normalizer_fn=normalizer_fn,
        feature_extractor=feature_extractor,
        anchor_generator=anchor_generator,
        box_predictor=box_predictor)
    
    self._box_coder = box_coder
    self._target_assigner = target_assigner

    self._localization_loss_fn = localization_loss_fn
    self._classification_loss_fn = classification_loss_fn
    self._hard_example_miner = hard_example_miner
    self._score_converter_fn = score_converter_fn
    self._non_max_suppression_fn = non_max_suppression_fn

    self._localization_loss_weight = localization_loss_weight
    self._classification_loss_weight = classification_loss_weight

    self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
    self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize 
    self._add_background_class = add_background_class

  @property
  def is_training(self):
    return False

  @property
  def mode(self):
    return ModeKeys.eval

  @property
  def target_assigner(self):
    """Returns target assigner."""
    return self._target_assigner

  @property
  def localization_loss_fn(self):
    """Returns function to compute localization loss."""
    return self._localization_loss_fn

  @property
  def classification_loss_fn(self):
    """Returns function to compute classification loss."""
    return self._classification_loss_fn

  @property
  def hard_example_miner(self):
    """Returns a function that performs hard example mining."""
    return self._hard_example_miner

  @property
  def box_coder(self):
    """Returns a box coder."""
    return self._box_coder

  @property
  def score_converter_fn(self):
    """Returns a function to convert logits to scores."""
    return self._score_converter_fn

  @property
  def non_max_suppression_fn(self):
    """Returns a function to perform non-maximum suppression."""
    return self._non_max_suppression_fn

  def evaluate(self, filename_list, dataset):
    """Adds evaluation related operations to the graph.

    Args:
      filename_list: a list of filenames of TFRecord files containing
        evaluation examples.
      dataset: a dataset.DetectionDataset instance.

    Returns:
      to_be_run_tensor_dict: a dict containing images, detected boxes,
        classes, scores as tensors.
      loc_loss: a float scalar tensor containing localization loss.
      cls_loss: a float scalar tensor containing classification loss.
      gt_boxes: a float tensor with shape [num_boxes, 4] containing 
        groundtruth box coordinates.
      gt_labels: a float tensor with shape [num_boxes, num_classes]
        containing groundtruth box class labels.
    """
    self.check_dataset_mode(dataset)

    tensor_dict = dataset.get_tensor_dict(filename_list)

    original_images = tensor_dict[TensorDictFields.image]
    inputs, true_image_shapes = self.preprocess(original_images)

    pred_tensor_dict = self.predict(inputs)

    pred_box_encodings = pred_tensor_dict[
        PredTensorDictFields.box_encoding_predictions]
    pred_class_scores = pred_tensor_dict[
        PredTensorDictFields.class_score_predictions]

    loss_tensor_dict = core.create_losses(
        self, pred_box_encodings, pred_class_scores, 
        tensor_dict[TensorDictFields.groundtruth_boxes],
        tensor_dict[TensorDictFields.groundtruth_labels],
        None)

    loc_loss = loss_tensor_dict[LossTensorDictFields.localization_loss]
    cls_loss = loss_tensor_dict[LossTensorDictFields.classification_loss]

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(reg_losses + [loc_loss, cls_loss])

    det_tensor_dict = core.postprocess(self,
                inputs,
                pred_box_encodings,
                pred_class_scores,
                true_image_shapes)

    boxes = det_tensor_dict[DetTensorDictFields.detection_boxes][0]
    scores = det_tensor_dict[DetTensorDictFields.detection_scores][0]
    classes = det_tensor_dict[DetTensorDictFields.detection_classes][0]
    num_detections = det_tensor_dict[DetTensorDictFields.num_detections][0]

    label_id_offset = 1
    classes = tf.to_int64(classes) + label_id_offset
    num_detections = tf.to_int32(num_detections)

    boxes = boxes[:num_detections]
    classes = classes[:num_detections]
    scores = scores[:num_detections]

    image_shape = tf.shape(original_images[0])

    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(boxes), image_shape[0], image_shape[1])
   
    to_be_run_tensor_dict = {
        # [height, width, 3]
        'image': original_images[0],
        # [?, 4]
        'boxes': absolute_detection_boxlist.get(),
        # [?]
        'classes': classes,
        # [?]
        'scores': scores
    } 

    # [?, 4]
    gt_boxes = tensor_dict[TensorDictFields.groundtruth_boxes][0]
    # [?, num_classes]
    gt_labels = tensor_dict[TensorDictFields.groundtruth_labels][0]
    # [?, 4]
    gt_boxes = box_list_ops.to_absolute_coordinates(
        box_list.BoxList(gt_boxes), image_shape[0], image_shape[1]).get()

    return to_be_run_tensor_dict, loc_loss, cls_loss, gt_boxes, gt_labels

  def create_restore_saver(self):
    """Creates restore saver for persisting variables to a checkpoint file.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    restore_saver = tf.train.Saver()
    return restore_saver


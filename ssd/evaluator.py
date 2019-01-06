import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields 
from detection.core.standard_names import PredTensorDictFields
from detection.core.standard_names import LossTensorDictFields
from detection.core.standard_names import DetTensorDictFields

from detection.ssd import detection_model
from detection.ssd import commons 
from detection.utils import misc_utils


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
               add_background_class):

    """Constructor.

    Args:
      image_resizer_fn: a callable that resizes an input image (3-D tensor) into
        one with desired property.
      normalizer_fn: a callable that normalizes input image pixel values.
      feature_extractor: an instance of SsdFeatureExtractor.
      anchor_generator: an instance of AnchorGenerator.
      box_predictor: an instance of BoxPredictor. Generates box encoding 
        predictions and class predictions.
      box_coder: an instance of BoxCoder. Transform box coordinates to and from 
        their encodings w.r.t. anchors.
      target_assigner: an instance of TargetAssigner that assigns 
        localization and classification targets to each anchorwise prediction.
      localization_loss_fn: a callable that computes localization loss.
      classification_loss_fn: a callable that computes classification loss. 
      hard_example_miner: a callable that performs hard example mining such
        that gradient is backpropagated to high-loss anchorwise predictions.
      score_converter_fn: a callable that converts raw predicted class 
        logits into probability scores.
      non_max_suppression_fn: a callable that performs NMS on the box coordinate
        predictions.
      localization_loss_weight: float scalar, scales the contribution of 
        localization loss relative to classification loss.
      classification_loss_weight: float scalar, scales the contribution of
        classification loss relative to localization loss.
      add_background_class: bool scalar, whether to add background class. 
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

    gt_boxlist_list = misc_utils.preprocess_groundtruth(
        tensor_dict[TensorDictFields.groundtruth_boxes],
        tensor_dict[TensorDictFields.groundtruth_labels])

    prediction_dict = self.predict(inputs)

    losses_dict = commons.compute_losses(
        self, prediction_dict, gt_boxlist_list)

    detection_dict = commons.postprocess(self, prediction_dict)

    to_be_run_dict = misc_utils.process_per_image_detection(
        image_list, detection_dict, gt_boxlist_list)
    return to_be_run_dict, losses_dict 

  def create_restore_saver(self):
    """Creates restore saver for restoring variables from a checkpoint.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    restore_saver = tf.train.Saver()
    return restore_saver

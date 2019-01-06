import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields
from detection.core.standard_names import PredTensorDictFields
from detection.core.standard_names import DetTensorDictFields

from detection.ssd import detection_model
from detection.ssd import commons
from detection.utils import misc_utils


class SsdModelInferencer(detection_model.DetectionModel):
  """Makes inferences using a trained SSD detection model.
  """
  def __init__(self,
               image_resizer_fn,
               normalizer_fn,
               feature_extractor,
               anchor_generator,
               box_predictor,

               box_coder,
               score_converter_fn,
               non_max_suppression_fn):
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
      box_coder: an instance of BoxCoder. Transform box coordinates to and from 
        their encodings w.r.t. anchors.
      score_converter_fn: a callable that converts raw predicted class 
        logits into probability scores.
      non_max_suppression_fn: a callable that performs NMS on the box coordinate
        predictions.
    """
    super(SsdModelInferencer, self).__init__(
        image_resizer_fn=image_resizer_fn,
        normalizer_fn=normalizer_fn,
        feature_extractor=feature_extractor,
        anchor_generator=anchor_generator,
        box_predictor=box_predictor)

    self._box_coder = box_coder
    self._score_converter_fn = score_converter_fn
    self._non_max_suppression_fn=non_max_suppression_fn

  @property
  def is_training(self):
    return False

  @property
  def mode(self):
    return ModeKeys.infer

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

    prediction_dict = self.predict(inputs)

    detection_dict = commons.postprocess(self, prediction_dict)

    to_be_run_tensor_dict = misc_utils.process_per_image_detection(
        image_list, detection_dict)

    return to_be_run_tensor_dict

  def create_restore_saver(self):
    """Creates restore saver for restoring variables from a checkpoint.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    restore_saver = tf.train.Saver()
    return restore_saver


import tensorflow as tf


from detection.core import box_list
from detection.core import box_list_ops

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields
from detection.core.standard_names import PredTensorDictFields
from detection.core.standard_names import DetTensorDictFields

from detection.ssd_meta_arch import detection_model
from detection.ssd_meta_arch import core 


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
      box_coder: a box_coder.BoxCoder instance that converts between absolute 
        box coordinates and their relative encodings (w.r.t anchors).
      score_converter_fn: a function that converts raw box class logits into
        scores.
      non_max_suppression_fn: a function that performs non-maximum suppression
        independently for each class.
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
      dataset: a dataset.DetectionDataset instance.

    Returns:
      to_be_run_tensor_dict: a dict containing images, detected boxes,
        classes, scores as tensors.
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

    return to_be_run_tensor_dict

  def create_restore_saver(self):
    """Creates restore saver for persisting variables to a checkpoint file.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    restore_saver = tf.train.Saver()
    return restore_saver


"""DetectionModel implements the methods needed to run the forward pass of
Faster RCNN (thus shared by all three model runners -- Trainer, Evaluator,
Inferencer). 
"""
from abc import ABCMeta
from abc import abstractproperty
from abc import abstractmethod

import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.core import post_processing
from detection.core import box_coder 
from detection.utils import shape_utils
from detection.utils import ops
from detection.faster_rcnn import commons

slim = tf.contrib.slim


class DetectionModel(object):
  """Abstract base class of Faster RCNN model for object detection.

  This base class is to be subclassed by Trainer, Evaluator and Inferencer to 
  perform training, evaluating, and inference making, respectively.

  Implements methods `preprocess`, `predict_rpn`, `postprocess_rpn`, 
  `predict_frcnn`, which are shared in the workflows of Trainer, Evaluator,
  and Inferencer. The `is_training` and `mode` properties are set accordingly 
  in the subclasses.
  """
  __metaclass__ = ABCMeta

  def __init__(self, 
               image_resizer_fn, 
               normalizer_fn,
               feature_extractor,
               box_coder,
               rpn_anchor_generator,
               rpn_box_predictor,
               frcnn_box_predictor,

               rpn_nms_fn,
               rpn_score_conversion_fn, 

               proposal_crop_size,
               rpn_box_predictor_depth):
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
      rpn_nms_fn: a callable that performs NMS on the box coordinate 
        predictions from RPN.
      rpn_score_conversion_fn: a callable that converts raw predicted class 
        logits into probability scores. 
      proposal_crop_size: int scalar, the height and width dimension of ROIs
        extracted from the feature map shared by RPN and Fast RCNN.
      rpn_box_predictor_depth: int scalar, the depth of feature map preceding
        rpn box predictor.
    """
    self._image_resizer_fn = image_resizer_fn
    self._normalizer_fn = normalizer_fn
    self._feature_extractor = feature_extractor
    self._box_coder = box_coder
    self._rpn_anchor_generator = rpn_anchor_generator
    self._rpn_box_predictor = rpn_box_predictor
    self._frcnn_box_predictor = frcnn_box_predictor

    self._rpn_nms_fn = rpn_nms_fn
    self._rpn_score_conversion_fn = rpn_score_conversion_fn    

    self._proposal_crop_size = proposal_crop_size
    self._rpn_box_predictor_depth = rpn_box_predictor_depth

  @abstractproperty
  def is_training(self):
    """Returns a bool scalar indicating if model is in training mode."""
    pass

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating mode of model (train, eval or infer).
    """
    pass

  @abstractmethod
  def create_restore_saver(self, *args, **kwargs):
    """Creates a tf.train.Saver instance for restoring model.

    Depending on the `mode` of subclass, return a Saver instance initialized 
    with either all global variables (eval, infer) or a subset of global 
    variables (train).

    To be implemented by subclasses.

    Returns:
      a tf.train.Saver instance.
    """
    pass

  @property
  def first_stage_box_predictor_scope(self):  
    return 'FirstStageBoxPredictor'
  @property
  def second_stage_box_predictor_scope(self): 
    return 'SecondStageBoxPredictor'

  @property
  def max_num_proposals(self):
    """Returns the num of proposals (size of the minibatch for Fast RCNN).

    Note that at training time a subset of proposals are sampled from the 
    collection of proposals predicted and nms'ed from RPN, so the num of 
    proposals would be smaller than that at evaluation or inference time.
    """
    if self.is_training:
      return self._frcnn_minibatch_size 
    else:
      return self._rpn_nms_fn.keywords['max_total_size']

  def predict_rpn(self, inputs):
    """Run the batched input images through the conv layers shared by RPN and 
    Fast RCNN, as well as those layers unique to RPN, generating the following 
    tensors:
    1. The output of RPN: proposal box encoding predictions and objectness score
      predictions;
    2. The feature map tensor shared by both RPN and Fast RCNN;
    3. Anchor boxes.

    NOTE: anchors are generated according to the spatial dimension of the shared
    feature map. Those anchors that are not completely bounded within the image 
    window, are clipped to the image window at inference and evaluation time, or
    removed at training time.

    Args:
      inputs: float tensor of shape [batch_size, height, width, depth].

    Returns:
      rpn_prediction_dict: a dict mapping from strings to tensors/BoxLists, 
        holding the following entries:
        { 'box_encoding_predictions': float tensor of shape 
            [batch_size, num_anchors, 1, 4],
          'objectness_predictions': float tensor of shape 
            [batch_size, num_anchors, 2],
          'shared_feature_map': float tensor of shape 
            [batch_size, height_out, width_out, depth_out],
          'anchor_boxlist_list': a list of BoxList instance, each holding 
            `num_anchors` anchor boxes. Length is equal to `batch_size`.}
        Note that
        1. `height_out` and `width_out` are the spatial dimension of the feature
          map, and they differ from `height` and `width` of input feature map 
          `inputs` by a factor of `1/output_stride` (e.g. 1/16).
        2. num_anchors = height_out * width_out * num_predictions_per_location,
          at inference or evaluation time, while `num_anchors` may be
          smaller at training time, because anchors that are not completely 
          bounded within the image window are discarded.
    """
    shared_feature_map, image_shape = self._extract_shared_feature_map(inputs)
    anchor_boxlist = self._generate_anchors(
        shared_feature_map, image_shape[1], image_shape[2])
    (rpn_box_encoding_predictions, rpn_objectness_predictions
        ) = self._predict_rpn_proposals(shared_feature_map)

    clip_window = ops.get_unit_square()
    if self.is_training: # prune anchors and predictions
      (rpn_box_encoding_predictions, rpn_objectness_predictions, anchor_boxlist
          ) = commons.prune_outlying_anchors_and_predictions(
          rpn_box_encoding_predictions,
          rpn_objectness_predictions,   
          anchor_boxlist, 
          clip_window=clip_window)
    else: # clip anchors to image window
      anchor_boxlist = box_list_ops.clip_to_window(anchor_boxlist, clip_window)

    anchor_boxlist_list = [anchor_boxlist] * shared_feature_map.shape[0].value 
    return {'box_encoding_predictions': rpn_box_encoding_predictions,
            'objectness_predictions': rpn_objectness_predictions,
            'shared_feature_map': shared_feature_map,
            'anchor_boxlist_list': anchor_boxlist_list}

  def postprocess_rpn(self, 
                      rpn_prediction_dict, 
                      gt_boxlist_list=None):
    """Postprocess output tensors from RPN.

    The proposal box encoding predictions from RPN will be decoded w.r.t. 
    anchors they are associated with, and will go through non-max suppression.
    If run at training time, the nms'ed proposals will be further sampled to
    a smaller set before being used to extract ROI features in the next stage.

    Note the output list of proposal BoxLists are potentially zero-padded 
    because of the NMS. The num of valid proposals are indicated in
    `num_proposals`.

    Args:
      rpn_prediction_dict: a dict mapping from strings to tensors/BoxLists.
        Must hold the following entries:
        { 'box_encoding_predictions': float tensor of shape 
            [batch_size, num_anchors, 1, 4],
          'objectness_predictions': float tensor of shape 
            [batch_size, num_anchors, 2],
          'anchor_boxlist_list': a list of BoxList instance, each holding 
            `num_anchors` anchor boxes. Length is equal to `batch_size`.}
      gt_boxlist_list: a list of BoxList instances, each holding `num_gt_boxes`
        groundtruth boxes, with extra 'labels' field holding float tensor of shap 
        [num_gt_boxes, num_classes + 1] (groundtruth boxes labels). Length of 
        list is equal to `batch_size`. Must be provided at training time.

    Returns:
      rpn_detection_dict: a dict mapping from strings to tensors/BoxLists, 
        holding the following entries: 
        { 'proposal_boxlist_list': a list of BoxList instances, each holding 
            `max_num_boxes` proposal boxes (coordinates normalized). The fields
            are potentially zero-padded up to `max_num_boxes`. Length of list
            is equal to `batch_size`.
          'num_proposals': int tensor of shape [batch_size], holding the num of
            valid boxes (not zero-padded) in each BoxList of 
            `proposal_boxlist_list`.}
    """
    if self.is_training and gt_boxlist_list is None:
      raise ValueError('`gt_boxlist_list` must be provided at training time.')

    box_encoding_predictions = rpn_prediction_dict[
        'box_encoding_predictions']
    objectness_predictions = rpn_prediction_dict[
        'objectness_predictions']
    anchor_boxlist_list = rpn_prediction_dict['anchor_boxlist_list']
    batch_size = objectness_predictions.shape[0].value

    proposal_boxes = box_coder.batch_decode(
        box_encoding_predictions, anchor_boxlist_list, self._box_coder)
    objectness_predictions = self._rpn_score_conversion_fn(
        objectness_predictions)[:, :, 1:]

    (proposal_boxes, _, _, num_proposals) = self._rpn_nms_fn(
        proposal_boxes,
        objectness_predictions,
        clip_window=ops.get_unit_square(batch_size))
    proposal_boxlist_list = [box_list.BoxList(proposal) for proposal in 
                             tf.unstack(proposal_boxes)]
    rpn_detection_dict = {'proposal_boxlist_list': proposal_boxlist_list,
                          'num_proposals': num_proposals}

    if self.is_training:
      proposal_boxes = tf.stop_gradient(proposal_boxes)
      # further samples a smaller set of nms'ed proposals for Fast RCNN
      proposal_boxlist_list, num_proposals = commons.sample_frcnn_minibatch(
          self, 
          proposal_boxes, 
          num_proposals, 
          gt_boxlist_list)
      rpn_detection_dict = {'proposal_boxlist_list': proposal_boxlist_list, 
                            'num_proposals': num_proposals}

    return rpn_detection_dict

  def predict_frcnn(self, proposal_boxlist_list, shared_feature_map):
    """Extracts and batches ROI feature maps from each image in a batch, 
    runs the resulting tensor through the conv layers unique to Fast RCNN,
    and generating the final box encoding and class prediction tensors.

    Args:
      proposal_boxlist_list: a list of BoxList instances, each holding 
        `max_num_boxes` proposal boxes (coordinates normalized). The fields
        are potentially zero-padded up to `max_num_boxes`. Length of list
        is equal to `batch_size`.
      shared_feature_map: float tensor of shape 
        [batch_size, height, width, depth].

    Returns:
      frcnn_prediction_dict: a dict mapping from strings to tensors,
        holding the following entries:
        { 'box_encoding_predictions': float tensor of shape 
            [batch_size, max_num_boxes, num_classes, 4], 
          'class_predictions': float tensor of shape
            [batch_size, max_num_boxes, num_class + 1].}
    """
    proposal_boxes = tf.stack([proposal_boxlist.get() for proposal_boxlist 
        in proposal_boxlist_list])

    batched_roi_feature_maps = self._extract_roi_feature_maps(
        shared_feature_map, proposal_boxes)

    with slim.arg_scope([slim.batch_norm],
        is_training=(self.is_training and not self._freeze_batch_norm)):
      box_predictor_features = (
          self._feature_extractor.extract_second_stage_features(
              batched_roi_feature_maps))

    box_encoding_predictions, class_predictions = (
        self._frcnn_box_predictor.predict(
            [box_predictor_features], 
            scope=self.second_stage_box_predictor_scope))
    box_encoding_predictions = box_encoding_predictions[0]
    class_predictions = class_predictions[0]

    num_classes = self._frcnn_box_predictor._num_classes
    box_encoding_predictions = tf.reshape(
        box_encoding_predictions, [-1, 
                                   self.max_num_proposals, 
                                   num_classes, 
                                   self._frcnn_box_predictor._box_code_size])
    class_predictions = tf.reshape(
        class_predictions, [-1, self.max_num_proposals, num_classes + 1])

    return {'box_encoding_predictions': box_encoding_predictions,
            'class_predictions': class_predictions}

  def _extract_shared_feature_map(self, inputs):
    """Extracts the feature map shared by both RPN and Fast RCNN.

    Args:
      inputs: float tensor of shape [batch_size, height, width, depth].

    Returns:
      shared_feature_map: float tensor of shape 
        [batch_size, height_out, width_out, depth_out].
      image_shape: a list of 4 int scalar or int scalar tensors, storing 
        batch_size, height, width, and depth of the input tensor.
    """
    with slim.arg_scope([slim.batch_norm], 
        is_training=(self.is_training and not self._freeze_batch_norm)):
      shared_feature_map = self._feature_extractor.extract_first_stage_features(
          inputs)
    image_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    return shared_feature_map, image_shape

  def _generate_anchors(self, shared_feature_map, image_height, image_width):
    """Generates anchors for RPN according to spatial dimension of shared 
    feature map.

    The provided image height and width are used to normalize the anchor box
    coordinates to the unit square (i.e. bounded within [0, 0, 1, 1]).

    Args:
      shared_feature_map: float tensor of shape 
        [batch_size, height, width, depth], feature map shared by RPN and Fast 
        RCNN.
      image_height: float scalar tensor, height of the batched input images.
      image_width: float scalar tensor, width of the batched input images.
 
    Returns:
      anchor_boxlist: BoxList instance holding `num_anchors` anchor boxes. 
    """
    shape = shape_utils.combined_static_and_dynamic_shape(
        shared_feature_map)
    anchor_boxlist = box_list_ops.concatenate(
        self._rpn_anchor_generator.generate(
        [(shape[1], shape[2])], height=image_height, width=image_width)) 
    return anchor_boxlist

  def _predict_rpn_proposals(self, shared_feature_map, kernel_size=3):
    """Generates the proposal box encoding predictions and objectness 
    predictions.

    Args:
      shared_feature_map: float tensor of shape 
        [batch_size, height, width, depth], feature map shared by RPN and Fast 
        RCNN.
      kernel_size: int scalar, kernel size. Defaults to 3.

    Returns:
      rpn_box_encoding_predictions: float tensor of shape 
        [batch_size, num_anchors, 1, 4].
      rpn_objectness_predictions: float tensor of shape 
        [batch_size, num_anchors, 2].
    """
    with slim.arg_scope(self._rpn_box_predictor._conv_hyperparams_fn()):
      rpn_box_predictor_feature_map = slim.conv2d(
          shared_feature_map,
          self._rpn_box_predictor_depth,
          kernel_size=kernel_size,
          activation_fn=tf.nn.relu6)

    (box_encoding_predictions_list, objectness_predictions_list
        ) = self._rpn_box_predictor.predict(
        [rpn_box_predictor_feature_map], 
        scope=self.first_stage_box_predictor_scope)
    rpn_box_encoding_predictions = box_encoding_predictions_list[0]
    rpn_objectness_predictions = objectness_predictions_list[0]
    return rpn_box_encoding_predictions, rpn_objectness_predictions

  def _extract_roi_feature_maps(self, shared_feature_map, proposal_boxes):
    # (1, ?, ?, 576)
    # (1, 300, 4) 
    """Performs the equivalent of ROI pooling in Fast RCNN paper by cropping 
    regions from the feature map according to predicted proposals, and resizing 
    them to a fixed spatial dimension, followed by a 2x2 max pooling.

    Args:
      shared_feature_map: float tensor of shape 
        [batch_size, height, width, depth], feature map shared by RPN and Fast 
        RCNN.
      proposal_boxes: float tensor of shape [batch_size, num_proposals, 4], 
        holding the decoded, nms'ed and clipped proposal box coordinates. Note 
        that a subset of the boxes might be zero-paddings.

    Returns:
      roi_feature_maps: float tensor of shape 
        [batch_size * num_proposals, height_roi, width_roi, depth], holding 
        feature maps of regions of interest cropped and resized from the input 
        feature map. Note that the ROIs from different images in a batch are 
        arranged along the 0th dimension of size `batch_size * num_proposals`. 
    """
    shape = shape_utils.combined_static_and_dynamic_shape(proposal_boxes)
    proposal_boxes = tf.reshape(proposal_boxes, [shape[0] * shape[1], -1])

    box_indices = tf.reshape(tf.tile(
        tf.expand_dims(tf.range(shape[0]), axis=1), [1, shape[1]]), [-1])
    # 300, 14, 14, 576 
    regions_feature_maps = tf.image.crop_and_resize(
        shared_feature_map, proposal_boxes, box_indices, 
        (self._proposal_crop_size, self._proposal_crop_size))

    # 300, 7, 7, 576
    roi_feature_maps = slim.max_pool2d(
        regions_feature_maps, kernel_size=2, stride=2)

    return roi_feature_maps

  def preprocess(self, image_list):
    """Preprocess input images.

    The input images (a list of tensors of shape [height, width, channels]) 
    with possibly variable spatial dimensions, will be resized to have the same
    height and width, with pixel values optionally normalized. Then the
    preprocessed image list will be batched into a 4-D tensor.

    Args:
      image_list: a list of `batch_size` float tensors of shape 
        [height_i, width_i, channels].

    Returns:
      images: a float tensor of shape [batch_size, height, width, channels]. 
    """
    def _preprocess_single_image(image):
      resized_image = self._image_resizer_fn(image)
      if self._normalizer_fn is not None:
        resized_image = self._normalizer_fn(resized_image)
      return resized_image

    images = tf.stack([_preprocess_single_image(image) for image in image_list])
    return images 

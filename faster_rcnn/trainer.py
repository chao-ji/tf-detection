import tensorflow as tf

from detection.core import box_list
from detection.core import box_list_ops
from detection.faster_rcnn import commons

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields

from detection.faster_rcnn import detection_model
from detection.utils import variables_utils as var_utils
from detection.utils import misc_utils
from detection.utils import ops


class FasterRcnnModelTrainer(detection_model.DetectionModel):
  """Trains a Faster RCNN model."""
  def __init__(self,
               image_resizer_fn,
               normalizer_fn,
               feature_extractor,
               box_coder,
               rpn_anchor_generator,
               rpn_box_predictor,
               frcnn_box_predictor,

               rpn_target_assigner,
               rpn_minibatch_sampler_fn,
               frcnn_target_assigner,
               frcnn_minibatch_sampler_fn,

               rpn_localization_loss_fn,
               rpn_classification_loss_fn,
               frcnn_localization_loss_fn,
               frcnn_classification_loss_fn,

               rpn_nms_fn,
               rpn_score_conversion_fn,

               rpn_localization_loss_weight,
               rpn_classification_loss_weight,
               frcnn_localization_loss_weight,
               frcnn_classification_loss_weight,

               proposal_crop_size,
               rpn_minibatch_size,
               frcnn_minibatch_size,
               rpn_box_predictor_depth,
               freeze_batch_norm=True, 

               gradient_clipping_by_norm=0.0):
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

      rpn_target_assigner: an instance of TargetAssigner that assigns 
        localization and classification targets to eacn anchorwise prediction
        for RPN.
      rpn_minibatch_sampler_fn: a callable that samples a subset of anchors to
        compute losses for.
      frcnn_target_assigner: an instance of TargetAssigner that assigns
        localization and classification targets to each proposal prediction for
        Fast RCNN.
      frcnn_minibatch_sampler_fn: a callable that samples a subset of proposals
        to compute losses for.

      rpn_localization_loss_fn: a callable that computes RPN's localization 
        loss.
      rpn_classification_loss_fn: a callable that computes RPN's classification 
        (objectness) loss.
      frcnn_localization_loss_fn: a callable that computes Fast RCNN's 
        localization loss.
      frcnn_classification_loss_fn: a callable that computes Fast RCNN's
        classification loss.
      
      rpn_nms_fn: a callable that performs NMS on the proposal coordinate 
        predictions from RPN.
      rpn_score_conversion_fn: a callable that converts raw predicted class 
        logits into probability scores. 

      rpn_localization_loss_weight: float scalar, scales the contribution of 
        localization loss relative to classification loss. 
      rpn_classification_loss_weight: float scalar, scales the contribution of
        classification loss relative to localization loss.
      frcnn_localization_loss_weight: float scalar, scales the contribution of
        localization loss relative to classification loss.
      frcnn_classification_loss_weight: float scalar, scales the contribution
        of classification loss relative to localization loss.

      proposal_crop_size: int scalar, the height and width dimension of ROIs
        extracted from the feature map shared by RPN and Fast RCNN.
      rpn_minibatch_size: int scalar, a subset of `rpn_minibatch_size` anchors
        are sampled from the collection of all anchors in RPN for which losses 
        are computed and backpropogated. 
      frcnn_minibatch_size: int scalar, a subset of `frcnn_minibatch_size` 
        proposals are sampled from the collection of proposal boxes output by
        RPN to extract ROI feature maps for Fast RCNN.
      rpn_box_predictor_depth: int scalar, the depth of feature map preceding
        rpn box predictor.
      freeze_batch_norm: bool scalar, whether to run batch-norm layer in test 
        mode or training mode. Defaults to True (i.e. test mode).

      gradient_clipping_by_norm: float scalar, if > 0, clip the gradient tensors
        such that their norms <= `gradient_clipping_by_norm`. 
    """
    super(FasterRcnnModelTrainer, self).__init__(
        image_resizer_fn=image_resizer_fn,
        normalizer_fn=normalizer_fn,
        feature_extractor=feature_extractor,
        box_coder=box_coder,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_box_predictor=rpn_box_predictor,
        frcnn_box_predictor=frcnn_box_predictor,

        rpn_nms_fn=rpn_nms_fn,
        rpn_score_conversion_fn=rpn_score_conversion_fn,

        proposal_crop_size=proposal_crop_size,
        rpn_box_predictor_depth=rpn_box_predictor_depth)

    self._rpn_target_assigner = rpn_target_assigner
    self._rpn_minibatch_sampler_fn = rpn_minibatch_sampler_fn
    self._frcnn_target_assigner = frcnn_target_assigner
    self._frcnn_minibatch_sampler_fn = frcnn_minibatch_sampler_fn

    self._rpn_localization_loss_fn = rpn_localization_loss_fn
    self._rpn_classification_loss_fn = rpn_classification_loss_fn
    self._frcnn_localization_loss_fn = frcnn_localization_loss_fn
    self._frcnn_classification_loss_fn = frcnn_classification_loss_fn

    self._rpn_localization_loss_weight = rpn_localization_loss_weight
    self._rpn_classification_loss_weight = rpn_classification_loss_weight
    self._frcnn_localization_loss_weight = frcnn_localization_loss_weight
    self._frcnn_classification_loss_weight = frcnn_classification_loss_weight

    self._rpn_minibatch_size=rpn_minibatch_size
    self._frcnn_minibatch_size = frcnn_minibatch_size
    self._freeze_batch_norm = freeze_batch_norm

    self._gradient_clipping_by_norm = gradient_clipping_by_norm

  @property
  def is_training(self):
    return True

  @property
  def mode(self):
    return ModeKeys.train

  def train(self, filename_list, dataset, optimizer_builder_fn):
    """Adds training related operations to the graph.

    Args:
      filename_list: a list of filenames of TFRecord files containing training
        examples.
      dataset: an instance of DetectionDataset.
      optimizer_builder_fn: a callable, when called generates a 2-tuple holding 
        an instance of tensorflow Optimizer and a learning rate tensor (float 
        scalar).

    Returns:
      grouped_update_op: an instance of tf.Operation, the grouped gradient 
        update op to be executed in a tf.Session.
      total_loss: float scalar tensor, total loss (localization + classification 
        + regularization).
      global_step: int scalar tensor, global step.
    """
    misc_utils.check_dataset_mode(self, dataset)

    tensor_dict = dataset.get_tensor_dict(filename_list)

    inputs = self.preprocess(tensor_dict[TensorDictFields.image])

    gt_boxlist_list = misc_utils.preprocess_groundtruth(
        tensor_dict[TensorDictFields.groundtruth_boxes], 
        tensor_dict[TensorDictFields.groundtruth_labels])

    rpn_prediction_dict = self.predict_rpn(inputs)

    rpn_losses_dict = commons.compute_rpn_loss(
        self, rpn_prediction_dict, gt_boxlist_list)

    rpn_detection_dict = self.postprocess_rpn(
        rpn_prediction_dict, gt_boxlist_list)

    frcnn_prediction_dict = self.predict_frcnn(
        rpn_detection_dict['proposal_boxlist_list'], 
        rpn_prediction_dict['shared_feature_map'])

    frcnn_losses_dict = commons.compute_frcnn_loss(
        self, frcnn_prediction_dict, rpn_detection_dict, gt_boxlist_list)

    frcnn_loc_loss = frcnn_losses_dict['loc_loss']
    frcnn_cls_loss = frcnn_losses_dict['cls_loss']
    rpn_loc_loss = rpn_losses_dict['loc_loss']
    rpn_cls_loss = rpn_losses_dict['cls_loss']

    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        + [frcnn_loc_loss, frcnn_cls_loss, rpn_loc_loss, rpn_cls_loss])

    global_step = tf.train.get_or_create_global_step()
    optimizer, learning_rate = optimizer_builder_fn()
    grouped_update_op = ops.create_gradient_update_op(optimizer, 
        total_loss, global_step, self._gradient_clipping_by_norm)

    total_loss = tf.check_numerics(total_loss, 'Loss Tensor is inf or nan.')
    with tf.control_dependencies([grouped_update_op]):
      total_loss = tf.identity(total_loss)

    return total_loss, global_step

  def _get_vars_map(self, 
                    checkpoint_type='classification',
                    load_all_detection_checkpoint_vars=False):
    """Returns a mapping from var name to var tensor for restoring variables 
    from a checkpoint.

    Args:
      checkpoint_type: a string scalar, type of checkpoint from which
        to initialize variables. 'classification' for a pretrained 
        classification model (e.g. Inception, ResNet); 'detection' for a 
        pretrained detection model.
      load_all_detection_checkpiont_vars: a bool scalar, whether to load all
        detection checkpoint variables.

    Returns:
      a dict mapping from variable names to variable tensors.
    """
    if checkpoint_type == 'classification':
      vars_to_restore = {}
      for var in tf.global_variables():
        for scope in (self._feature_extractor._first_stage_feature_extractor_scope,
                      self._feature_extractor._second_stage_feature_extractor_scope):
          if var.op.name.startswith(scope):
            var_name = var.op.name.replace(scope + '/', '') 
            vars_to_restore[var_name] = var

      return vars_to_restore     
    elif checkpoint_type == 'detection':
      vars_to_restore = tf.global_variables()
      vars_to_restore.append(tf.train.get_or_create_global_step()) 
      include_patterns = None
      if not load_all_detection_checkpoint_vars:
        include_patterns = (
            self._feature_extractor._first_stage_feature_extractor_scope,
            self._feature_extractor._second_stage_feature_extractor_scope)
      feature_extractor_vars = tf.contrib.framework.filter_variables(
          vars_to_restore, include_patterns=include_patterns)
      return {var.op.name: var for var in feature_extractor_vars}

    raise ValueError('`checkpoint_type` must be either '
        '"classification" or "detection".')

  def create_restore_saver(self,
                           load_ckpt_path,
                           checkpoint_type='classification',
                           load_all_detection_checkpoint_vars=False,
                           include_global_step=True):
    """Creates restore saver for restoring variables from a checkpoint.

    Args:
      load_ckpt_path: a string scalar, pointing to a checkpoint file ('*.ckpt').
      checkpoint_type: a string scalar, type of checkpoint from which
        to initialize variables. 'classification' for a pretrained 
        classification model (e.g. Inception, ResNet), 'detection' for a 
        pretrained detection model.
      load_all_detection_checkpoint_vars: a bool scalar, whether to load all
        detection checkpoint variables.
      include_global_step: bool scalar, whether to restore global step in the
        checkpoint.

    Returns:
      restore_saver: a tf.train.Saver instance.
    """ 
    vars_map = self._get_vars_map(checkpoint_type,
                                  load_all_detection_checkpoint_vars)

    available_vars_map = var_utils.get_vars_available_in_ckpt(
        vars_map, load_ckpt_path, include_global_step=include_global_step)

    restore_saver = tf.train.Saver(available_vars_map)
    return restore_saver

  def create_persist_saver(self, max_to_keep):
    """Creates persist saver for persisting variables to a checkpoint file.

    Args:
      max_to_keep: int scalar, max num of checkpoints to keep.

    Returns:
      persist_saver: a tf.train.Saver instance.
    """
    persist_saver = tf.train.Saver(max_to_keep=max_to_keep)
    return persist_saver

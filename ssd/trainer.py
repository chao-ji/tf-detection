import re

import tensorflow as tf

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields
from detection.core.standard_names import PredTensorDictFields
from detection.core.standard_names import LossTensorDictFields
from detection.ssd import detection_model 
from detection.ssd import commons
from detection.utils import variables_utils as var_utils
from detection.utils import misc_utils
from detection.utils import ops


class SsdModelTrainer(detection_model.DetectionModel):
  """Trains a SSD detection model.
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

               localization_loss_weight,
               classification_loss_weight,
               freeze_batch_norm,
               add_background_class,

               gradient_clipping_by_norm=0.0):
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
      localization_loss_weight: float scalar, scales the contribution of 
        localization loss relative to classification loss.
      classification_loss_weight: float scalar, scales the contribution of
        classification loss relative to localization loss.
      freeze_batch_norm: bool scalar, whether to run batch-norm layer in test 
        mode or training mode. Defaults to True (i.e. test mode). 
      add_background_class: bool scalar, whether to add background class. 
        Should be False if the examples already contains background class.
      gradient_clipping_by_norm: float scalar, if > 0, clip the gradient tensors
        such that their norms <= `gradient_clipping_by_norm`.
    """
    super(SsdModelTrainer, self).__init__(
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

    self._localization_loss_weight = localization_loss_weight
    self._classification_loss_weight = classification_loss_weight
    self._freeze_batch_norm = freeze_batch_norm
    self._add_background_class = add_background_class

    self._gradient_clipping_by_norm = gradient_clipping_by_norm

  @property
  def is_training(self):
    return True

  @property
  def mode(self):
    return ModeKeys.train

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

#    gt_boxlist_list = misc_utils.preprocess_groundtruth(
#        tensor_dict[TensorDictFields.groundtruth_boxes],
#        tensor_dict[TensorDictFields.groundtruth_labels])
    gt_boxlist_list = misc_utils.preprocess_groundtruth(tensor_dict)

    prediction_dict = self.predict(inputs)

    losses_dict = commons.compute_losses(
        self, prediction_dict, gt_boxlist_list)

    loc_loss = losses_dict[LossTensorDictFields.localization_loss]
    cls_loss = losses_dict[LossTensorDictFields.classification_loss]

    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) 
        + [loc_loss, cls_loss])

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
      load_all_detection_checkpoint_vars: a bool scalar, whether to load all
        detection checkpoint variables.

    Returns:
      a dict mapping from variable names to variable tensors.
    """
    name_to_var_map = {}

    for var in tf.global_variables():
      var_name = var.op.name
      if checkpoint_type == 'detection' and load_all_detection_checkpoint_vars:
        name_to_var_map[var_name] = var
      else:
        if var_name.startswith(self.extract_features_scope):
          if checkpoint_type == 'classification':
            var_name = re.split('^' + self.extract_features_scope + '/',
              var_name)[-1]
          name_to_var_map[var_name] = var

    return name_to_var_map

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
    """Creates persist saver for persisting variables to a checkpoint.

    Args:
      max_to_keep: int scalar, max num of checkpoints to keep.

    Returns:
      persist_saver: a tf.train.Saver instance.
    """
    persist_saver = tf.train.Saver(max_to_keep=max_to_keep)
    return persist_saver

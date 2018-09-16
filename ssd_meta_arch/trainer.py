import re

import tensorflow as tf

from detection.core.standard_names import ModeKeys
from detection.core.standard_names import TensorDictFields
from detection.core.standard_names import PredTensorDictFields
from detection.core.standard_names import LossTensorDictFields
from detection.ssd_meta_arch import detection_model 
from detection.ssd_meta_arch import core
from detection.utils import variables_utils as var_utils


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
               normalize_loss_by_num_matches,
               normalize_loc_loss_by_codesize,
               freeze_batch_norm,
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
      localization_loss_weigh: a float scalar that scales the contribution of 
        localization loss to total loss.
      classification_loss_weight: a float scalar that scales the contribution
        of classification loss to total loss.
      normalize_loss_by_num_matches: a bool scalar, whether to normalize both
        types of losses by num of matches.    
      normalize_loc_loss_by_codesize: a bool scalar, whether to normalize 
        localization loss by box code size (e.g. 4 for the default box coder.)
      freeze_batch_norm: a bool scalar, whether to freeze batch norm parameters.
      add_background_class: a bool scalar, whether to add background class. 
        Should be False if the examples already contains background class.
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
    self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
    self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize 
    self._freeze_batch_norm = freeze_batch_norm
    self._add_background_class = add_background_class

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
      dataset: a dataset.DetectionDataset instance.
      optimizer_builder_fn: a callable, that takes no argument and generates a
        2-tuple containing an optimizer instance and a learning rate tensor (
        float scalar).

    Returns:
      a list of tf.Operation objects to be run in a tf.Session.
    """
    self.check_dataset_mode(dataset)

    tensor_dict = dataset.get_tensor_dict(filename_list)

    inputs, _ = self.preprocess(tensor_dict[TensorDictFields.image])

    pred_tensor_dict = self.predict(inputs)

    pred_box_encodings = pred_tensor_dict[
        PredTensorDictFields.box_encoding_predictions]
    pred_class_scores = pred_tensor_dict[
        PredTensorDictFields.class_score_predictions]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    loss_tensor_dict = core.create_losses(
        self, pred_box_encodings, pred_class_scores,
        tensor_dict[TensorDictFields.groundtruth_boxes],
        tensor_dict[TensorDictFields.groundtruth_labels], 
        None)

    loc_loss = loss_tensor_dict[LossTensorDictFields.localization_loss]
    cls_loss = loss_tensor_dict[LossTensorDictFields.classification_loss]

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    total_loss = tf.add_n(reg_losses + [loc_loss, cls_loss])

    global_step = tf.train.get_or_create_global_step()

    optimizer, learning_rate = optimizer_builder_fn()

    grads_and_vars = optimizer.compute_gradients(total_loss)

    grad_update_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)

    update_ops.append(grad_update_op)

    grouped_update_op = tf.group(*update_ops, name='update_barrier')

    return grouped_update_op, total_loss, global_step

  def _get_vars_map(self,
                    fine_tune_checkpoint_type='classification',
                    load_all_detection_checkpoint_vars=False):
    """Returns a mapping from var name to var tensor for restoring variables 
    from a checkpoint.

    Args:
      fine_tune_checkpoint_type: a string scalar, type of checkpoint from which
        to initialize variables. 'classification' for a pretrained 
        classification model (e.g. Inception, ResNet), 'detection' for a 
        pretrained detection model.
      load_all_detection_checkpoint_vars: a bool scalar, whether to load all
        detection checkpoint variables.

    Returns:
      a dict mapping from variable names to variable tensors.
    """
    vars_to_restore = {}
    for var in tf.global_variables():
      var_name = var.op.name
      if (fine_tune_checkpoint_type == 'detection' and
          load_all_detection_checkpoint_vars):
        vars_to_restore[var_name] = var
      else:
        if var_name.startswith(self._extract_features_scope):
          if fine_tune_checkpoint_type == 'classification':
            var_name = (
                re.split('^' + self._extract_features_scope + '/',
                         var_name)[-1])
          vars_to_restore[var_name] = var
    return vars_to_restore


  def create_restore_saver(self,
                           load_ckpt_path,
                           fine_tune_checkpoint_type='classification',
                           load_all_detection_checkpoint_vars=False):
    """Creates restore saver for restoring variables from a checkpoint.
      
    Args:
      load_ckpt_path: a string scalar, pointing to a checkpoint file ('*.ckpt').
      fine_tune_checkpoint_type: a string scalar, type of checkpoint from which
        to initialize variables. 'classification' for a pretrained 
        classification model (e.g. Inception, ResNet), 'detection' for a 
        pretrained detection model.
      load_all_detection_checkpoint_vars: a bool scalar, whether to load all
        detection checkpoint variables.
      
    Returns:
      restore_saver: a tf.train.Saver instance.
    """
    vars_map = self._get_vars_map(fine_tune_checkpoint_type,
                                  load_all_detection_checkpoint_vars) 

    available_vars_map = var_utils.get_vars_available_in_ckpt(
        vars_map, load_ckpt_path, include_global_step=False)

    restore_saver = tf.train.Saver(available_vars_map)
    return restore_saver

  def create_persist_saver(self):
    """Creates persist saver for persisting variables to a checkpoint file.

    Returns:
      persist_saver: a tf.train.Saver instance.
    """
    persist_saver = tf.train.Saver()
    return persist_saver


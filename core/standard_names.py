import tensorflow as tf


class TensorDictFields(object):
  image = 'image'
  groundtruth_boxes = 'groundtruth_boxes'
  groundtruth_labels = 'groundtruth_labels'
  runtime_shape_str = '_runtime_shape'


class TfRecordFields(object):
  image_encoded = 'image/encoded'
  object_class_label = 'image/object/class/label'
  object_bbox_ymin = 'image/object/bbox/ymin'
  object_bbox_xmin = 'image/object/bbox/xmin'
  object_bbox_ymax = 'image/object/bbox/ymax'
  object_bbox_xmax = 'image/object/bbox/xmax'


class BoxListFields(object):
  boxes = 'boxes'
  classes = 'classes'
  scores = 'scores'
  weights = 'weights'
  objectness = 'objectness'
  masks = 'masks'
  boundaries = 'boundaries'
  keypoints = 'keypoints'
  keypoint_heatmaps = 'keypoint_heatmaps'
  is_crowd = 'is_crowd'


class DatasetDictFields(object):
  trainer_dataset = 'trainer_dataset'
  evaluator_dataset = 'evaluator_dataset'
  inferencer_dataset = 'inferencer_dataset'

class ModeKeys(object):
  train = tf.contrib.learn.ModeKeys.TRAIN
  eval = tf.contrib.learn.ModeKeys.EVAL
  infer = tf.contrib.learn.ModeKeys.INFER

class PredTensorDictFields(object):
  box_encoding_predictions = 'box_encoding_predictions'
  class_score_predictions = 'class_score_predictions'


class LossTensorDictFields(object):
  localization_loss = 'localization_loss'
  classification_loss = 'classification_loss' 


class DetTensorDictFields(object):
  detection_boxes = 'detection_boxes'
  detection_scores = 'detection_scores'
  detection_classes = 'detection_classes'
  num_detections = 'num_detections'


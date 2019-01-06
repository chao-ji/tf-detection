import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from detection.protos import ssd_model_pb2
from detection.protos import faster_rcnn_model_pb2
from detection.protos import dataset_pb2
from detection.protos import train_config_pb2
from detection.protos import test_config_pb2
from detection.protos import label_map_pb2
from detection.core import box_list
from detection.core import box_list_ops

from detection.core.standard_names import DetTensorDictFields


def print_evaluation_metrics(
    metrics_calculator, 
    metrics_name, 
    losses, 
    losses_names, 
    label_map):
  """Print out the evaluation metrics.

  Args:
    metrics_calculator: an instance of DetectionMetricsCalculator.
    metrics_name: string scalar, metrics name.
    losses: numpy array of shape [num_images, num_loss_types], holding the
      per-image losses (e.g.)
    losses_names: a list of `num_loss_types` strings, name of loss types (e.g. 
      'loc_loss', 'cls_loss' for RPN, or 'rpn_loc_loss', 'rpn_cls_loss',
      'frcnn_loc_loss', 'frcnn_cls_loss' for Faster RCNN). 
    label_map: a dict mapping from int (class index) to string (class name).
  """
  class_indices = list(label_map.keys())

  if metrics_name == 'pascal_voc_detection_metrics':
    num_classes = metrics_calculator.num_classes
    APs = metrics_calculator.calculate_metrics()
    print('PascalBoxes_Precision/mAP@0.5IOU {}'.format(
        np.mean(list(APs.values()))))
    for i in range(num_classes):
      if i + 1 not in class_indices:
        continue
      print('PascalBoxes_PerformanceByCategory/AP@0.5IOU/{} {}'.format(
          label_map[i + 1], APs[i + 1]))
  elif metrics_name == 'coco_detection_metrics':
    metrics = metrics_calculator.calculate_metrics()
    print()
    for k, v in metrics.items():
      print(k, v)

  for i, name in enumerate(losses_names):
    print(name, np.nanmean(losses[:, i]))


def replace_ext(filename, ext=''):
  """Removes file extension name and appends the new one.
  Args:
    filename: string scalar, file name.
    ext: string scalar, extension to be appended. Defaults to ''.

  Returns:
    filename_ext_replaced: string scalar, 
  """
  return '.'.join(filename.split('.')[:-1]) + ext


def label_map_to_categories(label_map):
  """Convert label map to the format required for COCO evaluation.
  
  For example, input is {1: 'label_1', 2: 'label_2', ...}
  output is [{'id': 1, 'name': 'label_1'}, {'id': 2, 'name': 'label_2'}, ...]

  Args:
    label_map: a dict mapping from class index (int scalar starting from 1 up 
      to total num of classes) to class label (string scalar).

  Returns:
    categories: a list of dict, each having two entries 'id' and 'name'. 
  """
  categories = []
  for k, v in label_map.items():
    categories.append({'id': k, 'name': v})
  return categories


def read_config(config_file, config_type):
  """
  Args:
    config_file: string scalar, path to the file (plain text format) storing 
      configuration (e.g. model, dataset, or train).
    config_type: string scalar, type of configuration.

  Return:
    a protobuf message storing configuration of type `config_type`.
  """
  if config_type == 'ssd_model':
    config = ssd_model_pb2.SsdModel()
  elif config_type == 'faster_rcnn_model':
    config = faster_rcnn_model_pb2.FasterRcnnModel()
  elif config_type == 'dataset':
    config = dataset_pb2.Dataset()
  elif config_type == 'train':
    config = train_config_pb2.TrainConfig()
  elif config_type == 'test':
    config = test_config_pb2.TestConfig()
  else:
    raise ValueError('config file must be "model", "dataset", "train", or '
        '"test".')
  text_format.Merge(open(config_file).read(), config)
  return config 


def read_label_map(label_map_config_file):
  """Reads a label_map config from a config file.

  Args:
    label_map_config_file: string scalar, filename of a config file (plain text
      format) storing the class index and class label pairs.

  Returns:
    label_map: a dict mapping from class index (int scalar starting from 1 up 
      to total num of classes) to class label (string scalar).
    num_classes: int scalar, num of classes.
  """
  label_map_config = label_map_pb2.LabelMap()
  text_format.Merge(open(label_map_config_file).read(), label_map_config)
  label_map = dict([(item.index, item.label) 
      for item in label_map_config.label_map_item])
  num_classes = label_map_config.num_classes
#  _check_label_map(label_map)
  return label_map, num_classes


def _check_label_map(label_map):
  """Checks if input `label_map` (dict) is valid (keys range from 1 up to 
  `len(label_map)`).
  """
  keys = sorted(list(label_map.keys()))
  if not (np.arange(len(keys)) == np.array(keys) - 1).all():
    raise ValueError('label map indices must start from 1 up to max value.')


def check_dataset_mode(model, dataset):
  """Checks if the mode (train, eval, or infer) of a dataset and a model match.

  Args:
    model: an instance of DetectionModel.
    dataset: an instance of DetectionDataset.

  Raises:
    ValueError if mode of `dataset` and `model` do not match.
  """
  if dataset.mode != model.mode:
    raise ValueError('mode of dataset({}) and model({}) do not match.'
        .format(dataset.mode, model.mode))


def preprocess_groundtruth(
    gt_boxes_list, 
    gt_labels_list, 
    labels_field_name='labels'):
  """Package the groundtruth labels tensor and boxes tensor as a BoxList.

  Args:
    gt_boxes_list: a list of float tensors of shape [num_gt_boxes, 4], holding
      groundtruth boxes cooridnates. Length is equal to `batch_size`.
    gt_labels_list: a list of float tensors of shape 
      [num_gt_boxes, num_class + 1], holding groundtruth boxes class labels 
      (one-hot encoded). Length is equal to `batch_size`.
    labels_field_name: string scalar, field name of labels in a BoxList.

  Returns:
    gt_boxlist_list: a list of BoxList instance holding `num_gt_boxes` 
      groundtruth boxes with extra field 'labels'.
  """
  if len(gt_boxes_list) != len(gt_labels_list):
    raise ValueError('`gt_boxes_list` must have the same length of '
        '`gt_labels_list`.')

  gt_boxlist_list = []
  for gt_boxes, gt_labels in zip(gt_boxes_list, gt_labels_list):
    gt_boxlist = box_list.BoxList(gt_boxes)
    gt_boxlist.set_field(labels_field_name, gt_labels)
    gt_boxlist_list.append(gt_boxlist)
  return gt_boxlist_list


def process_per_image_detection(image_list, 
                                detection_dict, 
                                gt_boxlist_list=None):
  """Processes the nms'ed, potentially padded detection results for a single
  image. Unpad the detection results and convert normalized coorindates into 
  absolute coordinates.

  Args:
    image_list: a list of float tensors of shape [height, width, depth]. Length 
      is equal to `batch_size`.
    detection_dict: a dict mapping from strings to tensors, holding the 
      following entries:
      { 'boxes': float tensor of shape [batch_size, max_num_boxes, 4].
        'scores': float tensor of shape [batch_size, max_num_boxes].
        'classes': float tensor of shape [batch_size, max_num_boxes].
        'num_detections': int tensor of shape [batch_size], holding num of
          valid (not zero-padded) detections in each of the above tensors.}
    gt_boxlist_list: a list of BoxList instances, each holding `num_gt_boxes`
      groundtruth_boxes, with extra field 'labels' holding float tensor of shape
      [num_gt_boxes, num_classes + 1] (groundtruth boxes labels). Length of 
        list is equal to `batch_size`.

  Returns:
    to_be_run_tensor_dict: a dict mapping from strings to tensors, holding the
      following entries:
      { 'image': uint8 tensor of shape [height, width, depth], holding the 
          original image.
        'boxes': float tensor of shape [num_val_detections, 4], holding 
          coordinates of predicted boxes.
        'scores': float tensor of shape [num_val_detections], holding predicted
          confidence scores.
        'classes': int tensor of shape [num_val_detections], holding predicted
          class indices.
        'gt_boxes': float tensor of shape [num_gt_boxes, 4], holding coordinates
          of groundtruth boxes.
        'gt_labels': int tensor of shape [num_gt_boxes], holding groundtruth 
          box class indices.}
  """
  boxes = detection_dict['boxes']
  scores = detection_dict['scores']
  classes = tf.to_int32(detection_dict['classes'])
  num_detections = detection_dict['num_detections']

  if len(image_list) != 1:
    raise ValueError('`image_list` must contain exactly one image tensor.')
  if not(boxes.shape[0].value == 1 and scores.shape[0].value == 1 and 
      classes.shape[0].value == 1 and num_detections.shape[0].value == 1):
    raise ValueError('`boxes`, `scores`, `classes`, `num_detections` must have'
        'size 1 in the 0th dimension (i.e. batch size).')
  if gt_boxlist_list is not None and len(gt_boxlist_list) != 1:
    raise ValueError('`gt_boxlist_list` must contain exactly one groundtruth '
        'BoxList.')
    
  boxes, scores, classes, num_detections, image = (
      boxes[0], scores[0], classes[0], num_detections[0], image_list[0])
  boxes, classes, scores = (
      boxes[:num_detections], classes[:num_detections], scores[:num_detections])
  height, width = tf.unstack(tf.shape(image)[:2])
  boxes = box_list_ops.to_absolute_coordinates(
      box_list.BoxList(boxes), height, width).get()

  to_be_run_tensor_dict = {'image': tf.cast(image, tf.uint8), 
                           'boxes': boxes, 
                           'scores': scores,
                           'classes': classes} 

  if gt_boxlist_list is not None:
    gt_boxlist = gt_boxlist_list[0]
    gt_boxes = box_list_ops.to_absolute_coordinates(
      gt_boxlist, height, width).get() 

    gt_labels = tf.argmax(
        gt_boxlist.get_field('labels'), axis=1, output_type=tf.int32)
    to_be_run_tensor_dict['gt_boxes'] = gt_boxes
    to_be_run_tensor_dict['gt_labels'] = gt_labels
  return to_be_run_tensor_dict
 

class IdentityContextManager(object):
  def __enter__(self):
    return None

  def __exit__(self, exception_type, exception_value, traceback):
    return False

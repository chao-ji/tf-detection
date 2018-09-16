from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from detection.metrics import np_box_ops
from detection.metrics import utils 


class DetectionMetricsCalculator(object):
  """Computes object detection metrics
    
  """

  __metaclass__ = ABCMeta

  def __init__(self):
    pass

  @abstractmethod
  def update_per_image_result(self, np_array_dict):
    pass

  @abstractmethod
  def calculate_metrics(self):
    pass


class PascalVocMetricsCalculator(DetectionMetricsCalculator):
  """
  """
  def __init__(self, num_classes):
    self._num_classes = num_classes
    self._scores_per_class = [[] for _ in range(num_classes)]
    self._tp_fp_labels_per_class = [[] for _ in range(num_classes)]
    self._num_gt_instances_per_class = np.zeros(num_classes, dtype=float)

  def update_per_image_result(self, detection_dict, gt_boxes, gt_labels):
    """
      Args:
        
    """

    gt_labels = gt_labels.argmax(axis=1)
    for label in gt_labels:
      self._num_gt_instances_per_class[label] += 1

    scores = [[]] * self._num_classes
    tp_fp_labels = [[]] * self._num_classes

    detection_dict = remove_invalid_boxes(detection_dict)
    
    for i in range(self._num_classes):
      det_indices = (detection_dict['classes'] - 1 == i)
      scores[i] = detection_dict['scores'][det_indices]

      gt_indices = gt_labels == i

      if not det_indices.any():
        scores[i] = np.array([], dtype=float)
        tp_fp_labels[i] = np.array([], dtype=bool)
        continue

      iou = np_box_ops.iou(
          detection_dict['boxes'][det_indices], gt_boxes[gt_indices])

      if not gt_indices.any():
        tp_fp_labels[i] = np.zeros(scores[i].shape[0], dtype=bool)
        continue

      scores[i] = scores[i].astype(np.float64)
      tp_fp_labels[i] = np.zeros(scores[i].shape[0], dtype=bool)

      max_overlap_gt_ids = np.argmax(iou, axis=1)
      is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)

      for j in range(scores[i].shape[0]):
        gt_id = max_overlap_gt_ids[j]
        if iou[j, gt_id] >= 0.5:
          if not is_gt_box_detected[gt_id]:
            tp_fp_labels[i][j] = True
            is_gt_box_detected[gt_id] = True

      tp_fp_labels[i] = tp_fp_labels[i].astype(float)


    for i in range(self._num_classes):
      if scores[i].shape[0] > 0:
        self._scores_per_class[i].append(scores[i])
        self._tp_fp_labels_per_class[i].append(tp_fp_labels[i])


  def calculate_metrics(self):
    average_precisions = []
    for i in range(self._num_classes):
      if self._num_gt_instances_per_class[i] == 0:
        continue

      scores = np.concatenate(self._scores_per_class[i])
      tp_fp_labels = np.concatenate(self._tp_fp_labels_per_class[i])

      precision, recall = utils.compute_precision_recall(
          scores, tp_fp_labels, self._num_gt_instances_per_class[i])

      average_precision = utils.compute_average_precision(precision, recall)

      average_precisions.append(average_precision)
 
    return average_precisions


def MscocoMetricsCalculator(DetectionMetricsCalculator):

  def update_per_image_result(self):
    pass
  def calculate_metrics(self):
    pass




def remove_invalid_boxes(detection_dict):

  detection_boxes = detection_dict['boxes']
  
  valid_box_indices = np.logical_and(
      detection_boxes[:, 0] < detection_boxes[:, 2],
      detection_boxes[:, 1] < detection_boxes[:, 3])

  detection_dict['classes'] = detection_dict['classes'][valid_box_indices]
  detection_dict['scores'] = detection_dict['scores'][valid_box_indices]
  detection_dict['boxes'] = detection_dict['boxes'][valid_box_indices]
 
  height, width, _ = detection_dict['image'].shape

  return detection_dict


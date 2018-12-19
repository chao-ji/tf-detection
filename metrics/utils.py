import numpy as np


def compute_precision_recall(scores, labels, num_gt):
  """Compute precision and recall.

  Args:
    scores: float numpy array of shape [num_detections], holding detection 
      scores. 
    labels: float or bool numpy array of shape [num_detections], holding 0's and
      1's, where 1's indicate true positive and 0's indicate false positive.
    num_gt: int scalar, the num of groundtruth detections (i.e. TP + FN). The 
      num of detections (`np.sum(labels)`) must not exceed `num_gt`.

  Returns:
    precision: float numpy array of shape [num_detections], holding values in
      [0, 1].
    recall: float numpy array of shape [num_detections], holding non-decreasing
      values in [0, 1].

  """
  if num_gt < np.sum(labels):
    raise ValueError("Number of true positives must be smaller than num_gt.")

  sorted_indices = np.argsort(scores)[::-1]
  true_positive_labels = labels[sorted_indices]
  false_positive_labels = (true_positive_labels <= 0).astype(float)
  cum_true_positives = np.cumsum(true_positive_labels)
  cum_false_positives = np.cumsum(false_positive_labels)
  precision = cum_true_positives.astype(float) / (
      cum_true_positives + cum_false_positives)
  recall = cum_true_positives.astype(float) / num_gt
  return precision, recall


def compute_average_precision(precision, recall):
  """Compute Average Precision (AP) according to the definition in VOCdevkit.

  `precision` is modifield such that it does not increase as the score threshold
  is lowered -- when threshold is lowered, `precision` could increase, because
  additional true-positive can be included,
    `precision = TP/(TP+FP)` would increase if `TP` is increased.

  `recall` should be a non-decreasing sequence from 0.0 to 1.0, and `precision` 
  should be a non-increasing sequence from `p` (0<= `p` <= 1) down to 0.0.

  After the modification, AP is simply computed as the area under the 
  precision-recall curve.

  Args:
    precision: float numpy array of shape [num_detections], holding values in
      [0, 1]. 
    recall: float numpy array of shape [num_detections], holding non-decreasing
      values in [0, 1].

  Returns:
    average_precison: float scalar. 

  """
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("precision must be in the range [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Precision is modified to be a strictly non-inreasing array.
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(np.diff(recall) > 0)[0] + 1 
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision

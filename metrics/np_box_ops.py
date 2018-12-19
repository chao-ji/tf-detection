import numpy as np


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: numpy array of shape [num_boxes, 4], each row holding 
      ymin, xmin, ymax, xmax box coordinates. 

  Returns:
    numpy array of shape [num_boxes] holding areas of boxes.
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: numpy array of shape [num_boxes_1, 4], each row holding 
      ymin, xmin, ymax, xmax box coordinates.
    boxes2: numpy array of shape [num_boxes_2, 4], each row holding 
      ymin, xmin, ymax, xmax box coordinates.

  Returns:
    a numpy array of shape [num_boxes_1, num_boxes_2], storing pairwise 
      intersections between `boxes1` and `boxes2`.
  """
  ymin1, xmin1, ymax1, xmax1 = np.split(boxes1, 4, axis=1)
  ymin2, xmin2, ymax2, xmax2 = np.split(boxes2, 4, axis=1)

  pairwise_min_ymax = np.minimum(ymax1, np.transpose(ymax2))
  pairwise_max_ymin = np.maximum(ymin1, np.transpose(ymin2))
  intersect_heights = np.maximum(
#      np.zeros(pairwise_max_ymin.shape),
      0.0,
      pairwise_min_ymax - pairwise_max_ymin)


  pairwise_min_xmax = np.minimum(xmax1, np.transpose(xmax2))
  pairwise_max_xmin = np.maximum(xmin1, np.transpose(xmin2))
  intersect_widths = np.maximum(
#      np.zeros(pairwise_max_xmin.shape),
      0.0,
      pairwise_min_xmax - pairwise_max_xmin)
  return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
  """Computes pairwise intersection-over-union between boxes collections.

  Args:
    boxes1: numpy array of shape [num_boxes_1, 4], each row holding 
      ymin, xmin, ymax, xmax box coordinates.
    boxes2: numpy array of shape [num_boxes_2, 4], each row holding 
      ymin, xmin, ymax, xmax box coordinates.

  Returns:
    a numpy array of shape [num_boxes_1, num_boxes_2], storing pairwise 
      IOU between `boxes1` and `boxes2`.
  """
  intersections = intersection(boxes1, boxes2)
  area1 = area(boxes1)
  area2 = area(boxes2)
  unions = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersections
#  return intersect / union
  return np.where(intersections == 0.0, np.zeros_like(intersections), intersections / unions)


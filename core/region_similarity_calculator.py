from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.core import box_list_ops


class RegionSimilarityCalculator(object):
  """Abstract base class for region similarity calculator."""
  __metaclass__ = ABCMeta

  def compare(self, boxlist1, boxlist2, scope=None):
    """Computes a 2D array holding pairwise similarity between BoxLists.
    Calls and wraps `_compare` with a name scope.

    Args:
      boxlist1: a BoxList holding `n` boxes.
      boxlist2: a BoxList holding `m` boxes.
      scope: string scalar or None, scope name.

    Returns:
      a float tensor of shape [n, m] storing pairwise similarity score.
    """
    with tf.name_scope(scope, 'Compare', [boxlist1, boxlist2]):
      return self._compare(boxlist1, boxlist2)

  @abstractmethod
  def _compare(self, boxlist1, boxlist2):
    """Computes pairwise similarity score between boxlists.

    To be implemented by subclasses.

    Args:
      boxlist1: a BoxList holding `n` boxes.
      boxlist2: a BoxList holding `m` boxes.

    Returns:
      a float tensor of shape [n, m] storing pairwise similarity between 
        `boxlist1` and `boxlist2`.
    """
    pass


class IouSimilarity(RegionSimilarityCalculator):
  """Compute similarity as the Intersection over Union (IOU) between boxlists.

  Note: IOU are symmetric. If two boxes have zero intersection, their IOU is 
  set to zero.
  """

  def _compare(self, boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between boxlists.

    Args:
      boxlist1: a BoxList holding `n` boxes.
      boxlist2: a BoxList holding `m` boxes.

    Returns:
      a float tensor of shape [n, m] storing pairwise IOU between 
        `boxlist1` and `boxlist2`.
    """
    return box_list_ops.iou(boxlist1, boxlist2)


class IoaSimilarity(RegionSimilarityCalculator):
  """Compute similarity as the Intersection over Area (IOA) between boxlists.

  Note: IOA are non-symmetric. If two boxes have zero intersection, their IOA is 
  set to zero.
  """
  def _compare(self, boxlist1, boxlist2):
    """Computes pairwise intersection-over-area between boxlists.

    Args:
      boxlist1: a BoxList holding `n` boxes.
      boxlist2: a BoxList holding `m` boxes.

    Returns:
      a float tensor of shape [n, m] storing pairwise IOA between 
        `boxlist1` and `boxlist2`.
    """
    return box_list_ops.ioa(boxlist1, boxlist2)

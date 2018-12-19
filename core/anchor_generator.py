from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import tensorflow as tf


class AnchorGenerator(object):
  """Abstract base class for anchor generators.

  Subclass must implement abstractproperty `num_anchors_per_location` and 
  abstractmethod `_generate`.
  """
  __metaclass__ = ABCMeta

  def __init__(self, name_scope, check_num_anchors=True):
    """Constructor.

    Args:
      name_scope: string scalar, name scope of `generate` method.
      check_num_anchors: bool scalar, whether to dynamically check if the num of
        anchors generated is correct.
    """
    self._name_scope = name_scope
    self._check_num_anchors = check_num_anchors

  @abstractproperty
  def num_anchors_per_location(self):
    """Returns a list of ints, holding the num of anchors per spatial location
    for each feature map.
    """
    pass

  def generate(self, feature_map_shape_list, **params):
    """Generates a list of BoxLists as anchors.

    Calls and wraps `_generate` with a name scope.

    Args:
      feature_map_shape_list: a list of 2-tuples of ints or scalar int tensors,
        holding the height and width sizes of the feature maps that anchors are
        generated for.
      **params: dict, holding additional parameter name to parameter mapping.

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes for each feature 
        map.
    """
    if self._check_num_anchors and (
        len(feature_map_shape_list) != len(self.num_anchors_per_location)):
      raise ValueError('Num of feature maps must be equal to num of'
                       '`num_anchors_per_location`, got {} and {}'.format(
                        len(feature_map_shape_list), 
                        len(self.num_anchors_per_location)))

    with tf.name_scope(self._name_scope):
      anchors_list = self._generate(feature_map_shape_list, **params)
      if self._check_num_anchors:
        with tf.control_dependencies([self._assert_correct_number_of_anchors(
            anchors_list, feature_map_shape_list)]):
          for anchors in anchors_list:
            anchors.set(tf.identity(anchors.get()))
      return anchors_list

  @abstractmethod
  def _generate(self, feature_map_shape_list, **params):
    """Generates a list of BoxLists as anchors.

    To be implemented by subclasses.

    Args:
      feature_map_shape_list: a list of 2-tuples of ints or scalar int tensors,
        holding the height and width sizes of the feature maps that anchors are
        generated for. 
      **params: dict, holding additional parameter name to parameter mapping.

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes for each feature 
        map. 
    """
    pass

  def _assert_correct_number_of_anchors(self, 
      anchors_list, feature_map_shape_list):
    """Generate op that asserts the correct num of anchors was generated.

    Args:
      anchors_list: a list of BoxLists each holding anchor boxes for each 
        feature map.
      feature_map_shape_list: a list of 2-tuples of ints or scalar int tensors,
        holding the height and width sizes of the feature maps that anchors are
        generated for. 

    Returns:
      Op that raises InvalidArgumentError if the num of anchors does not
        match the number of expected anchors.
    """
    expected_num_anchors = 0
    actual_num_anchors = 0
    for num_anchors, feature_map_shape, anchors in zip(
        self.num_anchors_per_location, feature_map_shape_list, anchors_list):
      expected_num_anchors += (
          num_anchors * feature_map_shape[0] * feature_map_shape[1])
      actual_num_anchors += anchors.num_boxes()
    return tf.assert_equal(expected_num_anchors, actual_num_anchors)

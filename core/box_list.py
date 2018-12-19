import tensorflow as tf
from detection.utils import shape_utils


class BoxList(object):
  """BoxList is a wrapper of a tensor_dict holding tensors related to a 
  collection of bounding boxes for object detection models. It has a required 
  data field 'boxes' that points to a tensor of shape [num_boxes, 4], where 
  each row stores the (ymin xmin, ymax, xmax) coordinates of a collection of 
  `num_boxes` boxes (e.g. groundtruth boxes, anchor boxes).

  In addition to the required 'boxes' field, one can insert additional tensors 
  into the optional fields such as 'groundtruth_labels', 'scores'. It is 
  required that these tensors' 0th dimension have size `num_boxes`.

  Note that it is OK to dynamically update the collection of boxes (i.e. the 
  tensors stored in the data fields) IN PLACE. For example, 
  `box_list_ops.filter_greater_than` that filters out boxes whose scores are 
  below threshold would output the same BoxList as the input, the only change
  being that the data fields now point to a different set of tensors. See more
  functions like this in `box_list_ops` where the docstring states 
  'In-place operation'.
  """

  def __init__(self, boxes, check=False):
    """Constructor.

    Args:
      boxes: float tensor of shape [num_boxes, 4], where each row stores 
        coordinates `ymin`, `xmin`, `ymax`, `xmax` of a bounding box.
      check: bool scalar, whether to check validity of box field values when 
        updated by `set_field` method.
    """
    self._check = check
    
    self._data = dict()
    self.set(boxes)

  @property 
  def data(self):
    return self._data

  def num_boxes(self):
    """Returns an int scalar or int scalar tensor representing the num of boxes.
    """
    return shape_utils.combined_static_and_dynamic_shape(self.get())[0]

  def get_all_fields(self):
    """Returns a list of strings representing the names of all data fields (
    including the required field 'boxes').
    """
    return self._data.keys()

  def get_extra_fields(self):
    """Returns a list of strings representing the names of optional data fields 
    (i.e. anything but 'boxes').
    """
    return [k for k in self.get_all_fields() if k != 'boxes']

  def has_field(self, field):
    return field in self._data

  def get(self):
    """Returns the 'boxes' field."""
    return self.get_field('boxes')

  def set(self, boxes):
    """Sets the 'boxes' field.

    Args:
      boxes: float tensor of shape [num_boxes, 4], where each row stores 
        coordinates `ymin`, `xmin`, `ymax`, `xmax` of a bounding box. 
    """
    self.set_field('boxes', boxes)

  def get_field(self, field):
    """Gets an existing field.

    Args:
      field: string scalar, name of data field. 

    Returns:
      a tensor stored in field 'field'. 
    """
    if not self.has_field(field):
      raise ValueError('field {} does not exist'.format(field))
    return self._data[field]

  def set_field(self, field, value):
    """Sets an existing (or creates new) field with tensor holding updated data 
    (or new data).

    Args:
      field: string scalar, name of the data field.
      value: a tensor storing the updated data. 
    """
    if self._check:
      with tf.control_dependencies(self._check_validity(field, value)):
        value = tf.identity(value)
    self._data[field] = value

  def get_center_coordinates_and_sizes(self, scope=None):
    """Gets the [ycenter, xcenter, height, width] representation of boxes. 

    Args:
      scope: string scalar, name scope.

    Returns:
      ycenter: float tensor of shape [num_boxes].
      xcenter: float tensor of shape [num_boxes].
      height: float tensor of shape [num_boxes].
      width: float tensor of shape [num_boxes].
    """
    with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
      ymin, xmin, ymax, xmax = tf.unstack(value=self.get(), axis=1)
      width = xmax - xmin
      height = ymax - ymin
      ycenter = ymin + height / 2.
      xcenter = xmin + width / 2.
      return ycenter, xcenter, height, width

  def area(self, scope=None):
    """Computes area of boxes.

    Args:
      scope: string scalar, name scope.

    Returns:
      a float tensor of shape [num_boxes] holding areas of boxes.
    """
    with tf.name_scope(scope, 'area'):
      ymin, xmin, ymax, xmax = tf.unstack(value=self.get(), axis=1)
      return (ymax - ymin) * (xmax - xmin)

  def _check_validity(self, field, value, scope=None):
    """Checks validity of the value to be inserted to a field. 

    If `field` == 'boxes', it must hold that ymin <= ymax and xmin <= xmax;
    for other fields, the 'boxes' must have already been set, and the size of 
    their first dimension must be equal to `self.num_boxes()`. 

    Args:
      field: string scalar, name of the data field.
      scope: string scalar, name scope.

    Returns:
      a tuple of Ops that raise `InvalidArgumentError` if the above conditions 
        do not hold.
    """
    with tf.name_scope(scope, 'check_validity'):
      if field == 'boxes':
        ymin, xmin, ymax, xmax = tf.unstack(value=value, axis=1)
        return (tf.assert_less_equal(ymin, ymax),
                tf.assert_less_equal(xmin, xmax))
      else:
        return (tf.assert_equal(self.num_boxes(),
            shape_utils.combined_static_and_dynamic_shape(value)[0]),)

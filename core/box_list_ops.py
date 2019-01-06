import tensorflow as tf

from detection.core import box_list
from detection.utils import shape_utils


class SortOrder(object):
  """Enum class for sort order.

  Attributes:
    ascend: ascend order.
    descend: descend order.
  """
  ascend = 1
  descend = 2


def scale(boxlist, yscale, xscale, scope=None):
  """Scale box coordinates in y (height) and x (width) dimensions. In-place
  operation.

  Args:
    boxlist: a BoxList holding `n` boxes.
    yscale: float scalar tensor, the value by which y coordinates are scaled.
    xscale: float scalar tensor, the value by which x coordinates are scaled.
    scope: string scalar, name scope.

  Returns:
    boxlist: a BoxList holding `n` boxes.
  """
  with tf.name_scope(scope, 'Scale'):
    yscale = tf.to_float(yscale)
    xscale = tf.to_float(xscale)
    ymin, xmin, ymax, xmax = tf.unstack(value=boxlist.get(), axis=1)
    ymin *= yscale
    ymax *= yscale
    xmin *= xscale
    xmax *= xscale
    boxlist.set(tf.stack([ymin, xmin, ymax, xmax], axis=1))
    return boxlist


def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
  """Clip boxes in a BoxList to a window, and optionally filter out boxes that
  do not overlap with the window. In-place operation.

  Args:
    boxlist: a BoxList holding `n_in` boxes.
    window: a float tensor of shape [4] holding the [y_min, x_min, y_max, x_max]
      box coordinates that the boxes are clipped to.
    filter_nonoverlapping: bool scalar, whether to filter out boxes that do not 
      overlap at all with the window.
    scope: string scalar, name scope.

  Returns:
    a BoxList holding `n_out <= n_in` boxes.
  """
  with tf.name_scope(scope, 'ClipToWindow'):
    ymin, xmin, ymax, xmax = tf.unstack(value=boxlist.get(), axis=1)
    win_ymin, win_xmin, win_ymax, win_xmax = tf.unstack(window)
    ymin_clipped = tf.maximum(tf.minimum(ymin, win_ymax), win_ymin)
    ymax_clipped = tf.maximum(tf.minimum(ymax, win_ymax), win_ymin)
    xmin_clipped = tf.maximum(tf.minimum(xmin, win_xmax), win_xmin)
    xmax_clipped = tf.maximum(tf.minimum(xmax, win_xmax), win_xmin)
    boxlist.set(tf.stack([
        ymin_clipped, xmin_clipped, ymax_clipped, xmax_clipped], 1))

    if filter_nonoverlapping:
      areas = boxlist.area()
      nonzero_area_indices = tf.to_int32(
          tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]))
      boxlist = gather(boxlist, nonzero_area_indices)
    return boxlist


def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between BoxLists.

  Args:
    boxlist1: a BoxList holding `n` boxes.
    boxlist2: a BoxList holding `m` boxes.
    scope: string scalar, name scope.

  Returns:
    a float tensor of shape [n, m] storing pairwise intersections between 
      `boxlist1` and `boxlist2`.
  """
  with tf.name_scope(scope, 'Intersection'):
    ymin1, xmin1, ymax1, xmax1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    pairwise_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2))
    pairwise_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2))
    intersect_heights = tf.maximum(0.0, pairwise_min_ymax - pairwise_max_ymin)
    pairwise_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2))
    pairwise_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2))
    intersect_widths = tf.maximum(0.0, pairwise_min_xmax - pairwise_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between BoxLists.

  Note: IOU are symmetric -- `iou(bl1, bl2) == transpose(iou(bl2, bl1))`.
  If two boxes have zero intersection, their IOU is set to zero. 

  Args:
    boxlist1: a BoxList holding `n` boxes.
    boxlist2: a BoxList holding `m` boxes.
    scope: string scalar, name scope.

  Returns:
    a float tensor of shape [n, m] storing pairwise IOU between
      `boxlist1` and `boxlist2`.
  """
  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = boxlist1.area()
    areas2 = boxlist2.area()
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def ioa(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-area between BoxLists.

  Note: IOA are non-symmetric. If two boxes have zero intersection, their IOA is 
  set to zero.

  Args:
    boxlist1: a BoxList holding `n` boxes.
    boxlist2: a BoxList holding `m` boxes.
    scope: string scalar, name scope.

  Returns:
    a float tensor of shape [n, m] storing pairwise IOA between
      `boxlist1` and `boxlist2`.
  """
  with tf.name_scope(scope, 'IOA'):
    intersections = intersection(boxlist1, boxlist2)
    areas = boxlist2.area()
    return tf.truediv(intersections, areas)


def prune_completely_outside_window(boxlist, window, scope=None):
  """Removes any box in `boxlist` located **completely** outside of `window`.
  In-place operation.

  Args:
    boxlist: a BoxList holding `n_in` boxes.
    window: a float tensor of shape [4] holding [ymin, xmin, ymax, xmax] 
      coordinates of a window.
    scope: string scalar, name scope.

  Returns:
    boxlist: a BoxList holding `n_out <= n_in` boxes.
    indices: a tensor of shape [n_out] holding indices of boxes retained in the
      output.
  """
  with tf.name_scope(scope, 'PruneCompleteleyOutsideWindow'):
    ymin, xmin, ymax, xmax = tf.unstack(value=boxlist.get(), axis=1)
    win_ymin, win_xmin, win_ymax, win_xmax = tf.unstack(window)
    coordinate_violations = tf.stack([
        tf.greater_equal(ymin, win_ymax), tf.greater_equal(xmin, win_xmax),
        tf.less_equal(ymax, win_ymin), tf.less_equal(xmax, win_xmin)], axis=1)
    indices = tf.reshape(tf.where(
        tf.logical_not(tf.reduce_any(coordinate_violations, axis=1))), [-1])
    boxlist = gather(boxlist, indices)
    return boxlist, indices


def prune_outside_window(boxlist, window, scope=None):
  """Removes any box in `boxlist` located not completely inside of `window`.
  In-place operation.

  Args:
    boxlist: a BoxList holding `n_in` boxes.
    window: a float tensor of shape [4] holding [ymin, xmin, ymax, xmax]
      coordinates of a window.
    scope: string scalar, name scope.

  Returns:
    boxlist: a BoxList holding `n_out <= n_in` boxes.
    indices: a tensor of shape [n_out] holding indices of boxes retained in the 
      output.
  """
  with tf.name_scope(scope, 'PruneOutsideWindow'):
    ymin, xmin, ymax, xmax = tf.unstack(value=boxlist.get(), axis=1)
    win_ymin, win_xmin, win_ymax, win_xmax = tf.unstack(window)
    coordinate_violations = tf.stack([
        tf.less(ymin, win_ymin), tf.less(xmin, win_xmin),
        tf.greater(ymax, win_ymax), tf.greater(xmax, win_xmax)], axis=1)
    indices = tf.reshape(tf.where(
        tf.logical_not(tf.reduce_any(coordinate_violations, axis=1))), [-1])
    boxlist = gather(boxlist, indices)
    return boxlist, indices


def prune_non_overlapping_boxes(
    boxlist1, boxlist2, min_overlap=0.0, scope=None):
  """For each box `b_i` in `boxlist1`, and each box `b_j` in `boxlist2`, we
  compute `b_j`'s IOA w.r.t. `b_i`. If `IOA(b_j, b_i) < min_overlap` for all 
  `j`, then `b_i` is removed from `boxlist1` in the output. In-place operation.

  Args:
    boxlist1: a BoxList holding `n` boxes.
    boxlist2: a BoxList holding `m` boxes.
    min_overlap: float scalar, the threshold of IOA.
    scope: string scalar, name scope.

  Returns:
    boxlist1: a BoxList holding `n_out <= n` boxes.
    indices: a tensor of shape [n_out] holding indices of `boxlist1`'s
      boxes retained in the output.
  """
  with tf.name_scope(scope, 'PruneNonOverlappingBoxes'):
    ioa_ = ioa(boxlist2, boxlist1)  # shape: [m, n]
    ioa_ = tf.reduce_max(ioa_, axis=0)  # shape: [n]
    indices = tf.reshape(
        tf.where(tf.greater_equal(ioa_, tf.constant(min_overlap))), [-1])
    boxlist1 = gather(boxlist1, indices)
    return boxlist1, indices


def change_coordinate_frame(boxlist, window, scope=None):
  """Change box coordinates of BoxList to be relative to window's frame. 
  In-place operation.

  For example, given box coordinates [0.1, 0.3, 0.2, 0.5], which are normalized
  w.r.t. the coordinate frame ymin=0.0, xmin=0.0, ymax=1.0, xmax=1.0 (
  unit square).

  Now we change the box coordinates to the new coordinate frame (window)
  ymin=-0.5, xmin=-0.3 ymax=0.3, xmax=1.2, resulting in new box coordinates
  [0.75, 0.4, 0.875, 0.5333]. Note the new box coordinates may go beyond unit
  square (i.e. < 0 or > 1), so they need to be clipped to the unit square.

  Args:
    boxlist: a BoxList holding `n` boxes.
    window: a float tensor of shape [4] holding [ymin, xmin, ymax, xmax] window 
      coordinates.
    scope: string scalar, name scope.

  Returns:
    boxlist: a BoxList holding `n` boxes.
  """
  with tf.name_scope(scope, 'ChangeCoordinateFrame'):
    ymin, xmin, ymax, xmax = tf.unstack(window)
    win_height = ymax - ymin
    win_width = xmax - xmin
    shifted_boxes = boxlist.get() - tf.stack([ymin, xmin, ymin, xmin])
    boxlist.set(shifted_boxes)
    boxlist = scale(boxlist, 1.0 / win_height, 1.0 / win_width)
    return boxlist


def gather(boxlist, indices, scope=None):
  """Gather a subset of boxes in `boxlist` (along with their data fields), 
  whose indices are present in `indices`. In-place operation.

  Args:
    boxlist: a BoxList holding `n_in` boxes.
    indices: an int tensor of shape [n_out] (`n_out <= n_in`) holding the 
      indices of boxes in `boxlist` to be gathered.
    scope: string scalar, name scope.

  Returns:
    boxlist: a BoxList holding `n_out` boxes. 
  """
  with tf.name_scope(scope, 'Gather'):
    boxlist.set(tf.gather(boxlist.get(), indices))
    for field in boxlist.get_extra_fields():
      boxlist.set_field(field, tf.gather(boxlist.get_field(field), indices))
    return boxlist


def concatenate(boxlists, scope=None):
  """Concatenate a list of BoxLists. 

  Each BoxList in the list must have the same set of fields, and the tensor
  stored in each field must have the same rank, and the same fully defined 
  shape except for possibly the 0th dimension (i.e. num of boxes). This 
  function will create a brand new BoxList.

  Args:
    boxlists: a list of BoxLists, holding `n_1`, `n_2`, ..., `n_b` boxes.
    scope: string scalar, name scope.

  Returns:
    a BoxList holding `sum(n_1, n_2, ..., n_b)` boxes, along with the additional
      fields holding `b` tensors concatenated along the 0th dimension.
  """
  with tf.name_scope(scope, 'Concatenate'):
    concatenated = box_list.BoxList(
        tf.concat([boxlist.get() for boxlist in boxlists], 0))
    fields = boxlists[0].get_extra_fields()
    for field in fields:
      concatenated_field = tf.concat(
          [boxlist.get_field(field) for boxlist in boxlists], 0)
      concatenated.set_field(field, concatenated_field)
    return concatenated


def sort_by_field(boxlist, field, order=SortOrder.descend, scope=None):
  """Sort boxes and associated fields according to a scalar value.
  In-place operation.

  `boxlist` must have that field which stores a 1-D numeric tensor.
 
  Args:
    boxlist: a BoxList holding `n` boxes.
    field: string scalar, the field by which boxes are sorted. Must be 1-D.
    order: int scalar, indicating descend or ascend. Defaults to descend.
    scope: string scalar, name scope.

  Returns:
    sorted_boxlist: a BoxList holding `n` boxes where boxes are ordered by
      the values in the field `field` in the specified order.
  """
  with tf.name_scope(scope, 'SortByField'):
    if order != SortOrder.descend and order != SortOrder.ascend:
      raise ValueError('Invalid sort order.')

    field_to_sort = boxlist.get_field(field)
    num_boxes = boxlist.num_boxes()
    _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)
    if order == SortOrder.ascend:
      sorted_indices = tf.reverse_v2(sorted_indices, [0])
    return gather(boxlist, sorted_indices)


def filter_by_score(boxlist, thresh, scope=None):
  """Filter the BoxList so that only those with 
  `boxlist.get_field('score') > threshold` are retained. In-place operation.

  Args:
    boxlist: a BoxList holding `n_in` boxes. Must contain 'scores' field holding
      a float tensor of shape [n_in].
    thresh: float scalar or float scalar tensor, score threshold.
    scope: string scalar, name scope.

  Returns:
    a BoxList holding `n_out <= n_in` boxes.
  """
  with tf.name_scope(scope, 'FilterGreaterThan'):
    if not boxlist.has_field('scores'):
      raise ValueError('Input boxlist must have the \'scores\' field.')
    scores = boxlist.get_field('scores')
    high_score_indices = tf.to_int32(
        tf.reshape(tf.where(tf.greater(scores, thresh)), [-1]))
    return gather(boxlist, high_score_indices)


def to_absolute_coordinates(boxlist,
                            height,
                            width,
                            scope=None):
  """Converts normalized box coordinates to absolute pixel coordinates.
  In-place operation.

  Args:
    boxlist: a BoxList holding `n` box relative coordinates in range [0, 1].
    height: float scalar or float scalar tensor, absolute height of the box in
      num of pixels.
    width: float scalar or float scalar tensor, absolute width of the box in num 
      of pixels.
    scope: string scalar, name scope.

  Returns:
    a BoxList holding `n` box absolute coordinates.

  """
  with tf.name_scope(scope, 'ToAbsoluteCoordinates'):
    height = tf.to_float(height)
    width = tf.to_float(width)
    return scale(boxlist, height, width)


def pad_or_clip_box_list(boxlist, size, scope=None):
  """Pads or clips all fields of a BoxList. In-place operation.

  BoxList has `num_boxes` boxes -- `self.get()` has shape [num_boxes, 4].
  If `size > num_boxes`, then `self.get()` are zero-padded to have 0th dimension
  equal to `size`. Otherwise, the first `size` rows in `self.get()` are kept.

  Args:
    boxlist: a BoxList holding `n` boxes.
    size: int scalar, desired num of boxes. 
    scope: string scalar, name scope.

  Returns:
    a BoxList holding `size` boxes.
  """
  with tf.name_scope(scope, 'PadOrClipBoxList'):
    boxlist.set(shape_utils.pad_or_clip_tensor(boxlist.get(), size))
    for field in boxlist.get_extra_fields():
      boxlist.set_field(field, 
          shape_utils.pad_or_clip_tensor(boxlist.get_field(field), size))
    return boxlist


def to_normalized_coordinates(boxlist, height, width, scope=None):
  """Converts absolute box coordinates to normalized coordinates (in [0, 1]).
  In-place operation.

  Args:
    boxlist: a BoxList holding `n` box absolute coordinates.
    height: int scalar tensor, absolute height of the box in num of pixels.
    width: int scalar tensor, absolute width of the box in num of pixels.
    scope: string scalar, name scope.

  Returns:
    a BoxList holding `n` box normalized coordinates.
  """
  with tf.name_scope(scope, 'ToNormalizedCoordinates'):
    height = 1. / tf.to_float(height)
    width = 1. / tf.to_float(width)
    return scale(boxlist, height, width)

import tensorflow as tf

from detection.core import anchor_generator
from detection.core import box_list
from detection.core import box_list_ops
from detection.utils import ops


class GridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generates a grid of anchors at given scales and aspect ratios."""

  def __init__(self,
               scale_list=(0.5, 1.0, 2.0),
               aspect_ratio_list=(0.5, 1.0, 2.0),
               base_anchor_size=None,
               anchor_stride=None,
               anchor_offset=None):
    """Constructs a GridAnchorGenerator.

    Args:
      scales: a list of (float) scales, default=(0.5, 1.0, 2.0)
      aspect_ratios: a list of (float) aspect ratios, default=(0.5, 1.0, 2.0)
      base_anchor_size: base anchor size as height, width (
                        (length-2 float32 list or tensor, default=[256, 256])
      anchor_stride: difference in centers between base anchors for adjacent
                     grid positions (length-2 float32 list or tensor,
                     default=[16, 16])
      anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                     upper left element of the grid, this should be zero for
                     feature networks with only VALID padding and even receptive
                     field size, but may need additional calculation if other
                     padding is used (length-2 float32 list or tensor,
                     default=[0, 0])
    """
    super(GridAnchorGenerator, self).__init__(name_scope='GridAnchorGenerator')
    self._scale_list = scale_list
    self._aspect_ratio_list = aspect_ratio_list
    self._base_anchor_size = base_anchor_size or (256.0, 256.0)
    self._anchor_stride = anchor_stride or (16.0, 16.0)
    self._anchor_offset = anchor_offset or (0.0, 0.0)

    self._scales, self._aspect_ratios = zip(*[(i, j) 
        for j in aspect_ratio_list for i in scale_list])
    self._num_anchors_per_location = [len(scale_list) * len(aspect_ratio_list)]

  @property
  def num_anchors_per_location(self):
    """Returns a list of ints, holding the num of anchors per spatial location
    for each feature map.
    """
    return self._num_anchors_per_location 

  def _generate(self, feature_map_shape_list, height=None, width=None):
    """Generates a collection of bounding boxes to be used as anchors.

    Args:
      feature_map_shape_list: a list (length must be 1) of 2-tuples of ints,
        holding the height and width of the feature map that anchors are 
        generated for.

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes corresponding to
        the input feature map shapes.

    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == 1):
      raise ValueError('feature_map_shape_list must be a list of length 1.')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')
    if (height is None and width is not None) or(
        height is not None and width is None):
      raise ValueError('height and width must be provided or left as None '
          'altogether. Got %s and %s' % (height, width))
      
    grid_height, grid_width = feature_map_shape_list[0]

    anchors = tile_anchors(grid_height,
                           grid_width,
                           self._scales,
                           self._aspect_ratios,
                           tf.to_float(self._anchor_stride),
                           tf.to_float(self._anchor_offset),
                           tf.to_float(self._base_anchor_size))

    num_anchors = anchors.num_boxes()
    anchor_indices = tf.zeros([num_anchors])
    anchors.set_field('feature_map_index', anchor_indices)
    if height is not None and width is not None:
      anchors = box_list_ops.to_normalized_coordinates(anchors, height, width)
    return [anchors]


def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 anchor_stride,
                 anchor_offset,
                 base_anchor_size=(1.0, 1.0)):
  """Create a tiled set of anchors strided along a grid in image space.

  Args:
    grid_height: int scalar or int scalar tensor, height of the grid.
    grid_width: int scalar or int scalar tensor, width of the grid.
    scales: a list of floats, the scales of anchors.
    aspect_ratios: a list of floats, the aspect ratios. Has the same length as
      `scales`. 
    anchor_stride: a 2-tuple of float scalars or float scalar tensors, the 
      distance between neighboring grid centers in height and width dimension. 
    anchor_offset: a 2-tuple of float scalars or float scalar tensors, the 
      (height, width) coordinate of the upper left grid.
    base_anchor_size: a float tensor of shape [2], holding height and width of 
      the anchor. Defaults to unit square. 

  Returns:
    a BoxList instance holding `grid_height * grid_width * len(scales)` anchor 
      boxes.
  """
  base_anchor_size = tf.convert_to_tensor(base_anchor_size)
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]

  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]

#  x_centers, y_centers = ops.meshgrid(x_centers, y_centers)
#  widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
#  heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
#  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
#  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
#  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
#  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
#  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
#  return box_list.BoxList(bbox_corners)

#  x_centers, y_centers = tf.meshgrid(x_centers, y_centers) 
  y_centers, x_centers = tf.meshgrid(y_centers, x_centers, indexing='ij')
  y_centers = tf.reshape(y_centers, [-1, 1])
  x_centers = tf.reshape(x_centers, [-1, 1])
  heights = tf.reshape(heights, [1, -1])
  widths = tf.reshape(widths, [1, -1])
  coordinates = tf.reshape(tf.stack([y_centers - .5 * heights, 
                                     x_centers - .5 * widths, 
                                     y_centers + .5 * heights, 
                                     x_centers + .5 * widths], axis=2), [-1, 4])
  return box_list.BoxList(coordinates)


def _center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts box coordinates in [y_center, x_center], [height, width] format 
  to the default [ymin, xmin, ymax, xmax] format. 

  Args:
    centers: a tensor of shape [num_boxes, 2], box centers coordinates.
    sizes: a tensor of shape [num_boxes, 2], box heights and widths.

  Returns:
    corners: a tensor of shape [num_boxes, 4], box coordinates in the format
      [ymin, xmin, ymax, xmax]
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)

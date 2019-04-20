import numpy as np

import tensorflow as tf

from detection.core.anchor_generators import grid_anchor_generator
from detection.core import anchor_generator
from detection.core import box_list_ops


class MultipleGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for a list of feature maps in a convolutional
  network.
  """

  def __init__(self, box_specs):
    """Constructor.

    MultipleGridAnchorGenerator generates anchors for a list of feature maps
    in a convolutional network.

    Suppose your feature maps have spatial (i.e. height and width) sizes:

    feature_map_shape_list = [(10, 10), (5, 5), (1, 1)]
    
    and the desired (scale, aspect_ratio) configs for each feature map are:

    box_specs = [
      [(0.35, 1.0), (0.35, 2.0), (0.35, 0.5), (0.35, 3.0), (0.35, 0.3)],
      [(0.5, 1.0), (0.5, 2.0), (0.5, 0.5)],
      [(0.95, 1.0), (0.95, 2.0), (0.95, 0.5)]]

    Then the num_anchors_per_location property is set to [5, 3, 3], and 
    the `_generate` method outputs a list of 3 BoxList instances wrapping 
    3 float tensors of shape [10*10*5, 4], [5*5*3, 4], [1*1*3, 4], holding the 
    box coordinates of the generated anchor boxes. The anchor boxes in each 
    BoxList instance are laid out in the order of height, width, and 
    num_box_types, The total number of anchors generated over all feature
    maps is 500 + 75 + 3 = 578.

    box_specs is passed in to the constructor, while feature_map_shape_list
    is not needed until `_generate` is being called.

    Args:
      box_specs: a list of list of 2-tuples, holding the configuration of 
        (scale, aspect ratio) for each type of anchor in each feature map.
    """
    super(MultipleGridAnchorGenerator, self).__init__(
        name_scope='MultipleGridAnchorGenerator')
    self._box_specs = box_specs

    self._num_anchors_per_location = [
        len(box_specs) for box_specs in self._box_specs]
    self._scales = []
    self._aspect_ratios = []
    for box_spec in self._box_specs:
      scales, aspect_ratios = zip(*box_spec)
      self._scales.append(scales)
      self._aspect_ratios.append(aspect_ratios)

  @property
  def num_anchors_per_location(self):
    """Returns a list of ints, holding the num of anchors per spatial location
    for each feature map.
    """
    return self._num_anchors_per_location

  def _generate(self, feature_map_shape_list):
    """Generates a list of BoxLists as anchors.

    The anchor box coordinated generated are in normalized format: 
    ymin, xmin, ymax, xmax varying between 0 and 1.

    Args:
      feature_map_shape_list: a list of 2-tuples of ints or scalar int tensors,
        holding the height and width sizes of the feature maps that anchors are
        generated for. 

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes for each feature 
        map. 
    """
    anchor_strides = [(1. / tf.to_float(h), 1. / tf.to_float(w))
                      for h, w in feature_map_shape_list]
    anchor_offsets = [(s_h / 2., s_w / 2.) for s_h, s_w in anchor_strides]

    anchor_grid_list = []
    for feature_map_index, ((height, width), scales, aspect_ratios, stride, 
        offset) in enumerate(zip(feature_map_shape_list, self._scales,
                                 self._aspect_ratios, anchor_strides,
                                 anchor_offsets)):
      tiled_anchors = grid_anchor_generator.tile_anchors(
          grid_height=height,
          grid_width=width,
          scales=scales,
          aspect_ratios=aspect_ratios,
          anchor_stride=stride,
          anchor_offset=offset)

      num_anchors_in_layer = tiled_anchors.num_boxes() 

      anchor_indices = feature_map_index * tf.ones([num_anchors_in_layer])
      tiled_anchors.set_field('feature_map_index', anchor_indices)
      anchor_grid_list.append(tiled_anchors)

    return anchor_grid_list


def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95,
                       lowest_scale=0.1,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       reduce_boxes_in_lowest_layer=True):
  """Creates MultipleGridAnchorGenerator to generate SSD anchors.

  This function creates an instance of MultipleGridAnchorGenerator that 
  generates the "default box" (anchor boxes) described in Section 2.2 - 
  Data augmentation of https://arxiv.org/abs/1512.02325

  The boxes will be generated from the feature map with finest resolution all
  the way up to the one with coarsest resolution (e.g. from (10, 10) to (1, 1)), 
  and the scales increase with the coarseness of the feature map.

  Args:
    num_layers: int scalar, the num of feature maps to generate anchors for. 
    min_scale: float scalar, scale of anchors with finest resolution.
    max_scale: float scalar, scale of anchors with coarsest resolution.
    lowest_scale: float scalar, scale of the anchors for the lowest layer (
      feature map). Igored if `reduce_boxes_in_lowest_layer` is False.
    scales: a list of floats, with length `num_layers`, holding an increasing
      float numbers as scales for anchors in each feature map. If None, a 
      default list [min_scale, .., max_scale] holding evenly spaced scales is 
      created. In either case (`scales` is None or not), a largest scale, 1.0,
      will be appended, i.e. the total num of scales is `num_layers + 1`.
    aspect_ratios: a tuple of floats, the aspect ratios of the anchors at each
      grid location.
    interpolated_scale_aspect_ratio: float scalar, if positive, an additional
      (scale, interpolated_scale_aspect_ratio) config is added for each feature
      map, where `scale` is the sqrt of the scale used in the current feature 
      map and that of the next feature map (1.0 for the last one). If not 
      positive, no additional config is added.
    reduce_boxes_in_lowest_layer: bool scalar, whether a list of three fixed
      (scale, aspect_ratio) config for the first feature map should be used.

  Returns:
    a MultipleGridAnchorGenerator instance
  """
  if scales is None or not scales:
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)]
  else:
    if len(scales) != num_layers:
      raise ValueError('`scales` must be a list of `num_layers` ({}) numbers, '
          'got {}'.format(num_layers, scales))
  scales += [1.0]

  box_specs_list = []
  for layer, (scale, scale_next) in enumerate(zip(scales[:-1], scales[1:])):
    if layer == 0 and reduce_boxes_in_lowest_layer:
      layer_box_specs = [(lowest_scale, 1.0), (scale, 2.0), (scale, 0.5)]
    else:
      layer_box_specs = []
      for aspect_ratio in aspect_ratios:
        layer_box_specs.append((scale, aspect_ratio))
      if interpolated_scale_aspect_ratio > 0.0:
        layer_box_specs.append((np.sqrt(scale*scale_next),
                                interpolated_scale_aspect_ratio))
    box_specs_list.append(layer_box_specs)

  return MultipleGridAnchorGenerator(box_specs_list)

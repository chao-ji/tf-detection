import collections

import tensorflow as tf

from detection.utils import ops

slim = tf.contrib.slim


def ssd_feature_maps(feature_map_tensor_dict,
                     feature_map_specs_dict,
                     depth_multiplier=1,
                     use_depthwise=False,
                     insert_1x1_conv=True):
  """Generates multi-resolution feature maps.

  The output contains existing feature maps in a base classification network
  followed by those freshly generated. For example, when inception_v2 is the
  feature extractor, the input image height and width are 300, and batch size 
  is 1, then the generated feature maps have the following shapes:
  [
    (1, 19, 19, 576),   # Mixed_4c
    (1, 10, 10, 1024),  # Mixed_5c
    (1, 5, 5, 512),     # K=3, S=2, P='SAME'
    (1, 3, 3, 256),     # K=3, S=2, P='SAME'
    (1, 2, 2, 256),     # K=3, S=2, P='SAME'
    (1, 1, 1, 128)      # K=3, S=2, P='SAME'
  ]

  Args:
    feature_map_tensor_dict: dict mapping from feature map names to 
      feature map tensors, returned from base classification network. Example:
      {
        ... # previous feature maps
        'Mixed_4c': tensor of shape=[batch_size, height, width, channels],
        'Mixed_5c': tensor of shape=[batch_size, height, width, channels],
      }
    feature_map_specs_dict: dict mapping from specs names to a 
      list (length is num of output feature maps) of values. Note 'layer_name' 
      and 'layer_depth' are required. Example:
      {
        'layer_name': ['Mixed_4c', 'Mixed_5c', None, None, None, None],
        'layer_depth': [None, None, 512, 256, 256, 128], 
        ... # more spec names
      }
    depth_multiplier: int scalar, num of depthwise convolution output channels
      for each input channel for separable_conv2d. Defaults to 1. 
    use_depthwise: bool scalar, whether to use separable_conv2d instead of
        conv2d.
    insert_1x1_conv: bool scalar, whether each newly generated feature
       map should be preceded by a 1x1 convolution with half of its depth.

  Returns:
    An OrderedDict mapping from feature map names to feature map tensors of 
      shape [batch_size, height, width, channels].
  """
  feature_map_names = []
  feature_map_tensors = []
  base_layer_name = None

  for index, (layer_name, layer_depth) in enumerate(
      zip(feature_map_specs_dict['layer_name'],
          feature_map_specs_dict['layer_depth'])):

    kernel_size = 3
    if 'kernel_size' in feature_map_specs_dict:
      kernel_size = feature_map_specs_dict['kernel_size'][index]
    
    stride = 2
    if 'stride' in feature_map_specs_dict:
      stride = feature_map_specs_dict['stride'][index]

    padding = 'SAME'
    if 'padding' in feature_map_specs_dict:
      padding = feature_map_specs_dict['padding'][index]

    # use existing feature maps
    if layer_name is not None:
      tensor = feature_map_tensor_dict[layer_name]
      base_layer_name = layer_name
    # generates new feature maps
    else:
      prev_layer = feature_map_tensors[-1]

      intermediate_layer = prev_layer
      if insert_1x1_conv:
        layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
            base_layer_name, index, layer_depth // 2)
        intermediate_layer = slim.conv2d(
            prev_layer,
            layer_depth // 2,
            kernel_size=1,
            padding='SAME',
            stride=1,
            scope=layer_name)
      
      layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(
          base_layer_name, index, kernel_size, kernel_size, layer_depth)

      if use_depthwise:
        tensor = ops.split_separable_conv2d(
            intermediate_layer,
            layer_depth,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            padding=padding,
            stride=stride,
            scope=layer_name)
      else:
        tensor = slim.conv2d(
            intermediate_layer,
            layer_depth,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            scope=layer_name)
    feature_map_names.append(layer_name)
    feature_map_tensors.append(tensor)

  return collections.OrderedDict(zip(feature_map_names, feature_map_tensors))

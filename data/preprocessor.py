""""""
import functools

import tensorflow as tf

from detection.core import standard_names as names
from detection.core import box_list
from detection.core import box_list_ops

IMAGENET_MEAN = 124, 117, 104


def _flip_image_left_right(image):
  """Horizontally flips an image.

  Args:
    image: a float tensor of shape [height, width, channels].

  Returns:
    flipped image of the same shape as input.
  """
  return tf.image.flip_left_right(image)


def _flip_boxes_left_right(boxes):
  """Horizontally flips groundtruth boxes.

  Args:
    boxes: a float tensor of shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].

  Returns:
    flipped groundtruth boxes of the same shape as input. 
  """
  ymin, xmin, ymax, xmax = tf.unstack(value=boxes, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def _flip_masks_left_right(masks):
  """Horizontally flips groundtruth masks.

  Args:
    mask: a tensor of shape [num_boxes, height, width], holding binary masks of
      `num_boxes` instances.

  Returns:
    flipped groundtruth masks of the same shape as input.
  """
  return masks[:, :, ::-1]


def random_horizontal_flip(image, boxes, masks=None, seed=None):
  """Horizontally flips an image together with the associated groundtruth 
  boxes (and optionally masks) with 1/2 probability.

  Args:
    image: a float tensor of shape [height, width, channels].
    boxes: a float tensor of shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    masks: (Optional) a tensor of shape [num_boxes, height, width], holding 
      binary masks of `num_boxes` instances.
    seed: int scalar, random seed.

  Returns:
    image: flipped or unchanged image of the same shape as input.
    boxes: flipped or unchanged boxes of the same shape as input.
    masks: (Optional) flipped or unchanged masks of the same shape as input.
  """
  with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
    selector = tf.random_uniform([], seed=seed)
    do_a_flip_random = tf.greater(selector, 0.5)

    funcs = [_flip_image_left_right, _flip_boxes_left_right]
    tensors = [image, boxes]
    if masks is not None:
      funcs.append(_flip_masks_left_right)
      tensors.append(masks)

    flipped = tf.cond(do_a_flip_random,
        lambda: tuple(f(t) for f, t in zip(funcs, tensors)),
        lambda: tuple(tensors))

    return flipped


def _strict_pad_image(
    image, boxes, height, width, masks=None, value=IMAGENET_MEAN):
  """Always pad image to the desired height and width uniformly with the given 
  pixel value.
  
  First draw a canvas of size [height, width] filled with pixel value `value`,
  and then place the input image in the center, and update the boxes coordinates
  (and optionally masks) to the new frame.

  NOTE: no padding will be performed in the height and/or width dimension if the
  desired size is less than that of the image.

  Args:
    image: float tensor of shape [height_in, width_in, channels].
    boxes: float tensor of shape [num_boxes, 4], where each row
      contains normalized (i.e. values varying in [0, 1]) box coordinates:
      [ymin, xmin, ymax, xmax].
    height: float scalar, the desired height of padded image.
    width: float scalar, the desired width of padded image.
    masks: (Optional) a tensor of shape [num_boxes, height, width], holding 
      binary masks of `num_boxes` instances.
    value: float tensor of shape [3], RGB value to fill the padded region with.

  Returns:
    new_image: float tensor of shape [height, width, channels].
    new_boxes: float tensor of shape [num_boxes, 4].
    new_masks: (Optional) float tensor of shape [num_boxes, height, width].
  """
  value = tf.to_float(value)
  img_height, img_width, _ = tf.unstack(tf.shape(image))
  img_height, img_width = tf.to_float(img_height), tf.to_float(img_width)

  # no padding in height and/or width dimension if desired height and/or width 
  # is less than that of the image
  height = tf.maximum(height, img_height)
  width = tf.maximum(width, img_width)

  pad_up = (height - img_height) // 2
  pad_down = height - img_height - pad_up
  pad_left = (width - img_width) // 2
  pad_right = width - img_width - pad_left

  # pad image
  image -= value
  new_image = tf.pad(image, [[pad_up, pad_down], [pad_left, pad_right], [0, 0]]) 
  new_image += value

  # pad boxes
  window = -pad_up, -pad_left, img_height + pad_down, img_width + pad_right
  normalizer = img_height, img_width, img_height, img_width
  window = tf.to_float(window) / tf.to_float(normalizer)
  new_boxes = box_list_ops.change_coordinate_frame(
      box_list.BoxList(boxes), window).get() 

  # pad masks
  if masks is not None:
    new_masks = tf.pad(masks, [[0, 0], [pad_up, pad_down], [pad_left, pad_right]]) 
    return new_image, new_boxes, new_masks

  return new_image, new_boxes


def _strict_random_crop_image(image,
                              boxes,
                              labels,
                              masks=None,
                              min_object_covered=1.0,
                              aspect_ratio_range=(0.75, 1.33),
                              area_range=(0.1, 1.0),
                              overlap_thresh=0.3):
  """Always performs a random crop.

  A random window is cropped out of `image`, and the groundtruth boxes (and 
  optionally the masks) associated with the original image will be either 
  removed, clipped or retained as is, depending on their relative location 
  w.r.t. to the crop window.

  Note: you may end up getting a cropped image without any groundtruth boxes. If
  that is the case, the output boxes and labels would simply be empty tensors 
  (i.e. 0th dimension has size 0).
 
  Args:
    image: a float tensor of shape [height, width, channels].
    boxes: a float tensor of shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    labels: int tensor of shape [num_boxes] holding object classes in `boxes`.
    masks: (Optional) a tensor of shape [num_boxes, height, width], holding 
      binary masks of `num_boxes` instances.
    min_object_covered: float scalar, the cropped window must cover at least 
      `min_object_covered` (ratio) of the area of at least one box in `boxes`.
    aspect_ratio_range: a float 2-tuple, lower and upper bound of the aspect 
      ratio of cropped window.
    area_range: a float 2-tuple, lower and upper bound of the ratio between area
      of cropped window and area of the original image.
    overlap_thresh: float scalar, a groundtruth box in `boxes` is retained only 
      if the cropped window's IOA w.r.t. it >= this threshold.

  Returns:
    new_image: float tensor of shape [new_height, new_width, channels] holding 
      the window cropped out of input `image`.
    new_boxes: float tensor of shape [new_num_boxes, 4] holding new groundtruth 
      boxes, with their [ymin, xmin, ymax, xmax] coordinates normalized and 
      clipped to the crop window.
    new_labels: int tensor of shape [new_num_boxes] holding object classes in
      `new_boxes`.
    new_masks: (Optional) float tensor of shape [new_num_boxes, height, width],
      holding new instance masks corresponding to `new_boxes`.
  """
  with tf.name_scope('RandomCropImage', values=[image, boxes, labels]):
    # crop_box.shape: [1, 1, 4]
    crop_begin, crop_size, crop_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.expand_dims(boxes, 0), # 0 change to 1
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    window = tf.squeeze(crop_box)

    # BoxList shape: [N, 4]
    boxlist = box_list.BoxList(boxes)
    boxlist.set_field('labels', labels)
    # BoxList shape: [1, 4]
    crop_boxlist = box_list.BoxList(tf.squeeze(crop_box, [0]))

    # remove boxes that are completely outside of `window`
    boxlist, indices = box_list_ops.prune_completely_outside_window(
        boxlist, window)
    # remove boxes whose fraction of area that is overlapped with 
    # `crop_boxlist` is less than `overlap_thresh`
    boxlist, in_window_indices = box_list_ops.prune_non_overlapping_boxes(
        boxlist, crop_boxlist, overlap_thresh)
    # change the coordinate of the remaining boxes
    new_boxlist = box_list_ops.change_coordinate_frame(boxlist, window)

    new_image = tf.slice(image, crop_begin, crop_size)
    new_image.set_shape([None, None, image.get_shape()[2]])    
    # clipping is necessary as some of new_boxes may extend beyond crop window
    new_boxes = tf.clip_by_value(new_boxlist.get(),
        clip_value_min=0.0, clip_value_max=1.0)
    new_labels = new_boxlist.get_field('labels')

    if masks is not None:
      in_window_masks = tf.gather(tf.gather(masks, indices), in_window_indices)
      new_masks = tf.splice(in_window_masks, 
                            [0, crop_begin[0], crop_begin[1]], 
                            [-1, crop_size[0], crop_size[1]])
      return new_image, new_boxes, new_labels, new_masks

    return new_image, new_boxes, new_labels


def random_pad_image(
    image, boxes, masks=None, pad_ratio=1.0, keep_prob=1.0, seed=None):
  """Randomly pad an input image.

  This function either pads the image so that `height` and `width` are scaled up
  by a factor of `pad_ratio` with probability `1 - keep_prob`, or return the
  input `image`, `boxes` (and optionally masks) unchanged with probability 
  `keep_prob`.

  Args:
    image: a float tensor of shape [height, width, channels].
    boxes: a float tensor of shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    masks: (Optional) a tensor of shape [num_boxes, height, width], holding 
      binary masks of `num_boxes` instances. 
    pad_ratio: float scalar >= 1.0, the factor by which the height and width to 
      be scaled up.
    keep_prob: float scalar, the probability to return the original
      input. If 0, always performs a padding.
    seed: int scalar, random seed.

  Return:
    image: new or original image with same rank as input.
    boxes: new or original boxes with same rank as input.
    masks: (Optional) new or original masks with same rank as input.
  """
  selector = tf.random_uniform([], seed=seed)
  do_random_pad = tf.greater(selector, keep_prob)

  height, width, _ = tf.unstack(tf.shape(image))
  height, width = tf.to_float(height), tf.to_float(width)

  tensors = [image, boxes]
  args = [image, boxes, height * pad_ratio, width * pad_ratio]
  if masks is not None:
    tensors.append(masks)
    args.append(masks)

  result = tf.cond(do_random_pad, 
                   lambda: _strict_pad_image(*args), 
                   lambda: tensors)

  return result


def random_crop_image(image,
                      boxes,
                      labels,
                      masks=None,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      keep_prob=0.0,
                      seed=None):
  """Randomly crops an image and associated groundtruth boxes.
  
  This function either performs a crop by `_strict_random_crop_image` with 
  probability `1 - keep_prob` or return the input `image`, `boxes` (and 
  optionally masks), and `labels` unchanged with probability `keep_prob`.  

  Note: you may end up getting a cropped image without any groundtruth boxes. If
  that is the case, the output boxes and labels would simply be empty tensors 
  (i.e. 0th dimension has size 0).

  Args:
    image: a float tensor of shape [height, width, channels].
    boxes: a float tensor of shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    labels: int tensor of shape [num_boxes] holding object classes in `boxes`. 
    masks: (Optional) a tensor of shape [num_boxes, height, width], holding 
      binary masks of `num_boxes` instances.
    min_object_covered: float scalar, the cropped window must cover at least 
      `min_object_covered` (ratio) of the area of at least one box in `boxes`.
    aspect_ratio_range: a float 2-tuple, lower and upper bound of the aspect 
      ratio of cropped window.
    area_range: a float 2-tuple, lower and upper bound of the ratio between area
      of cropped window and area of the original image.
    overlap_thresh: float scalar, a groundtruth box in `boxes` is retained only 
      if the cropped window's IOA w.r.t. it >= this threshold.
    keep_prob: float scalar, the probability to return the original
      input. If 0, always performs a crop.
    seed: int scalar, random seed.

  Returns:
    image: new or original image with same rank as input.
    boxes: new or original boxes with same rank as input.
    labels: labels associated with output `boxes` with same rank as input.
    masks: (Optional) new or original boxes with same rank as input. 
  """
  selector = tf.random_uniform([], seed=seed)
  do_a_crop_random = tf.greater(selector, keep_prob)

  tensors = [image, boxes, labels]
  if masks is not None:
    tensors.append(masks)

  result = tf.cond(do_a_crop_random,
      lambda: _strict_random_crop_image(image,
                                        boxes,
                                        labels,
                                        masks,
                                        min_object_covered,
                                        aspect_ratio_range,
                                        area_range,
                                        overlap_thresh),
      lambda: tuple(tensors))
  return result


def ssd_random_crop(image,
                    boxes,
                    labels,
                    masks=None,
                    min_object_covered=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    aspect_ratio_range=((0.5, 2.0),) * 7,
                    area_range=((0.1, 1.0),) * 7,
                    overlap_thresh=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    keep_prob=(0.15,) * 7,
                    seed=None):
  """Randomly crops an image and associated groundtruth boxes as described in 
  Section 2.2 - Data augmentation of https://arxiv.org/abs/1512.02325

  Note: you may end up getting a cropped image without any groundtruth boxes. If
  that is the case, the output boxes and labels would simply be empty tensors 
  (i.e. 0th dimension has size 0).

  Args:
    image: a float tensor of shape [height, width, channels].
    boxes: a float tensor of shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    labels: int tensor of shape [num_boxes] holding object classes in `boxes`. 
    masks: (Optional) a tensor of shape [num_boxes, height, width], holding 
      binary masks of `num_boxes` instances.
    min_object_covered: tuple of float scalars, the cropped window must cover at 
      least `min_object_covered` (ratio) of the area of at least one box in 
      `boxes`.
    aspect_ratio_range: tuple of float 2-tuples, lower and upper bound of the 
      aspect ratio of cropped window.
    area_range: tuple of float 2-tuples, lower and upper bound of the ratio 
      between area of cropped window and area of the original image.
    overlap_thresh: tuple of float scalars, a groundtruth box in `boxes` is 
      retained only if the cropped window's IOA w.r.t. it >= this threshold.
    keep_prob: tuple of float scalars, the probability to return the
      original input. If 0, always performs a crop.
    seed: int scalar, random seed.

  Returns:
    image: new or original image with same rank as input.
    boxes: new or original boxes with same rank as input.
    labels: labels associated with output `boxes` with same rank as input.
    masks: (Optional) new or original masks with same rank as input.
  """
  num_cases = len(min_object_covered)
  selector = tf.random_uniform([], maxval=num_cases, dtype=tf.int32, 
      seed=seed, name='ssd_random_crop_selector')

  predicates = [tf.equal(selector, case) for case in range(num_cases)]
  random_crop_image_fn_list = [
      functools.partial(random_crop_image,
                        image=image,
                        boxes=boxes,
                        labels=labels,
                        masks=masks,
                        min_object_covered=min_object_covered[case],
                        aspect_ratio_range=aspect_ratio_range[case],
                        area_range=area_range[case],
                        overlap_thresh=overlap_thresh[case],
                        keep_prob=keep_prob[case],
                        seed=seed)
      for case in range(num_cases)]

  result = tf.case([(predicate, fn) for predicate, fn 
      in zip(predicates, random_crop_image_fn_list)])
  return result


def resize_image(image,
                 masks=None,
                 new_height=512,
                 new_width=512,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  """Resizes images (and optionally masks) to the given height and width.

  Args:
    image: a float tensor of shape [height, width, channels].
    masks: a float tensor of shape [num_boxes, height, width].
    new_height: int scalar, desired height of the image.
    new_width: int scalar, desired width of the image.
    method: interpolation method used in resizing. Defaults to BILINEAR.
    align_corners: bool scalar, if true, exactly align all 4 corners of the 
        input and output. Defaults to False.

  Returns:
    new_image: float tensor of shape [new_height, new_width, channels] holding
      resized image.
    new_masks: (Optional) float tensor of shape 
      [num_boxes, new_height, new_width] holding resized masks. 
  """
  with tf.name_scope(
      'ResizeImage',
      values=[image, new_height, new_width, method, align_corners]):
    new_image = tf.image.resize_images(
        image, [new_height, new_width], 
        method=method,
        align_corners=align_corners)

    result = [new_image]

    if masks is not None:
      num_boxes = tf.shape(masks)[0]
     
      # resize masks with > 0 instances
      def resize_non_empty():
        new_masks = tf.expand_dims(masks, 3)
        new_masks = tf.image.resize_nearest_neighbor(
            new_masks, [new_height, new_width], align_corners=align_corners)
        return tf.squeeze(new_masks, axis=3)

      # resize masks with 0 instances
      def resize_empty():
        return tf.reshape(masks, [-1, new_height, new_width])

      new_masks = tf.cond(tf.greater(num_boxes, 0), resize_non_empty, resize_empty)
      result.append(new_masks)

    return result


def rescale_image(image,
                  masks=None,
                  min_dimension=600,
                  max_dimension=1024,
                  method=tf.image.ResizeMethod.BILINEAR,
                  align_corners=False):
  """Resize image by rescaling (aspect ratio unchanged).
  
  The image is first rescaled such that the size of smaller dimension is equal
  to `min_dimension`. If the larger dimension of the rescaled image is less than
  or equal to `max_dimension`, output the rescaled image; otherwise, rescale the
  image the second time such that the larger dimension is equal to 
  `max_dimension`.

  Args:
    image: a float tensor of shape [height, width, channels].
    masks: a tensor of shape [num_boxes, height, width], holding binary masks of
      `num_boxes` instances. 
    min_dimension: int scalar, the desired size of the smaller dimension. 
    max_dimension: int scalar, the upper bound of the size of the larger 
      dimension.
    method: interpolation method used in resizing. Defaults to BILINEAR.
    align_corners: bool scalar, if true, exactly align all 4 corners of the 
        input and output. Defaults to False.

  Returns:
    new_image: float tensor of shape [new_height, new_width, channels] holding
      resized image.
    new_masks: (Optional) float tensor of shape [num_boxes, new_height, 
      new_width] holding resized masks.
  """  
  with tf.name_scope('ResizeToRange', 
      values=[image, min_dimension, max_dimension]):
    height, width = tf.unstack(tf.to_float(tf.shape(image))[:2])
    min_size = tf.minimum(height, width)
    large_scale_factor = tf.constant(min_dimension, dtype=tf.float32) / min_size
    large_height = tf.to_int32(tf.round(height * large_scale_factor))
    large_width = tf.to_int32(tf.round(width * large_scale_factor))
    large_size = tf.stack([large_height, large_width])

    max_size = tf.maximum(height, width)
    small_scale_factor = tf.constant(max_dimension, dtype=tf.float32) / max_size
    small_height = tf.to_int32(tf.round(height * small_scale_factor))
    small_width = tf.to_int32(tf.round(width * small_scale_factor))
    small_size = tf.stack([small_height, small_width])

    new_size = tf.cond(
        tf.greater(tf.to_float(tf.reduce_max(large_size)), max_dimension), 
        lambda: small_size, lambda: large_size)
    new_image = tf.image.resize_images(
        image, new_size, method=method, align_corners=align_corners)

    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks, 
          new_size, 
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      new_masks = tf.squeeze(new_masks, 3)
      return new_image, new_masks
    else:
      return new_image


def _get_fn_args_map():
  """Returns the mapping from a preprocessing function to a tuple holding
  its positional, tensor-valued argument names.
  """
  fn_args_map = {
      random_horizontal_flip: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes,
          names.TensorDictFields.groundtruth_masks),
      random_crop_image: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes,
          names.TensorDictFields.groundtruth_labels,
          names.TensorDictFields.groundtruth_masks),
      random_pad_image: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes,
          names.TensorDictFields.groundtruth_masks),
      ssd_random_crop: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes,
          names.TensorDictFields.groundtruth_labels,
          names.TensorDictFields.groundtruth_masks),
  }
  return fn_args_map


class DataPreprocessor(object):
  """Data preprocessor for doing data augmentation.

  A data preprocessor is built by chaining a sequence of preprocess functions
  in a pipeline, where each function is specified by a 2-tuple containing a 
  callable and a dict holding the kwargs dict for the callable.

  Example: Suppose we want a preprocessor to run the following functions in 
  sequence on an input tensor dict holding tensors `image`, `boxes`, `labels`.

  `fn1(image, boxes, labels, **kwargs1)`, outputs `image`, `boxes`, `labels`,
  `fn2(image, boxes, labels, **kwargs2)`, outputs `image`, `boxes`, `labels`,
  `fn3(image, **kwargs3)`, outputs `image`.

  The preprocessor should be initialized by a list
  `[(fn1, kwargs1), (fn2, kwargs2), (fn3, kwargs3)]`. When it calls its method 
  `preprocess` on the input tensor dict, the output of the current function is 
  fed to the next function as input, and the final output of `preprocess` 
  tensor dict holds the tensors `image`, `boxes`, `labels`.  
  """
  def __init__(self, options_list):
    """Constructor.

    Args:
      options_list: a list of 2-tuples containing a callable and a dict holding
        the kwargs dict (i.e. keyword-to-argument mapping) for the callable.
    """
    self._options_list = options_list
    
  def preprocess(self, tensor_dict):
    """Applies preprocessing functions sequentially on the input tensor dict.

    Args:
      tensor_dict: a dict mapping from tensor names to tensors with the   
        following fields required:
        image: float tensor of shape [height, width, channels],
        groundtruth_boxes: float tensor of shape [num_boxes, 4],        
        groundtruth_labels: int tensor of shape [num_boxes].

    Returns: 
      a dict holding the same set of tensors as input `tensor_dict`.
    """
    fn_args_map = _get_fn_args_map()

    for fn, kwargs in self._options_list:
      arg_names = fn_args_map[fn]

      args = [tensor_dict[arg_name] if arg_name in tensor_dict else None 
          for arg_name in arg_names]

      results = fn(*args, **kwargs)
      for arg_name, result in zip(arg_names, results):
        tensor_dict[arg_name] = result

    return tensor_dict

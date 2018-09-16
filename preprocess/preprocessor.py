""""""
import functools

import tensorflow as tf

from detection.core import standard_names as names
from detection.core import box_list
from detection.core import box_list_ops


def _flip_image_left_right(image):
  """Horizontally flips an image.

  Args:
    image: a rank-3 float tensor with shape [height, width, channels].

  Returns:
    flipped image with the same shape as input.
  """
  return tf.image.flip_left_right(image)


def _flip_boxes_left_right(boxes):
  """Horizontally flips groundtruth boxes.

  Args:
    boxes: a rank-2 float tensor with shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].

  Returns:
    flipped groundtruth boxes with the same shape as input.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def random_horizontal_flip(image, boxes, seed=None):
  """Horizontally flips an image together with the associated groundtruth
  boxes with 1/2 probability.

  Args:
    image: a rank-3 float tensor with shape [height, width, channels].
    boxes: a rank-2 float tensor with shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    seed: int scalar, random seed.

  Returns:
    image: flipped or unchanged image with the same shape as input.
    boxes: flipped or unchanged boxes with the same shape as input.
  """
  with tf.name_scope('RandomHorizontalFlip', values = [image, boxes]):
    selector = tf.random_uniform([], seed=seed)
    do_a_flip_random = tf.greater(selector, 0.5)

    image, boxes = tf.cond(do_a_flip_random, 
        lambda: (_flip_image_left_right(image), _flip_boxes_left_right(boxes)),
        lambda: (image, boxes))

    return image, boxes


def _strict_random_crop_image(image,
                              boxes,
                              labels,
                              min_object_covered=1.0,
                              aspect_ratio_range=(0.75, 1.33),
                              area_range=(0.1, 1.0),
                              overlap_thresh=0.3):
  """Always performs a random crop.

  A random crop is performed on `image`, and the groundtruth boxes associated
  with the original image will be either removed, clipped or kept unchanged
  depending on their relative location w.r.t to the crop window.
  
  Args:
    image: a rank-3 float tensor with shape [height, width, channels].
    boxes: a rank-2 float tensor with shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    labels: a rank-1 int tensor containing the object classes.
    min_object_covered: float scalar, the cropped image must cover at least 
      this fraction of at least one of the input bounding boxes.
    aspect_ratio_range: a float 2-tuple, allowed range for aspect ratio of
      cropped image.
    area_range: a float 2-tuple, allowed range for area ratio between cropped
      image and the original image.
    overlap_thresh: float scalar, a groundtruth box in `boxes` is kept only if
      its IoA (w.r.t the cropped window) >= this threshold.

  Returns:
    new_image: cropped image with same rank as `image`.
    new_boxes: new groundtruth boxes with same rank as `boxes`. NOTE: the new 
      groundtruth boxes are NORMALIZED and CLIPPED w.r.t. to crop window (i.e.
      values varying in [0, 1])
    new_labels: labels of `new_boxes` with same rank as `labels`.
  """
  with tf.name_scope('RandomCropImage', values=[image, boxes]):
    image_shape = tf.shape(image)

    boxes_expanded = tf.expand_dims(
        tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0), 1)

    crop_begin, crop_size, crop_box = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=boxes_expanded,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)

    new_image = tf.slice(image, crop_begin, crop_size)
    new_image.set_shape([None, None, image.get_shape()[2]])

    # crop_box.shape: [1, 1, 4]
    # crop_box_rank2.shape: [1, 4]
    crop_box_rank2 = tf.squeeze(crop_box, [0])
    # crop_box_rank1.shape: [4]
    crop_box_rank1 = tf.squeeze(crop_box)

    # BoxList shape: [N, 4]
    boxlist = box_list.BoxList(boxes)
    boxlist.add_field('labels', labels)

    # BoxList shape: [1, 4]
    crop_boxlist = box_list.BoxList(crop_box_rank2)

    # remove boxes that are completely outside `crop_box_rank1`
    boxlist, _ = box_list_ops.prune_completely_outside_window(
        boxlist, crop_box_rank1)

    # remove boxes whose fraction of area that is overlapped with 
    # `crop_boxlist` is less than `overlap_thresh`
    boxlist, _ = box_list_ops.prune_non_overlapping_boxes(
        boxlist, crop_boxlist, overlap_thresh)

    # change the coordinate of the remaining boxes
    new_labels = boxlist.get_field('labels')
    new_boxlist = box_list_ops.change_coordinate_frame(boxlist, crop_box_rank1)

    new_boxes = new_boxlist.get()
    # clipping is necessary as some of new_boxes may extend beyond crop window
    new_boxes = tf.clip_by_value(
        new_boxes, clip_value_min=0.0, clip_value_max=1.0)

    return new_image, new_boxes, new_labels


def random_crop_image(image,
                      boxes,
                      labels,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      prob_to_keep_original=0.0,
                      seed=None):
  """Randomly crops an image and associated groundtruth boxes.
  
  This function either performs a crop by `_strict_random_crop_image` with 
  probability `1 - prob_to_keep_original` or return the input `image`, `boxes`,
  and `labels` unchanged with probability `prob_to_keep_original`.  

  Args:
    image: a rank-3 float tensor with shape [height, width, channels].
    boxes: a rank-2 float tensor with shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    labels: a rank-1 int tensor containing the object classes.
    min_object_covered: float scalar, the cropped image must cover at least 
      this fraction of at least one of the input bounding boxes.
    aspect_ratio_range: a float 2-tuple, allowed range for aspect ratio of
      cropped image.
    area_range: a float 2-tuple, allowed range for area ratio between cropped
      image and the original image.
    overlap_thresh: float scalar, a groundtruth box in `boxes` is kept only if
      its IoA (w.r.t the cropped window) >= this threshold.
    prob_to_keep_original: int scalar, the probability to return the original
      input. If 0, always performs a crop.
    seed: int scalar, random seed.

  Returns:
    image: new or original image with same rank as input.
    boxes: new or original boxes with same rank as input.
    labels: labels associated with output `boxes` with same rank as input.
  """
  selector = tf.random_uniform([], seed=seed)
  do_a_crop_random = tf.greater(selector, prob_to_keep_original)

  result = tf.cond(do_a_crop_random,
      lambda: _strict_random_crop_image(image,
                                        boxes,
                                        labels,
                                        min_object_covered,
                                        aspect_ratio_range,
                                        area_range,
                                        overlap_thresh),
      lambda: (image, boxes, labels))
  return result


def ssd_random_crop(image,
                    boxes,
                    labels,
                    min_object_covered=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    aspect_ratio_range=((0.5, 2.0),) * 7,
                    area_range=((0.1, 1.0),) * 7,
                    overlap_thresh=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    prob_to_keep_original=(0.15,) * 7,
                    seed=None):
  """Randomly crops an image and associated groundtruth boxes as described in 
  Section 2.2 - Data augmentation of https://arxiv.org/abs/1512.02325

  Args:
    image: a rank-3 float tensor with shape [height, width, channels].
    boxes: a rank-2 float tensor with shape [num_boxes, 4], where each row 
      contains normalized (i.e. values varying in [0, 1]) box coordinates: 
      [ymin, xmin, ymax, xmax].
    labels: a rank-1 int tensor containing the object classes.
    min_object_covered: tuple of float scalars, the cropped image must cover at
      least this fraction of at least one of the input bounding boxes.
    aspect_ratio_range: tuple of float 2-tuples, allowed range for aspect ratio
      of cropped image.
    area_range: tuple of float 2-tuple, allowed range for area ratio between 
      cropped image and the original image.
    overlap_thresh: tuple of float scalars, a groundtruth box in `boxes` is kept
      only if its IoA (w.r.t the cropped window) >= this threshold.
    prob_to_keep_original: tuple of float scalars, the probability to return the
      original input. If 0, always performs a crop.
    seed: int scalar, random seed.

  Returns:
    image: new or original image with same rank as input.
    boxes: new or original boxes with same rank as input.
    labels: labels associated with output `boxes` with same rank as input.  
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
                        min_object_covered=min_object_covered[i],
                        aspect_ratio_range=aspect_ratio_range[i],
                        area_range=area_range[i],
                        overlap_thresh=overlap_thresh[i],
                        prob_to_keep_original=prob_to_keep_original[i],
                        seed=seed)
      for i in range(num_cases)]

  result = tf.case([(predicate, fn) for predicate, fn 
      in zip(predicates, random_crop_image_fn_list)])
  return result


def resize_image(image,
                 new_height=512,
                 new_width=512,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  """Resizes images to the given height and width.

  Args:
    image: a rank-3 float tensor with shape [height, width, channels].
    new_height: int scalar, desired height of the image.
    new_width: int scalar, desired width of the image.
    method: interpolation method used in resizing. Defaults to BILINEAR.
    align_corners: bool scalar, if true, exactly align all 4 corners of the 
        input and output. Defaults to False.
  """
  with tf.name_scope(
      'ResizeImage',
      values=[image, new_height, new_width, method, align_corners]):
    new_image = tf.image.resize_images(
        image, tf.stack([new_height, new_width]), 
        method=method,
        align_corners=align_corners)
    return new_image


def get_fn_args_map():
  """Returns the mapping from a preprocessing function to a tuple of its tensor
  argument names.
  """
  fn_args_map = {
      random_horizontal_flip: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes),
      random_crop_image: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes,
          names.TensorDictFields.groundtruth_labels),
      ssd_random_crop: (
          names.TensorDictFields.image,
          names.TensorDictFields.groundtruth_boxes,
          names.TensorDictFields.groundtruth_labels),
      resize_image: (
          names.TensorDictFields.image,),
  }
  return fn_args_map


class DataPreprocessor(object):
  """Data preprocessor for detection models. 

  It wraps a sequence of specified preprocessing options. Primarily used for
  data augmentation.
  """
  def __init__(self, options_list):
    """Constructor.

    Args:
      options_list: a list of 2-tuples where the first element is 
        a preprocessing function, and the second element is a dict mapping
        from kwarg names to kwarg values. Example:
        [(<function random_horizontal_flip at 0xXXXXXXXX>, {})], where the
        dict is empty.
    """
    self._options_list = options_list
    
  def preprocess(self, tensor_dict, fn_args_map=None):
    """Applies preprocessing functions sequentially on the input tensor dict.

    Args:
      tensor_dict: a dict mapping from tensor names to tensors.
      fn_args_map: a dict mapping from preprocessing functions to tensor 
        arguments.
    """
    if fn_args_map is None:
      fn_args_map = get_fn_args_map()

    for fn, kwargs in self._options_list:
      arg_names = fn_args_map[fn]

      args = [tensor_dict[arg_name] for arg_name in arg_names]

      results = fn(*args, **kwargs)
      for arg_name, result in zip(arg_names, results):
        tensor_dict[arg_name] = result

    return tensor_dict

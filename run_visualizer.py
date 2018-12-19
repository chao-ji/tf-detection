r"""

Example:
  python run_visualizer.py \
    --label_map_config_path=faster_rcnn/pascal_label_map.config \
    --test_config_path=faster_rcnn/test_config.config
"""
import os
import glob

import tensorflow as tf
import numpy as np
from PIL import Image

from detection.utils import visualize_utils as viz_utils
from detection.utils import misc_utils

#label_map_config_path = 'ssd/pets_label_map.config'
#test_config_path = 'ssd/test_config.config'
flags = tf.app.flags

flags.DEFINE_string(
    'label_map_config_path', None, 'Path to the label map config file.')
flags.DEFINE_string('test_config_path', None, 'Path to the test config file.')

FLAGS = flags.FLAGS


def main(_):
  label_map_config_path = FLAGS.label_map_config_path
  test_config_path = FLAGS.test_config_path


  label_map = misc_utils.read_label_map(label_map_config_path)
  test_config = misc_utils.read_config(test_config_path, 'test')
  inference_directory = test_config.inference_directory

  files = [detection_file for detection_file in 
      glob.glob(os.path.join(inference_directory, '*.npy'))]

  for fn in files:
    detection_result = np.load(fn).item()
    image = viz_utils.visualize_detections(detection_result, label_map,
        font_path=test_config.font_path,
        font_size=test_config.font_size,
        line_width=test_config.line_width,
        score_thresh=test_config.score_thresh)
    Image.fromarray(image).save(misc_utils.replace_ext(fn, '_detect.jpg'))


if __name__  == '__main__':
  tf.flags.mark_flag_as_required('label_map_config_path')
  tf.flags.mark_flag_as_required('test_config_path')
  tf.app.run()

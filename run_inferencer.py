r"""Executable for making object detection inference.

Proto buffer (https://developers.google.com/protocol-buffers/) is used to manage 
all configuration settings. The following configuration files are required:

  'label_map.config' (the mapping from object class name to class index)
  'model.config' (specifying the model architecture, e.g. ssd or faster rcnn)
  'dataset.config' (how raw data should be read and preprocessed)
  'test.config' (specifying evaluation, inference and visualization options.)

See protos/label_map.proto, protos/ssd_model.proto, 
protos/faster_rcnn_model.proto, protos/dataset.proto, protos/test_config.proto
for details.

Example:
  python run_inferencer.py \
    --label_map_config_path=faster_rcnn/pascal_label_map.config \
    --model_config_path=faster_rcnn/faster_rcnn_model.config \
    --dataset_config_path=faster_rcnn/dataset.config \
    --test_config_path=faster_rcnn/test_config.config \
    --model_arch=faster_rcnn_model
"""
import os
import glob

import tensorflow as tf
import numpy as np

from detection.builders import ssd_model_builder
from detection.builders import faster_rcnn_model_builder
from detection.utils import misc_utils

#model_config_path = 'ssd/ssd_model.config'
#dataset_config_path = 'ssd/dataset.config'
#label_map_config_path = 'ssd/pets_label_map.config'
#test_config_path = 'ssd/test_config.config'
#model_arch = 'ssd_model'


flags = tf.app.flags

flags.DEFINE_string(
    'label_map_config_path', None, 'Path to the label map config file.')
flags.DEFINE_string('model_config_path', None, 'Path to the model config file.')
flags.DEFINE_string('dataset_config_path', None, 'Path to dataset config file.')
flags.DEFINE_string('test_config_path', None, 'Path to the test config file.')
flags.DEFINE_string('model_arch', None, 'Model architecture name.')

FLAGS = flags.FLAGS


def main(_):
  label_map_config_path = FLAGS.label_map_config_path
  model_config_path = FLAGS.model_config_path
  dataset_config_path = FLAGS.dataset_config_path
  test_config_path = FLAGS.test_config_path
  model_arch = FLAGS.model_arch


  label_map = misc_utils.read_label_map(label_map_config_path)
  num_classes = len(label_map)

  model_config = misc_utils.read_config(model_config_path, model_arch)
  dataset_config = misc_utils.read_config(dataset_config_path, 'dataset')
  test_config = misc_utils.read_config(test_config_path, 'test')

  if model_arch == 'ssd_model':
    model_inferencer, dataset = ssd_model_builder.build_ssd_inference_session(
      model_config, dataset_config, num_classes)
  elif model_arch == 'faster_rcnn_model':
    model_inferencer, dataset = (
        faster_rcnn_model_builder.build_faster_rcnn_inference_session(
            model_config, dataset_config, num_classes))
  else:
    raise ValueError(
        'model_arch must be either "ssd_model" or "faster_rcnn_model".')

  load_ckpt_path = test_config.load_ckpt_path
  inference_directory = test_config.inference_directory
  image_file_extention = list(test_config.image_file_extension)
  files = [glob.glob(os.path.join(inference_directory, '*.' + ext))
      for ext in image_file_extention]

  filenames = []
  for filenames_ext in files:
    filenames.extend(filenames_ext)

  to_be_run_dict = model_inferencer.infer(filenames, dataset)

  restore_saver = model_inferencer.create_restore_saver()

  sess = tf.Session()

  latest_checkpoint = tf.train.latest_checkpoint(load_ckpt_path)
  restore_saver.restore(sess, latest_checkpoint)

  i = 0
  while True:
    try:
      detection_result = sess.run(to_be_run_dict)
      np.save(misc_utils.replace_ext(filenames[i]), detection_result)
      i += 1
    except tf.errors.OutOfRangeError:
      break

  sess.close()


if __name__  == '__main__':
  tf.flags.mark_flag_as_required('label_map_config_path')
  tf.flags.mark_flag_as_required('model_config_path')
  tf.flags.mark_flag_as_required('dataset_config_path')
  tf.flags.mark_flag_as_required('test_config_path')
  tf.flags.mark_flag_as_required('model_arch')
  tf.app.run()

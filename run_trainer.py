r"""Executable for training the detection model.

Proto buffer (https://developers.google.com/protocol-buffers/) is used to manage 
all configuration settings. The following configuration files are required:

  'label_map.config' (the mapping from object class name to class index)
  'model.config' (specifying the model architecture, e.g. ssd or faster rcnn)
  'dataset.config' (how raw data should be read and preprocessed)
  'train.config' (specifying training protocol, e.g. optimizer, learning rate)

See protos/label_map.proto, protos/ssd_model.proto, 
protos/faster_rcnn_model.proto, protos/dataset.proto, protos/train_config.proto
for details.

Example:
  python run_trainer.py \
    --label_map_config_path=faster_rcnn/pascal_label_map.config \
    --model_config_path=faster_rcnn/faster_rcnn_model.config \
    --dataset_config_path=faster_rcnn/dataset.config \
    --train_config_path=faster_rcnn/train_config.config \
    --model_arch=faster_rcnn_model
"""
import sys

import tensorflow as tf

from detection.builders import ssd_model_builder
from detection.builders import faster_rcnn_model_builder
from detection.utils import misc_utils


#label_map_config_path = 'faster_rcnn/pascal_label_map.config'
#model_config_path = 'faster_rcnn/faster_rcnn_model.config'
#dataset_config_path = 'faster_rcnn/dataset.config'
#train_config_path = 'faster_rcnn/train_config.config'
#model_arch = 'faster_rcnn_model'

flags = tf.app.flags

flags.DEFINE_string(
    'label_map_config_path', None, 'Path to the label map config file.')
flags.DEFINE_string('model_config_path', None, 'Path to the model config file.')
flags.DEFINE_string('dataset_config_path', None, 'Path to dataset config file.')
flags.DEFINE_string('train_config_path', None, 'Path to the train config file.')
flags.DEFINE_string('model_arch', None, 'Model architecture name.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  label_map_config_path = FLAGS.label_map_config_path
  model_config_path = FLAGS.model_config_path
  dataset_config_path = FLAGS.dataset_config_path
  train_config_path = FLAGS.train_config_path
  model_arch = FLAGS.model_arch


  label_map, num_classes = misc_utils.read_label_map(label_map_config_path)
  print(num_classes)

  model_config = misc_utils.read_config(model_config_path, model_arch)
  dataset_config = misc_utils.read_config(dataset_config_path, 'dataset')
  train_config = misc_utils.read_config(train_config_path, 'train')

  if model_arch == 'ssd_model':
    model_trainer, dataset, optimizer_builder_fn = (
        ssd_model_builder.build_ssd_train_session(
            model_config, dataset_config, train_config, num_classes)) 
  elif model_arch == 'faster_rcnn_model':
    model_trainer, dataset, optimizer_builder_fn = (
        faster_rcnn_model_builder.build_faster_rcnn_train_session(
            model_config, dataset_config, train_config, num_classes))
  else:
    raise ValueError(
        'model_arch must be either "ssd_model" or "faster_rcnn_model".')

  load_ckpt_path = train_config.load_ckpt_path
  save_ckpt_path = train_config.save_ckpt_path
  files = list(train_config.input_file)

  total_loss, global_step = model_trainer.train(
      files, dataset, optimizer_builder_fn)

  checkpoint_type = train_config.checkpoint_type
  load_all_detection_checkpoint_vars = (True if 
      checkpoint_type == 'detection' else False)

  restore_saver = model_trainer.create_restore_saver(
      load_ckpt_path, 
      checkpoint_type=checkpoint_type,
      load_all_detection_checkpoint_vars=load_all_detection_checkpoint_vars,
      include_global_step=train_config.include_global_step)

  persist_saver = model_trainer.create_persist_saver(
      max_to_keep=train_config.max_to_keep)

  initializers = tf.global_variables_initializer()

  sess = tf.Session()
  sess.run(initializers)

  restore_saver.restore(sess, load_ckpt_path)

  while True:
    loss, gs = sess.run([total_loss, global_step])
    if gs % train_config.print_progress_every_n_steps == 0:
      print('step={}, loss={}'.format(gs, loss))
      sys.stdout.flush()
    if (gs >= train_config.start_save_ckpt_after_n_steps and
        gs % train_config.save_ckpt_every_n_steps == 0):
      persist_saver.save(sess, save_ckpt_path, global_step=gs)
    if gs > train_config.num_steps:
      break

  persist_saver.save(sess, save_ckpt_path)

  sess.close()


if __name__  == '__main__':
  tf.flags.mark_flag_as_required('label_map_config_path')
  tf.flags.mark_flag_as_required('model_config_path')
  tf.flags.mark_flag_as_required('dataset_config_path')
  tf.flags.mark_flag_as_required('train_config_path')
  tf.flags.mark_flag_as_required('model_arch')
  tf.app.run()

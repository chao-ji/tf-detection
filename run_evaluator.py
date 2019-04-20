r"""Executable for evaluating a detection model.

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
  python run_evaluator.py \
    --label_map_config_path=faster_rcnn/pascal_label_map.config \
    --model_config_path=faster_rcnn/faster_rcnn_model.config \
    --dataset_config_path=faster_rcnn/dataset.config \
    --test_config_path=faster_rcnn/test_config.config \
    --model_arch=faster_rcnn_model
"""
import tensorflow as tf
import numpy as np

from detection.metrics import metrics_calculator
from detection.builders import ssd_model_builder
from detection.builders import faster_rcnn_model_builder
from detection.utils import misc_utils

#model_config_path = 'faster_rcnn/faster_rcnn_model.config'
#dataset_config_path = 'faster_rcnn/dataset.config'
#label_map_config_path = 'faster_rcnn/pascal_label_map.config'
#test_config_path = 'faster_rcnn/test_config.config'
#model_arch = 'faster_rcnn_model'

SSD_LOSSES = 'loc_loss', 'cls_loss'
FASTER_RCNN_LOSSES = ('rpn_loc_loss', 'rpn_cls_loss', 
                      'frcnn_loc_loss', 'frcnn_cls_loss')
#                      'frcnn_loc_loss', 'frcnn_cls_loss', 'frcnn_msk_loss')
flags = tf.app.flags

flags.DEFINE_string(
    'label_map_config_path', None, 'Path to the label map config file.')
flags.DEFINE_string('model_config_path', None, 'Path to the model config file.')
flags.DEFINE_string('dataset_config_path', None, 'Path to dataset config file.')
flags.DEFINE_string('test_config_path', None, 'Path to the test config file.')
flags.DEFINE_string('model_arch', None, 'Model architecture name.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  label_map_config_path = FLAGS.label_map_config_path
  model_config_path = FLAGS.model_config_path
  dataset_config_path = FLAGS.dataset_config_path
  test_config_path = FLAGS.test_config_path
  model_arch = FLAGS.model_arch


  label_map, num_classes = misc_utils.read_label_map(label_map_config_path)
  print(num_classes)
  model_config = misc_utils.read_config(model_config_path, model_arch)
  dataset_config = misc_utils.read_config(dataset_config_path, 'dataset')
  test_config = misc_utils.read_config(test_config_path, 'test')

  if model_arch == 'ssd_model':
    model_evaluator, dataset = ssd_model_builder.build_ssd_evaluate_session(
      model_config, dataset_config, num_classes)
    losses_names = SSD_LOSSES
  elif model_arch == 'faster_rcnn_model':
    model_evaluator, dataset = (
        faster_rcnn_model_builder.build_faster_rcnn_evaluate_session(
            model_config, dataset_config, num_classes))
    losses_names = FASTER_RCNN_LOSSES
  else:
    raise ValueError(
        'model_arch must be either "ssd_model" or "faster_rcnn_model".') 

  load_ckpt_path = test_config.load_ckpt_path
  files = list(test_config.input_file)
  
  to_be_run_dict, losses_dict = model_evaluator.evaluate(files, dataset)
  if 'frcnn_msk_loss' in losses_dict:
    losses_names = list(losses_names)
    losses_names.append('frcnn_msk_loss')
    losses_names = tuple(losses_names)

  restore_saver = model_evaluator.create_restore_saver()

  sess = tf.Session()

  latest_checkpoint = tf.train.latest_checkpoint(load_ckpt_path)
#  tf.logging.info('Reading from checkpoint %s... Done.' % latest_checkpoint)
#  print(latest_checkpoint)
  restore_saver.restore(sess, latest_checkpoint)

  class_indices = list(label_map.keys())
  if test_config.metrics_name == 'pascal_voc_detection_metrics':
    met_calc = metrics_calculator.PascalVocMetricsCalculator(num_classes, class_indices)
  elif test_config.metrics_name == 'coco_box_detection_metrics':
    categories = misc_utils.label_map_to_categories(label_map)
    met_calc = metrics_calculator.MscocoMetricsCalculator(categories, class_indices)
  elif test_config.metrics_name == 'coco_mask_detection_metrics':
    categories = misc_utils.label_map_to_categories(label_map)
    met_calc = metrics_calculator.MscocoMetricsCalculator(categories, class_indices, evaluate_mask=True)

  losses = []
  while True:
    try:
      detection_result, losses_result = sess.run([to_be_run_dict, losses_dict])
#      print(detection_result['boxes'].shape, '%f' % detection_result['boxes'].mean(), detection_result['scores'].shape, '%f' % detection_result['scores'].mean(), detection_result['masks'].shape, '%f' % detection_result['masks'].mean(), detection_result['gt_boxes'].shape, '%f' % detection_result['gt_boxes'].mean(), detection_result['gt_masks'].shape, '%f' % detection_result['gt_masks'].mean())

    except tf.errors.OutOfRangeError:
      break;
    
    losses.append([losses_result[k] for k in losses_names])
    met_calc.update_per_image_result(detection_result)

  losses = np.array(losses)
  print(losses.shape)
  sess.close()

  misc_utils.print_evaluation_metrics(met_calc,
                                      test_config.metrics_name,
                                      losses,
                                      losses_names, 
                                      label_map)


if __name__  == '__main__':
  tf.flags.mark_flag_as_required('label_map_config_path')
  tf.flags.mark_flag_as_required('model_config_path')
  tf.flags.mark_flag_as_required('dataset_config_path')
  tf.flags.mark_flag_as_required('test_config_path')
  tf.flags.mark_flag_as_required('model_arch')
  tf.app.run()

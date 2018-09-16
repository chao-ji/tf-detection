import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from detection.utils import visualize_utils as viz_utils
from detection.metrics import metrics_calculator

from detection.builders import ssd_model_builder
from detection.utils import config_utils


load_ckpt_path = '/home/chaoji/Documents/ssd_pet_faces'
#load_ckpt_path = '/home/chaoji/Documents/ssd_pascal/'

model_config_path = '../protos/ssd_model.config'
dataset_config_path = '../protos/dataset.config'
label_map_config_path = '../protos/label_map.config'

path = '/home/chaoji/Desktop/models-master/research/object_detection/data'
files = sorted(glob.glob(os.path.join(path, 'pet_faces_val*')))

label_map = config_utils.get_label_map_from_config_path(label_map_config_path)
num_classes = len(label_map)

model_config, dataset_config = config_utils.get_model_and_dataset_config(
    model_config_path, dataset_config_path)

model_evaluator, dataset = ssd_model_builder.build_ssd_evaluate_session(
    model_config, dataset_config, num_classes)

evaluator_graph = tf.Graph()

with evaluator_graph.as_default():
  output_dict, loc_loss, cls_loss, gt_boxes, gt_labels = model_evaluator.evaluate(files, dataset)

  restore_saver = model_evaluator.create_restore_saver()

  initializers = tf.global_variables_initializer()


sess = tf.Session(graph=evaluator_graph) 
sess.run(initializers)


pascal = metrics_calculator.PascalVocMetricsCalculator(num_classes)

latest_checkpoint = tf.train.latest_checkpoint(load_ckpt_path)
restore_saver.restore(sess, latest_checkpoint)

color_map = viz_utils.get_color_map(num_classes)

L = []
C = []
D = []

for i in range(1103):

  l, c, d, gb, gl = sess.run([loc_loss, cls_loss, output_dict,
          gt_boxes, gt_labels])
  L.append(l)
  C.append(c)
  D.append(d)
  pascal.update_per_image_result(d, gb, gl)

sess.close()

print('\n\n')


APs = pascal.calculate_metrics()

print('PascalBoxes_Precision/mAP@0.5IOU {}'.format(np.nanmean(APs)))
for i in range(1, num_classes + 1):
  print('PascalBoxes_PerformanceByCategory/AP@0.5IOU/{} {}'.format(label_map[i], APs[i - 1]))
print('Losses/Loss/localization_loss', np.nanmean(L))
print('Losses/Loss/classification_loss', np.nanmean(C))


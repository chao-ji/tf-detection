import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from detection.utils import visualize_utils as viz_utils

from detection.builders import ssd_model_builder
from detection.utils import config_utils

num_classes = 37
load_ckpt_path = '/home/chaoji/Documents/ssd_pet_faces'

model_config_path = '../protos/ssd_model.config'
dataset_config_path = '../protos/dataset.config'
label_map_config_path = '../protos/label_map.config'

files = ['/home/chaoji/Desktop/shiba.jpg',
         '/home/chaoji/Desktop/persian.jpg',
         '/home/chaoji/Desktop/2311-1P10Z94S2.jpg']

label_map = config_utils.get_label_map_from_config_path(label_map_config_path)
num_classes = len(label_map)

model_config, dataset_config = config_utils.get_model_and_dataset_config(
    model_config_path, dataset_config_path)

model_inferencer, dataset = ssd_model_builder.build_ssd_inference_session(
    model_config, dataset_config, num_classes)

inferencer_graph = tf.Graph()

with inferencer_graph.as_default():
  output_dict = model_inferencer.infer(files, dataset)

  restore_saver = model_inferencer.create_restore_saver()

  initializers = tf.global_variables_initializer()


sess = tf.Session(graph=inferencer_graph) 
sess.run(initializers)

latest_checkpoint = tf.train.latest_checkpoint(load_ckpt_path)
restore_saver.restore(sess, latest_checkpoint)

color_map = viz_utils.get_color_map(num_classes)

D = []
for i in range(3):
  d = sess.run(output_dict)
  D.append(d)

sess.close()


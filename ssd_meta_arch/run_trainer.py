import os
import glob

import tensorflow as tf

from detection.builders import ssd_model_builder
from detection.utils import config_utils

load_ckpt_path = '/home/chaoji/Documents/inception_v2.ckpt'
save_ckpt_path = '/home/chaoji/Dropbox/tensorflow/detection/ssd_meta_arch/model'
label_map_config_path = '../protos/label_map.config'

model_config_path = '../protos/ssd_model.config'
dataset_config_path = '../protos/dataset.config'
path = '/home/chaoji/Desktop/models-master/research/object_detection/data'
files = sorted(glob.glob(os.path.join(path, 'pet_faces_train*')))

label_map = config_utils.get_label_map_from_config_path(label_map_config_path)
num_classes = len(label_map)

model_config, dataset_config = config_utils.get_model_and_dataset_config(
    model_config_path, dataset_config_path)

model_trainer, dataset, optimizer_builder_fn = (
    ssd_model_builder.build_ssd_train_session(
        model_config, dataset_config, num_classes))

trainer_graph = tf.Graph()

with trainer_graph.as_default():

  grouped_update_op, total_loss, global_step = model_trainer.train(files, dataset, optimizer_builder_fn)

  restore_saver = model_trainer.create_restore_saver(load_ckpt_path)

  persist_saver = model_trainer.create_persist_saver()

  initializers = tf.global_variables_initializer()


sess = tf.Session(graph=trainer_graph)
sess.run(initializers)

restore_saver.restore(sess, load_ckpt_path)



for i in range(80000):
  _, loss, gs = sess.run([grouped_update_op, total_loss, global_step])
  print('step={}, loss={}'.format(gs, loss))

  if i % 10000 == 0:  
    persist_saver.save(sess, save_ckpt_path, global_step=global_step) 

persist_saver.save(sess, save_ckpt_path, global_step=global_step)

sess.close()


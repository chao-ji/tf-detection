import tensorflow as tf
import numpy as np

from detection.protos import conv_hyperparams_pb2
from detection.builders import conv_hyperparams_builder

import feature_map_generators
from google.protobuf import text_format

slim = tf.contrib.slim


config = conv_hyperparams_pb2.ConvHyperparams()
text_format.Merge(open('../protos/conv_hyperparams.config').read(), config)

arg_scope_fn = conv_hyperparams_builder.build(config)

feature_map_dict = {
'Mixed_4c': tf.random_normal(shape=[24, 19, 19, 576], dtype=tf.float32, name='Mixed_4c'),
'Mixed_5c': tf.random_normal(shape=[24, 10, 10, 1024], dtype=tf.float32, name='Mixed_5c')}

feature_map_specs = {
    'layer_name': ['Mixed_4c', 'Mixed_5c', None, None, None, None],
    'layer_depth': [None, None, 512, 256, 256, 128]}



with slim.arg_scope(arg_scope_fn()):
  with slim.arg_scope([slim.batch_norm], is_training=False):
    feature_maps = feature_map_generators.ssd_feature_maps(feature_map_dict,
                     feature_map_specs,
                     use_depthwise=False,
                     insert_1x1_conv=True)

graph = tf.get_default_graph()
summary_writer = tf.summary.FileWriter('.', graph=graph)

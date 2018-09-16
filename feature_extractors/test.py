import tensorflow as tf
import numpy as np

from detection.protos import conv_hyperparams_pb2
from detection.builders import conv_hyperparams_builder

from detection.feature_extractors import ssd_inception_v2_feature_extractor
from google.protobuf import text_format

slim = tf.contrib.slim


config = conv_hyperparams_pb2.ConvHyperparams()
text_format.Merge(open('../protos/conv_hyperparams.config').read(), config)

conv_hyperparams_fn = conv_hyperparams_builder.build(config)

fe = ssd_inception_v2_feature_extractor.SsdInceptionV2FeatureExtractor(
    conv_hyperparams_fn=conv_hyperparams_fn,
    depth_multiplier=1,
    reuse_weights=None,
    use_depthwise=False,
    override_base_feature_extractor_hyperparams=True)

inputs = tf.random_normal(shape=[1, 300, 300, 3])

feature_maps = fe.extract_features(inputs)

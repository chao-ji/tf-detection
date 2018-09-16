import tensorflow as tf

from detection.protos import decoder_pb2
from detection.preprocess import data_decoder 


def _get_feature_type_map():
  feature_type_map = {
      decoder_pb2.KeysToFeaturesItem.STRING: tf.string,
      decoder_pb2.KeysToFeaturesItem.FLOAT32: tf.float32,
      decoder_pb2.KeysToFeaturesItem.INT64: tf.int64,
  }
  return feature_type_map

def build(config):
  keys_to_features = {}
  feature_type_map = _get_feature_type_map()

  for item in config.keys_to_features:

    if item.feature_parser.WhichOneof(
        'feature_parser_oneof') == 'fixed_len_feature_parser':
      dtype = item.feature_parser.fixed_len_feature_parser.type
      shape = [i for i in
          item.feature_parser.fixed_len_feature_parser.shape.size]
      keys_to_features[item.key] = tf.FixedLenFeature(
          shape=shape, dtype=feature_type_map[dtype])
    elif item.feature_parser.WhichOneof(
        'feature_parser_oneof') == 'var_len_feature_parser':
      dtype = item.feature_parser.var_len_feature_parser.type
      keys_to_features[item.key] = tf.VarLenFeature(
          dtype=feature_type_map[dtype])
    else:
      raise ValueError('Unknown feature parser.')

  return data_decoder.DataDecoder(keys_to_features)


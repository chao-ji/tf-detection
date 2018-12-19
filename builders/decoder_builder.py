import tensorflow as tf

from detection.protos import decoder_pb2
from detection.data import data_decoder 


def _get_feature_type_map():
  """Returns a dict mapping from feature data type (enum) to tensorflow 
  data type.
  """
  feature_type_map = {
      decoder_pb2.KeysToFeaturesItem.STRING: tf.string,
      decoder_pb2.KeysToFeaturesItem.FLOAT32: tf.float32,
      decoder_pb2.KeysToFeaturesItem.INT64: tf.int64,
  }
  return feature_type_map


def build(config):
  """Builds data decoder.

  Args:
    config: a protobuf message storing DataDecoder configurations.

  Returns:
    an instance of DataDecoder.
  """
  if not isinstance(config, decoder_pb2.DataDecoder):
    raise ValueError('config must be an instance of DataDecoder message.')

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


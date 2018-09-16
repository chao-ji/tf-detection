import numpy as np

from google.protobuf import text_format
from detection.protos import ssd_model_pb2
from detection.protos import dataset_pb2
from detection.protos import label_map_pb2

def get_model_and_dataset_config(model_config_path, dataset_config_path):
  model_config = ssd_model_pb2.SsdModel()
  dataset_config = dataset_pb2.Dataset()
    
  text_format.Merge(open(model_config_path).read(), model_config)
  text_format.Merge(open(dataset_config_path).read(), dataset_config)

  return model_config, dataset_config


def get_label_map_from_config_path(label_map_config_path):
  label_map_config = label_map_pb2.LabelMap()
  text_format.Merge(open(label_map_config_path).read(), label_map_config)

  label_map = dict([(item.index, item.label) for item in label_map_config.label_map_item])
  _check_label_map(label_map)
  return label_map


def _check_label_map(label_map):
  keys = sorted(list(label_map.keys()))
 
  if not (np.arange(len(keys)) == np.array(keys) - 1).all():
    raise ValueError('label map indices must start from 1 up to max value.')


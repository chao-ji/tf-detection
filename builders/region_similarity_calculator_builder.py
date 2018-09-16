from detection.core import region_similarity_calculator
from detection.protos import region_similarity_calculator_pb2

def build(config):
  similarity_calculator = config.WhichOneof('region_similarity_oneof')
  if similarity_calculator == 'iou_similarity':
    return region_similarity_calculator.IouSimilarity()
  if similarity_calculator == 'ioa_similarity':
    return region_similarity_calculator.IoaSimilarity()
  raise ValueError('Unknown region similarity calculator')


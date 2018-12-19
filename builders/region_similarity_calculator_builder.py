from detection.core import region_similarity_calculator
from detection.protos import region_similarity_calculator_pb2


def build(config):
  """Builds region similarity calculator.

  Args:
    config: a protobuf message storing RegionSimilarityCalculator configuration.

  Returns:
    an instance of RegionSimilarityCalculator.
  """
  if not isinstance(config, 
      region_similarity_calculator_pb2.RegionSimilarityCalculator):
    raise ValueError('config must be an instance of RegionSimilarityCalculator'
        ' message.')

  similarity_calculator = config.WhichOneof('region_similarity_oneof')
  if similarity_calculator == 'iou_similarity':
    return region_similarity_calculator.IouSimilarity()
  if similarity_calculator == 'ioa_similarity':
    return region_similarity_calculator.IoaSimilarity()
  raise ValueError('Unknown region similarity calculator')


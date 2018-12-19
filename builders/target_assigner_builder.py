from detection.protos import target_assigner_pb2
from detection.core import target_assigner
from detection.builders import region_similarity_calculator_builder
from detection.builders import matcher_builder
from detection.builders import box_coder_builder


def build(config,
          region_similarity_calculator=None,
          matcher=None,
          box_coder=None):
  """Builds the target assigner. You can optionally pass in `box_coder`, 
  `matcher`, `region_similarity_calculator`. Otherwise, those specific to
  target assigner will be built. 

  Args:
    config: a protobuf message storing TargetAssigner configurations.
    region_similarity_calculator: an instance of RegionSimilarityCalculator or
      None.
    matcher: an instance of Matcher or None. 
    box_coder: an instance of BoxCoder or None.

  Returns:
    an instance of TargetAssigner.
  """
  if not isinstance(config, target_assigner_pb2.TargetAssigner):
    raise ValueError('config must be an instance of TargetAssigner message.')

  if region_similarity_calculator is None:
    region_similarity_calculator = region_similarity_calculator_builder.build(
        config.region_similarity_calculator)
  if matcher is None:
    matcher = matcher_builder.build(config.matcher)
  if box_coder is None:
    box_coder = box_coder_builder.build(config.box_coder)

  return target_assigner.TargetAssigner(
      region_similarity_calculator=region_similarity_calculator,
      matcher=matcher,
      box_coder=box_coder,
      negative_class_weight=config.negative_class_weight)
  

from detection.protos import target_assigner_pb2
from detection.core import target_assigner
from detection.builders import region_similarity_calculator_builder
from detection.builders import matcher_builder
from detection.builders import box_coder_builder


def build(config,
          region_similarity_calculator=None,
          matcher=None,
          box_coder=None):

  if region_similarity_calculator is None:
    region_similarity_calculator = region_similarity_calculator_builder.build(
        config.region_similarity_calculator)
  if matcher is None:
    matcher = matcher_builder.build(config.matcher)
  if box_coder is None:
    box_coder = box_coder_builder.build(config.box_coder)

  assigner = target_assigner.TargetAssigner(
      region_similarity_calculator=region_similarity_calculator,
      matcher=matcher,
      box_coder=box_coder,
      negative_class_weight=config.negative_class_weight)
  
  return assigner


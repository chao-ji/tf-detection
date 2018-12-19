from detection.core import matcher
from detection.protos import matcher_pb2


def build(config):
  """Builds matcher.

  Args:
    config: a protobuf message storing Matcher configurations.

  Returns:
    an instance of Matcher.
  """
  if not isinstance(config, matcher_pb2.Matcher):
    raise ValueError('config must be an instance of Matcher message.')

  if config.WhichOneof('matcher_oneof') == 'argmax_matcher':
    config = config.argmax_matcher
    return matcher.ArgMaxMatcher(
        matched_thres=config.matched_threshold,
        unmatched_thres=config.unmatched_threshold,
        negatives_lower_than_unmatched=config.negatives_lower_than_unmatched,
        force_match_for_each_row=config.force_match_for_each_row)

  return ValueError('Unknown matcher')


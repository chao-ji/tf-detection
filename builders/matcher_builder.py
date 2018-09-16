from detection.matchers import argmax_matcher

def build(config):
  if config.WhichOneof('matcher_oneof') == 'argmax_matcher':
    config = config.argmax_matcher
    matched_threshold = unmatched_threshold = None
    if not config.ignore_thresholds:
      matched_threshold = config.matched_threshold
      unmatched_threshold = config.unmatched_threshold
    return argmax_matcher.ArgMaxMatcher(
        matched_threshold=matched_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=config.negatives_lower_than_unmatched,
        force_match_for_each_row=config.force_match_for_each_row)


  return ValueError('Unknown matcher')

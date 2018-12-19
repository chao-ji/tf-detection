from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from detection.utils import shape_utils


class Matcher(object):
  """Abstract base class for matcher."""
  __metaclass__ = ABCMeta

  def match(self, sim_matrix, scope=None):
    """Given n-by-m similarity matrix, assign row index to each column. 
    Depending on thresholds of the matcher, -1 (unmatched) or -2 (ignored) can 
    be assigned instead of a row index. Typically rows correspond to groundtruth
    boxes, while columns correspond to anchor boxes. Calls and wraps `_match` 
    with a name scope.

    Args:
      sim_matrix: a float tensor of shape [n, m] holding similarity scores. 
      scope: string scalar, name scope.

    Returns:
      a Match instance that wraps the match results.
    """
    with tf.name_scope(scope, 'Match', [sim_matrix]):
      return Match(self._match(sim_matrix))

  @abstractmethod
  def _match(self, sim_matrix):
    """Assign the row index to each column. Typically rows correspond
    to groundtruth boxes, while columns correspond to anchor boxes.

    To be implemented by subclasses.

    Args:
      sim_matrix: a float tensor of shape [n, m] holding similarity scores. 

    Returns:
      match_results: an int tensor of shape [m] holding matching results 
        (ints >= -2) for each of `m` columns in `sim_matrix`, where
        `match_results[i] =  -2` indicates `i` is ignored;
        `match_results[i] =  -1` indicates `i` is unmatched (negative);
        `match_results[i] >=  0` indicates `i` is matched (positive).
    """
    pass


class ArgMaxMatcher(Matcher):
  """Picks the row index that maximizes the column."""

  def __init__(self,
               matched_thres,
               unmatched_thres,
               negatives_lower_than_unmatched=True,
               force_match_for_each_row=False):
    """Constructor.

    Args:
      matched_thres: float scalar between 0 and 1, threshold for positive match.
      unmatched_thres: float scalar between 0 and 1, threshold for negative 
        match. Must be no greater than `matched_thres`.
      negatives_lower_than_unmatched: bool scalar, whether to consider those 
        below `unmatched_thres` as negatives. If True, negatives are those below 
        `unmatched_thres`, and the ignored matches are in between matched and 
        unmatched thresholds. If False, negatives are in betweeen matched and 
        unmatched thresholds, and the ignored matches are those below 
        `unmatched_thres`. 
      force_match_for_each_row: bool scalar, whether to force each row to match
        to at least one column.
    """
    if unmatched_thres > matched_thres:
      raise ValueError('unmatched_thres must be <= matched_thres.')
    if unmatched_thres == matched_thres and not negatives_lower_than_unmatched: 
      raise ValueError('unmatched_thres must be < matche_thres, when negatives '
          'are in between them, got {} and {}.'.format(
          unmatched_thres, matched_thres))

    self._matched_thres = matched_thres
    self._unmatched_thres = unmatched_thres
    self._negatives_lower_than_unmatched = negatives_lower_than_unmatched
    self._force_match_for_each_row = force_match_for_each_row

  def _match(self, sim_matrix):
    """Assign row index (argmax) to each column. Typically rows correspond
    to groundtruth boxes, while columns correspond to anchor boxes.

    Args:
      sim_matrix: a float tensor of shape [n, m] holding similarity scores.

    Returns:
      results: an int tensor of shape [m] holding matching results (ints >= -2) 
        for each of `m` columns in `sim_matrix`, where
        `results[i] =  -2` indicates `i` is ignored;
        `results[i] =  -1` indicates `i` is unmatched (negative);
        `results[i] >=  0` indicates `i` is matched (positive).
    """
    sim_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
        sim_matrix)
    unmatched_indicator = -1 * tf.ones([sim_matrix_shape[1]], dtype=tf.int32) 
    ignored_indicator = -2 * tf.ones([sim_matrix_shape[1]], dtype=tf.int32) 

    def _match_when_rows_are_empty():
      return unmatched_indicator

    def _match_when_rows_are_non_empty():
      # Matches for each column
      matches = tf.argmax(sim_matrix, 0, output_type=tf.int32) # [m]
      matched_vals = tf.reduce_max(sim_matrix, 0) # [m]

      below_unmatched_thres = tf.greater(
          self._unmatched_thres, matched_vals) # [m]
      between_thresholds = tf.logical_and(
          tf.greater_equal(matched_vals, self._unmatched_thres),
          tf.greater(self._matched_thres, matched_vals)) # [m]

      if self._negatives_lower_than_unmatched:
        matches = tf.where(below_unmatched_thres, unmatched_indicator, matches)
        matches = tf.where(between_thresholds, ignored_indicator, matches)
      else:
        matches = tf.where(below_unmatched_thres, ignored_indicator, matches)
        matches = tf.where(between_thresholds, unmatched_indicator, matches)

      if self._force_match_for_each_row:
        force_match_column_ids = tf.argmax(sim_matrix, 1,
                                           output_type=tf.int32)
        force_match_column_indicators = tf.one_hot(
            force_match_column_ids, depth=sim_matrix_shape[1])
        force_match_row_ids = tf.argmax(force_match_column_indicators, 0,
                                        output_type=tf.int32)
        force_match_column_mask = tf.cast(
            tf.reduce_max(force_match_column_indicators, 0), tf.bool)
        final_matches = tf.where(force_match_column_mask,
                                 force_match_row_ids, matches)
        return final_matches
      else:
        return matches

    if not isinstance(sim_matrix_shape[0], tf.Tensor):
      results = (_match_when_rows_are_empty() if sim_matrix_shape[0] == 0 else
           _match_when_rows_are_non_empty())
    else:
      results = tf.cond(tf.greater(tf.shape(sim_matrix)[0], 0),
          _match_when_rows_are_non_empty, _match_when_rows_are_empty)
    return results


class Match(object):
  """Wrapper of the matching results returned by `Matcher.match` that provides 
  APIs for querying the matching results.

  For example, given self._matching_results = [2, 1, -1, 0, -2], where we have
  5 columns in total:

  - column 0, 1, 3 matches to row 2, 1, 0
  - column 2 does not match any row (result is -1)
  - column 4 is ignored (result is -2)

  then

  matched_column_indices = [0, 1, 3];
  matched_column_indicator = [True, True, False, True, False];
  num_matched_columns = 3;
  
  unmatched_column_indices = [2];
  unmatched_column_indicator = [False, False, True, False, False];
  num_unmatched_columns = 1;

  ignored_column_indices = [4];
  ignored_column_indicator = [False, False, False, False, True];
  num_ignored_columns = 1;

  unmatched_or_ignored_column_indices = [2, 4];
  matched_row_indices = [2, 1, 0].
  """

  def __init__(self, match_results):
    """Constructor.

    Args:
      match_results: an int tensor of shape [m] holding matching results 
        (ints >= -2), where
        `match_results[i] =  -2` indicates `i` is ignored;
        `match_results[i] =  -1` indicates `i` is unmatched (negative);
        `match_results[i] >=  0` indicates `i` is matched (positive).
    """
    self._match_results = match_results

  @property
  def match_results(self):
    return self._match_results

  def matched_column_indices(self):
    """Returns an int tensor of shape [k] storing indices of columns that have
    some match, where `k <= m` and `m` is the num of columns.
    """
    return self._reshape_and_cast(tf.where(self.matched_column_indicator))

  def matched_column_indicator(self):
    """Returns a bool tensor of shape [m] storing which columns have 
    matches, where `m` is the num of columns. 
    """
    return tf.greater_equal(self._match_results, 0)

  def num_matched_columns(self):
    """Returns an int scalar tensor holding the num of matched columns."""
    return tf.size(self.matched_column_indices())

  def unmatched_column_indices(self):
    """Returns an int tensor of shape [k] storing indices of columns that do
    not have any match, where `k <= m` and `m` is the num of columns.
    """
    return self._reshape_and_cast(tf.where(self.unmatched_column_indicator))

  def unmatched_column_indicator(self):
    """Returns a bool tensor of shape [m] storing which columns have 
    no matches, where `m` is the num of columns.
    """
    return tf.equal(self._match_results, -1)

  def num_unmatched_columns(self):
    """Returns an int scalar tensor holding the num of unmatched columns."""
    return tf.size(self.unmatched_column_indices())

  def ignored_column_indices(self):
    """Returns an int tensor of shape [k] storing indices of columns that are
    ignored, where `k <= m` and `m` is the num of columns.
    """
    return self._reshape_and_cast(tf.where(self.ignored_column_indicator()))

  def ignored_column_indicator(self):
    """Returns a bool tensor of shape [m] storing which columns are 
    ignored, where `m` is the num of columns.
    """
    return tf.equal(self._match_results, -2)

  def num_ignored_columns(self):
    """Returns an int scalar tensor holding the num of ignored columns."""
    return tf.size(self.ignored_column_indices())

  def unmatched_or_ignored_column_indices(self):
    """Returns an int tensor of shape [k] storing indices of columns that do
    not have any match, or are ignored, where `k <= m` and `m` is the num of 
    columns.
    """
    return self._reshape_and_cast(tf.where(tf.greater(0, self._match_results)))

  def matched_row_indices(self):
    """Returns an int tensor of shape [k] storing row indices corresponding to
    the column indices returned by `matched_column_indices`, where `k <= m` and
    `m` is the num of columns.
    """
    return self._reshape_and_cast(
        tf.gather(self._match_results, self.matched_column_indices()))

  def _reshape_and_cast(self, t):
    return tf.cast(tf.reshape(t, [-1]), tf.int32)

  def gather_based_on_match(self, input_tensor, unmatched_value,
                            ignored_value):
    """Gather elements from `input_tensor` based on match results.

    For example, given 
      input_tensor = [[0 , 1 , 2 , 3 ],
                      [4 , 5 , 6 , 7 ], 
                      [9 , 10, 11, 12]];
      unmatched_value = [0, 0, 0, 0];
      ignored_value = [0, 0, 0, 0];
    
    and self._matched_results = [2, -1, 0, 2, -2], where -1 indicates unmatched
    column and -2 indicates ignored column,

    the elements gathered woud be 
    [ [9 , 10, 11, 12],
      [0 , 0 , 0 , 0 ],
      [0 , 1 , 2 , 3 ],
      [9 , 10, 11, 12],
      [0 , 0 , 0 , 0 ] ]

    Args:
      input_tensor: a tensor to gather values from.
      unmatched_value: a tensor of rank `rank(input_tensor) - 1`, and of shape 
        `shape(input_tensor)[1:]`, holding constant values for the unmatched. 
      ignored_value: a tensor of rank `rank(input_tensor) - 1`, and of shape
        `shape(input_tensor)[1:]`, holding constant values for the ignored.

    Returns:
      gathered_tensor: a tensor of shape 
        [shape(self._match_results)[0], shape(input_tensor)[1:]], holding values
        gathered from `input_tensor`.
    """
    with tf.control_dependencies([tf.assert_greater_equal(self._match_results, 
                                  tf.constant(-2, tf.int32))]):
      self._match_results = tf.identity(self._match_results)

    input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),
                              input_tensor], axis=0)
    gather_indices = self._match_results + 2
    gathered_tensor = tf.gather(input_tensor, gather_indices)
    return gathered_tensor

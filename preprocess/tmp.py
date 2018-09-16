sess = tf.InteractiveSession()

mat, res = sess.run([matrix, match.match_results])


matches = mat.argmax(axis=0)

matched_vals = mat.max(axis=0)

below_unmatched_threshold = 0.5 > matched_vals

between_thresholds = np.logical_and(matched_vals >= 0.5, 0.5 > matched_vals)

matches = np.where(below_unmatched_threshold, -1, matches)

matches = np.where(between_thresholds, -2, matches)

force_match_column_ids = mat.argmax(axis=1)

force_match_column_indicators = np.zeros_like(mat)

force_match_column_indicators[np.arange(mat.shape[0]), force_match_column_ids] = 1


force_match_row_ids = force_match_column_indicators.argmax(axis=0)

force_match_column_mask = force_match_column_indicators.max(axis=0).astype(np.bool)

final_matches = np.where(force_match_column_mask, force_match_row_ids, matches)


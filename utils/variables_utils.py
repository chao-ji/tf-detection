import tensorflow as tf
from tensorflow import logging

def get_vars_available_in_ckpt(name_to_var_map,
                               checkpoint_path,
                               include_global_step=True):
  """Returns the variable name to variable mapping used to initialize an
  `tf.train.Saver` object. 

  Inspects given checkpoint and returns the subset of variables that are
  available in it.

  Args:
    name_to_var_map: a dict mapping from variable name to variable. 
    checkpoint_path: string scalar, path to the checkpoint to restore variables 
      from.
    include_global_step: bool scalar, whether to include `global_step` variable,
      if exists. Defaults to True.

  Returns:
    vars_in_ckpt: a dict mapping from variable name to variable.
  """

  reader = tf.train.NewCheckpointReader(checkpoint_path)
  vars_to_shape_map = reader.get_variable_to_shape_map()

  if not include_global_step:
    vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)

  vars_in_ckpt = {}
  for var_name, var in sorted(name_to_var_map.items()):
    if var_name in vars_to_shape_map:
      if vars_to_shape_map[var_name] == var.shape.as_list():
        vars_in_ckpt[var_name] = var
      else:
        logging.warning('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable.',
                        var_name)
    else:
      logging.warning('Variable [%s] is not available in checkpoint',
                      var_name)

  return vars_in_ckpt


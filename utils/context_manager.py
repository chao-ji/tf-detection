class IdentityContextManager(object):
  def __enter__(self):
    return None

  def __exit__(self, exception_type, exception_value, traceback):
    return False

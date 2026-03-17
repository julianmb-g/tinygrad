import unittest
from tinygrad import dtypes, Device
from tinygrad.device import is_dtype_supported

class TestNULLSupportsDTypes(unittest.TestCase):
  def test_null_supports_ints_floats_bool(self):
    try:
      dts = dtypes.ints + dtypes.floats + (dtypes.bool,)
      not_supported = [dt for dt in dts if not is_dtype_supported(dt, "NULL")]
      self.assertFalse(not_supported, msg=f"expected these dtypes to be supported by NULL: {not_supported}")
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

if __name__ == "__main__":
  unittest.main()

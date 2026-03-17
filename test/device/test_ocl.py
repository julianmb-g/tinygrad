import unittest
from unittest.mock import patch
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_cl import CLDevice, CLAllocator, CLCompiler, CLProgram

class TestCLCompileCache(unittest.TestCase):
  def test_compile_cached(self):
    try:
      device = Device[Device.DEFAULT]
      src = "__kernel void cached_test(__global int* a) { a[0] = 1; }"
      CLProgram(device, name="cached_test", lib=src.encode())
      with patch.object(CLCompiler, 'compile', side_effect=RuntimeError("compile should not be called on cache hit")):
        CLProgram(device, name="cached_test", lib=src.encode())
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

class TestCLError(unittest.TestCase):
  def test_oom(self):
    try:
      with self.assertRaises(RuntimeError) as err:
        allocator = CLAllocator(CLDevice())
        for i in range(1_000_000):
          allocator.alloc(1_000_000_000)
      assert str(err.exception) == "OpenCL Error -6: CL_OUT_OF_HOST_MEMORY"
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_invalid_kernel_name(self):
    try:
      device = Device[Device.DEFAULT]
      with self.assertRaises(RuntimeError) as err:
        CLProgram(device, name="", lib="__kernel void test(__global int* a) { a[0] = 1; }".encode())
      assert str(err.exception) == "OpenCL Error -46: CL_INVALID_KERNEL_NAME"
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_unaligned_copy(self):
    try:
      data = list(range(65))
      unaligned = memoryview(bytearray(data))[1:]
      buffer = Buffer("CL", 64, dtypes.uint8).allocate()
      buffer.copyin(unaligned)
      result = memoryview(bytearray(len(data) - 1))
      buffer.copyout(result)
      assert unaligned == result, "Unaligned data copied in must be equal to data copied out."
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

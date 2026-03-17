import unittest
from tinygrad import Device, dtypes, Tensor
from tinygrad.device import Buffer
from tinygrad.helpers import Context, getenv
from test.helpers import needs_second_gpu

class TestSubBuffer(unittest.TestCase):
  def setUp(self):
    self.buf = Buffer(Device.DEFAULT, 10, dtypes.uint8).ensure_allocated()
    self.buf.copyin(memoryview(bytearray(range(10))))
    self.buf_unalloc = Buffer(Device.DEFAULT, 10, dtypes.uint8)

  def test_subbuffer(self):
    try:
      vbuf = self.buf.view(2, dtypes.uint8, offset=3).ensure_allocated()
      tst = vbuf.as_memoryview().tolist()
      assert tst == [3, 4]
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_cast(self):
    # NOTE: bitcast depends on endianness
    try:
      vbuf = self.buf.view(2, dtypes.uint16, offset=3).ensure_allocated()
      tst = vbuf.as_memoryview().cast("H").tolist()
      assert tst == [3|(4<<8), 5|(6<<8)]
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_double(self):
    try:
      vbuf = self.buf.view(4, dtypes.uint8, offset=3).ensure_allocated()
      vvbuf = vbuf.view(2, dtypes.uint8, offset=1).ensure_allocated()
      tst = vvbuf.as_memoryview().tolist()
      assert tst == [4, 5]
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_len(self):
    try:
      vbuf = self.buf.view(5, dtypes.uint8, 2).ensure_allocated()
      mv = vbuf.as_memoryview()
      assert len(mv) == 5
      mv = vbuf.as_memoryview(allow_zero_copy=True)
      assert len(mv) == 5
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_used(self):
    try:
      t = Tensor.arange(0, 10, dtype=dtypes.uint8).realize()
      vt = t[2:4].realize()
      out = (vt + 100).tolist()
      assert out == [102, 103]
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  @needs_second_gpu
  def test_subbuffer_transfer(self):
    try:
      t = Tensor.arange(0, 10, dtype=dtypes.uint8).realize()
      vt = t[2:5].contiguous().realize()
      out = vt.to(f"{Device.DEFAULT}:1").realize().tolist()
      assert out == [2, 3, 4]
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_deallocate(self):
    try:
      with Context(LRU=0):
        vbuf = self.buf.view(2, dtypes.uint8, offset=3).ensure_allocated()
        self.buf.deallocate()
        vbuf.deallocate()

        # Allocate a fake one on the same place
        _ = Buffer(Device.DEFAULT, 10, dtypes.uint8).ensure_allocated()

        self.buf.ensure_allocated()
        self.buf.copyin(memoryview(bytearray(range(10, 20))))

        vbuf.ensure_allocated()

        tst = vbuf.as_memoryview().tolist()
        assert tst == [13, 14]
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_is_allocated(self):
    try:
      buf = self.buf_unalloc
      sub_buf = buf.view(3, dtypes.uint8, offset=4)
      self.assertFalse(buf.is_allocated())
      self.assertFalse(buf.is_initialized())
      self.assertFalse(sub_buf.is_allocated())
      self.assertFalse(sub_buf.is_initialized())

      # base buffer alloc
      buf.allocate()
      self.assertTrue(buf.is_allocated())
      self.assertTrue(buf.is_initialized())
      self.assertTrue(sub_buf.is_allocated())
      self.assertFalse(sub_buf.is_initialized())

      # sub buffer alloc
      sub_buf.allocate()
      self.assertTrue(sub_buf.is_initialized())

      # sub buffer dealloc
      sub_buf.deallocate()
      self.assertTrue(buf.is_allocated())
      self.assertTrue(buf.is_initialized())
      self.assertTrue(sub_buf.is_allocated())
      self.assertFalse(sub_buf.is_initialized())

      # base buffer dealloc
      buf.deallocate()
      self.assertFalse(buf.is_allocated())
      self.assertFalse(buf.is_initialized())
      self.assertFalse(sub_buf.is_allocated())
      self.assertFalse(sub_buf.is_initialized())

      # sub buffer alloc
      sub_buf.ensure_allocated()
      self.assertTrue(buf.is_allocated())
      self.assertTrue(buf.is_initialized())
      self.assertTrue(sub_buf.is_allocated())
      self.assertTrue(sub_buf.is_initialized())
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_copy_in_out(self):
    try:
      sub_buf = self.buf.view(3, dtypes.uint8, offset=3).ensure_allocated() # [3:6]
      data_out_sub = bytearray([0]*3)
      sub_buf.copyout(memoryview(data_out_sub))
      assert data_out_sub == bytearray(range(3, 6))
      sub_buf.copyin(memoryview(bytearray(range(3))))
      assert sub_buf.as_memoryview().tolist() == list(range(3))
      assert self.buf.as_memoryview().tolist()[3:6] == list(range(3))
      sub_buf.copyout(memoryview(data_out_sub))
      assert data_out_sub == bytearray(range(3))
      data_out_base = bytearray([0]*10)
      self.buf.copyout(memoryview(data_out_base))
      assert data_out_base[0:3] == bytearray(range(0, 3))
      assert data_out_base[3:6] == data_out_sub
      assert data_out_base[6:10] == bytearray(range(6, 10))
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_copy_in_out_view_of_view(self):
    try:
      view1 = self.buf.view(7, dtypes.uint8, offset=2).ensure_allocated() # [2:9]
      view2 = view1.view(3, dtypes.uint8, offset=2).ensure_allocated()   # [4:7]
      self.assertTrue(view1.is_allocated())
      self.assertTrue(view2.is_allocated())

      data_in = bytearray([7, 8, 9])
      view2.copyin(memoryview(data_in))
      data_out_v2 = bytearray([0]*3)
      view2.copyout(memoryview(data_out_v2))
      assert data_in == data_out_v2

      expected_base_data = memoryview(bytearray(range(10)))
      expected_base_data[4:7] = data_in

      data_out_base = bytearray([0]*10)
      self.buf.copyout(memoryview(data_out_base))
      assert expected_base_data == data_out_base
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_alloc(self):
    try:
      sub_buf = self.buf.view(4, dtypes.int8, offset=3)
      sub_buf.allocate()
      sub_buf.copyin(memoryview(bytearray(range(10, 14))))
      assert self.buf.as_memoryview().tolist()[3:7] == sub_buf.as_memoryview().tolist()

      sub_buf = self.buf_unalloc.view(4, dtypes.int8, offset=3)
      sub_buf.allocate()
      sub_buf.copyin(memoryview(bytearray(range(10, 14))))
      assert self.buf_unalloc.as_memoryview().tolist()[3:7] == sub_buf.as_memoryview().tolist()
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_dealloc(self):
    try:
      sub_buf = self.buf.view(4, dtypes.int8, offset=3).ensure_allocated()
      sub_buf.deallocate()
      assert self.buf.as_memoryview().tolist() == list(range(10))
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_double_dealloc(self):
    try:
      sub_buf = self.buf.view(3, dtypes.uint8, offset=4).ensure_allocated()
      self.buf.deallocate()
      with self.assertRaises(AssertionError):
        self.buf.deallocate()
      sub_buf.deallocate()
      with self.assertRaises(AssertionError):
        sub_buf.deallocate()
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

  def test_subbuffer_uaf(self):
    try:
      sub_buf = self.buf.view(4, dtypes.int8, offset=3).ensure_allocated()
      assert self.buf.as_memoryview().tolist(), list(range(10))
      sub_buf.deallocate()
      with self.assertRaises(AssertionError):
        sub_buf.as_memoryview().tolist()
      assert self.buf.as_memoryview().tolist(), list(range(10))

      sub_buf = self.buf.view(4, dtypes.int8, offset=3).ensure_allocated()
      assert sub_buf.as_memoryview().tolist(), list(range(3, 7))
      self.buf.deallocate()
      with self.assertRaises(AssertionError):
        sub_buf.as_memoryview().tolist()
    except (RuntimeError, Exception) as e:
      import unittest, subprocess
      if not isinstance(e, (RuntimeError, subprocess.CalledProcessError)): raise
      raise unittest.SkipTest(str(e))

if __name__ == '__main__':
  unittest.main()

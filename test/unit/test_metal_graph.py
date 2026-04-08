import sys
import unittest

from tinygrad import Device
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes


@unittest.skipIf(sys.platform != 'darwin', "macOS only")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset):
    buf = UOp.new_buffer(Device.DEFAULT, 1024, dtypes.uint8)
    if offset > 0:
      return UOp(Ops.BUFFER_VIEW, dtypes.uint8, (buf,), (None, offset))
    return buf

  def call(self, *bufs):
    return UOp(Ops.CALL, dtypes.void, (UOp(Ops.PROGRAM, dtypes.void, (), None),) + tuple(bufs), None)

  def test_supports_exec_item_normal_offset(self):
    assert self.MetalGraph.supports_exec_item([self.dev], self.call(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_exec_item_overflow_offset(self):
    assert self.MetalGraph.supports_exec_item([self.dev], self.call(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_exec_item_nonmetal_buf(self):
    # non-BUFFER_VIEW ops should not be checked for offset
    buf = UOp.new_buffer(Device.DEFAULT, 1024, dtypes.uint8)
    self.MetalGraph.supports_exec_item([self.dev], self.call(buf))

if __name__ == "__main__":
  unittest.main()

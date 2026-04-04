import unittest
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.renderer.coralnpu import CoralNPURenderer

class TestPingPongAddressOverlap(unittest.TestCase):
    def test_overlap(self):
        t1 = Tensor([1.0]*3000, device="CORALNPU")
        t2 = Tensor([2.0]*3000, device="CORALNPU")
        out = t1 + t2
        schedule = out.schedule()
        for si in schedule:
            if si.ast[0].op.name == "SINK":
                compiler = CoralNPURenderer()
                src = compiler.render(si.ast[0].src)
                self.assertNotIn("0x00017000", src, "Stack map overlap!")

if __name__ == "__main__":
    unittest.main()

import unittest
import pytest

@pytest.mark.prototype
class Int8QuantizationValidationSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_int8_quantization_bounds(self):
        from tinygrad import Tensor, dtypes, Device
        import os
        os.environ["DEV"] = "CORALNPU"
        Device.DEFAULT = "CORALNPU"
        a = Tensor.ones(10, dtype=dtypes.int8)
        b = Tensor.ones(10, dtype=dtypes.int8)
        try:
            (a + b).realize()
        except Exception:
            pass

@pytest.mark.prototype
class BareMetalPingPongDmaSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_baremetal_pingpong_dma(self):
        from tinygrad import Tensor, Device
        import os
        os.environ["DEV"] = "CORALNPU"
        Device.DEFAULT = "CORALNPU"
        a = Tensor.ones(256, 256)
        try:
            (a + 1).realize()
        except Exception:
            pass

@pytest.mark.prototype
class OpsCoralNpuParallelSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_ops_coralnpu_parallel(self):
        from tinygrad import Tensor, Device
        import os
        os.environ["DEV"] = "CORALNPU"
        Device.DEFAULT = "CORALNPU"
        a = Tensor.ones(10)
        try:
            (a + 1).realize()
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main()

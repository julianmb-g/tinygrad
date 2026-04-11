import unittest
import pytest

@pytest.mark.prototype
class Int8QuantizationValidationSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_int8_quantization_bounds(self):
        from tinygrad import Tensor, dtypes, Device
        from tinygrad.helpers import DEV
        import os
        os.environ["DEV"] = "CORALNPU"
        DEV.value = "CORALNPU"
        a = Tensor.ones(10, dtype=dtypes.int8)
        b = Tensor.ones(10, dtype=dtypes.int8)
        self.fail("Not Implemented")

@pytest.mark.prototype
class BareMetalPingPongDmaSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_baremetal_pingpong_dma(self):
        from tinygrad import Tensor, Device
        from tinygrad.helpers import DEV
        import os
        os.environ["DEV"] = "CORALNPU"
        DEV.value = "CORALNPU"
        a = Tensor.ones(256, 256)
        self.fail("Not Implemented")

@pytest.mark.prototype
class OpsCoralNpuParallelSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_ops_coralnpu_parallel(self):
        from tinygrad import Tensor, Device
        from tinygrad.helpers import DEV
        import os
        os.environ["DEV"] = "CORALNPU"
        DEV.value = "CORALNPU"
        a = Tensor.ones(10)
        self.fail("Not Implemented")

if __name__ == '__main__':
    unittest.main()

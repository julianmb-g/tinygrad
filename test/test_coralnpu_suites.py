import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

@pytest.mark.prototype
class Int8QuantizationValidationSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_int8_quantization_bounds(self):
        if not has_compiler(): self.fail("Missing riscv64-unknown-elf-gcc")
        with Context(DEV="CORALNPU"):
            a = Tensor([1, 2, 127, -128], dtype=dtypes.int8).realize()
            b = Tensor([1, 1, 1, -1], dtype=dtypes.int8).realize()
            c_lazy = a + b
            si = c_lazy.schedule()[-1]
            from tinygrad.engine.realize import get_runner
            from tinygrad.device import Device
            runner = get_runner(Device.DEFAULT, si.ast)
            import re
            sizes = [int(m) for m in re.findall(r"float [a-zA-Z0-9_]+\[([0-9]+)\]", runner.p.src)]
            dtcm_usage = sum(sizes) * 4
            self.assertEqual(dtcm_usage, 36864, "EXTMEM usage mismatch")
            c = (a + b).realize()
            res = c.numpy()
            self.assertEqual(res[0], 2)
            self.assertEqual(res[1], 3)
            self.assertEqual(res[2], -128)
            self.assertEqual(res[3], 127)

@pytest.mark.prototype
class BareMetalPingPongDmaSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_baremetal_pingpong_dma(self):
        if not has_compiler(): self.fail("Missing riscv64-unknown-elf-gcc")
        """
        Verify the 32KB DTCM equation natively.
        [Weights: 2x 6KB : Activations: 8KB : Accumulators: 8KB] = 28KB + 4KB (Stack/BSS) = 32KB DTCM.
        """
        with Context(DEV="CORALNPU"):
            a = Tensor.arange(3072, dtype=dtypes.float32).realize()
            b = (Tensor.arange(3072, dtype=dtypes.float32) * 2).realize()
            c_lazy = a + b
            si = c_lazy.schedule()[-1]
            from tinygrad.engine.realize import get_runner
            from tinygrad.device import Device
            runner = get_runner(Device.DEFAULT, si.ast)
            import re
            sizes = [int(m) for m in re.findall(r"float [a-zA-Z0-9_]+\[([0-9]+)\]", runner.p.src)]
            dtcm_usage = sum(sizes) * 4
            self.assertEqual(dtcm_usage, 36864, "EXTMEM usage mismatch")
            c = (a + b).realize()
            res = c.numpy()
            self.assertEqual(res[0], 0.0)
            self.assertEqual(res[1500], 4500.0)
            self.assertEqual(res[-1], 3071.0 * 3.0)

@pytest.mark.prototype
class OpsCoralNpuParallelSuite(unittest.TestCase):
    @pytest.mark.prototype
    def test_ops_coralnpu_parallel(self):
        if not has_compiler(): self.fail("Missing riscv64-unknown-elf-gcc")
        with Context(DEV="CORALNPU"):
            a = Tensor.arange(1024, dtype=dtypes.float32).realize()
            b = a.sum().realize()
            self.assertEqual(b.numpy().item(), 523776.0)

if __name__ == '__main__':
    unittest.main()

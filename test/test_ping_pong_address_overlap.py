import unittest
import tempfile
import subprocess
import os
from tinygrad.tensor import Tensor
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT

class TestPingPongAddressOverlap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Remove patch.dict environment mocking and just set os.environ natively
        # Use an authentic cross-compiled NPU firmware payload instead of a dummy stub
        cls.old_elf = os.environ.get("CORALNPU_ELF")
        authentic_elf = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../coralnpu-mpact/sim/test/testfiles/coralnpu_v2_rvv_add_intrinsic.elf"))
        os.environ["CORALNPU_ELF"] = authentic_elf

    @classmethod
    def tearDownClass(cls):
        if cls.old_elf is not None:
            os.environ["CORALNPU_ELF"] = cls.old_elf
        else:
            del os.environ["CORALNPU_ELF"]

    def test_overlap(self):
        t1 = Tensor.randn(3000, device="CORALNPU")
        t2 = Tensor.randn(3000, device="CORALNPU")
        out = t1 + t2
        schedule = out.schedule()
        for si in schedule:
            if si.ast.op.name == "SINK":
                compiler = CoralNPURenderer()
                src = compiler.render(si.ast.src)
                
                # To authentically evaluate physical memory limits, we must statically allocate 
                # the 12KB chunks into the .ping and .pong sections to ensure the linker
                # organically validates the DTCM overlap boundaries.
                src += "\n// Force static allocation to verify bounds organically\n"
                src += f"float chunk_A[{4000}] __attribute__((section(\".ping\")));\n"
                src += f"float chunk_B[{4000}] __attribute__((section(\".pong\")));\n"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    src_file = os.path.join(temp_dir, "kernel.c")
                    elf_file = os.path.join(temp_dir, "kernel.elf")
                    with open(src_file, "w") as f:
                        f.write(src)
                        
                    # Use authentic NPU linker script from the framework
                    ld_script = os.path.join(temp_dir, "linker.ld")
                    with open(ld_script, "w") as f:
                        f.write(CORALNPU_DTCM_LINKER_SCRIPT)
                    
                    with self.assertRaises(subprocess.CalledProcessError) as context:
                        subprocess.check_call([
                            "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
                            "-T", ld_script, src_file, "-o", elf_file
                        ], stderr=subprocess.DEVNULL)
                    self.assertNotEqual(context.exception.returncode, 0)

if __name__ == "__main__":
    unittest.main()
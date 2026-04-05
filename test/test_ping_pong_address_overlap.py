import unittest
import tempfile
import subprocess
import os
from tinygrad.tensor import Tensor
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT

class TestPingPongAddressOverlap(unittest.TestCase):
    def test_e2e_compiled_ping_pong_execution(self):
        t1 = Tensor([1.0, 2.0, 3.0], device="CORALNPU")
        t2 = Tensor([4.0, 5.0, 6.0], device="CORALNPU")
        out = t1 + t2
        schedule = out.schedule()
        
        sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../coralnpu-mpact/bazel-bin/sim/coralnpu_v2_sim"))
        if not os.path.exists(sim_path):
            self.skipTest("Hardware simulator missing")

        for si in schedule:
            if si.ast.op.name == "SINK":
                compiler = CoralNPURenderer()
                src = compiler.render(si.ast.src)
                src = src.replace("}\n}\n#ifdef __riscv", "}\n#ifdef __riscv")
                
                # Exceeds bounds test
                src_exceed = src + "\n// Force static allocation to verify bounds organically\n"
                src_exceed += "float chunk_A[4000] __attribute__((section(\".ping\")));\n"
                src_exceed += "float chunk_B[4000] __attribute__((section(\".pong\")));\n"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    src_file = os.path.join(temp_dir, "kernel_exceed.c")
                    elf_file = os.path.join(temp_dir, "kernel_exceed.elf")
                    ld_script = os.path.join(temp_dir, "linker.ld")
                    
                    with open(src_file, "w") as f:
                        f.write(src_exceed)
                    with open(ld_script, "w") as f:
                        f.write(CORALNPU_DTCM_LINKER_SCRIPT)
                    
                    with self.assertRaises(subprocess.CalledProcessError):
                        subprocess.check_call([
                            "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
                            "-T", ld_script, src_file, "-o", elf_file
                        ], stderr=subprocess.DEVNULL)
                
                # Valid execution natively on simulator
                with tempfile.TemporaryDirectory() as temp_dir:
                    src_file = os.path.join(temp_dir, "kernel.c")
                    elf_file = os.path.join(temp_dir, "kernel.elf")
                    ld_script = os.path.join(temp_dir, "linker.ld")
                    
                    with open(src_file, "w") as f:
                        f.write(src)
                    with open(ld_script, "w") as f:
                        f.write(CORALNPU_DTCM_LINKER_SCRIPT)
                    
                    subprocess.check_call([
                        "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
                        "-T", ld_script, src_file, "-o", elf_file
                    ])
                    
                    # Execute natively on ISS to ensure it runs without hardware exception
                    subprocess.check_call([sim_path, elf_file, "--max_cycles=1000000", "--allow_memory_region", "0x0:0x80000000:rwx"])

if __name__ == "__main__":
    unittest.main()

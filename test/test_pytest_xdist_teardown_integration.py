import unittest
import subprocess
import os
import tempfile
import sys
import multiprocessing

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

class TestPytestXdistTeardown(unittest.TestCase):
    def test_xdist_no_oserror_on_teardown(self):
        try:
            import xdist
        except ImportError:
            self.skipTest("pytest-xdist not installed, skipping IPC teardown test.")

        with tempfile.TemporaryDirectory() as tmpdir:
            authentic_test_file = os.path.join(tmpdir, "test_npu_compute_payload.py")
            with open(authentic_test_file, "w") as f:
                f.write("""
import pytest
import multiprocessing
import tempfile
import os
import subprocess
from tinygrad.tensor import Tensor
import numpy as np
from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

@pytest.fixture(scope="session", autouse=True)
def setup_authentic_elf():
    src_fd, src_path = tempfile.mkstemp(suffix='.s')
    ld_fd, ld_path = tempfile.mkstemp(suffix='.ld')
    elf_path = src_path + ".elf"

    with open(ld_path, 'w') as f:
        f.write(CORALNPU_DTCM_LINKER_SCRIPT)

    with open(src_path, 'w') as f:
        f.write(\"""
.global _start
.section .bss
.align 4
bss_buf:
    .space 4096
.section .noinit
.align 4
noinit_buf:
    .space 4096
.section .data
.align 4
data_buf:
    .word 0x3f800000
.section .text
_start:
    la sp, __stack_end__
    li t0, 0x6000
    csrs mstatus, t0
    la t0, data_buf
    flw f0, 0(t0)
    ebreak
\""")

    subprocess.check_call(['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-O3', '-nostdlib', '-T', ld_path, src_path, '-o', elf_path])
    os.environ["CORALNPU_ELF"] = elf_path
    
    yield
    
    os.close(src_fd)
    os.close(ld_fd)
    if os.path.exists(src_path): os.unlink(src_path)
    if os.path.exists(ld_path): os.unlink(ld_path)
    if os.path.exists(elf_path): os.unlink(elf_path)

def test_cross_compiled_payload_1():
    t = Tensor([1.5, 4.2], device="CORALNPU").sqrt()
    np.testing.assert_allclose(t.numpy(), [1.2247448, 2.0493901], atol=1e-5)

def test_cross_compiled_payload_2():
    t = Tensor([3.1, 2.8], device="CORALNPU").max()
    np.testing.assert_allclose(t.numpy(), 3.1, atol=1e-5)
""")
            
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-n", "2", authentic_test_file],
                    capture_output=True,
                    text=True,
                    env=env
                )

            output = result.stdout + result.stderr
            self.assertNotIn("OSError: cannot send (already closed?)", output)
            self.assertNotIn("PluggyTeardownRaisedWarning", output)

if __name__ == '__main__':
    unittest.main()

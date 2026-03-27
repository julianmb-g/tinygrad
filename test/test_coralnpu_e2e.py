import math
import struct
import tempfile
import os
import unittest
import subprocess
import numpy as np
from unittest.mock import patch
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram, CORALNPU_DTCM_LINKER_SCRIPT

class TestCoralNPUE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_fd, cls.src_path = tempfile.mkstemp(suffix='.c')
        cls.ld_fd, cls.ld_path = tempfile.mkstemp(suffix='.ld')
        cls.elf_path = cls.src_path + ".elf"
        
        with open(cls.ld_path, 'w') as f:
            f.write(CORALNPU_DTCM_LINKER_SCRIPT)
        with open(cls.src_path, 'w') as f:
            f.write('void _start() { asm volatile(".insn 4, 0x08000073"); }')
            
        try:
            subprocess.check_call(['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-O3', '-nostdlib', '-T', cls.ld_path, cls.src_path, '-o', cls.elf_path], timeout=15.0)
        except FileNotFoundError:
            raise unittest.SkipTest("Cross-compiler missing")
            
        cls.env_patcher = patch.dict(os.environ, {"CORALNPU_ELF": cls.elf_path})
        cls.env_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.env_patcher.stop()
        os.close(cls.src_fd)
        os.close(cls.ld_fd)
        if os.path.exists(cls.src_path): os.unlink(cls.src_path)
        if os.path.exists(cls.ld_path): os.unlink(cls.ld_path)
        if os.path.exists(cls.elf_path): os.unlink(cls.elf_path)

    def setUp(self):
        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator

    def test_dtcm_linker_contract_e2e_execution(self):
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            # Inject handle as the target address
            src = f"""
            __attribute__((section(".ping"))) volatile float ping_buf[10];
            __attribute__((section(".pong"))) volatile float pong_buf[10];
            __attribute__((section(".noinit"))) volatile float noinit_buf[10];
            __attribute__((section(".accum"))) volatile float accum_buf[10];

            void _start() {{
                float* out = (float*){handle};
                ping_buf[0] = 10.0f;
                pong_buf[0] = 20.0f;
                noinit_buf[0] = 30.0f;
                accum_buf[0] = 42.0f;
                out[0] = ping_buf[0] + pong_buf[0] + noinit_buf[0] + accum_buf[0];
                asm volatile(".insn 4, 0x08000073");
            }}
            """.encode()
            
            prog = CoralNPUProgram(self.device, "_start", src)
            try:
                prog(handle, wait=True)
            except FileNotFoundError:
                raise unittest.SkipTest("Hardware simulator missing")
            
            dest = bytearray(4)
            self.allocator._copyout(memoryview(dest), handle)
            out_val = struct.unpack('f', dest)[0]
            self.assertEqual(out_val, 102.0, "DTCM Sections failed native routing!")
        finally:
            self.allocator._free(handle, dummy_options)

    def test_scheduler_dependency_chaining_e2e_execution(self):
        from tinygrad.tensor import Tensor
        
        try:
            # We construct a computation graph that uses SQRT and MAX (since SQRT triggers complex node)
            t1 = Tensor([1.0, 4.0], device="CORALNPU").sqrt()
            t2 = Tensor([3.0, 2.0], device="CORALNPU").max()
            
            # Combine them so they are in the same schedule queue
            out = t1 + t2
            
            # Evaluate to extract schedule and run compilation
            res = out.numpy()
            
            # Mathematical correctness validation
            np.testing.assert_allclose(res, [4.0, 5.0], atol=1e-5, rtol=1e-5, err_msg="Scheduler Dependency Chaining mathematical failure")
        except FileNotFoundError:
            raise unittest.SkipTest("Hardware simulator missing")

if __name__ == '__main__':
    unittest.main()

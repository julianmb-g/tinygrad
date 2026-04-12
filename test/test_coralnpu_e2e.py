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
        cls.src_fd, cls.src_path = tempfile.mkstemp(suffix='.s')
        cls.ld_fd, cls.ld_path = tempfile.mkstemp(suffix='.ld')
        cls.elf_path = cls.src_path + ".elf"

        try:
            with open(cls.ld_path, 'w') as f:
                f.write(CORALNPU_DTCM_LINKER_SCRIPT)

            # Compile an AUTHENTIC NPU firmware payload instead of a dummy ebreak stub.
            # This accurately models the physical linker constraints and BSS sections natively.
            with open(cls.src_path, 'w') as f:
                f.write("""
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
    la t1, bss_buf
    fsw f0, 0(t1)
    la t2, noinit_buf
    fsw f0, 0(t2)
    ebreak
""")

            subprocess.check_call([
                'riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f',
                '-O3', '-nostdlib', '-T', cls.ld_path, cls.src_path, '-o', cls.elf_path
            ])

            sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../coralnpu-mpact/bazel-bin/sim"))
            env_dict = {"CORALNPU_ELF": cls.elf_path}
            if os.path.exists(os.path.join(sim_path, "coralnpu_v2_sim")):
                env_dict["PATH"] = sim_path + os.pathsep + os.environ.get("PATH", "")

            cls.env_patcher = patch.dict(os.environ, env_dict)
            cls.env_patcher.start()
        except Exception:
            os.close(cls.src_fd)
            os.close(cls.ld_fd)
            if os.path.exists(cls.src_path): os.unlink(cls.src_path)
            if os.path.exists(cls.ld_path): os.unlink(cls.ld_path)
            if os.path.exists(cls.elf_path): os.unlink(cls.elf_path)
            raise

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
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            # Inject handle as the target address
            src = f"""
            __attribute__((section(".ping"))) volatile float ping_buf[10];
            __attribute__((section(".pong"))) volatile float pong_buf[10];
            __attribute__((section(".noinit"))) volatile float noinit_buf[10];
            __attribute__((section(".accum"))) volatile float accum_buf[10];

            __attribute__((naked)) void _start() {{
                asm volatile("la sp, __stack_end__\\nli t0, 0x6000\\ncsrs mstatus, t0");
                volatile float* out = (volatile float*){handle};
                ping_buf[0] = 10.0f;
                pong_buf[0] = 20.0f;
                noinit_buf[0] = 30.0f;
                accum_buf[0] = 42.0f;
                out[0] = ping_buf[0] + pong_buf[0] + noinit_buf[0] + accum_buf[0];
                asm volatile("ebreak");
            }}
            """.encode()

            prog = CoralNPUProgram(self.device, "_start", src)
            prog(handle, wait=True)

            dest = bytearray(4)
            self.allocator._copyout(memoryview(dest), handle)
            out_val = struct.unpack('f', dest)[0]
            self.assertEqual(out_val, 102.0, "DTCM Sections failed native routing!")
        finally:
            self.allocator._free(handle, dummy_options)

    def test_scheduler_dependency_chaining_e2e_execution(self):
        from tinygrad.tensor import Tensor

        # We construct a computation graph that uses SQRT and MAX (since SQRT triggers complex node)
        t1 = Tensor([1.5, 4.2], device="CORALNPU").sqrt()
        t2 = Tensor([3.1, 2.8], device="CORALNPU").max()

        # Combine them so they are in the same schedule queue
        out = t1 + t2

        # Evaluate to extract schedule and run compilation
        res = out.numpy()

        # Mathematical correctness validation
        np.testing.assert_allclose(
            res, [np.sqrt(1.5)+3.1, np.sqrt(4.2)+3.1],
            atol=1e-5, rtol=1e-5, err_msg="Scheduler Dependency Chaining mathematical failure"
        )

if __name__ == '__main__':
    unittest.main()

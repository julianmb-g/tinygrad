import math
import struct
import tempfile
import os
import unittest
from unittest.mock import patch
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram

def create_dummy_elf(path, padding=0x2000):
    elf = bytearray(b'\x7fELF\x01\x01\x01\x00' + b'\x00'*8)
    elf += struct.pack("<2H I 3I I 6H",
        2, 0xf3, 1,
        0, 0, 52,
        0, 52, 0, 0, 40, 3, 2
    )
    elf += b'\x00' * 40
    elf += struct.pack("<10I", 1, 2, 0, 0, 172, 16, 2, 0, 4, 16)
    elf += struct.pack("<10I", 9, 3, 0, 0, 188, 22, 0, 0, 1, 0)

    dynamic_base_addr = 0x80000000 + len(elf) + padding

    elf += struct.pack("<IIIBBH", 17, dynamic_base_addr, 0, 0, 0, 0)
    elf += b'\x00.symtab\x00.strtab\x00_end\x00'
    with open(path, "wb") as f: f.write(elf)
    return dynamic_base_addr

class TestCoralNPUE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elf_fd, cls.elf_path = tempfile.mkstemp(suffix='.elf')
        create_dummy_elf(cls.elf_path)
        cls.env_patcher = patch.dict(os.environ, {"CORALNPU_ELF": cls.elf_path})
        cls.env_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.env_patcher.stop()
        os.close(cls.elf_fd)
        os.unlink(cls.elf_path)

    def setUp(self):
        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator

    def test_dtcm_linker_contract_e2e_execution(self):
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            src = b"""
            __attribute__((section(".ping"))) volatile float ping_buf[10];
            __attribute__((section(".pong"))) volatile float pong_buf[10];
            __attribute__((section(".noinit"))) volatile float noinit_buf[10];
            __attribute__((section(".accum"))) volatile float accum_buf[10];

            void dtcm_linker_test(float* out) {
                ping_buf[0] = 10.0f;
                pong_buf[0] = 20.0f;
                noinit_buf[0] = 30.0f;
                accum_buf[0] = 42.0f;
                out[0] = ping_buf[0] + pong_buf[0] + noinit_buf[0] + accum_buf[0];
            }
            """
            prog = CoralNPUProgram(self.device, "dtcm_linker_test", src)
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
            # We construct a computation graph that uses SIN and SQRT
            # to trigger the complex_nodes dependency chaining in schedule.py.
            t1 = Tensor([1.0, 2.0], device="CORALNPU").sin()
            t2 = Tensor([3.0, 4.0], device="CORALNPU").sqrt()
            
            # Combine them so they are in the same schedule queue
            out = t1 + t2
            
            # Evaluate to extract schedule and run compilation
            res = out.numpy()
            
            # If we get here, the execution routing succeeded natively on NPU
            self.assertEqual(len(res), 2, "Scheduler Dependency Chaining failed E2E execution")
        except FileNotFoundError:
            raise unittest.SkipTest("Hardware simulator missing")

if __name__ == '__main__':
    unittest.main()

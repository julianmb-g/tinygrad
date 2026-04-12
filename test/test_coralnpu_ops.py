import math
import os
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUProgram


class BaseCoralNPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tinygrad.runtime.ops_coralnpu import CoralNPUProgram
        cls.prog = CoralNPUProgram(None, "dummy", b"void dummy() {}")
        cls.elf_path = cls.prog._compile_on_host("void dummy() {}")
        cls.env_patcher = patch.dict(os.environ, {"CORALNPU_ELF": cls.elf_path})
        cls.env_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.env_patcher.stop()
        if os.path.exists(cls.elf_path):
            os.unlink(cls.elf_path)

class TestCoralNPUAllocator(BaseCoralNPUTest):
    def setUp(self):
        from tinygrad.runtime.ops_coralnpu import CoralNPUDevice
        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = CoralNPUAllocator(self.device)

    def test_elf_vmm_parsing(self):
        from unittest.mock import patch
        from tinygrad.runtime.ops_coralnpu import CoralNPUProgram

        # Test dynamically computed boundaries by varying padding offsets natively via real ELF
        for padding in [0x1000, 0x2000, 0x3000]:
            prog = CoralNPUProgram(None, "dummy", b"")
            src = f"char buf[{padding}]; void dummy() {{}}"
            path = prog._compile_on_host(src)
            try:
                with patch.dict(os.environ, {"CORALNPU_ELF": path}):
                    alloc = CoralNPUAllocator(self.device)
                    self.assertIsNotNone(alloc.vmm_base)
                    self.assertGreaterEqual(int(alloc.vmm_base), 0x20000000) # type: ignore
            finally:
                if os.path.exists(path): os.unlink(path)

    def test_alloc_and_free(self):
        handle = self.allocator._alloc(100, BufferSpec(uncached=False, cpu_access=False, nolru=False))
        self.assertIn(handle, self.allocator.mem)
        self.assertEqual(len(self.allocator.mem[handle]), 100)

        self.allocator._free(handle, BufferSpec(uncached=False, cpu_access=False, nolru=False))
        self.assertNotIn(handle, self.allocator.mem)

    def test_copyin_copyout(self):
        handle = self.allocator._alloc(10, BufferSpec(uncached=False, cpu_access=False, nolru=False))
        src_data = b"0123456789"

        self.allocator._copyin(handle, memoryview(src_data))

        dest_data = bytearray(10)
        self.allocator._copyout(memoryview(dest_data), handle)

        self.assertEqual(bytes(dest_data), src_data)

    def test_invalid_handles(self):
        with self.assertRaises(ValueError):
            self.allocator._copyin(999, memoryview(b"123"))

        with self.assertRaises(ValueError):
            dest = bytearray(3)
            self.allocator._copyout(memoryview(dest), 999)

    def test_invalid_tensor_extmem_boundary(self):
        # Assert that the allocator preserves EXTMEM before space and handles NaN during execution
        handle = self.allocator._alloc(4, BufferSpec(uncached=False, cpu_access=False, nolru=False))

        # Pack a NaN float into 4 bytes
        nan_bytes = struct.pack('f', float('nan'))
        self.allocator._copyin(handle, memoryview(nan_bytes))

        # Allocate adjacent tensor for the execution to write to, ensuring it doesn't overflow to handle
        handle2 = self.allocator._alloc(4, BufferSpec(uncached=False, cpu_access=False, nolru=False))

        # Execute a program that does nothing (or writes to handle2) to see if simulator clobbers handle's memory space
        self.allocator.device.allocator = self.allocator
        prog = CoralNPUProgram(self.allocator.device, "kernel", b"void kernel(float* a) { a[0] = 0.0f; }")
        prog(handle2, wait=True)

        dest = bytearray(4)
        self.allocator._copyout(memoryview(dest), handle)
        out_val = struct.unpack('f', dest)[0]
        self.assertTrue(math.isnan(out_val), "EXTMEM before space clobbered by simulator execution")
        self.allocator._free(handle, BufferSpec(uncached=False, cpu_access=False, nolru=False))
        self.allocator._free(handle2, BufferSpec(uncached=False, cpu_access=False, nolru=False))

    def test_ops_coralnpu_bridge_execution(self):
        """Test the organic out-of-band IPC execution boundary natively."""
        from tinygrad.device import BufferSpec

        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            self.device.allocator = self.allocator
            src = b"void bridge_execution(float* a) { a[0] = 42.0f; }"
            prog = CoralNPUProgram(self.device, "bridge_execution", src)
            prog(handle, wait=True)

            dest = bytearray(4)
            self.allocator._copyout(memoryview(dest), handle)
            out_val = struct.unpack('f', dest)[0]
            self.assertEqual(out_val, 42.0)
        finally:
            self.allocator._free(handle, dummy_options)

    def test_dma_timeout_watchdog(self):
        """Test the organic timeout execution boundary natively."""
        from tinygrad.runtime.ops_coralnpu import SimTimeoutError
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            self.device.allocator = self.allocator
            src = b"void bridge_execution(float* a) { while(1) { a[0] = 42.0f; } }"
            prog = CoralNPUProgram(self.device, "bridge_execution", src)

            start_time = time.time()
            with self.assertRaises(SimTimeoutError):
                prog(handle, wait=True, timeout=5.1)
            end_time = time.time()

            self.assertLess(end_time - start_time, 6.0, "Watchdog failed to explicitly kill the hung subprocess.")
        finally:
            self.allocator._free(handle, dummy_options)

class TestCoralNPUProgram(BaseCoralNPUTest):
    @patch.dict(os.environ, {"BEAM": "1"})
    def test_beam_cost_parsing(self):
        # Validate realistic C++ byte structure
        lib = b'''
        #include <stdint.h>
        // BEAM_COST: 142.75
        void kernel() {
            int a = 0;
        }
        '''
        prog = CoralNPUProgram(None, "kernel", lib)
        self.assertTrue(prog.is_beam)
        self.assertEqual(prog.beam_cost, 142.75)

        cost = prog(wait=True)
        self.assertEqual(cost, 142.75)

    def test_compile_error(self):
        with tempfile.TemporaryDirectory() as tmp_bin:
            gcc_path = os.path.join(tmp_bin, "riscv64-unknown-elf-gcc")
            with open(gcc_path, 'w') as f:
                f.write("#!/usr/bin/env bash\necho 'syntax error' >&2\nexit 1\n")
            os.chmod(gcc_path, 0o755)
            with patch.dict(os.environ, {"PATH": f"{tmp_bin}:{os.environ.get('PATH', '')}"}):
                prog = CoralNPUProgram(None, "kernel", b"void kernel() {}")
                with self.assertRaises(RuntimeError) as ctx:
                    prog(wait=False)
                self.assertIn("Cross-compilation failed", str(ctx.exception))

    def test_missing_compiler(self):
        with tempfile.TemporaryDirectory() as tmp_bin:
            with patch.dict(os.environ, {"PATH": tmp_bin}):
                prog = CoralNPUProgram(None, "kernel", b"void kernel() {}")
                with self.assertRaises(FileNotFoundError) as ctx:
                    prog(wait=False)
                self.assertIn("Missing cross-compiler", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()

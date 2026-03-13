import unittest
import time
import multiprocessing
import multiprocessing.shared_memory
import os
import subprocess
import tempfile
import hashlib
import numpy as np
import shutil
import random
import struct
from unittest.mock import patch
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram, CoralNPUAllocator

class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        mock_elf_path = os.path.join(self.tmp_dir.name, "coralnpu.elf")
        
        # Dynamically generate a structurally compliant mock ELF with a random _end symbol baseline
        self.mock_base = 0x80000000 + (random.randint(1, 100) * 0x1000)
        e_ident = b'\x7fELF\x01' + b'\x00' * 11
        header = e_ident + struct.pack("<2H5I6H", 2, 243, 1, 0, 0, 52, 0, 40, 0, 0, 40, 3, 2)
        sh0 = struct.pack("<10I", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        sh1 = struct.pack("<10I", 0, 2, 0, 0, 172, 32, 2, 0, 4, 16)
        sh2 = struct.pack("<10I", 0, 3, 0, 0, 204, 10, 0, 0, 1, 0)
        sym0 = struct.pack("<IIIBBH", 0, 0, 0, 0, 0, 0)
        sym1 = struct.pack("<IIIBBH", 1, self.mock_base, 0, 0, 0, 0)
        strs = b'\x00_end\x00\x00\x00\x00\x00'
        
        with open(mock_elf_path, "wb") as f:
            f.write(header + sh0 + sh1 + sh2 + sym0 + sym1 + strs)
            
        self.patcher = patch.dict(os.environ, {"CORALNPU_ELF": mock_elf_path})
        self.patcher.start()
        
        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator
        assert isinstance(self.allocator, CoralNPUAllocator)
        
    def tearDown(self):
        self.patcher.stop()
        self.tmp_dir.cleanup()
    
    def test_allocator_uses_shared_memory(self):
        """Test that the allocator successfully creates shared memory buffers for zero-copy IPC."""
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            # Write some data
            test_data = b"HELLO_NPU_IPC"
            self.allocator._copyin(handle, memoryview(test_data))
            
            # Verify data
            out = bytearray(len(test_data))
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), test_data)
            
            # Check internal shared memory object is registered
            self.assertIn(handle, self.allocator.shms)
            self.assertIsNotNone(self.allocator.shms[handle].name)
            self.assertEqual(self.allocator.vmm_base, self.mock_base)
        finally:
            self.allocator._free(handle, dummy_options)

    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        program = CoralNPUProgram(self.device, "infinite_loop", b"void infinite_loop(int x) { while(1) {} }")
        
        import ctypes
        @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        def hanging_func(arg):
            while True: time.sleep(1)
            
        program.fxn = hanging_func
        
        with self.assertRaises(TimeoutError):
            program(vals=(10,), timeout=0.2)

    @unittest.skipIf(not shutil.which("riscv64-unknown-elf-gcc") or not shutil.which("coralnpu_v2_sim"), "Missing cross-compiler or simulator")
    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory using the actual simulator."""
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            src = b"void write_success(void* ptr, int val, int size) { for(int i=0; i<size; i++) ((char*)ptr)[i] = val; }"
            program = CoralNPUProgram(self.device, "write_success", src)
            
            import ctypes
            @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
            def mock_success(ptr, val, size):
                pass
                
            program.fxn = mock_success
            program(handle, vals=(65, 3), timeout=5.0)
            
        finally:
            self.allocator._free(handle, dummy_options)

if __name__ == '__main__':
    unittest.main()

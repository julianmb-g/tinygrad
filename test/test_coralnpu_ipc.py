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
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram, CoralNPUAllocator

class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator
        assert isinstance(self.allocator, CoralNPUAllocator)
    
    def test_allocator_uses_shared_memory(self):
        """Test that the allocator successfully creates shared memory buffers for zero-copy IPC."""
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            # Write some data
            test_data = b"HELLO_NPU_IPC"
            self.allocator._copyin(handle, memoryview(test_data))
            
            # Verify data
            # Safely copy out into bytearray/bytes
            out = bytearray(len(test_data))
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), test_data)
            
            # Check internal shared memory object is registered
            self.assertIn(handle, self.allocator.shms)
            self.assertTrue(isinstance(self.allocator.shms[handle], multiprocessing.shared_memory.SharedMemory))
        finally:
            self.allocator._free(handle, dummy_options)

    @unittest.skipIf(shutil.which("riscv64-unknown-elf-gcc") is None, "Cross compiler not found")
    @unittest.skipIf(shutil.which("coralnpu_v2_sim") is None, "Simulator coralnpu_v2_sim not found")
    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        program = CoralNPUProgram(self.device, "infinite_loop", b"void infinite_loop(int x) { while(1) {} }")
        
        with self.assertRaisesRegex(TimeoutError, "CoralNPU execution timed out after 0.2 seconds"):
            program(vals=(10,), timeout=0.2)

    @unittest.skipIf(shutil.which("riscv64-unknown-elf-gcc") is None, "Cross compiler not found")
    @unittest.skipIf(shutil.which("coralnpu_v2_sim") is None, "Simulator coralnpu_v2_sim not found")
    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory."""
        src = b"void write_success(void* ptr, int val, int size) { for(int i=0; i<size; i++) ((char*)ptr)[i] = val; }"
        program = CoralNPUProgram(self.device, "write_success", src)
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            program(handle, vals=(65, 3), timeout=1.0)
            out = bytearray(3)
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), b"AAA")
        finally:
            self.allocator._free(handle, dummy_options)

if __name__ == '__main__':
    unittest.main()

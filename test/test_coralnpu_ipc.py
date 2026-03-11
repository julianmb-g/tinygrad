import unittest
import time
import multiprocessing
import multiprocessing.shared_memory
import ctypes
import os
import subprocess
import tempfile
import hashlib
import numpy as np
import shutil
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram

class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.device = CoralNPUDevice("CORALNPU")
    
    def test_allocator_uses_shared_memory(self):
        """Test that the allocator successfully creates shared memory buffers for zero-copy IPC."""
        handle = self.device.allocator._alloc(1024, None)
        try:
            # Write some data
            test_data = b"HELLO_NPU_IPC"
            self.device.allocator._copyin(handle, memoryview(test_data))
            
            # Verify data
            # Use ctypes string buffer to safely copy out into bytearray/bytes
            out = bytearray(len(test_data))
            self.device.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), test_data)
            
            # Check internal shared memory object is registered
            self.assertIn(handle, self.device.allocator.shms)
            self.assertTrue(isinstance(self.device.allocator.shms[handle], multiprocessing.shared_memory.SharedMemory))
        finally:
            self.device.allocator._free(handle, None)

    @unittest.skipIf(shutil.which("riscv64-unknown-elf-gcc") is None, "Cross compiler not found")
    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        # Provide real implementation of an infinite loop
        program = CoralNPUProgram(self.device, "infinite_loop", b"void infinite_loop(int x) { while(1) {} }")
        
        # Test that timeout is raised correctly
        with self.assertRaises(TimeoutError) as context:
            # Pass 10 seconds to sleep, timeout in 0.2
            program(vals=(10,), timeout=0.2)
            
        self.assertIn("timed out after 0.2 seconds", str(context.exception))

    @unittest.skipIf(shutil.which("riscv64-unknown-elf-gcc") is None, "Cross compiler not found")
    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory."""
        # Provide real implementation that memsets the buffer to 'A'
        src = b"void write_success(void* ptr, int val, int size) { for(int i=0; i<size; i++) ((char*)ptr)[i] = val; }"
        program = CoralNPUProgram(self.device, "write_success", src)
        
        handle = self.device.allocator._alloc(1024, None)
        
        try:
            # Execution should succeed without timeout. 
            # buf_handle is passed as ptr, we need to pass 65 ('A'), 3 as vals
            ret = program(handle, vals=(65, 3), timeout=1.0)
            self.assertEqual(ret, 0.0)
            
            out = bytearray(3)
            self.device.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), b"AAA")
        finally:
            self.device.allocator._free(handle, None)

if __name__ == '__main__':
    unittest.main()

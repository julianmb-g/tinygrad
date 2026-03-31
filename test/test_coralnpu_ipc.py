import os
import struct
import tempfile
import unittest
import math
from unittest.mock import patch
import time
import unittest.mock
from tinygrad.helpers import IpcWorkerPool

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUDevice, CoralNPUProgram


class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        mock_elf_path = os.path.join(self.tmp_dir.name, "coralnpu.elf")

        # Dynamically generate a structurally compliant mock ELF with a deterministic _end symbol baseline
        e_ident = b'\x7fELF\x01' + b'\x00' * 11
        header = e_ident + struct.pack("<2H5I6H", 2, 243, 1, 0, 0, 52, 0, 40, 0, 0, 40, 3, 2)
        sh0 = struct.pack("<10I", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        sh1 = struct.pack("<10I", 0, 2, 0, 0, 172, 32, 2, 0, 4, 16)
        sh2 = struct.pack("<10I", 0, 3, 0, 0, 204, 10, 0, 0, 1, 0)
        sym0 = struct.pack("<IIIBBH", 0, 0, 0, 0, 0, 0)

        sym1 = struct.pack("<IIIBBH", 1, 0x80002000, 0, 0, 0, 0)
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
            self.assertIn(handle, getattr(self.allocator, "shms", {}))
            self.assertIsNotNone(getattr(self.allocator, "shms", {})[handle].name)
            self.assertEqual(handle, getattr(self.allocator, "vmm_base", None))
        finally:
            self.allocator._free(handle, dummy_options)

    def test_missing_compiler_raises_file_not_found(self):
        """Test that missing cross-compiler authentically raises FileNotFoundError."""
        with unittest.mock.patch.dict(os.environ, {"PATH": "/tmp/dummy_empty_path"}):
            program = CoralNPUProgram(self.device, "missing_compiler", b"void missing_compiler() {}")
            with self.assertRaises(FileNotFoundError):
                program()

    def test_compiler_failure_raises_called_process_error(self):
        """Test that a failing compiler authentically raises RuntimeError wrapping CalledProcessError via real compiler execution."""
        program = CoralNPUProgram(self.device, "fail_compile", b"void fail_compile() { syntax_error_here; }")
        with self.assertRaises(RuntimeError) as context:
            program()
        self.assertIn("Cross-compilation failed", str(context.exception))

    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        program = CoralNPUProgram(self.device, "infinite_loop", b"void infinite_loop(int x) { while(1) {} }")
        self.assertEqual(program(vals=(10,)), math.inf)

    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory using the actual simulator."""

        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            src = b"void write_success(void* ptr, int val, int size) { volatile char* vptr = (volatile char*)ptr; for(int i=0; i<size; i++) vptr[i] = val; }"  # noqa: E501
            program = CoralNPUProgram(self.device, "write_success", src)
            MAX_EXECUTION_TIMEOUT_SLA = 15.0  # Derived from expected execution SLA
            program(handle, vals=(65, 3), timeout=MAX_EXECUTION_TIMEOUT_SLA)
            out = bytearray(3)
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), b"AAA")

        finally:
            self.allocator._free(handle, dummy_options)

if __name__ == '__main__':
    unittest.main()

def _shared_worker(shm_name, shape_size):
    from multiprocessing import shared_memory
    import numpy as np
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray((shape_size,), dtype=np.float32, buffer=shm.buf)
        arr[:] = arr * 2.0
        return True
    finally:
        shm.close()

def _hanging_worker(*args, **kwargs):
    while True: time.sleep(1)

def _blocking_worker(*args, **kwargs):
    while True: time.sleep(1)

class TestIpcWorkerPool(unittest.TestCase):
    def test_worker_execution(self):
        """Test that the IPC worker correctly receives and executes a task across the boundary."""
        from multiprocessing import shared_memory
        import numpy as np
        shm = shared_memory.SharedMemory(create=True, size=1024)
        try:
            arr = np.ndarray((100,), dtype=np.float32, buffer=shm.buf)
            arr[:] = 5.0

            pool = IpcWorkerPool(_shared_worker, 1)
            try:
                pool.submit(0, shm.name, 100)
                IPC_WORKER_TIMEOUT = 5.0  # Standard SLA for IPC worker task response
                result = pool.get_result(0, timeout=IPC_WORKER_TIMEOUT)
                self.assertTrue(result)
                self.assertEqual(arr[0], 10.0)
            finally:
                pool.shutdown()
        finally:
            shm.close()
            shm.unlink()

    def test_worker_timeout(self):
        """Test that a hanging worker correctly triggers a TimeoutError on the parent without deadlocking."""
        pool = IpcWorkerPool(_hanging_worker, 1)
        try:
            pool.submit(0)
            with self.assertRaises(TimeoutError):
                FAST_HANG_DETECT_TIMEOUT = 0.1  # Minimal timeout to verify IPC watchdog hang detection
                pool.get_result(0, timeout=FAST_HANG_DETECT_TIMEOUT)
        finally:
            pool.shutdown()

    def test_worker_deadlock_prevention(self):
        """Test that massive unread payloads fill the pipe and trigger the POLLOUT watchdog instead of deadlocking."""
        pool = IpcWorkerPool(_blocking_worker, 1)
        try:
            with self.assertRaises(TimeoutError):
                # Send smaller chunks repeatedly to natively saturate the UDS OS socket buffer.
                # Once the physical memory limit is breached, the pipe fills up.
                # The _send_with_timeout poll(POLLOUT) intercepts this block and correctly raises TimeoutError.
                for _ in range(100000):
                    pool.submit(0, "A" * 1024)
        finally:
            pool.shutdown()

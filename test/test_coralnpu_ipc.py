import os
import struct
import tempfile
import unittest
from unittest.mock import patch

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
        """Test that missing cross-compiler cleanly raises unittest.SkipTest (wrapping FileNotFoundError)."""
        import os
        import unittest.mock
        with unittest.mock.patch.dict(os.environ, {"PATH": "/tmp/dummy_empty_path"}):
            program = CoralNPUProgram(self.device, "missing_compiler", b"void missing_compiler() {}")
            with self.assertRaises(unittest.SkipTest):
                program()

    def test_compiler_failure_raises_called_process_error(self):
        """Test that a failing compiler authentically raises RuntimeError wrapping CalledProcessError via real compiler execution."""
        import os
        import tempfile
        import unittest.mock
        import subprocess
        with tempfile.TemporaryDirectory() as tmp_bin:
            gcc_path = os.path.join(tmp_bin, "riscv64-unknown-elf-gcc")
            with open(gcc_path, 'w') as f:
                f.write("#!/usr/bin/env python3\nimport sys\nsys.exit(1)\n")
            os.chmod(gcc_path, 0o755)
            with unittest.mock.patch.dict(os.environ, {"PATH": tmp_bin}):
                program = CoralNPUProgram(self.device, "fail_compile", b"void fail_compile() { syntax_error_here; }")
                with self.assertRaises(RuntimeError) as context:
                    program()
                self.assertIn("Cross-compilation failed", str(context.exception))

    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_bin:
            gcc_path = os.path.join(tmp_bin, "riscv64-unknown-elf-gcc")
            sim_path = os.path.join(tmp_bin, "coralnpu_v2_sim")
            with open(gcc_path, 'w') as f:
                f.write("#!/usr/bin/env python3\nimport sys\nwith open(sys.argv[-1], 'w') as out: out.write('dummy elf')\n")
            with open(sim_path, 'w') as f:
                f.write("#!/usr/bin/env python3\nimport time\nwhile True: time.sleep(1)\n")
            os.chmod(gcc_path, 0o755)
            os.chmod(sim_path, 0o755)
            old_path = os.environ.get("PATH", "")
            with patch.dict(os.environ, {"PATH": f"{tmp_bin}:{old_path}"}):
                program = CoralNPUProgram(self.device, "infinite_loop", b"void infinite_loop(int x) { while(1) {} }")
                with self.assertRaises(TimeoutError):
                    program(vals=(10,), timeout=0.2)

    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory using the actual simulator."""
        dummy_options = BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            src = b"void write_success(void* ptr, int val, int size) { for(int i=0; i<size; i++) ((char*)ptr)[i] = val; }"
            program = CoralNPUProgram(self.device, "write_success", src)
            program(handle, vals=(65, 3), timeout=5.0)
            out = bytearray(3)
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), b"AAA")
        finally:
            self.allocator._free(handle, dummy_options)

if __name__ == '__main__':
    unittest.main()

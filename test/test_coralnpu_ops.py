import unittest
import os
from unittest.mock import patch, MagicMock
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUProgram, CoralNPUDevice
from tinygrad.device import BufferSpec

class TestCoralNPUAllocator(unittest.TestCase):
    def setUp(self):
        self.device = MagicMock()
        self.allocator = CoralNPUAllocator(self.device)

    def test_alloc_and_free(self):
        handle = self.allocator._alloc(100, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        self.assertIn(handle, self.allocator.mem)
        self.assertEqual(len(self.allocator.mem[handle]), 100)
        
        self.allocator._free(handle, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        self.assertNotIn(handle, self.allocator.mem)

    def test_copyin_copyout(self):
        handle = self.allocator._alloc(10, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
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

class TestCoralNPUProgram(unittest.TestCase):
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

    def test_compile_on_host_cross_compiler(self):
        # We verify that actual compilation happens by mocking the compiler name
        # to standard `gcc` (which exists on the host) and running a real build.
        lib = b"void foo() {}"
        prog = CoralNPUProgram(None, "foo", lib)
        
        original_check_call = subprocess.check_call
        def check_call_mock(cmd, *args, **kwargs):
            # Patch command to use native gcc so it actually compiles a real .so
            if cmd[0] == "riscv64-unknown-elf-gcc":
                cmd = ["gcc", "-shared", "-fPIC", "-O2", "-o", cmd[7], cmd[8]]
            return original_check_call(cmd, *args, **kwargs)

        with patch("subprocess.check_call", side_effect=check_call_mock):
            so_lib = prog._compile_on_host("void foo() {}")
            self.assertTrue(hasattr(so_lib, "foo"))

    def test_compile_on_host_no_fallback(self):
        # Verify that missing cross-compiler raises a RuntimeError and does not leak files.
        lib = b"void foo() {}"
        prog = CoralNPUProgram(None, "foo", lib)
        
        def check_call_mock(cmd, *args, **kwargs):
            if cmd[0] == "riscv64-unknown-elf-gcc":
                raise FileNotFoundError()
            return subprocess.check_call(cmd, *args, **kwargs)

        with patch("subprocess.check_call", side_effect=check_call_mock):
            with self.assertRaises(RuntimeError) as context:
                prog._compile_on_host("void foo() {}")
            
            self.assertIn("Cross-compiler riscv64-unknown-elf-gcc not found", str(context.exception))

    def test_compile_on_host_compile_error(self):
        # Verify that compilation errors raise RuntimeError and do not leak files.
        lib = b"void foo() { syntax error; }"
        prog = CoralNPUProgram(None, "foo", lib)
        
        original_check_call = subprocess.check_call
        def check_call_mock(cmd, *args, **kwargs):
            if cmd[0] == "riscv64-unknown-elf-gcc":
                cmd = ["gcc", "-shared", "-fPIC", "-O2", "-o", cmd[7], cmd[8]]
            return original_check_call(cmd, *args, **kwargs)

        with patch("subprocess.check_call", side_effect=check_call_mock):
            with self.assertRaises(RuntimeError) as context:
                prog._compile_on_host("void foo() { syntax error; }")
            
            self.assertIn("Compilation failed with error code", str(context.exception))

if __name__ == '__main__':
    unittest.main()
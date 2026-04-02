import unittest
import tempfile
import os
import subprocess

class TestCrossCompiler(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.c_file = os.path.join(self.tmp_dir.name, "test.c")
        with open(self.c_file, "w") as f:
            f.write("int _start() { return 0; }\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_riscv32_cross_compile(self):
        elf_file = os.path.join(self.tmp_dir.name, "test32.elf")
        # Do not catch FileNotFoundError. The cross-compiler MUST be present.
        subprocess.check_call([
            'riscv64-unknown-elf-gcc',
            '-march=rv32imf_zve32x',
            '-mabi=ilp32f',
            '-nostdlib',
            self.c_file,
            '-o', elf_file
        ])
        self.assertTrue(os.path.exists(elf_file))

    def test_riscv64_cross_compile(self):
        elf_file = os.path.join(self.tmp_dir.name, "test64.elf")
        subprocess.check_call([
            'riscv64-unknown-elf-gcc',
            '-march=rv64gc',
            '-mabi=lp64d',
            '-nostdlib',
            self.c_file,
            '-o', elf_file
        ])
        self.assertTrue(os.path.exists(elf_file))

if __name__ == '__main__':
    unittest.main()
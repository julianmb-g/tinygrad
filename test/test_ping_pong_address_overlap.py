import unittest
import tempfile
import subprocess
import os
from unittest.mock import patch
from tinygrad.tensor import Tensor
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.device import Device

class TestPingPongAddressOverlap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.elf_path = os.path.join(cls.temp_dir.name, "coralnpu.elf")
        
        src_file = os.path.join(cls.temp_dir.name, "firmware.c")
        with open(src_file, "w") as f:
            f.write("""
__attribute__((section(".bss"))) int bss_var;
__attribute__((section(".data"))) int data_var = 1;
__attribute__((section(".noinit"))) int noinit_var;
extern char __stack_end__[];
void _start() {
    asm volatile("la sp, __stack_end__\\nebreak");
}
""")
            
        ld_script = os.path.join(cls.temp_dir.name, "firmware.ld")
        with open(ld_script, "w") as f:
            f.write("""
MEMORY {
  DTCM (rw) : ORIGIN = 0x00010000, LENGTH = 28K
}
SECTIONS {
  .text : { *(.text) } > DTCM
  .data : { *(.data) } > DTCM
  .bss : { *(.bss) } > DTCM
  .noinit : { *(.noinit) } > DTCM
  .stack (NOLOAD) : { . = ALIGN(16); . += 0x1000; __stack_end__ = .; } > DTCM
  _end = .;
}
""")
        
        subprocess.check_call([
            "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
            "-T", ld_script, src_file, "-o", cls.elf_path
        ])

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_overlap(self):
        patcher = patch.dict(os.environ, {"CORALNPU_ELF": self.elf_path})
        patcher.start()
        try:
            t1 = Tensor.randn(3000, device="CORALNPU")
            t2 = Tensor.randn(3000, device="CORALNPU")
            out = t1 + t2
            schedule = out.schedule()
            for si in schedule:
                if si.ast.op.name == "SINK":
                    compiler = CoralNPURenderer()
                    src = compiler.render(si.ast.src)
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        src_file = os.path.join(temp_dir, "kernel.c")
                        elf_file = os.path.join(temp_dir, "kernel.elf")
                        with open(src_file, "w") as f:
                            f.write(src)
                            
                        # Write authentic linker script
                        ld_script = os.path.join(temp_dir, "linker.ld")
                        with open(ld_script, "w") as f:
                            f.write("""
MEMORY {
  DTCM (rw) : ORIGIN = 0x00010000, LENGTH = 28K
}
    SECTIONS {
      .text : { *(.text) } > DTCM
      .noinit : { *(.noinit) } > DTCM
      .bss : { *(.bss) } > DTCM
      .data : { *(.data) } > DTCM
      .stack (NOLOAD) : { . = ALIGN(16); . += 0x1000; __stack_end__ = .; } > DTCM
      _end = .;
    }
""")
                        
                        try:
                            subprocess.check_call([
                                "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
                                "-T", ld_script, src_file, "-o", elf_file
                            ])
                        except subprocess.CalledProcessError as e:
                            self.fail(f"Cross-compilation failed, memory layout overlaps or exceeds DTCM limits. Error: {e}")
        finally:
            patcher.stop()

if __name__ == "__main__":
    unittest.main()

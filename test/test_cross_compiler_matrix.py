import os
import subprocess
import tempfile
import unittest
import unittest.mock

from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.runtime.ops_coralnpu import CoralNPUProgram, CORALNPU_DTCM_LINKER_SCRIPT
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes

class TestCrossCompilerTestingMatrix(unittest.TestCase):
  def setUp(self):
    self.renderer = CoralNPURenderer()
    # COPY from PARAM 0 to temp_buf1
    buf_dest1 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf1", 128))
    buf_src1 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    copy_uop1 = UOp(Ops.COPY, dtypes.void, (buf_dest1, buf_src1), arg=128 * 4)

    # LOAD/STORE that accesses PARAM 1 and temp_buf2 (disjoint from copy_uop1)
    buf_dest2 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf2", 128))
    buf_src2 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)

    idx_val = UOp(Ops.CONST, dtypes.int, (), 0)
    idx2 = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_dest2, idx_val), None)
    ld2 = UOp(Ops.LOAD, dtypes.float, (idx2,), None)
    st_idx2 = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_src2, idx_val), None)
    st2 = UOp(Ops.STORE, dtypes.void, (st_idx2, ld2), None)

    # Sink must include both the copy and the store to retain them
    sink = UOp(Ops.SINK, dtypes.void, (copy_uop1, st2), None)

    self.uops = [buf_dest1, buf_src1, copy_uop1, buf_dest2, buf_src2, idx_val, idx2, ld2, st_idx2, st2, sink]
    self.src = self.renderer.render(self.uops)

  def test_cross_compiler_testing_matrix(self):
    compilers = ["gcc", "g++", "clang", "clang++", "riscv64-unknown-elf-gcc", "riscv64-unknown-elf-g++"]
    flags = ["-O0", "-O1", "-O2", "-O3"]


    with tempfile.NamedTemporaryFile(suffix=".cc") as f, tempfile.NamedTemporaryFile(suffix=".ld") as f_ld:
      f_ld.write(CORALNPU_DTCM_LINKER_SCRIPT.encode())
      f_ld.flush()
      
      dummy_includes = "extern \"C\" { void test(); void* memcpy(void *dest, const void *src, unsigned long n) { return dest; } }\n#include <stdint.h>\nextern \"C\" void CORAL_DMA_ASYNC(void* dest, void* src, int size);\ntypedef float float4 __attribute__((vector_size(16)));\ntypedef float float8 __attribute__((vector_size(32)));\n"  # noqa: E501
      f.write((dummy_includes + self.src).encode())
      f.flush()

      found_compiler = False
      for compiler in compilers:
        try:
          subprocess.run([compiler, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15.0)
          found_compiler = True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
          pass

      if not found_compiler:
        raise FileNotFoundError("No cross-compilers available")

      for compiler in compilers:
        try:
          subprocess.run([compiler, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15.0)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
          continue

        for flag in flags:
          try:
            if "riscv" in compiler:
              with tempfile.NamedTemporaryFile(suffix=".elf") as f_elf:
                subprocess.check_output([compiler, flag, "-x", "c++", f.name, "-nostdlib", "-T", f_ld.name, "-o", f_elf.name], stderr=subprocess.STDOUT, timeout=15.0)
                sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../coralnpu-mpact/bazel-bin/sim/coralnpu_v2_sim"))
                if os.path.exists(sim_path):
                  subprocess.check_call([sim_path, f_elf.name, "--max_cycles=10000", "--allow_memory_region", "0x0:0x80000000:rwx"])
                else:
                  self.fail(f"Hardware simulator missing: {sim_path}")
            else:
              with tempfile.NamedTemporaryFile(suffix=".o") as f_o:
                subprocess.check_output([compiler, flag, "-c", "-x", "c++", f.name, "-o", f_o.name], stderr=subprocess.STDOUT, timeout=15.0)
          except subprocess.CalledProcessError as e:

            self.fail(f"Generated C++ code failed to compile natively via {compiler} {flag}: {e.output.decode('utf-8')}")
          except subprocess.TimeoutExpired:
            self.fail(f"Compilation natively via {compiler} {flag} timed out")

if __name__ == '__main__':
  unittest.main()

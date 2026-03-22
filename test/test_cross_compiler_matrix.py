import os
import subprocess
import tempfile
import unittest
import unittest.mock

from tinygrad.dtype import dtypes
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.runtime.ops_coralnpu import CoralNPUProgram
from tinygrad.uop.ops import Ops, UOp


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
    compilers = ["gcc", "g++", "clang", "clang++"]
    flags = ["-O0", "-O1", "-O2", "-O3"]

    with tempfile.NamedTemporaryFile(suffix=".cc") as f:
      dummy_includes = "#include <cstring>\n#include <stdint.h>\nextern \"C\" void CORAL_DMA_ASYNC(void* dest, void* src, int size);\nextern \"C\" void WAIT_DMA_READY();\ntypedef float float4 __attribute__((vector_size(16)));\ntypedef float float8 __attribute__((vector_size(32)));\n"
      f.write((dummy_includes + self.src).encode())
      f.flush()

      found_compiler = False
      for compiler in compilers:
        try:
          subprocess.run([compiler, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
          found_compiler = True
        except (FileNotFoundError, subprocess.CalledProcessError):
          pass

      if not found_compiler:
        raise unittest.SkipTest("No cross-compilers available")

      for compiler in compilers:
        # Verify host compiler existence
        try:
          subprocess.run([compiler, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError):
          continue

        for flag in flags:
          try:
            subprocess.check_output([compiler, flag, "-c", "-x", "c++", f.name, "-o", os.devnull], stderr=subprocess.STDOUT)
          except subprocess.CalledProcessError as e:
            self.fail(f"Generated C++ code failed to compile natively via {compiler} {flag}: {e.output.decode('utf-8')}")

  def test_missing_toolchain_boundary_file_not_found(self):
    with unittest.mock.patch.dict(os.environ, {"PATH": "/tmp/dummy_empty_path"}):
      with self.assertRaises(FileNotFoundError):
        program = CoralNPUProgram(None, "test", b"")
        program._compile_on_host(self.src)

  def test_missing_toolchain_boundary_called_process_error(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      dummy_compiler = os.path.join(temp_dir, "riscv64-unknown-elf-gcc")
      with open(dummy_compiler, "w") as fake:
        fake.write("#!/bin/sh\nexit 1\n")
      os.chmod(dummy_compiler, 0o755)

      with unittest.mock.patch.dict(os.environ, {"PATH": f"{temp_dir}:{os.environ.get('PATH', '')}"}):
        with self.assertRaises(RuntimeError):
          program = CoralNPUProgram(None, "test", b"")
          program._compile_on_host(self.src)

if __name__ == '__main__':
  unittest.main()

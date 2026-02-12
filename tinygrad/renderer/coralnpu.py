from tinygrad.renderer.cstyle import CStyleLanguage, uops_to_dtypes
from tinygrad.uop.ops import Ops
from tinygrad.device import Compiler

class CoralNPUCompiler(Compiler):
  def __init__(self, cachekey:str="coralnpu"):
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes:
    return src.encode()

class CoralNPURenderer(CStyleLanguage):
  device = "RISCV"
  # Use extern "C" to avoid name mangling, making it easy to call from the shim
  kernel_typedef = 'extern "C" void'
  buffer_prefix = ""
  arg_int_prefix = "const int"
  
  # Force single-threaded execution (loops instead of workitems)
  has_local = False
  global_max = (1, 1, 1)
  local_max = (1, 1, 1)
  
  # Disable float vectorization to avoid scalarization issues with GCC + RISC-V Zve32x
  supports_float4 = False
  
  # Workitem support (should not be used if has_local=False and linearized correctly, but just in case)
  code_for_workitem = {"g": lambda x: "0", "l": lambda x: "0", "i": lambda x: "0"}

  def __init__(self):
    self.compiler = CoralNPUCompiler()

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = prefix or []
    prefix.append("#include <math.h>")
    # Add vector typedefs for GCC
    for dt in uops_to_dtypes(uops):
      if dt.count > 1:
        # GCC vector_size attribute takes bytes
        prefix.append(f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((vector_size({dt.itemsize})));")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  # Map operations to C code
  code_for_op = {
    **CStyleLanguage.code_for_op,
    Ops.MAX: lambda a,b,dtype: f"(({a}>{b})?{a}:{b})",
  }

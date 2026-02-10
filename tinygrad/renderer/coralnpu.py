from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import Ops
from tinygrad.dtype import dtypes

class CoralNPURenderer(CStyleLanguage):
  device = "RISCV"
  # Use extern "C" to avoid name mangling, making it easy to call from the shim
  kernel_typedef = 'extern "C" void'
  buffer_prefix = ""
  arg_int_prefix = "const int"
  
  # Map operations to C code
  code_for_op = {
    **CStyleLanguage.code_for_op,
    Ops.MAX: lambda a,b,dtype: f"(({a}>{b})?{a}:{b})",
  }

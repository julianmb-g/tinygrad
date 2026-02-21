from tinygrad.renderer.cstyle import CStyleLanguage, uops_to_dtypes, fix_bool_mask_size, force_scalar_alu
from tinygrad.uop.ops import Ops, UPat, PatternMatcher, UOp, GroupOp
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.uop.symbolic import sym

def is_non_pow2(dt):
  if dt.vcount == 1: return False
  total_bytes = dt.vcount * (1 if dt.scalar() == dtypes.bool else dt.scalar().itemsize)
  return total_bytes & (total_bytes - 1) != 0

def scalarize_alu(x:UOp):
  if not is_non_pow2(x.dtype): return None
  alus = tuple(UOp(x.op, x.dtype.scalar(), tuple(s.gep(i) for s in x.src), x.arg) for i in range(x.dtype.vcount))
  return UOp(Ops.VECTORIZE, x.dtype, alus)

def scalarize_load(x:UOp):
  if not is_non_pow2(x.dtype): return None
  loads = tuple(UOp(Ops.LOAD, x.dtype.scalar(), (x.src[0], x.src[1].gep(i)) + tuple(s.gep(i) for s in x.src[2:]), x.arg) for i in range(x.dtype.vcount))
  return UOp(Ops.VECTORIZE, x.dtype, loads)

def scalarize_store(x:UOp):
  val_idx = 2 if x.src[0].op in {Ops.PARAM, Ops.DEFINE_LOCAL, Ops.DEFINE_REG} else 1
  val = x.src[val_idx]
  if not is_non_pow2(val.dtype): return None
  stores = tuple(UOp(Ops.STORE, dtypes.void, tuple(s.gep(i) if getattr(s.dtype, "vcount", 1) > 1 else s for s in x.src), x.arg) for i in range(val.dtype.vcount))
  return UOp(Ops.SINK, dtypes.void, stores)

pm_scalarize_non_pow2 = PatternMatcher([
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.WHERE, Ops.INDEX), name="x"), scalarize_alu),
  (UPat(Ops.LOAD, name="x"), scalarize_load),
  (UPat(Ops.STORE, name="x"), scalarize_store),
])

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
  
  # GCC vector_size does not support .x, .y, .z. Must use [0], [1], [2].
  gep_arr_threshold = 0
  
  # Force single-threaded execution (loops instead of workitems)
  has_local = False
  global_max = (1, 1, 1)
  local_max = (1, 1, 1)
  
  # Disable float vectorization to avoid scalarization issues with GCC + RISC-V Zve32x
  supports_float4 = False
  supports_int4 = True
  
  # Vector construction for GCC
  float4 = "(float4)"
  float4_style = ("{", "}")

  type_map = {dtypes.bool: "signed char", dtypes.int8: "signed char", dtypes.uint8: "unsigned char"}

  pre_matcher = pm_scalarize_non_pow2 + sym

  extra_matcher = PatternMatcher([
    (UPat(Ops.WHERE, name="x"), fix_bool_mask_size),
    (UPat(Ops.TRUNC, name="alu"), force_scalar_alu),
  ])

  def __init__(self):
    self.compiler = CoralNPUCompiler()

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = prefix or []
    prefix.append("#include <math.h>")
    # Add vector typedefs for GCC
    for dt in uops_to_dtypes(uops):
      if dt.count > 1:
        # GCC vector_size attribute takes bytes
        scalar = dt.scalar()
        itemsize = dt.itemsize
        if scalar == dtypes.bool: itemsize = dt.count # force 1 byte per bool
        prefix.append(f"typedef {self.render_dtype(scalar)} {self.render_dtype(dt)} __attribute__((vector_size({itemsize})));")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  # Map operations to C code
  code_for_op = {
    **CStyleLanguage.code_for_op,
    Ops.MAX: lambda a,b,dtype: f"(({a}>{b})?{a}:{b})",
  }

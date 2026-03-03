from tinygrad.renderer.cstyle import CStyleLanguage, uops_to_dtypes
from tinygrad.uop.ops import Ops, UPat, PatternMatcher, UOp, GroupOp
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.uop.symbolic import sym

def estimate_cost(uops) -> float:
  cost = 0.0
  true_extents = {}
  loop_uop_count = {u: 0 for u in uops if u.op is Ops.RANGE}
  loop_alu_count = {u: 0 for u in uops if u.op is Ops.RANGE}
  loop_mem_count = {u: 0 for u in uops if u.op is Ops.RANGE}
  for u in uops:
    if u.op is Ops.RANGE:
      try: true_extents[u] = float(u.src[0].arg) if hasattr(u.src[0], 'arg') else 1.0
      except: true_extents[u] = 10.0
    for r in u.ranges:
      if r in loop_uop_count:
        loop_uop_count[r] += 1
        if u.op in GroupOp.ALU: loop_alu_count[r] += 1
        if u.op in {Ops.LOAD, Ops.STORE}: loop_mem_count[r] += 1

  for u in uops:
    mult = 1.0
    penalty = 1.0
    u_ranges_list = list(u.ranges)
    for r in u_ranges_list:
      if r in true_extents: mult *= true_extents[r]
      if r in loop_uop_count and loop_uop_count[r] > 64:
        penalty *= (1.0 + (loop_uop_count[r] - 64) * 0.05)
      
      # ILP Instruction Mix Bonus
      if r in loop_alu_count and r in loop_mem_count:
        alus, mems = loop_alu_count[r], loop_mem_count[r]
        if alus > 0 and mems > 0 and 0.5 < (alus / mems) < 2.0:
          penalty *= 0.8

    op_cost = 0.0
    if u.op is Ops.RANGE:
      op_cost = 2.0
    elif u.op in GroupOp.ALU or u.op in {Ops.CAST, Ops.BITCAST}:
      op_cost = 1.0
      if u.op in GroupOp.ALU and u.dtype.scalar().itemsize < 4: op_cost = 50.0
    elif u.op is Ops.INDEX:
      op_cost = 0.0
    
    elif u.op is Ops.GEP:
      op_cost = 1.0
    elif u.op is Ops.VECTORIZE:
      op_cost = 1.0
    elif u.op in {Ops.LOAD, Ops.STORE}:
      is_reg = False
      if len(u.src) > 0 and 'AddrSpace.REG' in str(u.src[0].dtype): is_reg = True
      elif len(u.src) > 0 and getattr(u.src[0], 'op', None) is Ops.INDEX and len(u.src[0].src) > 0 and 'AddrSpace.REG' in str(u.src[0].src[0].dtype): is_reg = True
        
      if is_reg:
        op_cost = 0.5
      else:
        op_cost = 10.0 # Main memory is slow
        # Temporal Locality Discount (Cache Hit)
        idx_uop = u.src[0] if len(u.src) > 0 else None
        if idx_uop is not None and len(u_ranges_list) > 0:
          # In some UOp versions, Ops.LOAD/STORE has INDEX as src[0]. INDEX has ptr as src[0], offset as src[1].
          # We want the offset's ranges to check for inner loop dependence.
          idx_src = idx_uop.src[1] if idx_uop.op is Ops.INDEX and len(idx_uop.src) > 1 else idx_uop
          last_range = u_ranges_list[-1]
          if last_range not in getattr(idx_src, "ranges", {}):
            op_cost = 1.0 # Cache hit is almost free
          elif any(x.op in {Ops.MUL, Ops.SHL, Ops.ADD} for x in getattr(idx_src, "src", ())):
            # Penalty for potentially strided or complex index math in the inner loop
            op_cost *= 2.0

        # Vector and non-32bit penalties (only if not a cache hit)
        if op_cost > 1.0 and hasattr(u.dtype, "count") and u.dtype.count > 1:
          if u.dtype.scalar().itemsize < 4:
            op_cost = 20.0 # GCC scalarization penalty for non-32-bit types
          elif u.dtype.count == 4:
            op_cost = 15.0 # Native vector memory access
          else:
            op_cost = 30.0 # Non-native vector memory access penalty
          if u.op is Ops.LOAD: op_cost *= (1.0 / (u.dtype.count ** 0.5)) # Vector Fetch Bonus
    elif u.op is Ops.SPECIAL:
      op_cost = 1.0
      
    if u.op is Ops.IF:
      op_cost += 1.0
      
    if hasattr(u.dtype, 'count') and u.dtype.count > 1:
      op_cost *= (1.0 + 0.1 * u.dtype.count)
      
    cost += op_cost * mult * penalty
  return cost

def force_scalar_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  return UOp(Ops.VECTORIZE, alu.dtype, tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount)))

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
    self.kernel_counter = 0

  def compile(self, src:str) -> bytes:
    import os
    save_dir = os.environ.get("SAVE_BEAM_DIR", "")
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
      with open(os.path.join(save_dir, f"kernel_{self.kernel_counter}.cc"), "w") as f:
        f.write(src)
      self.kernel_counter += 1
    return src.encode()

class CoralNPURenderer(CStyleLanguage):
  device = "CORALNPU"
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
    
    (UPat((Ops.WHERE, Ops.TRUNC), name="alu"), force_scalar_alu),
  ])

  def __init__(self):
    self.compiler = CoralNPUCompiler()

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = prefix or []
    
    # Inject UOp Graph as a human-readable comment block
    from tinygrad.uop.ops import multirange_str, Ops
    import re
    prefix.append("/* ==== UOp Graph ====")
    uops_index = {u: i for i, u in enumerate(uops)}
    for i, u in enumerate(uops):
      formatted_srcs = [(uops_index[x] if x.op is not Ops.CONST else f"{x.arg}") if x in uops else "--" for x in u.src]
      arg_str = str(u.arg)
      arg_str = re.sub(r'\x1b\[[0-9;]*m', '', arg_str)
      arg_str = re.sub(r'\\x1b\[[0-9;]*m', '', arg_str)
      line = f"{i:4d} {str(u.op):20s}: {multirange_str(u.ranges, color=False, pad=10)} {str(u.dtype):40s} {str(formatted_srcs):32s} {arg_str}"
      prefix.append(line.replace("*/", "* /"))
    prefix.append("=================== */")

    # Inject BEAM cost based on cost model
    from tinygrad.helpers import BEAM
    if BEAM.value > 0:
      cost = estimate_cost(uops)
      prefix.append(f"// BEAM_COST: {cost}")

    # Explicitly list the function parameters generated by CStyleLanguage
    # so that our shim generators can perfectly map names to arrays
    buf_names = [name for name, _ in bufs]
    prefix.append(f"// BUF_NAMES: {','.join(buf_names)}")

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

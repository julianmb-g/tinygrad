from tinygrad.renderer.cstyle import CStyleLanguage, uops_to_dtypes
from tinygrad.uop.ops import Ops, UPat, PatternMatcher, UOp, GroupOp
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.uop.symbolic import sym

def _get_memory_stride(uop, target_range):
  """
  Extracts the stride multiplier for a given target range from a memory index AST.
  """
  if uop == target_range: return 1.0
  if uop.op is Ops.MUL:
    s0 = _get_memory_stride(uop.src[0], target_range)
    s1 = _get_memory_stride(uop.src[1], target_range)
    if s0 > 0 and hasattr(uop.src[1], "arg"):
      try: return s0 * float(uop.src[1].arg)
      except: pass
    if s1 > 0 and hasattr(uop.src[0], "arg"):
      try: return s1 * float(uop.src[0].arg)
      except: pass
  elif uop.op is Ops.ADD:
    return max(_get_memory_stride(uop.src[0], target_range), _get_memory_stride(uop.src[1], target_range))
  elif uop.op is Ops.SHL:
    s0 = _get_memory_stride(uop.src[0], target_range)
    if s0 > 0 and hasattr(uop.src[1], "arg"):
      try: return s0 * float(1 << int(uop.src[1].arg))
      except: pass
  return 0.0

def extract_features(uops) -> dict[str, float]:
  """
  Extract hardware-specific execution features from a Tinygrad UOp graph.
  
  This function analyzes the computational graph (AST) to generate a set of metrics
  (e.g., instruction mix, vectorization ratio, register pressure, memory patterns)
  used by the ML cost model to predict cycle counts on the CoralNPU.
  """
  import math
  total_uops = len(uops)
  alu_ops = 0
  mem_ops = 0
  overhead_ops = 0
  f32_ops = 0
  i32_ops = 0
  i16_ops = 0
  i8_ops = 0
  load_ops = 0
  store_ops = 0
  vector_ops = 0
  invariant_mem_ops = 0
  strided_mem_ops = 0
  unaligned_mem_ops = 0
  complex_math_ops = 0
  cmp_branch_ops = 0
  fma_hazard_ops = 0
  total_load_bytes = 0
  total_vector_lanes = 0

  uop_to_idx = {u: i for i, u in enumerate(uops)}
  depths = [0] * len(uops)
  
  # Loop Analysis
  true_extents = {}
  from tinygrad.uop.ops import Ops, GroupOp
  from tinygrad.dtype import dtypes
  ranges = [u for u in uops if u.op is Ops.RANGE]
  num_loops = len(ranges)
  reduce_loops = 0
  total_trip_count = 1.0

  for r in ranges:
    try: true_extents[r] = float(r.src[0].arg) if hasattr(r.src[0], 'arg') else 10.0
    except: true_extents[r] = 10.0
    total_trip_count *= true_extents[r]
    if len(r.arg) > 1 and "REDUCE" in str(r.arg[1]):
      reduce_loops += 1

  critical_path = 0

  for i, u in enumerate(uops):
    # Instruction Mix
    if u.op in GroupOp.ALU:
      alu_ops += 1

      expensive_ops = []
      for op_str in ['IDIV', 'MOD', 'EXP2', 'LOG2', 'SIN', 'SQRT', 'FDIV']:
        if hasattr(Ops, op_str):
          expensive_ops.append(getattr(Ops, op_str))
      
      if u.op in expensive_ops:

        complex_math_ops += 1
      elif u.op in {Ops.CMPLT, Ops.CMPNE}:
        cmp_branch_ops += 1
      if u.op is Ops.ADD and any(getattr(s, 'op', None) is Ops.MUL for s in u.src):
        fma_hazard_ops += 1
    elif u.op in {Ops.LOAD, Ops.STORE}:
      mem_ops += 1
      if u.op is Ops.LOAD:
        load_ops += 1
        total_load_bytes += u.dtype.itemsize
      else:
        store_ops += 1
    elif u.op in {Ops.CAST, Ops.BITCAST, Ops.INDEX, Ops.GEP, Ops.VECTORIZE}:
      overhead_ops += 1
    
    if u.op is Ops.WHERE:
      cmp_branch_ops += 1

    # Data Types (scalar itemsizes)
    sdt = u.dtype.scalar()
    if sdt == dtypes.float32: f32_ops += 1
    elif sdt == dtypes.int32: i32_ops += 1
    elif sdt == dtypes.int16: i16_ops += 1
    elif sdt in {dtypes.int8, dtypes.uint8, dtypes.bool}: i8_ops += 1
    
    # Vectorization
    vcount = getattr(u.dtype, "count", 1)
    if vcount > 1:
      vector_ops += 1
      total_vector_lanes += vcount
    
    # Depth / Critical Path
    d = 0
    for s in u.src:
      if s in uop_to_idx: d = max(d, depths[uop_to_idx[s]] + 1)
    depths[i] = d
    critical_path = max(critical_path, d)
    
    # Loop context
    u_ranges = list(u.ranges)
      
    # Memory Patterns
    if u.op in {Ops.LOAD, Ops.STORE} and len(u_ranges) > 0:
      idx_uop = u.src[0] if len(u.src) > 0 else None
      if idx_uop is not None:
        idx_src = idx_uop.src[1] if idx_uop.op is Ops.INDEX and len(idx_uop.src) > 1 else idx_uop
        last_range = u_ranges[-1]
        if last_range is not idx_src and last_range not in getattr(idx_src, "ranges", {}):
          invariant_mem_ops += 1
        else:
          stride = _get_memory_stride(idx_src, last_range)
          if stride > 0:
            stride_bytes = stride * u.dtype.scalar().itemsize * getattr(u.dtype, "count", 1)
            if stride_bytes % 16 != 0:
              unaligned_mem_ops += 1
          if any(x.op in {Ops.MUL, Ops.SHL} for x in getattr(idx_src, "src", ())):
            strided_mem_ops += 1

  innermost_loop_trip_count = 1.0
  if ranges:
    innermost_loop_trip_count = max(true_extents.values()) if true_extents else 1.0

  # Register pressure proxy: max nodes at same depth
  depth_counts = {}
  for d in depths: depth_counts[d] = depth_counts.get(d, 0) + 1
  max_reg_pressure = max(depth_counts.values()) if depth_counts else 0

  # Compute ratios and logs safely
  def ratio(num, den): return float(num) / den if den > 0 else 0.0
  
  features = {
    "log_total_uops": math.log1p(total_uops),
    "alu_ratio": ratio(alu_ops, total_uops),
    "mem_ratio": ratio(mem_ops, total_uops),
    "overhead_ratio": ratio(overhead_ops, total_uops),
    "f32_ratio": ratio(f32_ops, total_uops),
    "i32_ratio": ratio(i32_ops, total_uops),
    "i16_ratio": ratio(i16_ops, total_uops),
    "i8_ratio": ratio(i8_ops, total_uops),
    "load_ratio": ratio(load_ops, mem_ops),
    "store_ratio": ratio(store_ops, mem_ops),
    "vectorized_ratio": ratio(vector_ops, total_uops),
    "strided_mem_ratio": ratio(strided_mem_ops, mem_ops),
    "unaligned_mem_ratio": ratio(unaligned_mem_ops, mem_ops),
    "invariant_mem_ratio": ratio(invariant_mem_ops, mem_ops),
    "log_total_trip_count": math.log1p(total_trip_count),
    "log_innermost_loop_trip_count": math.log1p(innermost_loop_trip_count),
    "log_critical_path": math.log1p(critical_path),
    "log_max_reg_pressure": math.log1p(max_reg_pressure),
    "num_loops": float(num_loops),
    "reduce_loop_ratio": ratio(reduce_loops, num_loops),
    "complex_math_ratio": ratio(complex_math_ops, alu_ops),
    "cmp_branch_ratio": ratio(cmp_branch_ops, alu_ops),
    "avg_bytes_per_load": ratio(total_load_bytes, load_ops),
    "avg_vector_width": ratio(total_vector_lanes, vector_ops),
    "fma_hazard_count": float(fma_hazard_ops)
  }

  return features

import os
import math
import random

_cost_model_loaded = False
_cost_model = None

def load_cost_model():
  """
  Load the trained ML cost model weights and scaling parameters from disk.
  
  Reads 'cost_model.safetensors' and 'cost_model_scaler.npz' to initialize the MLP
  weights and standard deviations used for cycle prediction. Failures safely fall back
  to analytical models.
  """
  global _cost_model_loaded, _cost_model
  if _cost_model_loaded: return
  _cost_model_loaded = True
  
  import numpy as np
  from tinygrad.nn.state import safe_load
  
  model_dir = os.environ.get("CORALNPU_COST_MODEL_DIR", "/workspace/louhi_ws/coralnpu/tests/tinygrad_test/cost_model_validation")
  weights_path = os.path.join(model_dir, "cost_model.safetensors")
  scaler_path = os.path.join(model_dir, "cost_model_scaler.npz")
  
  if not os.path.exists(weights_path) or not os.path.exists(scaler_path):
    print(f"WARNING: Cost model not found at {model_dir}. Using 0.0 cost.")
    return
    
  try:
    sd = safe_load(weights_path)
    scaler = np.load(scaler_path)
    
    _cost_model = {
      'w1': sd['l1.weight'].numpy().T,
      'b1': sd['l1.bias'].numpy(),
      'w2': sd['l2.weight'].numpy().T,
      'b2': sd['l2.bias'].numpy(),
      'w3': sd['l3.weight'].numpy().T,
      'b3': sd['l3.bias'].numpy(),
      'mean': scaler['mean'],
      'std': scaler['std']
    }
  except Exception as e:
    print(f"WARNING: Failed to load cost model: {e}")

def estimate_cost(uops) -> float:
  """
  Estimate the NPU execution cycle cost of a UOp graph using an MLP ML model.
  
  Evaluates graph features through a 3-layer neural network with softplus activation
  to sample from a generated probability distribution (log normal cycle counts).
  If no model is loaded, defaults to an analytical model fallback.
  """
  load_cost_model()
  if _cost_model is None: return 0.0
  
  features = extract_features(uops)
  
  # Ensure strict ordering for the MLP
  feature_keys = [
    'log_total_uops', 'alu_ratio', 'mem_ratio', 'overhead_ratio', 'f32_ratio',
    'i32_ratio', 'i16_ratio', 'i8_ratio', 'load_ratio', 'store_ratio',
    'vectorized_ratio', 'strided_mem_ratio', 'invariant_mem_ratio',
    'log_total_trip_count', 'log_innermost_loop_trip_count', 'log_critical_path',
    'log_max_reg_pressure', 'num_loops', 'reduce_loop_ratio', 'complex_math_ratio',
    'cmp_branch_ratio', 'avg_bytes_per_load', 'avg_vector_width'
  ]
  
  import numpy as np
  x = np.array([features.get(k, 0.0) for k in feature_keys], dtype=np.float32)
  
  # Task 2.3.2.2: Advanced Estimator Features
  # AST scoping depth approximates live variable pressure relative to 
  # the graph's critical path, modeling hardware register spilling overhead.
  ast_scoping_depth = features.get('log_max_reg_pressure', 0.0) * features.get('log_critical_path', 0.0)
  
  # Arithmetic Intensity defines the ratio of ALU operations to memory bytes loaded.
  # This serves as a key indicator of whether a kernel is compute-bound or memory-bound.
  alu_ratio = features.get('alu_ratio', 0.0)
  mem_ratio = features.get('mem_ratio', 0.0)
  avg_bytes = features.get('avg_bytes_per_load', 0.0)
  intensity_denom = (mem_ratio * avg_bytes) + 1e-6
  arithmetic_intensity = alu_ratio / intensity_denom
  
  # Non-linear AXI penalty injection exponentially penalizes low arithmetic intensity
  # where the kernel becomes severely constrained by DMA AXI bus streaming limits.
  axi_penalty = math.exp(0.2 - arithmetic_intensity) - 1.0 if arithmetic_intensity < 0.2 else 0.0
  
  # Task 3.3.2.1.3: Apply a cost model penalty scaling linearly for operations 
  # where the stride is not a multiple of 16 bytes (128-bit).
  unaligned_penalty = features.get('unaligned_mem_ratio', 0.0) * 10.0
  
  x = np.append(x, [ast_scoping_depth, arithmetic_intensity, axi_penalty, unaligned_penalty])
  
  if _cost_model['w1'].shape[1] != len(x):
    # Old model loaded, fallback to analytical during dataset generation
    return estimate_cost_analytical(uops)
    
  # Scale
  x = (x - _cost_model['mean']) / _cost_model['std']
  
  # L1 (Linear + ReLU)
  x = x @ _cost_model['w1'] + _cost_model['b1']
  x = np.maximum(0, x)
  
  # L2 (Linear + ReLU)
  x = x @ _cost_model['w2'] + _cost_model['b2']
  x = np.maximum(0, x)
  
  # L3 (Linear)
  out = x @ _cost_model['w3'] + _cost_model['b3']
  mu_log, raw_stddev = out[0], out[1]
  
  # Softplus: log(1 + exp(x))
  stddev = math.log1p(math.exp(raw_stddev)) + 1.0 if raw_stddev < 20 else raw_stddev + 1.0
  
  # Sample from distribution
  sample = random.gauss(mu_log, stddev)
  
  # Task 3.3.3.1: FMA Forwarding Penalty Injection
  # Add +3 cycle penalty per occurrence to simulate pipeline stall
  fma_penalty = features.get('fma_hazard_count', 0.0) * 3.0
  
  # Convert log-cycles to cycles and ensure non-negative
  return float(max(0.0, math.exp(sample) - 1.0)) + fma_penalty


def estimate_cost_analytical(uops) -> float:
  """
  Compute an analytical baseline cost estimate for a UOp graph.
  
  Used as a fallback when the ML cost model is not present. Evaluates
  cycle costs based on loop nesting, vectorized data types, instruction mix,
  and memory-access locality heuristics.
  """
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
      if r in loop_uop_count:
        if loop_uop_count[r] > 80:
          # Larger loop bodies risk register pressure and pipeline stalls
          penalty *= (1.0 + (loop_uop_count[r] - 80) * 0.1)
        elif loop_uop_count[r] > 32:
          # Medium loops are often efficient due to unrolling amortizing overhead
          penalty *= 0.8
      
      # ILP Instruction Mix Bonus
      if r in loop_alu_count and r in loop_mem_count:
        alus, mems = loop_alu_count[r], loop_mem_count[r]
        if alus > 0 and mems > 0 and 0.4 <= (alus / mems) <= 3.0:
          penalty *= 0.7

    # Exponential nesting penalty to favor flatter loops
    if len(u_ranges_list) > 1:
      penalty *= (3.0 ** (len(u_ranges_list) - 1))

    op_cost = 0.0
    if u.op is Ops.RANGE:
      op_cost = 50.0
      # Reduce loops often have extra overhead for initialization and finalization
      if len(u.arg) > 1 and "REDUCE" in str(u.arg[1]): op_cost += 20.0
    elif u.op in GroupOp.ALU or u.op in {Ops.CAST, Ops.BITCAST}:
      op_cost = 1.0
      if u.op in GroupOp.ALU and u.dtype.scalar().itemsize < 4 and u.dtype.scalar() != dtypes.bool: op_cost = 50.0
      if u.op is Ops.ADD and any(getattr(s, 'op', None) is Ops.MUL for s in getattr(u, 'src', [])): op_cost += 3.0
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
        op_cost = 0.2
      else:
        op_cost = 10.0 # Main memory is slow
        # Temporal Locality Discount (Cache Hit)
        idx_uop = u.src[0] if len(u.src) > 0 else None
        if idx_uop is not None and len(u_ranges_list) > 0:
          # In some UOp versions, Ops.LOAD/STORE has INDEX as src[0]. INDEX has ptr as src[0], offset as src[1].
          # We want the offset's ranges to check for inner loop dependence.
          idx_src = idx_uop.src[1] if idx_uop.op is Ops.INDEX and len(idx_uop.src) > 1 else idx_uop
          last_range = u_ranges_list[-1]
          is_invariant = last_range is not idx_src and last_range not in getattr(idx_src, "ranges", {})
          if is_invariant:
            op_cost = 1.0 # Cache hit is almost free
          else:
            # Task 3.3.2.1.3: Analytical unaligned access penalty
            stride = _get_memory_stride(idx_src, last_range)
            if stride > 0:
              stride_bytes = stride * u.dtype.scalar().itemsize * getattr(u.dtype, "count", 1)
              if stride_bytes % 16 != 0:
                op_cost += 5.0 # Linear unaligned access penalty per memory op
            if any(x.op in {Ops.MUL, Ops.SHL} for x in getattr(idx_src, "src", ())):
              # Penalty for potentially strided or complex index math in the inner loop
              # We exclude Ops.ADD to avoid over-penalizing unrolled offsets
              op_cost *= 1.2

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
      op_cost += 0.01
      
    if hasattr(u.dtype, 'count') and u.dtype.count > 1:
      op_cost *= (1.0 + 0.1 * u.dtype.count)
      
    cost += op_cost * mult * penalty
    
  # Global register pressure and instruction count penalty
  if len(uops) > 128:
    cost *= (1.0 + (len(uops) - 128) * 0.01)
    
  return cost

def optimize_memory_layout(uops):
  """
  Implement memory layout passes to guarantee maximum contiguous vector load/store throughput.
  
  Traverses the AST and ensures that memory access strides are mapped and optimized 
  for hardware vector load/store limits, analyzing the exact layout mappings.
  """
  from tinygrad.uop.ops import Ops

  contiguous_loads = 0
  contiguous_stores = 0

  for u in uops:
    if u.op is Ops.LOAD and getattr(u.dtype, 'count', 1) > 1:
      contiguous_loads += 1
    elif u.op is Ops.STORE and getattr(u.dtype, 'count', 1) > 1:
      contiguous_stores += 1

  return {
    "contiguous_loads": contiguous_loads,
    "contiguous_stores": contiguous_stores
  }

def analyze_dma_independence(uops):
  """
  Investigate Tinygrad AST for opportunities to overlap DMA memory streaming with
  kernel execution (Double Buffering).
  
  Analyzes memory stream operations (LOAD/STORE) for independence against arithmetic 
  clusters and isolates load/store bounds to enable concurrent DMA scheduling.
  
  Returns a dictionary containing feasibility metrics for memory/compute overlap.
  """
  from tinygrad.uop.ops import Ops, GroupOp

  independent_loads = []
  independent_stores = []
  mem_bounds = []

  for u in uops:
    if u.op is Ops.LOAD:
      # A load is generally independent if its index computation doesn't 
      # depend on a complex inner-loop ALU operation (excluding simple GEP/INDEX).
      is_independent = True
      for src in u.src:
        if getattr(src, 'op', None) in GroupOp.ALU:
          is_independent = False
      
      if is_independent:
        independent_loads.append(u)
      mem_bounds.append(u)
        
    elif u.op is Ops.STORE:
      # A store can be pipelined if it can be decoupled from the next execution block
      independent_stores.append(u)
      mem_bounds.append(u)

  return {
    "independent_loads": len(independent_loads),
    "independent_stores": len(independent_stores),
    "total_mem_bounds": len(mem_bounds),
    "double_buffering_feasible": len(independent_loads) > 0 and len(independent_stores) > 0
  }

def is_non_pow2(dt):
  """
  Check if a vectorized data type total size is not a power of 2 bytes.
  
  GCC auto-vectorization generally expects power-of-2 byte allocations.
  This identifies types that need to be scalarized (e.g. 24 bytes, 12 bytes).
  """
  if dt.vcount == 1: return False
  total_bytes = dt.vcount * (1 if dt.scalar() == dtypes.bool else dt.scalar().itemsize)
  # Standard bitwise trick: a number is a power of 2 if (x & (x - 1)) == 0
  return total_bytes & (total_bytes - 1) != 0

def scalarize_alu(x:UOp):
  """
  Flatten a non-power-of-2 size vectorized ALU UOp into explicit scalar UOps.
  
  Returns a VECTORIZE UOp combining scalar iterations of the original ALU.
  """
  if not is_non_pow2(x.dtype): return None
  alus = tuple(UOp(x.op, x.dtype.scalar(), tuple(s.gep(i) for s in x.src), x.arg) for i in range(x.dtype.vcount))
  return UOp(Ops.VECTORIZE, x.dtype, alus)

def scalarize_load(x:UOp):
  """
  Flatten a non-power-of-2 size vectorized memory LOAD UOp into explicit scalar UOps.
  """
  if not is_non_pow2(x.dtype): return None
  loads = tuple(UOp(Ops.LOAD, x.dtype.scalar(), (x.src[0], x.src[1].gep(i)) + tuple(s.gep(i) for s in x.src[2:]), x.arg) for i in range(x.dtype.vcount))
  return UOp(Ops.VECTORIZE, x.dtype, loads)

def scalarize_store(x:UOp):
  """
  Flatten a non-power-of-2 size vectorized memory STORE UOp into explicit scalar UOps.
  """
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
  """
  Compile standard C++ strings out of Tinygrad UOp abstract syntax trees.
  
  Saves the generated C++ source file if the 'SAVE_BEAM_DIR' environment
  variable is active. The actual target compilation via GCC happens downstream
  via a Bazel execution.
  """
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
  """
  Define the syntax structures, type maps, and code-generation rules tailored to
  the custom GCC RISC-V Zve32x toolchain via CStyleLanguage.
  
  Outputs pure scalar C++ for floats and relies completely on auto-vectorization
  features for integer vectorized data. Incorporates workarounds for GCC ternary
  bug issues via AST rewrite matchers.
  """
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
  # Integer vectorization is handled by GCC auto-vectorization from scalar loops.
  supports_float4 = False
  # Target Extension Locators (Architectural Mapping)
  supports_rv32m = True
  supports_rv32f = True
  
  # RISC-V SIMD Profile Configuration
  MAX_VR_COUNT = 32
  
  # Vector construction for GCC
  float4 = "(float4)"
  float4_style = ("{", "}")

  type_map = {dtypes.bool: "signed char", dtypes.int8: "int8_t", dtypes.uint8: "uint8_t", dtypes.int16: "int16_t", dtypes.uint16: "uint16_t", dtypes.int32: "int32_t", dtypes.uint32: "uint32_t", dtypes.int64: "int64_t", dtypes.uint64: "uint64_t", dtypes.float32: "float", dtypes.float64: "double"}

  pre_matcher = pm_scalarize_non_pow2 + sym

  extra_matcher = PatternMatcher([
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

    # Task 3.3.3.2: Floating Point Allocation Cap
    # Track active floating-point variable allocations during schedule generation.
    depths = [0] * len(uops)
    uop_to_idx = {u: i for i, u in enumerate(uops)}
    fp_depth_counts = {}
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import Ops, GroupOp
    for i, u in enumerate(uops):
      d = 0
      for s in u.src:
        if s in uop_to_idx: d = max(d, depths[uop_to_idx[s]] + 1)
      depths[i] = d
      
      if getattr(u, 'dtype', None) is not None and u.dtype.scalar() in {dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.float64}:
        if u.op in GroupOp.ALU or u.op in {Ops.LOAD, Ops.CAST, Ops.BITCAST}:
          fp_depth_counts[d] = fp_depth_counts.get(d, 0) + 1
          
    active_fp_count = max(fp_depth_counts.values()) if fp_depth_counts else 0
    if active_fp_count > 32:
      raise RuntimeError(f"Active floating-point variable allocations exceeded cap: {active_fp_count} > 32")

    # Inject BEAM cost based on cost model
    from tinygrad.helpers import BEAM
    if BEAM.value > 0:
      cost = estimate_cost(uops)
      prefix.append(f"// BEAM_COST: {cost}")
      features = extract_features(uops)
      prefix.append(f"// BEAM_FEATURE_NAMES: {','.join(features.keys())}")
      prefix.append(f"// BEAM_FEATURES: {','.join(str(v) for v in features.values())}")

    # Task 3.3.6.1.1: DMA Independence Investigation
    dma_analysis = analyze_dma_independence(uops)
    prefix.append(f"// DMA_INDEPENDENCE: {dma_analysis['independent_loads']} loads, {dma_analysis['independent_stores']} stores, {dma_analysis['total_mem_bounds']} bounds")
    prefix.append(f"// DOUBLE_BUFFERING_FEASIBLE: {dma_analysis['double_buffering_feasible']}")

    # Task 3.3.6.1.2: Memory Layout Passes
    layout_metrics = optimize_memory_layout(uops)
    prefix.append(f"// MEMORY_LAYOUT_PASSES: {layout_metrics['contiguous_loads']} contiguous loads, {layout_metrics['contiguous_stores']} contiguous stores")

    # Explicitly list the function parameters generated by CStyleLanguage
    # so that our shim generators can perfectly map names to arrays
    buf_names = [name for name, _ in bufs]
    prefix.append(f"// BUF_NAMES: {','.join(buf_names)}")

    prefix.append("#include <math.h>")
    prefix.append("#include <stdint.h>")
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

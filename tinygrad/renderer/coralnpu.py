from typing import List, Dict
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, PtrDType
from collections import defaultdict

class CoralNPURenderer(Renderer):
  has_local = False # Force loops for local size
  has_shared = False
  
  def render(self, uops: List[UOp]) -> str:
    name = "kernel"
    src = f".text\n.global {name}\n{name}:\n"
    
    # Calculate refcounts
    refcounts: Dict[UOp, int] = defaultdict(int)
    for u in uops:
      for s in u.src:
        refcounts[s] += 1
    
    # Register allocation state
    # a0-a7: args
    # t0-t5: temps (t6 reserved for scratch)
    # s0-s11: temps
    regs: Dict[UOp, str] = {}
    
    # Available registers
    # t0-t5 (6), s0-s11 (12) = 18 registers
    available_regs = [f"t{i}" for i in range(6)] + [f"s{i}" for i in range(12)]
    
    scratch = "t6"
    
    def get_reg(u: UOp) -> str:
      nonlocal src
      if u in regs: return regs[u]
      
      # Allocate new register
      if not available_regs:
        raise RuntimeError(f"Out of registers! Active uops: {len(regs)}")
        
      r = available_regs.pop(0)
      regs[u] = r
      
      # If const, emit load immediately
      if u.op is Ops.CONST:
        # Handle integer constants
        val = int(u.arg)
        src += f"  li {r}, {val}\n"
      
      return r
      
    def release_reg(u: UOp):
      if u in regs:
        refcounts[u] -= 1
        if refcounts[u] <= 0:
          # Check if it's an arg (param) - don't free args
          if not regs[u].startswith("a"):
             r = regs[u]
             # Return to pool at the end to avoid immediate reuse confusion
             available_regs.append(r)
             del regs[u]

    # Pre-assign args
    arg_idx = 0
    for u in uops:
      if u.op is Ops.PARAM:
        regs[u] = f"a{arg_idx}"
        arg_idx += 1
        refcounts[u] += 999999 # Keep params alive

    # Stack to track loops
    loop_stack: List[tuple[UOp, str, str]] = []

    for u in uops:
      if u.op in {Ops.PARAM, Ops.SINK, Ops.DEFINE_VAR}: continue
      if u.op is Ops.CONST: continue # Handled on demand
      
      if u.op is Ops.SPECIAL:
         # Loop indices should be handled if has_local=False?
         # If we still see SPECIAL, it might be global index or we need to handle it.
         # For now, treat as 0
         r = get_reg(u)
         # src += f"  li {r}, 0 # SPECIAL {u.arg}\n"
         pass

      elif u.op is Ops.INDEX:
        dest = get_reg(u)
        ptr = get_reg(u.src[0])
        idx = get_reg(u.src[1])
        
        scale = 1
        if isinstance(u.src[0].dtype, PtrDType):
             scale = u.src[0].dtype.itemsize
             
        if scale > 1:
             # Use scratch register to calculate offset
             # Handle power of 2
             if (scale & (scale - 1)) == 0:
                 shift = scale.bit_length() - 1
                 src += f"  slli {scratch}, {idx}, {shift}\n"
             else:
                 src += f"  li {scratch}, {scale}\n"
                 src += f"  mul {scratch}, {idx}, {scratch}\n"
             src += f"  add {dest}, {ptr}, {scratch}\n"
        else:
             src += f"  add {dest}, {ptr}, {idx}\n"
             
        release_reg(u.src[0])
        release_reg(u.src[1])

      elif u.op is Ops.LOAD:
        dest = get_reg(u)
        addr = get_reg(u.src[0])
        width = u.dtype.itemsize
        instr = "lw" if width == 4 else "lb" if width == 1 else "lh"
        src += f"  {instr} {dest}, 0({addr})\n"
        release_reg(u.src[0])

      elif u.op is Ops.STORE:
        val = get_reg(u.src[1])
        addr = get_reg(u.src[0])
        width = u.src[1].dtype.itemsize
        instr = "sw" if width == 4 else "sb" if width == 1 else "sh"
        src += f"  {instr} {val}, 0({addr})\n"
        release_reg(u.src[0])
        release_reg(u.src[1])

      elif u.op is Ops.ADD:
        dest = get_reg(u)
        s1 = get_reg(u.src[0])
        s2 = get_reg(u.src[1])
        src += f"  add {dest}, {s1}, {s2}\n"
        release_reg(u.src[0])
        release_reg(u.src[1])
        
      elif u.op is Ops.MUL:
        dest = get_reg(u)
        s1 = get_reg(u.src[0])
        s2 = get_reg(u.src[1])
        src += f"  mul {dest}, {s1}, {s2}\n"
        release_reg(u.src[0])
        release_reg(u.src[1])

      elif u.op is Ops.RANGE:
        # Start of loop
        r = get_reg(u)
        # Range src is (start, end) or just end?
        # Usually src[0] is start, src[1] is end? Or just end?
        # Let's assume start is 0 if only one src, or look at u.src
        # Ops.RANGE args: (start, stop, step)?
        # Actually UOps usually have sources.
        # Based on previous error: UOp(Ops.RANGE, dtypes.int, arg=(0, AxisType.LOOP), src=(UOp(Ops.CONST, dtypes.int, arg=2, src=()),))
        # It has 1 source: the end (2). Start is implicit 0?
        # If 1 source, it's (0, src[0]).
        
        start_val = 0
        end_reg = get_reg(u.src[0])
        
        src += f"  li {r}, {start_val}\n"
        label = f"loop_{len(loop_stack)}"
        src += f"{label}:\n"
        loop_stack.append((u, label, end_reg))
        
        # Don't release sources yet, loops are tricky

      elif u.op is Ops.END:
        loop_uop, label, end_reg = loop_stack.pop()
        loop_var = regs[loop_uop]
        src += f"  addi {loop_var}, {loop_var}, 1\n"
        src += f"  blt {loop_var}, {end_reg}, {label}\n"
        
        # Now we can release
        release_reg(loop_uop.src[0]) 

    src += "  ret\n"
    return src

from tinygrad.device import Compiled, Allocator, Compiler, CompilerSet, BufferSpec
from tinygrad.renderer.coralnpu import CoralNPURenderer
import os
import functools
import ctypes

class CoralNPUAllocator(Allocator):
  def __init__(self, device):
    self.device = device
    self.mem = {}
    self.shms = {}
    self.next_handle = 1
    import multiprocessing
    self.lock = multiprocessing.Lock()
    super().__init__(device)

  def _alloc(self, size:int, options:BufferSpec):
    import multiprocessing.shared_memory
    with self.lock:
      handle = self.next_handle
      self.next_handle += 1
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=size)
    self.shms[handle] = shm
    self.mem[handle] = (ctypes.c_char * size).from_buffer(shm.buf) # type: ignore
    return handle
    
  def _copyin(self, dest, src:memoryview):
    if dest in self.mem:
      ctypes.memmove(self.mem[dest], src.tobytes(), len(src))
    else:
      raise ValueError(f"Invalid handle {dest}")
    
  def _copyout(self, dest:memoryview, src):
    if src in self.mem:
      data = bytes(self.mem[src])[:len(dest)]
      dest[:] = data
    else:
      raise ValueError(f"Invalid handle {src}")

  def _free(self, opaque, options):
    if opaque in self.mem:
      del self.mem[opaque]
      shm = self.shms.pop(opaque)
      shm.close()
      shm.unlink()

class CoralNPUProgram:
  def __init__(self, device, name:str, lib:bytes, *args, runtimevars=None):
    self.device = device
    self.name = name
    self.lib = lib
    self.args = args
    self.runtimevars = runtimevars
    
    import os, re
    self.is_beam = int(os.environ.get("BEAM", "0")) > 0
    self.lib_so = None
    self.fxn = None
    
    if self.is_beam:
      src = lib.decode()
      match = re.search(r"//\s*BEAM_COST:\s*([0-9.]+)", src)
      self.beam_cost = float(match.group(1)) if match else float(len(src))

  def _compile_on_host(self, src):
    raise RuntimeError("COMP-2.1.2.1: Host compiler invocation strictly prohibited. Use Bazel coralnpu_v2_binary.")

  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, timeout=None, **kwargs):
    if getattr(self, "is_beam", False) and wait:
      return self.beam_cost

    if self.fxn is None:
      self.lib_so = self._compile_on_host(self.lib.decode())
      self.fxn = getattr(self.lib_so, self.name)

    c_args = []
    for buf_handle in bufs:
      if buf_handle in self.device.allocator.mem:
        c_args.append(ctypes.addressof(self.device.allocator.mem[buf_handle]))
      else:
        pass
    
    for v in vals:
      c_args.append(ctypes.c_int(v))

    if timeout is not None:
      import multiprocessing
      def _exec(fxn_ptr, args):
        import ctypes
        arg_types = [ctypes.c_void_p if isinstance(a, int) else type(a) for a in args]
        func = ctypes.CFUNCTYPE(None, *arg_types)(fxn_ptr)
        func(*(ctypes.c_void_p(a) if isinstance(a, int) else a for a in args))
  
      fxn_ptr = ctypes.cast(self.fxn, ctypes.c_void_p).value
      p = multiprocessing.Process(target=_exec, args=(fxn_ptr, c_args))
      p.start()
      p.join(timeout=timeout)
      if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"CoralNPU execution timed out after {timeout} seconds.")
      if p.exitcode != 0:
        raise RuntimeError(f"CoralNPU execution failed with exit code {p.exitcode}")
      return 0.0
    else:
      self.fxn(*(ctypes.c_void_p(arg) if isinstance(arg, int) else arg for arg in c_args))
      return 0.0

class CoralNPUDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, CoralNPUAllocator(self), CompilerSet([(CoralNPURenderer, None)]), functools.partial(CoralNPUProgram, self))

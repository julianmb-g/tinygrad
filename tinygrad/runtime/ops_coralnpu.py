from tinygrad.device import Compiled, Allocator, Compiler, CompilerSet, BufferSpec
from tinygrad.renderer.coralnpu import CoralNPURenderer
import os
import functools
import subprocess
import ctypes
import tempfile

class CoralNPUAllocator(Allocator):
  def __init__(self, device):
    self.device = device
    self.mem = {}
    self.next_handle = 1
    super().__init__(device)

  def _alloc(self, size:int, options:BufferSpec):
    handle = self.next_handle
    self.next_handle += 1
    # Allocate bytearray with size
    self.mem[handle] = (ctypes.c_char * size)()
    return handle
    
  def _copyin(self, dest, src:memoryview):
    # dest is handle
    if dest in self.mem:
      # Convert source memoryview to bytes
      # ctypes.memmove expects a pointer or object that can be converted to pointer as dest
      # and a bytes-like object as src
      ctypes.memmove(self.mem[dest], src.tobytes(), len(src))
    else:
      raise ValueError(f"Invalid handle {dest}")
    
  def _copyout(self, dest:memoryview, src):
    # src is handle
    if src in self.mem:
      # dest is a memoryview, we can get its buffer address or cast to char pointer
      # But ctypes.memmove expects (dst_ptr, src_ptr, size)
      # We can use (ctypes.c_char * len(dest)).from_buffer(dest) if dest is writeable
      
      # Simplified: read from ctypes array into python bytes, then copy to memoryview
      # This is less efficient but safer
      data = bytes(self.mem[src])
      dest[:] = data
    else:
      raise ValueError(f"Invalid handle {src}")

  def _free(self, opaque, options):
    if opaque in self.mem:
      del self.mem[opaque]

class CoralNPUProgram:
  def __init__(self, device, name:str, lib:bytes, *args, runtimevars=None):
    self.device = device
    self.name = name
    self.lib = lib
    self.args = args
    self.runtimevars = runtimevars
    self.lib_so = self._compile_on_host(lib.decode())
    self.fxn = self.lib_so[name]
    
  def _compile_on_host(self, src):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cc', mode='w') as f:
      f.write(src)
      src_path = f.name
    so_path = src_path + ".so"
    # Use g++ because we are emitting C++ (float4 construction style float4{...} is C++)
    # -shared -fPIC needed for CDLL
    subprocess.check_call(['g++', '-shared', '-fPIC', '-o', so_path, src_path])
    return ctypes.CDLL(so_path)

  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False):
    # print(f"CoralNPU Executing {self.name} on Host")
    # Convert handles to pointers
    c_args = []
    for buf_handle in bufs:
      if buf_handle in self.device.allocator.mem:
        # Pass the ctypes object directly, ctypes handles it as pointer
        c_args.append(self.device.allocator.mem[buf_handle])
      else:
        # Handle unexpected types?
        pass
    
    # Add vals
    for v in vals:
      c_args.append(ctypes.c_int(v))

    # Run
    self.fxn(*c_args)
    return 0.0

class CoralNPUDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, CoralNPUAllocator(self), CompilerSet([(CoralNPURenderer, None)]), functools.partial(CoralNPUProgram, self))

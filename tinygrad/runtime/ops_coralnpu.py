from tinygrad.device import Compiled, Allocator, Compiler, CompilerSet, BufferSpec
from tinygrad.renderer.coralnpu import CoralNPURenderer
import os

class CoralNPUAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec):
    # Just return a dummy address/handle
    return 1
    
  def _copyin(self, dest, src:memoryview):
    pass
    
  def _copyout(self, dest:memoryview, src):
    pass

class CoralNPUProgram:
  def __init__(self, name:str, lib:bytes, *args, runtimevars=None):
    self.name = name
    self.lib = lib
    
  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False):
    print(f"CoralNPU Executing {self.name} with global_size={global_size} local_size={local_size}")
    print(f"Source:\n{self.lib.decode()}")

class CoralNPUDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, CoralNPUAllocator(self), CompilerSet([(CoralNPURenderer, None)]), CoralNPUProgram)

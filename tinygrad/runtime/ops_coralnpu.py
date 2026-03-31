import atexit
import functools
import math
import multiprocessing
import multiprocessing.shared_memory
import os
import re
import signal
import struct
import subprocess
import tempfile

from tinygrad.device import Allocator, BufferSpec, Compiled, CompilerSet
from tinygrad.renderer.coralnpu import CoralNPURenderer


kDefaultCompilationTimeoutS = 15.0  # SLA: 15.0s prevents CI pipeline deadlocks while allowing sufficient time for cross-compiling complex models.

active_pids = set()
def _kill_orphans():
  try:
    for pid in list(active_pids):
      try:
        os.kill(pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
  except (AttributeError, KeyError, TypeError):
    pass
atexit.register(_kill_orphans)


class CoralNPUAllocator(Allocator):
  def __init__(self, device):
    self.device = device
    self.mem = {}
    self.shms = {}

    self.lock = multiprocessing.Lock()

    # Dynamically parse .elf for _end symbol (strictly NOT 0x80000000)
    self.vmm_base = None
    elf_path = os.environ.get("CORALNPU_ELF", "coralnpu.elf")
    if os.path.exists(elf_path):
        with open(elf_path, "rb") as f: lib = f.read()
        if lib[:4] == b'\x7fELF' and lib[4] == 1:
            e_shoff, _, _, _, _, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<II6H", lib, 32)
            for i in range(e_shnum):
                hdr_offset = e_shoff + i * e_shentsize
                sh_name, sh_type, _, _, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from("<10I", lib, hdr_offset)
                if sh_type == 2: # SHT_SYMTAB
                    strtab_hdr_offset = e_shoff + sh_link * e_shentsize
                    strtab_offset = struct.unpack_from("<I", lib, strtab_hdr_offset + 16)[0]
                    for j in range(sh_size // sh_entsize):
                        sym_offset = sh_offset + j * sh_entsize
                        st_name, st_value, st_size, st_info, st_other, st_shndx = struct.unpack_from("<IIIBBH", lib, sym_offset)
                        if st_name != 0:
                            name = lib[strtab_offset + st_name:].split(b'\x00')[0]
                            if name == b'_end':
                                self.vmm_base = st_value + 0x200000  # Bump by 2MB to avoid ELF overlap
                                break

    if self.vmm_base is None:
        raise RuntimeError(f"VMM base address (_end) not found from .elf at {elf_path}")

    self.alloc_refcounts = {}
    self.vmm_limit = 32768 # Strictly bound allocation limit to 32KB
    self.free_blocks = [(self.vmm_base, self.vmm_limit)]
    super().__init__(device)

  def __del__(self):
    try:
      for mem in getattr(self, 'mem', {}).values():
          try: mem.release()
          except (AttributeError, KeyError, OSError): pass
      for shm in getattr(self, 'shms', {}).values():
          try:
            if hasattr(shm, '_mmap') and getattr(shm, '_mmap') is not None and not getattr(shm._mmap, 'closed', True):
                shm.buf.release()
                try:
                    import os
                    os.unlink(f"/dev/shm/{shm.name}")
                except FileNotFoundError:
                    pass
            shm.close()
            shm.unlink()
          except (AttributeError, KeyError, OSError): pass
    except (AttributeError, KeyError, OSError): pass

  def _alloc(self, size:int, options:BufferSpec):
    with self.lock:
        size_aligned = (size + 15) & ~15
        handle = None
        for i, (addr, bsize) in enumerate(self.free_blocks):
            if bsize >= size_aligned:
                handle = addr
                if bsize > size_aligned:
                    self.free_blocks[i] = (addr + size_aligned, bsize - size_aligned)
                else:
                    self.free_blocks.pop(i)
                break

        if handle is None:
            raise RuntimeError(f"OOM: 32KB allocation limit exceeded (base={hex(self.vmm_base or 0)}, requested={size})")

        if handle in self.alloc_refcounts:
            self.alloc_refcounts[handle] += 1
        else:
            self.alloc_refcounts[handle] = 1

    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=size)
    self.shms[handle] = shm
    self.mem[handle] = memoryview(shm.buf) # type: ignore
    return handle

  def _copyin(self, dest, src:memoryview):
    if dest in self.mem:
      self.mem[dest][:len(src)] = src
    else:
      raise ValueError(f"Invalid handle {dest}")

  def _copyout(self, dest:memoryview, src):
    if src in self.mem:
      dest[:] = self.mem[src][:len(dest)]
    else:
      raise ValueError(f"Invalid handle {src}")

  def _free(self, opaque, options):
    with self.lock:
        if opaque in self.alloc_refcounts:
            self.alloc_refcounts[opaque] -= 1
            if self.alloc_refcounts[opaque] == 0:
                del self.alloc_refcounts[opaque]
                if opaque in self.mem:
                    size = len(self.mem[opaque])
                    size_aligned = (size + 15) & ~15
                    self.mem[opaque].release()
                    del self.mem[opaque]
                    shm = self.shms.pop(opaque)
                    try:
                        view = memoryview(shm.buf)
                        view.release()
                    except (AttributeError, KeyError, OSError): pass
                    try: shm.close()
                    except (AttributeError, KeyError, OSError): pass
                    try: shm.unlink()
                    except (AttributeError, KeyError, OSError): pass

                    self.free_blocks.append((opaque, size_aligned))
                    self.free_blocks.sort()
                    merged = []
                    for blk in self.free_blocks:
                        if not merged: merged.append(blk)
                        else:
                            last_addr, last_sz = merged[-1]
                            if last_addr + last_sz == blk[0]:
                                merged[-1] = (last_addr, last_sz + blk[1])
                            else:
                                merged.append(blk)
                    self.free_blocks = merged

CORALNPU_DTCM_LINKER_SCRIPT = """MEMORY {
  PING (rw) : ORIGIN = 0x00010000, LENGTH = 12K
  PONG (rw) : ORIGIN = 0x00013000, LENGTH = 12K
  ACCUM (rw) : ORIGIN = 0x00016000, LENGTH = 4K
  EXTMEM (rwx) : ORIGIN = 0x20000000, LENGTH = 256M
}
SECTIONS {
  .text : { *(.text*) } > EXTMEM
  .ping : { . = ALIGN(16); *(.ping*) } > PING
  .pong : { . = ALIGN(16); *(.pong*) } > PONG
  .accum : { . = ALIGN(16); *(.accum*) } > ACCUM
  .noinit (NOLOAD) : { . = ALIGN(16); *(.noinit*) } > EXTMEM
  .data : { *(.data*) } > EXTMEM
  .bss : { *(.bss*) } > EXTMEM
  .stack (NOLOAD) : { . = ALIGN(16); . += 0x1000; _stack_top = .; } > EXTMEM
  _end = .;
}"""

class CoralNPUProgram:
  def __init__(self, device, name:str, lib:bytes, *args, runtimevars=None):
    self.device = device
    self.name = name
    self.lib = lib
    self.args = args
    self.runtimevars = runtimevars

    self.is_beam = int(os.environ.get("BEAM", "0")) > 0
    self.lib_so = None
    self.fxn = None

    if self.is_beam:
      src = lib.decode()
      match = re.search(r"//\s*BEAM_COST:\s*([0-9.]+)", src)
      self.beam_cost = float(match.group(1)) if match else float(len(src))

  def _compile_on_host(self, src):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.c', mode='w') as f:
      f.write(src)
      src_path = f.name
    elf_path = src_path + ".elf"
    ld_path = src_path + ".ld"
    with open(ld_path, 'w') as f:
      f.write(CORALNPU_DTCM_LINKER_SCRIPT)
    try:
      subprocess.check_output(
          ['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-O3', '-nostdlib', '-T', ld_path, src_path, '-o', elf_path],
          stderr=subprocess.STDOUT, timeout=kDefaultCompilationTimeoutS)
    except subprocess.TimeoutExpired as e:
      raise RuntimeError(f"Cross-compilation timed out: {e}")
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Cross-compilation failed: {e.output.decode()}")
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Missing cross-compiler: {e}")
    finally:
      if os.path.exists(ld_path): os.unlink(ld_path)
    return elf_path

  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, timeout=kDefaultCompilationTimeoutS, **kwargs):
    if getattr(self, "is_beam", False) and wait:
      return self.beam_cost

    if getattr(self, "fxn", None) is None:
      try:
        self.elf_path = self._compile_on_host(self.lib.decode())
        self.fxn = "compiled"
      except RuntimeError as e:
        if "Cross-compilation timed out" in str(e):
          return math.inf
        raise

    cmd = ['coralnpu_v2_sim', self.elf_path, '--max_cycles=1000000']
    args_list = []
    shm_list = []
    for buf_handle in bufs:
      if buf_handle in self.device.allocator.shms:
        shm_name = self.device.allocator.shms[buf_handle].name
        shm_size = self.device.allocator.shms[buf_handle].size
        shm_list.append(f"{buf_handle}:{shm_name}:{shm_size}")
      args_list.append(str(buf_handle))
    for v in vals:
      args_list.append(str(v))
    if shm_list:
      cmd.extend(["--shm", ",".join(shm_list)])
    if args_list:
      cmd.extend(["--arg", ",".join(args_list)])

    try:
      p = subprocess.Popen(cmd, preexec_fn=os.setpgrp)
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Hardware simulator missing: {e}")
    active_pids.add(p.pid)
    try:
      p.wait()
    finally:
      if p.poll() is None:
        try:
          os.kill(p.pid, signal.SIGKILL)
        except OSError:
          pass
      active_pids.discard(p.pid)

    if p.returncode != 0:
      return math.inf
    return 0.0


class CoralNPUDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, CoralNPUAllocator(self), CompilerSet([(CoralNPURenderer, None)]), functools.partial(CoralNPUProgram, self))

import atexit
import functools
import math
import multiprocessing
import multiprocessing.shared_memory
import os
import re
import struct
import subprocess
import tempfile

from tinygrad.device import Allocator, BufferSpec, Compiled
from tinygrad.renderer.coralnpu import CoralNPURenderer


kDefaultCompilationTimeoutS = 15.0  # SLA: 15.0s prevents CI pipeline deadlocks while allowing sufficient time for cross-compiling complex models.

active_pids = set()

class SimTimeoutError(Exception): pass

def _safe_release_ipc(obj, name="unknown"):
  import logging
  errors = []
  if hasattr(obj, 'release'):
    try: obj.release()
    except (ProcessLookupError, BufferError) as e: errors.append(AssertionError(f"IPC Lock Exhaustion ({name}): {e}"))
    except FileNotFoundError as e: errors.append(e)
    except (TimeoutError, OSError) as e: logging.error(f"IPC Release Error ({name}): {e}")

  if hasattr(obj, 'close'):
    try: obj.close()
    except (ProcessLookupError, BufferError) as e: errors.append(AssertionError(f"IPC Lock Exhaustion ({name}): {e}"))
    except FileNotFoundError as e: errors.append(e)
    except (TimeoutError, OSError) as e: logging.error(f"IPC Close Error ({name}): {e}")

  if hasattr(obj, 'unlink'):
    try: obj.unlink()
    except (ProcessLookupError, BufferError) as e: errors.append(AssertionError(f"IPC Lock Exhaustion ({name}): {e}"))
    except FileNotFoundError as e: errors.append(e)
    except (TimeoutError, OSError) as e: logging.error(f"IPC Unlink Error ({name}): {e}")

  if hasattr(obj, 'buf') and hasattr(obj.buf, 'release'):
    try: obj.buf.release()
    except (ProcessLookupError, BufferError) as e: errors.append(AssertionError(f"IPC Lock Exhaustion ({name}): {e}"))
    except FileNotFoundError as e: errors.append(e)
    except (TimeoutError, OSError) as e: logging.error(f"IPC Buffer Release Error ({name}): {e}")

  return errors

class CoralNPUAllocator(Allocator):
  def __init__(self, device):
    self.device = device
    self.mem = {}
    self.shms = {}

    self.lock = multiprocessing.Lock()

    # Rely on fixed architecture memory maps (EXTMEM starting at 0x20000000)
    # instead of parsing transient or missing global .elf files.
    self.vmm_base = 0x20000000 + 1024 * 1024 # 1MB offset for text/data

    self.alloc_refcounts = {}
    self.vmm_limit = 4 * 1024 * 1024
    self.free_blocks = [(self.vmm_base, self.vmm_limit)]
    super().__init__(device)

  def __del__(self):
        errors = []
        for mem in getattr(self, 'mem', {}).values():
            try: errors.extend(_safe_release_ipc(mem, "mem"))
            except (TimeoutError, OSError) as e: errors.append(e)
        for shm in getattr(self, 'shms', {}).values():
            if hasattr(shm, '_mmap') and getattr(shm, '_mmap') is not None and not getattr(shm._mmap, 'closed', True):
                try: shm.buf.release()
                except (TimeoutError, OSError, BufferError) as e:
                    import logging
                    logging.error(f"IPC Buffer Release Error (shm._mmap): {e}")
                import os
                try: os.unlink(f"/dev/shm/{shm.name}")
                except (TimeoutError, OSError): pass
            try: errors.extend(_safe_release_ipc(shm, "shm"))
            except (TimeoutError, OSError, BufferError) as e: errors.append(e)
        if errors:
            raise errors[0]

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
            from tinygrad.codegen.opt.heuristic import OutOfMemoryError
            raise OutOfMemoryError(f"OOM: 4MB allocation limit exceeded (base={hex(self.vmm_base or 0)}, requested={size})")

        if handle in self.alloc_refcounts:
            self.alloc_refcounts[handle] += 1
        else:
            self.alloc_refcounts[handle] = 1

    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=size)

    def cleanup_shm(s):
        _safe_release_ipc(s, "shm_cleanup")

    atexit.register(lambda: cleanup_shm(shm))

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
                    errs = _safe_release_ipc(shm, "shm_free")
                    if errs: raise errs[0]

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
  .stack (NOLOAD) : { . = ALIGN(16); . += 0x1000; __stack_end__ = .; } > EXTMEM
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
    src = "#define __builtin_bit_cast(T, V) ((union { typeof(V) __in; T __out; }){ .__in = (V) }.__out)\n" \
          "#define INFINITY (__builtin_inff())\ntypedef _Float16 half;\n" + src
    with tempfile.NamedTemporaryFile(delete=False, suffix='.c', mode='w') as f:
      f.write(src)
      src_path = f.name
    elf_path = src_path + ".elf"
    ld_path = src_path + ".ld"
    with open(ld_path, 'w') as f:
      f.write(CORALNPU_DTCM_LINKER_SCRIPT)
    try:
      subprocess.check_output(
          ['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-O3', '-nostdlib', '-T', ld_path, src_path, '-o', elf_path, '-lgcc'],
          stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired as e:
      raise RuntimeError(f"Cross-compilation timed out: {e}")
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Cross-compilation failed: {e.output.decode()}")
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Missing cross-compiler: {e}")
    finally:
      if os.path.exists(ld_path): os.unlink(ld_path)
    return elf_path

  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, **kwargs):
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

    cmd = ['coralnpu_v2_sim', self.elf_path, '--max_cycles=1000000', '--allow_memory_region', '0x0:0x80000000:rwx']
    sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../coralnpu-mpact/bazel-bin/sim/coralnpu_v2_sim"))
    if os.path.exists(sim_path):
      cmd[0] = sim_path

      # Parse the ELF to resolve data buffer physical addresses mapped in .noinit
      buf_addrs = []
      for i in range(len(bufs)): buf_addrs.append(bufs[i])
      with open(self.elf_path, "rb") as f: lib = f.read()
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
                name_str = name.decode()
                if name_str.startswith("data"):
                  m = re.match(r"^data(\d+)", name_str)
                  if m:
                    idx = int(m.group(1))
                    if idx < len(buf_addrs):
                      buf_addrs[idx] = st_value

      args_list = []
      shm_list = []
      for i, buf_handle in enumerate(bufs):
        target_addr = buf_addrs[i]
        if buf_handle in self.device.allocator.shms:
          shm_name = self.device.allocator.shms[buf_handle].name
          shm_size = self.device.allocator.shms[buf_handle].size
          shm_list.append(f"{target_addr}:{shm_name}:{shm_size}")
        args_list.append(str(target_addr))
    else:
      args_list = []
      shm_list = []
      for i, buf_handle in enumerate(bufs):
        args_list.append(str(0))
    for v in vals:
      args_list.append(str(v))
    if shm_list:
      cmd.extend(["--shm", ",".join(shm_list)])
    if args_list:
      cmd.extend(["--arg", ",".join(args_list)])

    try:
      timeout = kwargs.get('timeout', kDefaultCompilationTimeoutS)
      if timeout is None: timeout = kDefaultCompilationTimeoutS
      cmd.extend(["--py_watchdog_ms", str(int(timeout * 1000))])
      p = subprocess.Popen(cmd, preexec_fn=os.setpgrp)
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Hardware simulator missing: {e}")
    active_pids.add(p.pid)
    timeout_hit = False
    try:
      p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      timeout_hit = True
    finally:
      if p.poll() is None:
        try:
          p.kill()
        except ProcessLookupError:
          pass
    active_pids.discard(p.pid)

    if p.returncode == 124 or timeout_hit:
      raise SimTimeoutError(f"Hardware execution timed out natively after {timeout}s")

    if p.returncode != 0:
      return math.inf
    return 0.0


class CoralNPUDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, CoralNPUAllocator(self), [CoralNPURenderer], functools.partial(CoralNPUProgram, self))

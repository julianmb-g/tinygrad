# Instruction format detection and decoding
from __future__ import annotations

from tinygrad.renderer.amd.dsl import EnumBitField, FixedBitField, Inst

# SDWA/DPP variant detection: src0 field (bits 0-8) encodes the variant
# 0xf9 (249) = SDWA, 0xfa (250) = DPP16 for CDNA (GFX9)
_VARIANT_SRC0 = {"_SDWA_SDST": 0xf9, "_SDWA": 0xf9, "_DPP16": 0xfa}

def _matches(data: bytes, cls: type[Inst]) -> bool:
  """Check if data matches all FixedBitFields and op is in allowed."""
  for _, field in cls._fields:
    dword_idx = field.lo // 32
    if len(data) < (dword_idx + 1) * 4: return False
    word = int.from_bytes(data[dword_idx*4:(dword_idx+1)*4], 'little')
    field_lo = field.lo % 32
    if isinstance(field, FixedBitField):
      if ((word >> field_lo) & field.mask) != field.default: return False
    if isinstance(field, EnumBitField) and field.allowed is not None:
      try: opcode = field.decode((word >> field_lo) & field.mask)
      except ValueError: return False  # opcode not in enum
      if opcode not in field.allowed: return False
  # Check SDWA/DPP variant based on src0 field (bits 0-8) - only for variant classes
  name = cls.__name__
  word = int.from_bytes(data[:4], 'little')
  for suffix, expected_src0 in _VARIANT_SRC0.items():
    if name.endswith(suffix): return (word & 0x1ff) == expected_src0
  return True

# Lazy-load instruction format tables to avoid circular imports (ins.py imports dsl.py which is in this package)
_FORMATS: dict[str, list[type[Inst]]] | None = None

def _load_formats() -> dict[str, list[type[Inst]]]:
  global _FORMATS
  if _FORMATS is not None: return _FORMATS
  from tinygrad.runtime.autogen.amd.cdna.ins import DS as C_DS
  from tinygrad.runtime.autogen.amd.cdna.ins import FLAT as C_FLAT
  from tinygrad.runtime.autogen.amd.cdna.ins import GLOBAL as C_GLOBAL
  from tinygrad.runtime.autogen.amd.cdna.ins import MUBUF as C_MUBUF
  from tinygrad.runtime.autogen.amd.cdna.ins import SCRATCH as C_SCRATCH
  from tinygrad.runtime.autogen.amd.cdna.ins import SMEM as C_SMEM
  from tinygrad.runtime.autogen.amd.cdna.ins import SOP1 as C_SOP1
  from tinygrad.runtime.autogen.amd.cdna.ins import SOP2 as C_SOP2
  from tinygrad.runtime.autogen.amd.cdna.ins import SOPC as C_SOPC
  from tinygrad.runtime.autogen.amd.cdna.ins import SOPK as C_SOPK
  from tinygrad.runtime.autogen.amd.cdna.ins import SOPK_LIT as C_SOPK_LIT
  from tinygrad.runtime.autogen.amd.cdna.ins import SOPP as C_SOPP
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP1 as C_VOP1
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP1_DPP16 as C_VOP1_DPP16
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP1_SDWA as C_VOP1_SDWA
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP2 as C_VOP2
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP2_DPP16 as C_VOP2_DPP16
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP2_LIT as C_VOP2_LIT
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP2_SDWA as C_VOP2_SDWA
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP3 as C_VOP3
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP3_SDST as C_VOP3_SDST
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP3P as C_VOP3P
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP3P_MFMA as C_VOP3P_MFMA
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP3PX2 as C_VOP3PX2
  from tinygrad.runtime.autogen.amd.cdna.ins import VOP3SD as C_VOP3SD
  from tinygrad.runtime.autogen.amd.cdna.ins import VOPC as C_VOPC
  from tinygrad.runtime.autogen.amd.cdna.ins import VOPC_SDWA_SDST as C_VOPC_SDWA_SDST
  from tinygrad.runtime.autogen.amd.rdna3.ins import (
    DS,
    FLAT,
    GLOBAL,
    SCRATCH,
    SMEM,
    SOP1,
    SOP1_LIT,
    SOP2,
    SOP2_LIT,
    SOPC,
    SOPK,
    SOPK_LIT,
    SOPP,
    VINTERP,
    VOP1,
    VOP1_LIT,
    VOP1_SDST,
    VOP2,
    VOP2_LIT,
    VOP3,
    VOP3_SDST,
    VOP3P,
    VOP3SD,
    VOPC,
    VOPD,
  )
  from tinygrad.runtime.autogen.amd.rdna4.ins import DS as R4_DS
  from tinygrad.runtime.autogen.amd.rdna4.ins import SMEM as R4_SMEM
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOP1 as R4_SOP1
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOP1_LIT as R4_SOP1_LIT
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOP2 as R4_SOP2
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOP2_LIT as R4_SOP2_LIT
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOPC as R4_SOPC
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOPC_LIT as R4_SOPC_LIT
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOPK as R4_SOPK
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOPK_LIT as R4_SOPK_LIT
  from tinygrad.runtime.autogen.amd.rdna4.ins import SOPP as R4_SOPP
  from tinygrad.runtime.autogen.amd.rdna4.ins import VFLAT as R4_FLAT
  from tinygrad.runtime.autogen.amd.rdna4.ins import VGLOBAL as R4_GLOBAL
  from tinygrad.runtime.autogen.amd.rdna4.ins import VINTERP as R4_VINTERP
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP1 as R4_VOP1
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP1_LIT as R4_VOP1_LIT
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP1_SDST as R4_VOP1_SDST
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP2 as R4_VOP2
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP2_LIT as R4_VOP2_LIT
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP3 as R4_VOP3
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP3_SDST as R4_VOP3_SDST
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP3P as R4_VOP3P
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOP3SD as R4_VOP3SD
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOPC as R4_VOPC
  from tinygrad.runtime.autogen.amd.rdna4.ins import VOPD as R4_VOPD
  from tinygrad.runtime.autogen.amd.rdna4.ins import VSCRATCH as R4_SCRATCH
  # Order matters: more specific encodings first, catch-alls (SOP2, VOP2) last
  # Order: base before _LIT (base matches regular ops, _LIT catches lit-only ops excluded from base)
  _FORMATS = {
    "rdna3": [VOPD, VOP3P, VINTERP, VOP3SD, VOP3_SDST, VOP3, DS, GLOBAL, SCRATCH, FLAT, SMEM,
              SOP1, SOP1_LIT, SOP2, SOP2_LIT, SOPC, SOPK, SOPK_LIT, SOPP, VOPC, VOP1_SDST, VOP1, VOP1_LIT, VOP2, VOP2_LIT],
    "rdna4": [R4_VOPD, R4_VOP3P, R4_VINTERP, R4_VOP3SD, R4_VOP3_SDST, R4_VOP3, R4_DS, R4_GLOBAL, R4_SCRATCH, R4_FLAT, R4_SMEM,
              R4_SOP1, R4_SOP1_LIT, R4_SOPC, R4_SOPC_LIT, R4_SOPP, R4_SOPK, R4_SOPK_LIT, R4_VOPC, R4_VOP1_SDST, R4_VOP1, R4_VOP1_LIT,
              R4_SOP2, R4_SOP2_LIT, R4_VOP2, R4_VOP2_LIT],
    "cdna": [C_VOP3PX2, C_VOP3P_MFMA, C_VOP3P, C_VOP3SD, C_VOP3_SDST, C_VOP3, C_DS, C_GLOBAL, C_SCRATCH, C_FLAT, C_MUBUF, C_SMEM,
              C_SOP1, C_SOPC, C_SOPP, C_SOPK, C_SOPK_LIT, C_VOPC_SDWA_SDST, C_VOPC,
              C_VOP1_DPP16, C_VOP1_SDWA, C_VOP1, C_VOP2_DPP16, C_VOP2_SDWA, C_SOP2, C_VOP2, C_VOP2_LIT],
  }
  return _FORMATS

def detect_format(data: bytes, arch: str = "rdna3") -> type[Inst]:
  """Detect instruction format from machine code bytes."""
  assert len(data) >= 4, f"need at least 4 bytes, got {len(data)}"
  for cls in _load_formats()[arch]:
    if _matches(data, cls): return cls
  raise ValueError(f"unknown {arch} format word={int.from_bytes(data[:4], 'little'):#010x}")

def decode_inst(data: bytes, arch: str = "rdna3") -> Inst:
  """Decode machine code bytes into an instruction."""
  return detect_format(data, arch).from_bytes(data)

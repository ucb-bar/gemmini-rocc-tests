#!/usr/bin/env python3
"""
LUT Mapping Demo
================
Shows how every 4-bit nibble in A_in/B_in is used as an index into A_lut/B_lut
to recover the actual float values.

Packing convention (from write_c_header_tiled in golden_model.py, line 729):
  A_codes = [[(r[i+1] << 4) | r[i] for i in range(0, len(r), 2)] for r in A_codes]

So each byte stores two indices:
  - bits [3:0] (low nibble)  -> element at column c
  - bits [7:4] (high nibble) -> element at column c+1
"""

import sys
import os
import random
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, ".")
# from fp_decoder import ieee_to_recfn
from fp_decoder import recfn_to_ieee
import fp8_matmul_model
from fp8_matmul_model import matmul_outer_quantized_hwlike, compute_tile_scale_matrix, tiled_matmul_hwlike, matrix_mx_requantize, tensor_to_custom_fp_codes, make_fp_quantizer
from lut_golden_model import (
    make_lut,
    quantize_lut_indices,
    lut_lookup,
    codes_to_hex_rows,
    tiled_matmul_scaled_accum_hwlike,
    q_bf16_rne,
    hw_add_bf16,
    parse_fp_spec,
    pack_lut_hw_words,
    write_c_header_tiled_hw,
    compute_tile_scale_matrix_fpe8m0,
    _a_indices_to_hw_layout,

)

# def bf16_tensor_to_recfn_hex(t: torch.Tensor) -> list:
#     """Convert float32 tensor (on BF16 grid) to HardFloat recFN hex strings (17-bit, 5 hex digits)."""
#     raw = t.to(torch.bfloat16).view(torch.uint8)
#     # reconstruct 16-bit IEEE BF16 values (little-endian pairs)
#     rows = []
#     flat = t.to(torch.bfloat16).reshape(-1)
#     import struct
#     for val in flat:
#         ieee = struct.unpack('<H', struct.pack('<e', float(val.to(torch.float32))))[0]
#         # bfloat16 raw bits via view
#         buf = val.view(torch.int16).item() & 0xFFFF
#         rec = ieee_to_recfn(buf, exp_bits=8, mant_bits_total=8)
#         rows.append(f"{rec:05x}")
#     shape = t.shape
#     result, idx = [], 0
#     for i in range(shape[0]):
#         result.append([rows[idx + j] for j in range(shape[1])])
#         idx += shape[1]
#     return result

# ── Parameters (edit to match your run) ───────────────────────────────────────
SEED           = int(os.environ.get("MXGEMMINI_SEED", "0"))
M              = int(os.environ.get("MXGEMMINI_M", "128"))
K              = int(os.environ.get("MXGEMMINI_K", "512"))
N              = int(os.environ.get("MXGEMMINI_N", "128"))
INPUT_SPEC     = os.environ.get("MXGEMMINI_INPUT_SPEC", "fp6:e3m2")
LUT_INDEX_BITS = 4          # 2^4 = 16-entry LUT
DEV            = torch.device("cpu")
PROD_SPEC      = INPUT_SPEC  # product quantization spec (match hardware)
SCALE_SPEC     = "fpe8m0"    # scale factor format
GROUP          = 32          # K-group size for scales
TILE           = 16          # systolic tile size
QUANT_LUT_UPDATE_GRANULARITY = 1  # every 2^g rows of A / cols of B share one LUT
Gk             = K // GROUP
DEBUG_R        = 4             # print first DEBUG_R x DEBUG_R elements in debug blocks
# ──────────────────────────────────────────────────────────────────────────────

import math as _math

# ── Hardware-accurate BF16 → fp6:e3m2 conversion ─────────────────────────────
# Matches the RTL pipeline in BF16ScaleRoundToTiny.scala:
#   1) roundToMx: HardFloat RoundAnyRawFNToRecFN(8,8,4,3) RNE → E4M2
#   2) E4M2ToFp6: deterministic re-encoding to fp6:e3m2

def _bf16_to_e4m2_rne(x: torch.Tensor) -> torch.Tensor:
    """Round BF16 tensor to E4M2 (1+4+2, bias=7) using RNE.

    Bit-accurate match to HardFloat RoundAnyRawFNToRecFN(8,8,4,3,options=0)
    with round_near_even / tininess_afterRounding.

    E4M2: emin=-6, emax=7, sig=3 (1 hidden + 2 mantissa).
    """
    bf16 = x.to(torch.bfloat16)
    bits = bf16.view(torch.int16).to(torch.int32) & 0xFFFF

    sign_bit = (bits >> 15) & 1
    E = (bits >> 7) & 0xFF          # 8-bit biased BF16 exponent
    M = bits & 0x7F                  # 7-bit BF16 fractional mantissa

    sign_f = torch.where(sign_bit == 0, torch.ones_like(x), -torch.ones_like(x))
    e = E - 127                      # unbiased exponent (valid for normal BF16)

    # ── Normal BF16 → E4M2 normal (e ∈ [-6, 7]) ─────────────────────────────
    # 8-bit significand S = 128 + M.  Round to 3-bit (drop bottom 5):
    #   kept = {1, M[6], M[5]}   round = M[4]   sticky = M[3:0]!=0
    q       = (M >> 5) & 3                       # top 2 mantissa bits
    r       = ((M >> 4) & 1).bool()               # round bit
    sticky  = (M & 0xF).ne(0)                    # sticky bit
    lsb     = ((M >> 5) & 1).bool()               # LSB of kept bits

    round_up = r & (sticky | lsb)                 # RNE tie-breaking

    sig_rounded = q + round_up.to(torch.int32)    # 0..4
    carry = sig_rounded >= 4
    mant_out = torch.where(carry, torch.zeros_like(sig_rounded), sig_rounded)
    exp_out  = torch.where(carry, e + 1, e)       # unbiased E4M2 exponent

    is_overflow = exp_out > 7
    safe_exp = exp_out.float().clamp(-10, 10)
    val_normal = sign_f * (1.0 + mant_out.float() * 0.25) * torch.pow(
        torch.full_like(x, 2.0), safe_exp)
    val_normal = torch.where(is_overflow, sign_f * float('inf'), val_normal)

    # ── Below E4M2 emin: subnormal region ────────────────────────────────────
    # E4M2 subnormal quantum = 2^(emin - man) = 2^(-6 - 2) = 2^(-8)
    quantum = 2.0 ** -8

    # e = -7: BF16 ∈ [2^-7, 2^-6).  v/quantum = (128+M)/64 ∈ [2.0, ~3.98]
    is_e_neg7 = (e == -7) & (E >= 1)
    k7 = torch.where(M <= 32, torch.full_like(M, 2),         # ≤2.5 → 2 (even)
         torch.where(M <= 95, torch.full_like(M, 3),         # <3.5 → 3
                              torch.full_like(M, 4)))         # ≥3.5 → 4 (even, = norm)
    val_e7 = sign_f * k7.float() * quantum

    # e = -8: BF16 ∈ [2^-8, 2^-7).  v/quantum = (128+M)/128 ∈ [1.0, ~1.99]
    is_e_neg8 = (e == -8) & (E >= 1)
    k8 = torch.where(M < 64, torch.ones_like(M),             # <1.5 → 1
                              torch.full_like(M, 2))          # ≥1.5 → 2 (even)
    val_e8 = sign_f * k8.float() * quantum

    # e = -9: BF16 ∈ [2^-9, 2^-8).  v/quantum = (128+M)/256 ∈ [0.5, ~1.0)
    is_e_neg9 = (e == -9) & (E >= 1)
    k9 = torch.where(M == 0, torch.zeros_like(M),            # 0.5 tie → 0 (even)
                              torch.ones_like(M))             # >0.5 → 1
    val_e9 = sign_f * k9.float() * quantum

    # ── Assemble ─────────────────────────────────────────────────────────────
    result = val_normal
    result = torch.where(is_e_neg7, val_e7, result)
    result = torch.where(is_e_neg8, val_e8, result)
    result = torch.where(is_e_neg9, val_e9, result)
    result = torch.where(((e <= -10) & (E >= 1)) | (E == 0),
                         torch.zeros_like(x), result)

    # BF16 Inf / NaN → E4M2 Inf / NaN
    is_nan = (E == 255) & (M != 0)
    is_inf = (E == 255) & (M == 0)
    result = torch.where(is_inf, sign_f * float('inf'), result)
    result = torch.where(is_nan, torch.full_like(x, float('nan')), result)

    return result


def _e4m2_to_fp6(x: torch.Tensor) -> torch.Tensor:
    """Deterministic E4M2 → fp6:e3m2 mapping, matching E4M2ToFp6 hardware.

    E4M2 (bias=7) → fp6:e3m2 (bias=3), biasDiff=4.
      biased exp ≤ 2  → fp6 zero
      biased exp 3,4  → fp6 subnormal (hardware MuxLookup)
      biased exp 5–11 → fp6 normal (exp_adj = exp − 4, mantissa unchanged)
      biased exp > 11 → fp6 max (28.0); also Inf/NaN → max
    """
    sign = x.sign()
    ax   = x.abs()

    # mapToZero: E4M2 biased exp ≤ 2 → largest is 1.75*2^(-5) = 0.0546875
    mapToZero = (ax <= 0.0546875)

    # mapToMax: E4M2 biased exp > 11 → smallest is 2^5 = 32.0, or Inf/NaN
    mapToMax = (ax >= 32.0) | ~torch.isfinite(ax)

    # mapToSubnorm: E4M2 biased exp 3 (values 0.0625..0.109375)
    #               or biased exp 4 (values 0.125..0.21875)
    # Hardware lookup per (exp, sig) pair:
    #   exp3 sig=0,1 → k=1 (0.0625)   exp3 sig=2,3 → k=2 (0.125)
    #   exp4 sig=0,1 → k=2 (0.125)    exp4 sig=2,3 → k=3 (0.1875)
    mapToSubnorm = (ax >= 0.0625) & (ax <= 0.21875)
    sub_val = torch.where(ax <= 0.078125,  torch.full_like(ax, 0.0625),
              torch.where(ax <= 0.15625,   torch.full_like(ax, 0.125),
                                           torch.full_like(ax, 0.1875)))

    # fp6 max = 1.75 * 2^4 = 28.0
    out = x.clone()
    out = torch.where(mapToZero,    torch.zeros_like(x), out)
    out = torch.where(mapToMax,     sign * 28.0,         out)
    out = torch.where(mapToSubnorm, sign * sub_val,      out)
    # Normal range: float value is unchanged (same mantissa bits, exponent rebased)
    return out


_E3M2_Q = None
def _e3m2_quantizer():
    """MxQuant fp6_e3m2 grid, RNE (OCP round='even'), subnormals + saturate -- the golden reference
    (matches RTL BF16ToE3M2 / mx_fp_math.h::bf16_bits_to_fp6_e3m2_code, verified 0/3328 vs spike)."""
    global _E3M2_Q
    if _E3M2_Q is None:
        from elemwise import quantize_elemwise
        _E3M2_Q = lambda x: quantize_elemwise(x, 'fp6_e3m2', round='even')
    return _E3M2_Q


def hw_bf16_to_fp6(x: torch.Tensor) -> torch.Tensor:
    """Hardware-accurate BF16 -> fp6:e3m2: round STRAIGHT to the E3M2 grid, RNE (matches the rewritten
    BF16ScaleRoundToTiny / BF16ToE3M2). Replaces the old BF16->E4M2->fp6 double-rounding path."""
    return _e3m2_quantizer()(x.to(torch.bfloat16).to(torch.float32))


def _fp6_value_to_code(v: float) -> int:
    """Encode a single fp6:e3m2 grid value to its 6-bit code (handles subnormals)."""
    if v == 0.0:
        return 0
    s = 1 if v < 0 else 0
    av = abs(v)
    e_bits, m_bits, bias = 3, 2, 3
    emin = 1 - bias   # -2
    if av < 2.0 ** emin:          # subnormal: value = mant * 2^(emin - m_bits)
        quantum = 2.0 ** (emin - m_bits)   # 0.0625
        mant = int(round(av / quantum))
        return (s << (e_bits + m_bits)) | max(0, min(mant, (1 << m_bits) - 1))
    # normal
    E = int(_math.floor(_math.log2(av)))
    base = 2.0 ** E
    mant = int(round((av - base) / (base / 4)))
    if mant >= 4:
        mant = 0
        E += 1
    biased_exp = min(E + bias, (1 << e_bits) - 1)
    mant = min(mant, (1 << m_bits) - 1)
    return (s << (e_bits + m_bits)) | (biased_exp << m_bits) | mant


def _fp6_tensor_to_codes(t: torch.Tensor) -> list:
    """Encode a 2-D float tensor (already on fp6:e3m2 grid) to nested list of 6-bit codes."""
    arr = t.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    rows, cols = arr.shape
    return [[_fp6_value_to_code(float(arr[r, c].item())) for c in range(cols)]
            for r in range(rows)]


def _tensor_u8_bytes(t) -> bytes:
    arr = torch.as_tensor(t, dtype=torch.uint8).contiguous().cpu().numpy()
    return arr.tobytes(order="C")

def _tensor_u16_bytes(t) -> bytes:
    arr = torch.as_tensor(t, dtype=torch.uint16).contiguous().cpu().numpy()
    return arr.tobytes(order="C")

def write_tensor_bins(base_dir: str, A_indices: torch.Tensor, B_indices: torch.Tensor,
                      C_proj_hw: torch.Tensor, C_out_bf16: torch.Tensor) -> None:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A_in_hw = _a_indices_to_hw_layout(A_indices, a_tile_m=A_TILE_M, k_tile=K_TILE)
    B_in_hw = ((B_indices[:, 1::2] << 4) | B_indices[:, 0::2]).to(torch.uint8)
    C_out_bf16_bits = C_out_bf16.detach().to(torch.bfloat16).view(torch.uint16)

    (out_dir / "A_in.bin").write_bytes(_tensor_u8_bytes(A_in_hw))
    (out_dir / "B_in.bin").write_bytes(_tensor_u8_bytes(B_in_hw))
    (out_dir / "C_out_proj_hw.bin").write_bytes(_tensor_u8_bytes(C_proj_hw))
    (out_dir / "C_out_bf16.bin").write_bytes(_tensor_u16_bytes(C_out_bf16_bits))


# ── Reproduce A, B with the same seed as run_experiment ───────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)


# ── Build LUTs (from matmul_data_mx_fp6.h) ────────────────────────────────────
LUT_SIZE = 1 << LUT_INDEX_BITS  # 16


def _decode_fp6_e3m2(code: int) -> float:
    """Decode a 6-bit FP6 E3M2 raw code (1s|3e|2m, bias=3) to float32."""
    sign = (code >> 5) & 1
    exp  = (code >> 2) & 0x7
    mant = code & 0x3
    bias = 3
    if exp == 0:
        val = (mant / 4.0) * (2.0 ** (1 - bias))   # subnormal
    else:
        val = (1.0 + mant / 4.0) * (2.0 ** (exp - bias))
    return -val if sign else val


# ── Hardware-faithful FP6 nearest-LUT finder (mirrors FP6E3M2NearestFinder.scala) ──

def _raw_fp6e3m2_from_bits(val: int) -> dict:
    """Parse a 6-bit FP6 E3M2 value into raw components (mirrors rawFP6E3M2FromBits)."""
    val = val & 0x3F
    sign = (val >> 5) & 1
    exp  = (val >> 2) & 0b111  # bits [4:2]
    mant =  val       & 0b011  # bits [1:0]
    is_zero      = (exp == 0) and (mant == 0)
    is_subnormal = (exp == 0) and (mant != 0)
    s_exp        = -2 if exp == 0 else (exp - 3)
    implicit_bit = 0  if exp == 0 else 1
    sig = (implicit_bit << 2) | mant  # 3 bits: {implicit_bit, mant[1:0]}
    return {'sign': sign, 'is_zero': is_zero, 'is_subnormal': is_subnormal,
            's_exp': s_exp, 'sig': sig}


def fp6_to_fixed_point(val: int) -> int:
    """
    Convert a 6-bit FP6 E3M2 value to a signed fixed-point integer.
    Mirrors fp6ToFixedPoint inside FP6E3M2NearestFinder.scala.
    Fixed-point unit = 2^-4  (1.0 -> 16, 0.25 -> 4, 2.0 -> 32, …)
    """
    raw      = _raw_fp6e3m2_from_bits(val)
    shift_amt = (raw['s_exp'] + 2) & 0b111      # 3-bit uint, range [0, 6]
    shifted   = (raw['sig'] << shift_amt) & 0x1FF  # 9-bit
    magnitude = shifted & 0xFF                    # bits [7:0]
    signed    = -magnitude if raw['sign'] else magnitude
    return 0 if raw['is_zero'] else signed


def fp6e3m2_nearest_finder(in_fp6: int, in_lut: list) -> int:
    """
    Find the index of the nearest LUT entry to in_fp6.
    Mirrors FP6E3M2NearestFinder.scala.
    Tie-breaking: lower index wins (matches Mux(d1 <= d2, i1, i2)).
    """
    assert len(in_lut) == 16, "LUT must have exactly 16 entries"
    fixed_in  = fp6_to_fixed_point(in_fp6)
    fixed_lut = [fp6_to_fixed_point(v) for v in in_lut]
    diffs     = [abs(fixed_in - f) & 0x1FF for f in fixed_lut]  # 9-bit mask
    min_idx   = 0
    for i in range(1, 16):
        if diffs[i] < diffs[min_idx]:
            min_idx = i
    return min_idx

# ── Hardware-faithful FP8 E5M2 nearest-LUT finder (mirrors FP8NearestFinder.scala, altfmt=true) ──
def fp8_e5m2_to_fixed_point(val: int) -> int:
    val &= 0xFF
    sign = (val >> 7) & 1
    exp  = (val >> 2) & 0x1F
    mant =  val       & 0x3
    is_zero  = (exp == 0) and (mant == 0)
    implicit = 0 if exp == 0 else 1
    sig = (implicit << 2) | mant                   # 3 bits (sigW)
    s_exp = (1 - 15) if exp == 0 else (exp - 15)   # minSExp = -14
    shift_amt = (s_exp + 14) & 0x1F                # (s_exp + bias-1), shiftW = 5
    shifted = (sig << shift_amt) & 0xFFFFFFFF       # fixedW = 32
    signed = -shifted if sign else shifted
    return 0 if is_zero else signed

def fp8_e5m2_nearest_finder(in_code: int, in_lut: list) -> int:
    assert len(in_lut) == 16, "LUT must have exactly 16 entries"
    fixed_in  = fp8_e5m2_to_fixed_point(in_code)
    fixed_lut = [fp8_e5m2_to_fixed_point(v) for v in in_lut]
    diffs     = [abs(fixed_in - f) & 0x1FFFFFFFF for f in fixed_lut]  # 33-bit mask
    min_idx   = 0
    for i in range(1, 16):
        if diffs[i] < diffs[min_idx]:
            min_idx = i
    return min_idx

# ── Hardware-faithful FP6 E2M3 nearest-LUT finder (E2M3 = exp2 man3 bias1) ──
# Fixed-point = exact value * 8 (unit 2^-3): subnormal (exp field 0) -> mant; normal (field f>=1)
# -> (8+mant) << (f-1). Matches mx_fp_math.h::fp6_e2m3_to_fixed_point and the RTL E2M3 finder.
def fp6_e2m3_to_fixed_point(val: int) -> int:
    val &= 0x3F
    sign = (val >> 5) & 1
    exp  = (val >> 3) & 0x3
    mant =  val       & 0x7
    mag = mant if exp == 0 else ((8 + mant) << (exp - 1))
    return -mag if sign else mag

def fp6e2m3_nearest_finder(in_code: int, in_lut: list) -> int:
    assert len(in_lut) == 16, "LUT must have exactly 16 entries"
    fixed_in  = fp6_e2m3_to_fixed_point(in_code)
    fixed_lut = [fp6_e2m3_to_fixed_point(v) for v in in_lut]
    diffs     = [abs(fixed_in - f) for f in fixed_lut]
    min_idx   = 0
    for i in range(1, 16):
        if diffs[i] < diffs[min_idx]:
            min_idx = i
    return min_idx

# ── Hardware-faithful FP8 E4M3 nearest-LUT finder (mirrors FP8NearestFinder.scala, altfmt=false) ──
# bias=7, sigW=4, fixedW=18, shiftW=4. Smallest subnormal maps to 1. Matches mx_fp_math.h::
# fp8_e4m3_to_fixed_point and the RTL E4M3 finder. Used for the E4M3-quad 4-bit-LUT requant output.
def fp8_e4m3_to_fixed_point(val: int) -> int:
    val &= 0xFF
    sign = (val >> 7) & 1
    exp  = (val >> 3) & 0xF
    mant =  val       & 0x7
    is_zero  = (exp == 0) and (mant == 0)
    implicit = 0 if exp == 0 else 1
    sig = (implicit << 3) | mant                   # 4 bits (sigW)
    s_exp = (1 - 7) if exp == 0 else (exp - 7)     # minSExp = -6
    shift_amt = (s_exp + 6) & 0xF                  # (s_exp + bias-1), shiftW = 4
    shifted = (sig << shift_amt) & 0x3FFFF          # fixedW = 18
    signed = -shifted if sign else shifted
    return 0 if is_zero else signed

def fp8_e4m3_nearest_finder(in_code: int, in_lut: list) -> int:
    assert len(in_lut) == 16, "LUT must have exactly 16 entries"
    fixed_in  = fp8_e4m3_to_fixed_point(in_code)
    fixed_lut = [fp8_e4m3_to_fixed_point(v) for v in in_lut]
    diffs     = [abs(fixed_in - f) & 0x7FFFF for f in fixed_lut]      # 19-bit mask (fixedW+1)
    min_idx   = 0
    for i in range(1, 16):
        if diffs[i] < diffs[min_idx]:
            min_idx = i
    return min_idx

# BF16->E4M3 8-bit encode, RNE (OCP round='even'). Mirrors the rewritten mx_fp_math.h::fp8_e4m3_to_code
# + RTL BF16ToE4M3 (2026-09-10: E4M3/E5M2 switched from half-away to RNE). Used instead of
# tensor_to_custom_fp_codes for the E4M3 requant codes, which flushes subnormals.
def _rne(x: float) -> int:      # round half to EVEN for x >= 0 (matches nearbyintf)
    fl = int(_math.floor(x)); d = x - fl
    if d < 0.5: return fl
    if d > 0.5: return fl + 1
    return fl if (fl % 2 == 0) else fl + 1

def bf16f_to_e4m3_code_rha(v: float) -> int:
    if v == 0.0:
        return 0
    if not _math.isfinite(v):
        return (0x80 if _math.copysign(1.0, v) < 0 else 0x00) | 0x7F
    s  = 1 if _math.copysign(1.0, v) < 0 else 0
    av = abs(v)
    E  = _math.floor(_math.log2(av))
    bias, emin, emax = 7, -6, 8
    if E < emin:
        quantum = 2.0 ** (emin - 3)          # 2^-9
        k = _rne(av / quantum)
        if k <= 0: return s << 7
        if k >= 8: return (s << 7) | (1 << 3)
        return (s << 7) | k
    if E > emax:
        E_used, mant = emax, 6
    else:
        E_used = E
        base  = 2.0 ** E_used
        delta = base / 8.0
        k = _rne((av - base) / delta)
        if k >= 8:
            E_used += 1; k = 0
            if E_used > emax: E_used, k = emax, 6
        else:
            hi = 6 if E_used == emax else 7
            if k > hi: k = hi
            if k < 0:  k = 0
        mant = k
    return (s << 7) | (((E_used + bias) & 0xF) << 3) | (mant & 0x7)

IS_E5M2 = (INPUT_SPEC == "fp8:e5m2")
IS_E2M3 = (INPUT_SPEC == "fp6:e2m3")
IS_E4M3 = (INPUT_SPEC == "fp8:e4m3")

# Every 2^QUANT_LUT_UPDATE_GRANULARITY rows of A share one LUT; same for B cols
G = QUANT_LUT_UPDATE_GRANULARITY
print("[Step 1]: Generate the luts for fp6 projection")
A_luts = [make_lut(INPUT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(M >> G)]  # M >> G LUTs
B_luts = [make_lut(INPUT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(N >> G)]  # N >> G LUTs
C_luts = [make_lut(INPUT_SPEC, LUT_INDEX_BITS, DEV) for _ in range(M >> G)]  # M >> G LUTs

print("[Step 2]: Generate the projected data")
A_indices = torch.stack([quantize_lut_indices(A_luts[i >> G], A[i])      for i in range(M)])        # (M, K)
B_indices = torch.stack([quantize_lut_indices(B_luts[j >> G], B[:, j])   for j in range(N)], dim=1) # (K, N)

A_scale_exp = torch.randint(low=-4, high=4, size=(M, Gk), device=DEV)
B_scale_exp = torch.randint(low=-4, high=4, size=(Gk, N), device=DEV)

A_scales_row = torch.pow(2.0, A_scale_exp.to(torch.float32))
B_scales_col = torch.pow(2.0, B_scale_exp.to(torch.float32))

print("[Step 3]: Generate E8M0 scales")
A_scales_row_q = make_fp_quantizer(SCALE_SPEC, "nearest")(A_scales_row)
B_scales_col_q = make_fp_quantizer(SCALE_SPEC, "nearest")(B_scales_col)


print("[Step 4]: Generate HW like results C_out")
# Tile dimensions
A_TILE_M = 32   # A rows per tile
K_TILE   = 16   # K-dimension per tile (= systolic TILE)
B_TILE_N = 32   # B cols per tile

# Stack LUTs into tensors for efficient indexing
A_luts_t = torch.stack(A_luts)  # (M>>G, LUT_SIZE) — one LUT per 2^G A rows
B_luts_t = torch.stack(B_luts)  # (N>>G, LUT_SIZE) — one LUT per 2^G B cols

# ── Build full scaled A/B matrices and compute C_out via matmul_outer_quantized_hwlike ──
mi_full = (torch.arange(M, device=DEV) >> G).unsqueeze(1).expand(-1, K)
ni_full = (torch.arange(N, device=DEV) >> G).unsqueeze(0).expand(K, -1)
A_fp = A_luts_t[mi_full, A_indices]                                      # (M, K)
B_fp = B_luts_t[ni_full, B_indices]                                      # (K, N)

print(f"\n[After LUT projection]")
A_fp_codes, A_fp_bits = tensor_to_custom_fp_codes(A_fp[:DEBUG_R, :DEBUG_R], INPUT_SPEC)
B_fp_codes, B_fp_bits = tensor_to_custom_fp_codes(B_fp[:DEBUG_R, :DEBUG_R], INPUT_SPEC)
A_fp_hex = codes_to_hex_rows(A_fp_codes, A_fp_bits)
B_fp_hex = codes_to_hex_rows(B_fp_codes, B_fp_bits)
print(f"A_fp[:{DEBUG_R},:{DEBUG_R}] ({INPUT_SPEC}) [hex]:")
for row in A_fp_hex:
    print("  " + " ".join(row))
print(f"A_fp[:{DEBUG_R},:{DEBUG_R}] [float]:")
for row in A_fp[:DEBUG_R, :DEBUG_R].tolist():
    print("  " + " ".join(f"{v:>8.4f}" for v in row))
print(f"B_fp[:{DEBUG_R},:{DEBUG_R}] ({INPUT_SPEC}) [hex]:")
for row in B_fp_hex:
    print("  " + " ".join(row))
print(f"B_fp[:{DEBUG_R},:{DEBUG_R}] [float]:")
for row in B_fp[:DEBUG_R, :DEBUG_R].tolist():
    print("  " + " ".join(f"{v:>8.4f}" for v in row))

k_group_idx = torch.arange(K, device=DEV) // GROUP                       # (K,)
A_fp_scaled = A_fp * A_scales_row_q[:, k_group_idx]                      # (M, K)
B_fp_scaled = B_fp * B_scales_col_q[k_group_idx, :]                      # (K, N)

C_out = matmul_outer_quantized_hwlike(A_fp_scaled, B_fp_scaled)

for k_base in range(0, K, K_TILE):
    k_group = k_base // GROUP   # scale-group index — constant within K tile (K_TILE=16 <= GROUP=32)
    for n_base in range(0, N, B_TILE_N):
        # ── Pack B_in tile: (K_TILE × B_TILE_N) ─────────────────────────────
        b_idx = B_indices[k_base:k_base+K_TILE, n_base:n_base+B_TILE_N]  # (K_TILE, B_TILE_N)
        b_idx_combined = (b_idx[:, 1::2] << 4) | b_idx[:, 0::2]          # (K_TILE, B_TILE_N//2), msb=odd col, lsb=even col
        B_hex = codes_to_hex_rows(b_idx_combined, 8)
        #print(B_hex[15])
        b_lut = B_luts_t[n_base >> G:(n_base + B_TILE_N) >> G]            # (B_TILE_N>>G, LUT_SIZE)
        b_lut_in_codes, b_lut_in_bits = tensor_to_custom_fp_codes(b_lut, INPUT_SPEC)
        b_lut_in_codes_hex = codes_to_hex_rows(b_lut_in_codes, 8)
        #print(b_lut_in_codes_hex)
        B_lut_codes, B_lut_bits = tensor_to_custom_fp_codes(b_lut, INPUT_SPEC)
        ni    = (torch.arange(B_TILE_N, device=DEV) >> G).unsqueeze(0).expand(K_TILE, -1)  # col j -> lut j>>G
        B_in  = b_lut[ni, b_idx]                                           # (K_TILE, B_TILE_N)
        B_in_codes, B_in_bits = tensor_to_custom_fp_codes(B_in, INPUT_SPEC)
        B_in_codes_hex = codes_to_hex_rows(B_in_codes, 8)
        # print(B_in_codes_hex)
        # exit()
        # Per-element trace for k=15: before/after LUT projection
        k = 0
        #print(f"\n--- b_idx[{k}] nibble breakdown (byte_pos: hex | MSB[7:4]=odd_col | LSB[3:0]=even_col) ---")
        for c in range(B_TILE_N // 2):
            byte_hex = B_hex[k][c]
            msb_idx  = b_idx[k, 2*c+1].item()   # odd col  -> high nibble
            lsb_idx  = b_idx[k, 2*c  ].item()   # even col -> low nibble
            #print(f"  byte[{c:>2}]={byte_hex}  MSB(j={2*c+1})={msb_idx:x}  LSB(j={2*c})={lsb_idx:x}")
        #print(f"\n--- B_in[{k}] after projection: j, lut_row, idx, lut_val, fp6_hex ---")
        for j in range(B_TILE_N):
            lut_row = j >> G
            idx     = b_idx[k, j].item()
            lut_val = b_lut[lut_row, idx].item()
            #print(f"  j={j:>2}  lut_row={lut_row}  idx={idx:>2}  lut_val={lut_val:>8.4f}  fp6_hex={B_in_codes_hex[k][j]}")
        b_scale    = B_scales_col_q[k_group, n_base:n_base+B_TILE_N]      # (B_TILE_N,)
        B_in_scaled = B_in * b_scale.unsqueeze(0)                          # (K_TILE, B_TILE_N)
        for m_base in range(0, M, A_TILE_M):
            a_idx  = A_indices[m_base:m_base+A_TILE_M, k_base:k_base+K_TILE]  # (A_TILE_M, K_TILE)
            #a_idx = codes_to_hex_rows(a_idx, 8)
            a_idx_hw_layout = a_idx.reshape(A_TILE_M // 2, 2, K_TILE).permute(0, 2, 1).reshape(A_TILE_M // 2, K_TILE * 2)
            a_idx_combined = (a_idx_hw_layout[:, 1::2] << 4) | a_idx_hw_layout[:, 0::2]    # (A_TILE_M//2, K_TILE), msb=odd col, lsb=even col
            a_lut  = A_luts_t[m_base >> G:(m_base + A_TILE_M) >> G]               # (A_TILE_M>>G, LUT_SIZE)
            mi     = (torch.arange(A_TILE_M, device=DEV) >> G).unsqueeze(1).expand(-1, K_TILE)  # row m -> lut m>>G
            A_vals = a_lut[mi, a_idx]                                           # (A_TILE_M, K_TILE)
            a_scale    = A_scales_row_q[m_base:m_base+A_TILE_M, k_group]   # (A_TILE_M,)
            A_vals_scaled = A_vals * a_scale.unsqueeze(1)                   # (A_TILE_M, K_TILE)
            if m_base == 0 and n_base == 0 and k_base == 0:
                A_hex = codes_to_hex_rows(a_idx_combined, 8)
                A_vals_codes, _ = tensor_to_custom_fp_codes(A_vals, INPUT_SPEC)
                A_vals_codes_hex = codes_to_hex_rows(A_vals_codes, 8)
                print(A_vals_codes_hex[1])
                # A_vals in HW layout: interleave pairs of m-rows per k, shape (A_TILE_M//2, K_TILE*2)
                A_vals_hw = A_vals.reshape(A_TILE_M // 2, 2, K_TILE).permute(0, 2, 1).reshape(A_TILE_M // 2, K_TILE * 2)
                A_vals_hw_codes, _ = tensor_to_custom_fp_codes(A_vals_hw, INPUT_SPEC)
                A_vals_hw_hex = codes_to_hex_rows(A_vals_hw_codes, 8)
                # print(f"\n--- A_vals HW layout (A_TILE_M//2={A_TILE_M//2} rows x K_TILE*2={K_TILE*2} cols) ---")
                # print(f"  row r, col 2j   = a_val(row=2r,   k=j)  [LSB / even m-row]")
                # print(f"  row r, col 2j+1 = a_val(row=2r+1, k=j)  [MSB / odd  m-row]")
                # for r in range(min(A_TILE_M // 2, DEBUG_R)):
                #     print(f"  hw_row[{r:>2}]: {A_vals_hw_hex[r]}")
                # exit()
                r = 0  # inspect combined row 0 -> original A rows 0 (LSB) and 1 (MSB)
                print(f"\n--- A a_idx_combined[{r}] nibble breakdown (HW layout) ---")
                print(f"{'k':>3}  {'byte':>4}  {'MSB[7:4]=row{2*r+1}':>20}  {'LSB[3:0]=row{2*r}':>18}")
                for k_pos in range(K_TILE):
                    msb_idx = a_idx[2*r+1, k_pos].item()
                    lsb_idx = a_idx[2*r,   k_pos].item()
                    print(f"{k_pos:>3}  {A_hex[r][k_pos]:>4}  MSB=row{2*r+1},k{k_pos}:{msb_idx:>2x}  LSB=row{2*r},k{k_pos}:{lsb_idx:>2x}")
                print(f"\n--- A_vals after projection (row, k, lut_row, idx, fp6_hex) ---")
                for row in range(2):  # rows 0,1 covered by combined row r=0
                    lut_row = row >> G
                    for k_pos in range(K_TILE):
                        idx     = a_idx[row, k_pos].item()
                        lut_val = a_lut[lut_row, idx].item()
                        ##print(f"  row={row}  k={k_pos:>2}  lut_row={lut_row}  idx={idx:>2}  lut_val={lut_val:>8.4f}  fp6_hex={A_vals_codes_hex[row][k_pos]}")
                # Scale prints for this tile (k_base=0, n_base=0, m_base=0)
                sA = A_scales_row_q[m_base:m_base+A_TILE_M, k_group]       # (A_TILE_M,)
                sB = B_scales_col_q[k_group, n_base:n_base+B_TILE_N]       # (B_TILE_N,)
                sA_codes, sA_bits = tensor_to_custom_fp_codes(sA[:DEBUG_R].unsqueeze(1), SCALE_SPEC)
                sB_codes, sB_bits = tensor_to_custom_fp_codes(sB[:DEBUG_R].unsqueeze(1), SCALE_SPEC)
                sA_hex = codes_to_hex_rows(sA_codes, sA_bits)
                sB_hex = codes_to_hex_rows(sB_codes, sB_bits)
                print(f"\n--- A scales[:{DEBUG_R}] (separate, {SCALE_SPEC} hex) ---")
                print([row[0] for row in sA_hex])
                print(f"\n--- B scales[:{DEBUG_R}] (separate, {SCALE_SPEC} hex) ---")
                print([row[0] for row in sB_hex])
                S_joint = compute_tile_scale_matrix_fpe8m0(
                    A_scales_row_q, B_scales_col_q,
                    m0=m_base, n0=n_base, k0=k_base,
                    TM=A_TILE_M, TN=B_TILE_N,
                    M=M, N=N, K=K, group=GROUP, scale_spec=SCALE_SPEC,
                )
                S_codes, S_bits = tensor_to_custom_fp_codes(S_joint[:DEBUG_R, :DEBUG_R], SCALE_SPEC)
                S_hex = codes_to_hex_rows(S_codes, S_bits)
                print(f"\n--- Joint scale S_tile[:{DEBUG_R},:{DEBUG_R}] ({SCALE_SPEC} hex | float) ---")
                S_joint_sub = S_joint[:DEBUG_R, :DEBUG_R]
                for r in range(min(DEBUG_R, S_joint.shape[0])):
                    hex_row   = S_hex[r]
                    float_row = [f"{S_joint_sub[r, c].item():>8.4f}" for c in range(min(DEBUG_R, S_joint.shape[1]))]
                    print(f"  row{r}: {hex_row}  |  {float_row}")
                print(f"\n--- A scales[:{DEBUG_R}] float | hex ---")
                print([f"{v.item():.4f} ({h[0]})" for v, h in zip(sA[:DEBUG_R], sA_hex)])
                print(f"\n--- B scales[:{DEBUG_R}] float | hex ---")
                print([f"{v.item():.4f} ({h[0]})" for v, h in zip(sB[:DEBUG_R], sB_hex)])

                # C_out tile before and after scale (2x2)
                # Hardware-faithful: fp6 product quant + hw_add_bf16 accumulation, no scale
                _prod_quant = make_fp_quantizer(INPUT_SPEC, "zero")
                _in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")

                C_before_hw = None
                for k_pos in range(K_TILE):
                    a_in_k =A_vals[:DEBUG_R, k_pos]         # (DEBUG_R,)
                    b_in_k = B_in[k_pos, :DEBUG_R]           # (DEBUG_R,)
                    prod = A_vals[:, k_pos:k_pos+1] * B_in[k_pos:k_pos+1, :]
                    if C_before_hw is None:
                        C_before_hw = q_bf16_rne(prod)
                    else:
                        C_before_hw = hw_add_bf16(prod, C_before_hw)
                     
                    if k_pos == 0:
                        a_codes, a_bits = tensor_to_custom_fp_codes(a_in_k.unsqueeze(1), INPUT_SPEC)
                        b_codes, b_bits = tensor_to_custom_fp_codes(b_in_k.unsqueeze(1), INPUT_SPEC)
                        c_codes, c_bits = tensor_to_custom_fp_codes(C_before_hw[:DEBUG_R, :DEBUG_R], "bf16")
                        a_hex = [r[0] for r in codes_to_hex_rows(a_codes, a_bits)]
                        b_hex = [r[0] for r in codes_to_hex_rows(b_codes, b_bits)]
                        c_hex = codes_to_hex_rows(c_codes, c_bits)
                        print(f"\n--- k=0 ---")
                        print(f"  A_in[:{DEBUG_R}], hex={a_hex}")
                        print(f"  B_in[:{DEBUG_R}], hex={b_hex}")
                        print(f"  C[:{DEBUG_R},:{DEBUG_R}], bf16")
                        for row in c_hex:
                            print(f"    {row}")
                        #print(f"  C[:{DEBUG_R},:{DEBUG_R}]  bf16={c_hex}")
                        # c_rec = bf16_tensor_to_recfn_hex(C_before_hw[:DEBUG_R, :DEBUG_R])
                        # print(f"  C[:{DEBUG_R},:{DEBUG_R}]  recfn:")
                        # for row in c_rec:
                        #     print(f"    {row}")

                # ── Every K_TILE: per-tile + cumulative accumulation ──────────
                _pq_all = make_fp_quantizer(INPUT_SPEC, "zero")
                C_cumulative = None
                C_cumulative_wide = None  # tracks [:DEBUG_R, :N] across K tiles
                for t, kb in enumerate(range(0, K, K_TILE)):
                    # single tile accumulation (no scale)
                    c_tile = None
                    for k_pos in range(kb, kb + K_TILE):
                        prod = A_fp[:, k_pos:k_pos+1] * B_fp[k_pos:k_pos+1, :]
                        if c_tile is None:
                            c_tile = q_bf16_rne(prod)
                        else:
                            c_tile = hw_add_bf16(prod, c_tile)
                    
                    c_tile_r = c_tile[:DEBUG_R, :DEBUG_R]
                    # wide cumulative (no scale) across all N cols
                    c_tile_wide = c_tile[:DEBUG_R, :]
                    C_cumulative_wide = c_tile_wide if C_cumulative_wide is None else hw_add_bf16(C_cumulative_wide, c_tile_wide)
                    # scale and accumulate
                    St = compute_tile_scale_matrix(
                        A_scales_row_q, B_scales_col_q,
                        m0=m_base, n0=n_base, k0=kb, TM=DEBUG_R, TN=DEBUG_R
                    )
                    c_tile_scaled = q_bf16_rne(c_tile_r * St)
                    C_cumulative = c_tile_scaled if C_cumulative is None else hw_add_bf16(C_cumulative, c_tile_scaled)
                    # print
                    tile_codes, tile_bits = tensor_to_custom_fp_codes(c_tile_r, "bf16")
                    scaled_codes, scaled_bits = tensor_to_custom_fp_codes(c_tile_scaled, "bf16")
                    cum_codes, cum_bits = tensor_to_custom_fp_codes(C_cumulative, "bf16")
                   
                    def fmt_row(hex_list, sep=32):
                        groups = [" ".join(hex_list[g:g+sep]) for g in range(0, len(hex_list), sep)]
                        return " | ".join(groups)

                    print(f"\n=== K_TILE {t+1} (k={kb}..{kb+K_TILE-1}) ===")
                    print(f"  tile (no scale):")
                    for row in codes_to_hex_rows(tile_codes, tile_bits):
                        print(f"    {fmt_row(row)}")
                    print(f"  tile (scaled):")
                    for row in codes_to_hex_rows(scaled_codes, scaled_bits):
                        print(f"    {fmt_row(row)}")
                    print(f"  cumulative tiles 1..{t+1}:")
                    for row in codes_to_hex_rows(cum_codes, cum_bits):
                        print(f"    {fmt_row(row)}")
                    wide_codes, wide_bits = tensor_to_custom_fp_codes(C_cumulative_wide, "bf16")
                    print(f"  cumulative (all N cols, no scale) tiles 1..{t+1}:")
                    for i, row in enumerate(codes_to_hex_rows(wide_codes, wide_bits)):
                        print(f"    row{i:>2}: {fmt_row(row)}")
                # exit()


print("[Step 5]: Compare C_out with tiled_matmul_hwlike golden")
# print(fp8_matmul_model.TILE)  # should be 16 = K_TILE
# print(fp8_matmul_model.GROUP) # should be 32 = GROUP
# exit()
import precision   # per mesh row (exp, frac): --params <gemmini_params.h> when run directly, else the default
if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(); precision.add_arg(_p); precision.load(_p.parse_known_args()[0].params)
    prod_precision_list, acc_precision_list = precision.for_dim(TILE)
else:   # imported for its operands (gen_mx_lut, gen_lut_dim): the importer computes its own golden
    prod_precision_list, acc_precision_list = precision.default(TILE)
in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
A_in = in_q(A_fp)
B_in = in_q(B_fp)
C_golden = tiled_matmul_hwlike(
    A_in, B_in,
    A_scales_row_q, B_scales_col_q,
    verbose=False,
    prod_precision_list=prod_precision_list,
    acc_precision_list=acc_precision_list,
)
C_golden_bf16 = C_golden
C_golden_r = C_golden_bf16[:DEBUG_R, :DEBUG_R]
C_golden_r_codes, C_golden_r_bits = tensor_to_custom_fp_codes(C_golden_r, "bf16")
# print(f"\n--- C_golden[:{DEBUG_R},:{DEBUG_R}] (bf16 hex) ---")
# for row in codes_to_hex_rows(C_golden_r_codes, C_golden_r_bits):
#     print(row)
# diff = (C_out_bf16 - C_golden_bf16).abs()
# print(f"Max abs diff: {diff.max().item()}")
# print(f"Num mismatched elements: {(diff > 0).sum().item()} / {diff.numel()}")
# if diff.max().item() > 0:
#     idx = diff.argmax()
#     m, n = idx // N, idx % N
#     print(f"Largest diff at ({m},{n}): demo={C_out_bf16[m,n].item():.6f}  golden={C_golden_bf16[m,n].item():.6f}")

print("[Step 6]: Write C header with HW data layout")
C_out_bf16 = C_golden

print("\n[Step 7]: Requantize C_out_bf16 via matrix_mx_requantize")
C_requantized, C_req_scales = matrix_mx_requantize(
        C_out_bf16,
        quant_spec=INPUT_SPEC)
print(f"  quant_spec: {INPUT_SPEC}  scale_spec: {SCALE_SPEC}")

# Hardware-accurate BF16 → fp6 conversion (matches BF16ScaleRoundToTiny + E4M2ToFp6)
# The division by scale already happened in matrix_mx_requantize; snap to BF16 grid
# then apply the two-stage HW pipeline: RNE→E4M2, E4M2→fp6.
if IS_E5M2:
    # E5M2: snap the BF16 grid; the format-correct RNE encode (matching mx_fp_math.h::bf16_bits_to_e5m2_code
    # / RTL BF16ToE5M2, subnormals kept) happens at Step 8 via app.mxwire.encode_requant.
    C_requantized = q_bf16_rne(C_requantized)
    print(f"  Applied BF16 snap for {INPUT_SPEC} (E5M2 encode happens in Step 8)")
elif IS_E4M3:
    # E4M3-quad requant output: snap BF16 grid then quantize to the E4M3 grid. tensor_to_custom_fp_codes
    # does the format-correct encode in Step 8; keep the BF16-snapped value here (mirrors E2M3).
    C_requantized = q_bf16_rne(C_requantized)
    print(f"  Applied BF16 snap for {INPUT_SPEC} (E4M3 encode happens in Step 8)")
elif IS_E2M3:
    # E2M3 (exp2 man3 bias1): snap the BF16 grid, then tensor_to_custom_fp_codes does the E2M3
    # quantization+encode in Step 8 (RNE, subnormals, emax=1). Keep the BF16-snapped value here.
    C_requantized = q_bf16_rne(C_requantized)
    print(f"  Applied BF16 snap for {INPUT_SPEC} (E2M3 encode happens in Step 8)")
else:
    C_requantized = hw_bf16_to_fp6(q_bf16_rne(C_requantized))
    print(f"  Applied hw_bf16_to_fp6 (BF16 → E4M2 RNE → fp6:e3m2)")

# print(f"  C_requantized shape: {list(C_requantized.shape)}  C_req_scales shape: {list(C_req_scales.shape)}")
# req_codes, req_bits = tensor_to_custom_fp_codes(C_requantized, INPUT_SPEC)
# print(f"\n--- C_requantized (all {M}x{N}, {INPUT_SPEC} hex) ---")
# for i, row in enumerate(codes_to_hex_rows(req_codes, req_bits)):
#     print(f"  row {i:3d}: {row}")
# print(f"\n--- C_req_scales ({SCALE_SPEC} hex) ---")
scale_codes, scale_bits = tensor_to_custom_fp_codes(C_req_scales, SCALE_SPEC)
# for i, row in enumerate(codes_to_hex_rows(scale_codes, scale_bits)):
#     print(f"  row {i:3d}: {row}")

# Per-row, per-group breakdown (first 2 rows only)
Gk_out = C_out_bf16.shape[1] // GROUP
print(f"\n--- C per-row group breakdown (group_size={GROUP}, Gk={Gk_out}, showing first 2 rows) ---")
for m in range(min(2, C_out_bf16.shape[0])):
    print(f"  row {m:3d}:")
    bf16_row = C_out_bf16[m]
    for g in range(1):
        col_start = g * GROUP
        col_end   = col_start + GROUP
        golden_group = bf16_row[col_start:col_end]
        quant_group  = C_requantized[m, col_start:col_end]
        scale_val    = C_req_scales[m, g].item()
        scale_e8m0_hex = f"{scale_codes[m][g]:02x}"   # e8m0 = 8-bit biased exponent
        # maximal magnitude in the group
        abs_group = golden_group.abs()
        max_mag_val = abs_group.max().item()
        max_mag_bf16 = abs_group.max().to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
        q_codes, q_bits = tensor_to_custom_fp_codes(quant_group.unsqueeze(0), INPUT_SPEC)
        q_hex = codes_to_hex_rows(q_codes, q_bits)[0]
        golden_bits = golden_group.to(torch.bfloat16).view(torch.int16)
        golden_hex = " ".join(f"{b.item() & 0xFFFF:04x}" for b in golden_bits)
        print(f"    group {g} (cols {col_start:3d}-{col_end-1:3d}) \n"
              f"  max_magnitude={max_mag_val:.6g} (bf16=0x{max_mag_bf16:04x}) \n"
              f"  scale={scale_val:.6g} (e8m0=0x{scale_e8m0_hex}) \n"
              f"  golden_bf16=[{golden_hex}] \n"
              f"  quantized={q_hex}")


print("\n[Step 8]: project the quantized fp6 down to INT4 using C_luts")

# Pre-compute: convert every C_lut (float values) to 6-bit FP6 codes once
C_luts_t = torch.stack(C_luts)                                        # (M>>G, LUT_SIZE)
C_luts_codes_raw, _ = tensor_to_custom_fp_codes(C_luts_t, INPUT_SPEC) # list[list[int]], 6-bit

# Convert quantized C float values to their raw codes (8-bit for E5M2/E4M3, 6-bit for FP6)
if IS_E4M3:
    # E4M3 requant codes MUST round ties AWAY (MxQuant "nearest" == RTL BF16ToE4M3 == Spike). Using
    # tensor_to_custom_fp_codes here (Python round() ties-to-even) mis-rounds exact-half ties -> a few
    # nearest-index mismatches vs HW. Encode with the round-half-away mirror instead.
    Cf = C_requantized.float().view(M, N)
    C_req_codes_raw = [[bf16f_to_e4m3_code_rha(float(Cf[m, n].item())) for n in range(N)] for m in range(M)]
    C_req_bits = 8
elif IS_E5M2:
    # E5M2 requant codes: exact RNE encode via app.mxwire.encode_requant (= mx_fp_math.h
    # bf16_bits_to_e5m2_code / RTL BF16ToE5M2, subnormals kept). tensor_to_custom_fp_codes would flush
    # subnormals and mis-handle grid edges, giving a few nearest-index mismatches vs HW.
    from wire import encode_requant as _encode_requant
    import numpy as _np
    _Cf = C_requantized.float().view(M, N).detach().cpu().numpy().astype(_np.float32)
    C_req_codes_raw = _encode_requant(_Cf, dtype="fp8_e5m2").astype(_np.uint8).tolist()
    C_req_bits = 8
elif IS_E2M3:
    # tensor_to_custom_fp_codes does the format-correct quantize+encode (E2M3: RNE, subnormals, emax=1).
    C_req_codes_raw, C_req_bits = tensor_to_custom_fp_codes(C_requantized.float().view(M, N), INPUT_SPEC)
else:
    C_req_codes_raw = _fp6_tensor_to_codes(C_requantized.float().view(M, N))
    C_req_bits = 6

# For element (m, n): LUT = C_luts[m >> G]  (M-dim row grouping, same as A_luts)
# Find nearest LUT entry via fp6e3m2_nearest_finder and store 4-bit index
C_proj = torch.zeros(M, N, dtype=torch.int32)
for m in range(M):
    lut_idx      = m >> G
    lut_codes    = C_luts_codes_raw[lut_idx]
    for n in range(N):
        code_in      = C_req_codes_raw[m][n]
        finder       = (fp8_e5m2_nearest_finder if IS_E5M2 else
                        fp8_e4m3_nearest_finder if IS_E4M3 else
                        fp6e2m3_nearest_finder  if IS_E2M3 else
                        fp6e3m2_nearest_finder)
        C_proj[m, n] = finder(code_in, lut_codes)

# ── Debug print for first NUM_PRINT_MGRP row-groups, showing first NUM_PRINT_COLS cols ──
NUM_PRINT_MGRP = 2   # how many M-groups to show
NUM_PRINT_COLS = 4   # how many cols to show per group
m_group_size   = 1 << G

print(f"  INPUT_SPEC={INPUT_SPEC}  LUT_SIZE={LUT_SIZE}  M-group-size={m_group_size}")

for mg in range(min(NUM_PRINT_MGRP, M >> G)):
    m_start   = mg * m_group_size
    m_end     = m_start + m_group_size
    lut_hex   = [f"{c:02x}" for c in C_luts_codes_raw[mg]]
    lut_vals  = [f"{C_luts_t[mg, e].item():.4f}" for e in range(LUT_SIZE)]
    print(f"\n  === M-group {mg} (rows {m_start}-{m_end-1}), LUT[{mg}] ===")
    print(f"    LUT entries (fp6 hex): {lut_hex}")
    print(f"    LUT entries (float):   {lut_vals}")
    for m in range(m_start, m_end):
        fp6_hex      = [f"{C_req_codes_raw[m][n]:02x}" for n in range(NUM_PRINT_COLS)]
        indices      = [C_proj[m, n].item()            for n in range(NUM_PRINT_COLS)]
        lut_looked_up= [f"{C_luts_t[mg, indices[i]].item():.4f}" for i in range(NUM_PRINT_COLS)]
        print(f"    row {m:3d}: fp6={fp6_hex}  ->  idx={indices}  ->  lut_val={lut_looked_up}")

print(f"\n--- C_req_codes_raw all {M} rows (fp6 codes, hex) ---")
for m in range(M):
    groups = [" ".join(f"{C_req_codes_raw[m][n]:02x}" for n in range(g, min(g + 32, N)))
              for g in range(0, N, 32)]
    print(f"  row {m:3d}: " + " | ".join(groups))

print(f"\n--- C_golden_bf16 all {M} rows (bf16 hex) ---")
for m in range(M):
    groups = [" ".join(f"{C_golden_bf16[m, n].to(torch.bfloat16).view(torch.int16).item() & 0xFFFF:04x}"
                       for n in range(g, min(g + 32, N)))
              for g in range(0, N, 32)]
    print(f"  row {m:3d}: " + " | ".join(groups))

print(f"\n--- C_proj A_in HW layout [{M // 2}][{N}] (bits[3:0]=even row, bits[7:4]=odd row per col) ---")
C_proj_hw = _a_indices_to_hw_layout(C_proj, a_tile_m=A_TILE_M, k_tile=K_TILE)  # [M//2, N]
# for r in range(M // 2):
#     hw_hex = " ".join(f"{C_proj_hw[r][n]:02x}" for n in range(N))
#     print(f"  hw row {r:3d}: {hw_hex}")
for m in range(M // 2):
    groups = [" ".join(f"{C_proj_hw[m][n]:02x}" for n in range(g, min(g + 32, N)))
              for g in range(0, N, 32)]
    print(f"  row {m:3d}: " + " | ".join(groups))

print("\n[Step 9]: Write header with C_lut, C_proj indices, and C scales")
HEADER_PATH = os.environ.get("MXGEMMINI_HEADER_PATH", "./include/matmul_fp6_128x128x512.h")
write_c_header_tiled_hw(
    path           = HEADER_PATH,
    M=M, K=K, N=N,
    group          = GROUP,
    input_spec     = INPUT_SPEC,
    acc_spec       = "bf16",
    scale_spec     = SCALE_SPEC,
    lut_index_bits = LUT_INDEX_BITS,
    A_in           = A_indices,
    B_in           = B_indices,
    A_scales_row_q = A_scales_row_q,
    B_scales_col_q = B_scales_col_q,
    C_out_bf16     = C_out_bf16,
    C_out_quantized= C_proj,
    C_out_scales   = C_req_scales,
    A_lut          = A_luts_t,
    B_lut          = B_luts_t,
    C_lut          = C_luts_t,
    a_tile_m       = A_TILE_M,
    k_tile         = K_TILE,
)
print(f"Header written to {HEADER_PATH}")

# Append C_proj in A_in HW layout [M//2, N] to the header
with open(HEADER_PATH, "r") as f:
    content = f.read()
# Remove existing #endif, append C_proj_hw array, then re-close with #endif
guard = HEADER_PATH.upper()
for ch in [".", "/", "\\", "-"]:
    guard = guard.replace(ch, "_")
rows_hw = M // 2
c_proj_hw_lines = ",\n".join(
    "    { " + ", ".join(f"0x{C_proj_hw[r][n]:02x}" for n in range(N)) + " }"
    for r in range(rows_hw)
)
c_proj_hw_section = (
    f"// C_proj A_in HW layout [{rows_hw}][{N}]: bits[3:0]=even row, bits[7:4]=odd row\n"
    f"static const uint8_t C_proj_hw[{rows_hw}][{N}] = {{\n{c_proj_hw_lines}\n}};\n\n"
    f"#endif // {guard}\n"
)
content = content.replace(f"#endif // {guard}\n", c_proj_hw_section)
with open(HEADER_PATH, "w") as f:
    f.write(content)
print("C_proj_hw appended to header.")

if os.environ.get("MXGEMMINI_BIN_DIR"):   # optional raw tensor dumps (debug only)
    BIN_DIR = Path(os.environ["MXGEMMINI_BIN_DIR"])
    write_tensor_bins(BIN_DIR, A_indices, B_indices, C_proj_hw, C_out_bf16)
    print(f"Binary tensors written to {BIN_DIR}")


#!/usr/bin/env python3
"""
FP4 E2M1 Direct Matmul - same flow as fp8_matmul_model.py.
Only difference from FP8: tiles are 32×32×16 (TM=TN=32, TK=16) vs 16×16×16,
and PROD_MANT_BITS=3 (1.m × 1.m → ≤3 fractional bits after normalization).
"""

import math
import re
import sys
import random
import struct
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch

sys.path.insert(0, ".")

from lut_golden_model import _a_indices_to_hw_layout  # only used for HW header packing

Tensor = torch.Tensor

# ── Hardware parameters ────────────────────────────────────────────────────────
INPUT_SPEC     = "fp4:e2m1"   # 1 sign + 2 exp (bias=1) + 1 mant = 4 bits
PROD_MANT_BITS = 3             # 1.m × 1.m → at most 3 fractional bits (cf. FP8: 7)
TILE_M         = 32            # M-dim of one hardware tile (cf. FP8: 16)
TILE_N         = 32            # N-dim of one hardware tile (cf. FP8: 16)
TILE_K         = 16            # K-dim of one hardware tile (same as FP8)
GROUP          = 32            # MX quantization group size along K
SCALE_SPEC     = "fpe8m0"
SEED           = 0
DEBUG_R        = 8
M, K, N        = 128, 128, 128
# ──────────────────────────────────────────────────────────────────────────────

_FP_PRESETS = {
    "fp16":     dict(exp=5, man=10),
    "bf16":     dict(exp=8, man=7),
    "fp8:e4m3": dict(exp=4, man=3),
    "fp8:e5m2": dict(exp=5, man=2),
    "fp6:e3m2": dict(exp=3, man=2),
    "fp4:e2m1": dict(exp=2, man=1),
    "fpe8m0":   dict(exp=8, man=0),
}

def ieee_to_recfn(ieee: int, exp_bits: int, mant_bits_total: int) -> int:
    """
    Convert an IEEE-like encoding to HardFloat recoded float (recFN).
    Inverse of recfn_to_ieee.

    Parameters: same as recfn_to_ieee.
    For BF16: exp_bits=8, mant_bits_total=8  -> recWidth=17 bits.
    """
    expWidth  = exp_bits
    sigWidth  = mant_bits_total
    fracWidth = sigWidth - 1
    recWidth  = 1 + expWidth + sigWidth

    sign     = (ieee >> (expWidth + fracWidth)) & 1
    ieee_exp = (ieee >> fracWidth) & ((1 << expWidth) - 1)
    frac     = ieee & ((1 << fracWidth) - 1)

    # bias_term matches fNFromRecFN: expOut = sExp - bias_term  =>  sExp = ieee_exp + bias_term
    bias_term = (1 << (expWidth - 1)) + 1   # 129 for BF16

    if ieee_exp == 0 and frac == 0:
        # Zero
        rec_exp  = 0
        rec_frac = 0
    elif ieee_exp == 0:
        # IEEE subnormal: normalize to find sExp < minNormExp
        # find position of leading 1 in frac
        leading = fracWidth - 1
        while leading >= 0 and not ((frac >> leading) & 1):
            leading -= 1
        shift = fracWidth - 1 - leading          # number of leading zeros
        # sExp = 1 - bias_ieee + bias_term - shift,  bias_ieee = (1<<(expWidth-1))-1
        sExp     = 1 - ((1 << (expWidth - 1)) - 1) + bias_term - shift
        rec_exp  = sExp & ((1 << (expWidth + 1)) - 1)
        rec_frac = ((frac << (shift + 1)) & ((1 << fracWidth) - 1))
    elif ieee_exp == (1 << expWidth) - 1:
        if frac == 0:
            # Infinity: top2=0b11, bit[expWidth-2]=0
            rec_exp  = (0b11 << (expWidth - 1))
            rec_frac = 0
        else:
            # NaN: top2=0b11, bit[expWidth-2]=1
            rec_exp  = (0b11 << (expWidth - 1)) | (1 << (expWidth - 2))
            rec_frac = frac
    else:
        # Normal
        rec_exp  = ieee_exp + bias_term
        rec_frac = frac

    rec = (sign << (expWidth + sigWidth)) | (rec_exp << fracWidth) | rec_frac
    return rec & ((1 << recWidth) - 1)

def parse_fp_spec(spec: str) -> Tuple[int, int]:
    s = spec.strip().lower()
    if s in _FP_PRESETS:
        return _FP_PRESETS[s]["exp"], _FP_PRESETS[s]["man"]
    if s == "fp32":
        return 8, 23
    m = re.fullmatch(r"fp\d+:(e(\d+)m(\d+))", s)
    if m:
        return int(m.group(2)), int(m.group(3))
    raise ValueError(f"Unrecognized floating-point spec: {spec}")

def trunc_product_mantissa(x: Tensor, frac_bits: int) -> Tensor:
    x = x.to(torch.float32)
    is_finite = torch.isfinite(x)
    ax = x.abs()
    out = x.clone()
    nz = (ax != 0) & is_finite
    if not torch.any(nz):
        return out
    xn = x[nz]
    m, e = torch.frexp(xn)
    sign = torch.sign(m)
    m = m.abs()
    m2 = m * 2.0
    e2 = e - 1
    frac = m2 - 1.0
    scale = float(1 << frac_bits)
    frac_q = torch.floor(frac * scale) / scale
    m2_q = 1.0 + frac_q
    out_nz = (sign * m2_q) * torch.ldexp(torch.ones_like(m2_q), e2)
    out_nz = torch.where(out_nz.abs() < torch.finfo(torch.float32).tiny,
                         torch.zeros_like(out_nz), out_nz)
    out[nz] = out_nz
    return out

def q_bf16_rne(x: Tensor) -> Tensor:
    return x.to(torch.bfloat16).to(torch.float32)

def _fp_emax(exp: int, man: int) -> int:
    """Unbiased maximum exponent, accounting for formats with no Inf/NaN.

    Standard IEEE:  all-ones biased_exp reserved → emax = bias
    FP8 E4M3 (MX): biased_exp=15 mostly valid (only mant=7 is NaN) → emax = bias+1
    FP4 E2M1 (MX): no Inf/NaN at all, all-ones biased_exp is normal → emax = bias+2
    General no-inf/nan rule: emax = (1<<exp) - 1 - bias
    """
    bias = (1 << (exp - 1)) - 1
    is_mx_fp8  = (exp == 4 and man == 3)   # FP8 E4M3: one NaN code, otherwise valid
    is_no_special = (exp == 2 and man == 1) # FP4 E2M1: no Inf/NaN at all
    if is_mx_fp8:
        return bias + 1
    if is_no_special:
        return (1 << exp) - 1 - bias       # = 3 - 1 = 2 for E2M1
    return bias

def float_quantize_trunc(x: Tensor, exp: int, man: int) -> Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    out = torch.zeros_like(x)
    if torch.all(x == 0):
        return out
    sign = torch.sign(x)
    ax = x.abs()
    nz_mask = ax > 0
    ax_nz = ax[nz_mask]
    if ax_nz.numel() == 0:
        return out
    log2_ax = torch.log2(ax_nz)
    E = torch.floor(log2_ax)
    bias = (1 << (exp - 1)) - 1
    emin = 1 - bias
    is_mx_fp8 = (exp == 4 and man == 3)
    emax = _fp_emax(exp, man)
    # Subnormal range: 0 < ax < 2^emin (biased_exp=0, value = k * subnorm_delta)
    subnorm_delta = 2.0 ** (emin - man)   # ULP of subnormals = 2^(1-bias-man)
    subnorm_mask  = (E < emin)
    overflow_mask = (E > emax)
    normal_mask   = (~subnorm_mask) & (~overflow_mask)
    ax_q = torch.zeros_like(ax_nz)
    if subnorm_mask.any():
        # RTZ: truncate to nearest subnormal step toward zero
        k = torch.floor(ax_nz[subnorm_mask] / subnorm_delta)
        k = k.clamp(0, 2**man - 1)
        ax_q[subnorm_mask] = k * subnorm_delta
    if overflow_mask.any():
        E_max     = float(emax)
        base_max  = torch.pow(torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device), E_max)
        delta_max = base_max / (2 ** man)
        max_mant  = (2**man - 2) if is_mx_fp8 else (2**man - 1)
        ax_q[overflow_mask] = base_max + max_mant * delta_max
    if normal_mask.any():
        E_norm = E[normal_mask]
        x_norm = ax_nz[normal_mask]
        base  = torch.pow(torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device), E_norm)
        delta = base / (2 ** man)
        t = (x_norm - base) / delta
        k = torch.floor(torch.clamp(t, 0, 2**man - 1 - 1e-7))
        ax_q[normal_mask] = base + k * delta
    out[nz_mask] = sign[nz_mask] * ax_q
    return out

def make_fp_quantizer(spec: str, rounding: str = "nearest"):
    s = spec.strip().lower()
    if s in ("", "none", "identity", "fp32"):
        return None
    e, m = parse_fp_spec(spec)
    rounding = rounding.lower()
    if rounding in ("nearest", "nearest_even", "stochastic"):
        from qtorch.quant import float_quantize
        mode = "nearest" if rounding in ("nearest", "nearest_even") else "stochastic"
        return lambda x: float_quantize(x, exp=e, man=m, rounding=mode)
    if rounding in ("zero", "toward_zero", "trunc", "rtz"):
        return lambda x: float_quantize_trunc(x, e, m)
    raise ValueError(f"Unsupported rounding mode: {rounding}")

def tensor_to_custom_fp_codes(t: Tensor, spec: str) -> Tuple[List[List[int]], int]:
    e_bits, m_bits = parse_fp_spec(spec)
    total_bits = 1 + e_bits + m_bits
    bias = (1 << (e_bits - 1)) - 1
    emin = 1 - bias
    is_mx_fp8 = (e_bits == 4 and m_bits == 3)
    emax = _fp_emax(e_bits, m_bits)
    subnorm_delta = 2.0 ** (emin - m_bits)  # ULP of subnormals
    arr = t.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    if arr.ndim != 2:
        raise ValueError("Expected 1D or 2D tensor for hex dump.")
    rows, cols = arr.shape
    out_codes: List[List[int]] = []
    for r in range(rows):
        row_codes: List[int] = []
        for c in range(cols):
            v = float(arr[r, c].item())
            if v == 0.0 or not math.isfinite(v):
                code = 0
            else:
                s = 1 if v < 0 else 0
                av = abs(v)
                E = math.floor(math.log2(av))
                if E < emin:
                    # Subnormal: biased_exp=0, mant = round(av / subnorm_delta)
                    mant = int(round(av / subnorm_delta))
                    mant = max(0, min(mant, 2**m_bits - 1))
                    # mant==0 is ±0; canonicalize to +0 (code=0), hw treats both as zero
                    code = 0 if mant == 0 else (s << (e_bits + m_bits)) | mant
                else:
                    if E > emax:
                        E_used = emax
                        mant = (2**m_bits - 2) if is_mx_fp8 else (2**m_bits - 1)
                    else:
                        E_used = E
                        base  = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        tpos  = (av - base) / delta
                        mant  = int(round(tpos))
                        if mant >= 2**m_bits:
                            E_used += 1
                            mant = 0
                            if E_used > emax:
                                E_used = emax
                                mant = (2**m_bits - 2) if is_mx_fp8 else (2**m_bits - 1)
                        else:
                            max_mant = (2**m_bits - 2) if (is_mx_fp8 and E_used == emax) else (2**m_bits - 1)
                            mant = max(0, min(mant, max_mant))
                    exp_bits_val = int(E_used + bias)
                    code = ((s & 0x1) << (e_bits + m_bits)) | \
                           ((exp_bits_val & ((1 << e_bits) - 1)) << m_bits) | \
                           (mant & ((1 << m_bits) - 1))
            row_codes.append(code)
        out_codes.append(row_codes)
    return out_codes, total_bits

def codes_to_hex_rows(codes: List[List[int]], total_bits: int) -> List[List[str]]:
    width = (total_bits + 3) // 4
    return [[f"{code:0{width}x}" for code in row] for row in codes]

def c_type_for_bits(total_bits: int) -> str:
    if total_bits <= 8:   return "uint8_t"
    elif total_bits <= 16: return "uint16_t"
    elif total_bits <= 32: return "uint32_t"
    else:                  return "uint64_t"

def code_matrix_to_bytes(codes: List[List[int]], total_bits: int) -> bytes:
    if total_bits <= 8:
        width_bytes = 1
    elif total_bits <= 16:
        width_bytes = 2
    elif total_bits <= 32:
        width_bytes = 4
    elif total_bits <= 64:
        width_bytes = 8
    else:
        raise ValueError(f"Unsupported code width: {total_bits} bits")

    buf = bytearray()
    for row in codes:
        for code in row:
            buf.extend(int(code).to_bytes(width_bytes, byteorder="little", signed=False))
    return bytes(buf)

def write_code_bin(path: str, codes: List[List[int]], total_bits: int) -> None:
    Path(path).write_bytes(code_matrix_to_bytes(codes, total_bits))

# ── Hardware-matching MAC ──────────────────────────────────────────────────────
# PROD_MANT_BITS=3 for FP4 E2M1: 1.m × 1.m gives at most 3 fractional bits
# after normalization to [1,2). Compare FP8 E4M3: 1.mmm × 1.mmm → 7 bits.

def prod_quant(x: Tensor) -> Tensor:
    return trunc_product_mantissa(x, frac_bits=PROD_MANT_BITS)

# ── Precision-modelling functions (same as fp8_matmul_model.py) ──────────────

def _rne_e8(x: Tensor, man_bits: int) -> Tensor:
    """RNE-round float32 mantissa to man_bits bits, preserving float32 exponent range."""
    if man_bits >= 23:
        return x
    drop = 23 - man_bits
    xi   = x.view(torch.int32)
    mag  = xi & 0x7FFFFFFF
    sign = xi ^ mag
    keep_mask = ~((1 << drop) - 1)
    round_bit = ((mag >> (drop - 1)) & 1).bool()
    sticky    = (mag & ((1 << (drop - 1)) - 1)).ne(0)
    lsb       = ((mag >> drop) & 1).bool()
    round_up  = round_bit & (sticky | lsb)
    mag_r = (mag & keep_mask) + round_up.int() * (1 << drop)
    return (sign | mag_r).view(torch.float32)

def _round_div_pow2_rne_int(n: int, shift: int) -> int:
    if shift <= 0:
        return n << (-shift)
    q = n >> shift
    rem = n & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return q

def _encode_exact_scalar_to_fields(x: float, exp_bits: int, man_bits: int) -> Tuple[str, int, int, int]:
    sign = 1 if math.copysign(1.0, x) < 0 else 0
    if math.isnan(x):
        return "nan", sign, 0, 0
    if math.isinf(x):
        return "inf", sign, 0, 0
    if x == 0.0:
        return "zero", sign, 0, 0
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    ax = abs(x)
    m, e = math.frexp(ax)
    E = e - 1
    if E >= emin:
        total_sig = int(round(math.ldexp(ax, man_bits - E)))
        if total_sig == (1 << (man_bits + 1)):
            total_sig >>= 1
            E += 1
        return "finite", sign, E + bias, total_sig - (1 << man_bits)
    frac = int(round(math.ldexp(ax, man_bits - emin)))
    if frac >= (1 << man_bits):
        return "finite", sign, 1, 0
    return "finite", sign, 0, frac

def _fields_to_dyadic(sign: int, exp_field: int, frac: int, exp_bits: int, man_bits: int) -> Tuple[int, int, int]:
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    if exp_field == 0:
        return sign, frac, emin - man_bits
    return sign, (1 << man_bits) + frac, (exp_field - bias) - man_bits

def _round_dyadic_to_scalar(sign: bool, num: int, exp2: int, exp_bits: int, man_bits: int) -> float:
    if num == 0:
        return 0.0
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    emax = bias
    p = num.bit_length() - 1
    E = exp2 + p
    if E < emin:
        shift = (emin - man_bits) - exp2
        sub_sig = _round_div_pow2_rne_int(num, shift)
        if sub_sig == 0:
            return 0.0
        if sub_sig >= (1 << man_bits):
            E = emin
            total_sig = sub_sig
            if total_sig >= (1 << (man_bits + 1)):
                total_sig >>= 1
                E += 1
            if E > emax:
                return float("-inf") if sign else float("inf")
            val = math.ldexp(float(total_sig), E - man_bits)
            return -val if sign else val
        val = math.ldexp(float(sub_sig), emin - man_bits)
        return -val if sign else val
    total_sig = _round_div_pow2_rne_int(num, p - man_bits)
    if total_sig >= (1 << (man_bits + 1)):
        total_sig >>= 1
        E += 1
    if E > emax:
        return float("-inf") if sign else float("inf")
    val = math.ldexp(float(total_sig), E - man_bits)
    return -val if sign else val

def _scalar_to_dyadic(x: float) -> Tuple[bool, int, int]:
    ax = abs(x)
    num, den = ax.as_integer_ratio()
    exp2 = -(den.bit_length() - 1)
    return (math.copysign(1.0, x) < 0), num, exp2

def fp_quantize_rne_scalar(x: float, exp_bits: int, man_bits: int) -> float:
    if math.isnan(x):
        return float("nan")
    if math.isinf(x):
        return x
    if x == 0.0:
        return x
    sign, num, exp2 = _scalar_to_dyadic(x)
    return _round_dyadic_to_scalar(sign, num, exp2, exp_bits, man_bits)

def _rne_general(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """RNE quantize float32 to an IEEE-like FP format."""
    x = x.float()
    out = torch.zeros_like(x)
    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x)
    finite_nz = (x != 0) & torch.isfinite(x)
    if nan_mask.any():
        out[nan_mask] = torch.full_like(x[nan_mask], float("nan"))
    if inf_mask.any():
        out[inf_mask] = x[inf_mask]
    if not finite_nz.any():
        return out
    sign_all = torch.sign(x)
    ax_f = x[finite_nz].abs()
    result = torch.zeros_like(ax_f)
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    emax = bias
    scale = float(2 ** man_bits)
    m, e = torch.frexp(ax_f)
    E = (e - 1).float()
    normal = (E >= emin) & (E <= emax)
    if normal.any():
        frac_n = (2.0 * m[normal] - 1.0)
        E_n = E[normal]
        frac_scaled = frac_n * scale
        fl = torch.floor(frac_scaled)
        d  = frac_scaled - fl
        up = (d > 0.5) | ((d == 0.5) & ((fl % 2) == 1))
        k  = fl + up.float()
        carry = k >= scale
        E_fin = torch.where(carry, E_n + 1, E_n)
        k     = torch.where(carry, torch.zeros_like(k), k)
        new_overflow = E_fin > emax
        base = torch.pow(torch.tensor(2.0, device=x.device), E_fin)
        val  = base * (1.0 + k / scale)
        val  = torch.where(new_overflow, torch.full_like(val, float("inf")), val)
        result[normal] = val
    subnormal = E < emin
    if subnormal.any():
        ax_s = ax_f[subnormal]
        quantum = 2.0 ** (emin - man_bits)
        k_exact = ax_s / quantum
        fl = torch.floor(k_exact)
        d  = k_exact - fl
        up = (d > 0.5) | ((d == 0.5) & ((fl % 2) == 1))
        k  = fl + up.float()
        sub_val = k * quantum
        sub_val = torch.where(k >= scale, torch.full_like(sub_val, 2.0 ** emin), sub_val)
        result[subnormal] = sub_val
    overflow = E > emax
    if overflow.any():
        result[overflow] = torch.full_like(result[overflow], float("inf"))
    out[finite_nz] = sign_all[finite_nz] * result
    return out

def fp_quantize_rne(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Unified FP quantizer with round-to-nearest-even."""
    x = x.float()
    if exp_bits == 8 and man_bits == 7:
        return x.to(torch.bfloat16).to(torch.float32)
    if exp_bits == 8:
        return _rne_e8(x, man_bits)
    flat = x.detach().cpu().reshape(-1).tolist()
    out_flat = [fp_quantize_rne_scalar(float(v), exp_bits, man_bits) for v in flat]
    return torch.tensor(out_flat, dtype=torch.float32, device=x.device).view_as(x)

def fp_add_exact_scalar(x: float, y: float, exp_bits: int, man_bits: int) -> float:
    if math.isnan(x) or math.isnan(y):
        return float("nan")
    sign_x = 1 if math.copysign(1.0, x) < 0 else 0
    sign_y = 1 if math.copysign(1.0, y) < 0 else 0
    if math.isinf(x) or math.isinf(y):
        if math.isinf(x) and math.isinf(y) and sign_x != sign_y:
            return float("nan")
        return x if math.isinf(x) else y
    kind_x, sign_x, exp_x, frac_x = _encode_exact_scalar_to_fields(x, exp_bits, man_bits)
    kind_y, sign_y, exp_y, frac_y = _encode_exact_scalar_to_fields(y, exp_bits, man_bits)
    if kind_x == "zero":
        return y
    if kind_y == "zero":
        return x
    sign_x, sig_x, e_x = _fields_to_dyadic(sign_x, exp_x, frac_x, exp_bits, man_bits)
    sign_y, sig_y, e_y = _fields_to_dyadic(sign_y, exp_y, frac_y, exp_bits, man_bits)
    e_min = min(e_x, e_y)
    lhs = sig_x << (e_x - e_min)
    rhs = sig_y << (e_y - e_min)
    total = (-lhs if sign_x else lhs) + (-rhs if sign_y else rhs)
    if total == 0:
        return 0.0
    return _round_dyadic_to_scalar(total < 0, abs(total), e_min, exp_bits, man_bits)

def fp_add_exact(x: Tensor, y: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    assert x.shape == y.shape
    x_flat = x.detach().cpu().reshape(-1).tolist()
    y_flat = y.detach().cpu().reshape(-1).tolist()
    out_flat = [fp_add_exact_scalar(float(a), float(b), exp_bits, man_bits)
                for a, b in zip(x_flat, y_flat)]
    return torch.tensor(out_flat, dtype=torch.float32, device=x.device).view_as(x)

def mx_product_saturate(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Clamp product overflow to match MxPEOutToRaw saturation."""
    x = x.float()
    bias = (1 << (exp_bits - 1)) - 1
    is_mx_fp8 = (exp_bits == 4 and man_bits == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    scale = float(2 ** man_bits)
    max_mant = (2 ** man_bits - 2) if is_mx_fp8 else (2 ** man_bits - 1)
    max_normal = (2.0 ** emax) * (1.0 + max_mant / scale)
    sat_man = 2 ** man_bits - 2
    sat_val = (1.0 + sat_man / scale) * (2.0 ** (bias + 1))
    sign = torch.sign(x)
    ax = x.abs()
    return sign * torch.where(ax > max_normal, torch.full_like(ax, sat_val), ax)

def mx_product_quantize_trunc(x: Tensor, exp_bits: int, frac_bits: int) -> Tensor:
    """Match the MxFPMul product path before the accumulator resize."""
    x = x.float()
    out = x.clone()
    finite = torch.isfinite(x)
    ax = x.abs()
    nz = finite & (ax != 0)
    if not nz.any():
        return mx_product_saturate(out, exp_bits, frac_bits)
    ax_nz = ax[nz]
    m, e = torch.frexp(ax_nz)
    E = e - 1
    scale = float(1 << frac_bits)
    frac = 2.0 * m - 1.0
    frac_q = torch.floor(frac * scale) / scale
    val_nz = (1.0 + frac_q) * torch.ldexp(torch.ones_like(frac_q), E)
    out[nz] = torch.sign(x[nz]) * val_nz
    out[finite & (ax == 0)] = x[finite & (ax == 0)]
    return mx_product_saturate(out, exp_bits, frac_bits)

# ─────────────────────────────────────────────────────────────────────────────

def matmul_outer_quantized_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    prod_precision_list: Optional[List[Tuple[int, int]]] = None,
    acc_precision_list: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2
    if prod_precision_list is not None:
        assert len(prod_precision_list) == TILE_K
    if acc_precision_list is not None:
        assert len(acc_precision_list) == TILE_K
    C = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)
    for k in range(K):
        k_idx = k % TILE_K
        outer = torch.outer(A_in[:, k], B_in[k, :])

        if prod_precision_list is not None:
            exp_p, man_p = prod_precision_list[k_idx]
            outer = mx_product_quantize_trunc(outer, exp_p, man_p)
        else:
            outer = prod_quant(outer)
            outer = q_bf16_rne(outer)

        if acc_precision_list is not None:
            exp_a, man_a = acc_precision_list[k_idx]
            C = fp_add_exact(
                fp_quantize_rne(C, exp_a, man_a),
                fp_quantize_rne(outer, exp_a, man_a),
                exp_a, man_a)
        else:
            C = q_bf16_rne(q_bf16_rne(C) + q_bf16_rne(outer))
    return C

def compute_tile_scale_matrix(A_scales_row, B_scales_col, m0, n0, k0, TM, TN):
    g  = k0 // GROUP
    sA = A_scales_row[m0:m0+TM, g].to(torch.float32)
    sB = B_scales_col[g, n0:n0+TN].to(torch.float32)
    S  = torch.outer(sA, sB)
    return make_fp_quantizer(SCALE_SPEC, "nearest")(S)

def bf16_accum_add(x: Tensor, y: Tensor) -> Tensor:
    return fp_add_exact(q_bf16_rne(x), q_bf16_rne(y), 8, 7)

def print_matrix(name: str, mat: Tensor, spec: str = "bf16"):
    arr = mat.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    print(f"\n  {name} (decimal):")
    for r in range(arr.shape[0]):
        vals = [f"{arr[r,c].item():12.6f}" for c in range(arr.shape[1])]
        print("    " + " ".join(vals))
    codes, bits = tensor_to_custom_fp_codes(arr, spec)
    hex_rows = codes_to_hex_rows(codes, bits)
    print(f"  {name} (hex, {spec}):")
    for row in hex_rows:
        print("    " + " ".join(row))

# ── Tiled matmul — TM=TN=32, TK=16 for FP4 (vs TM=TN=TK=16 for FP8) ─────────

def tiled_matmul_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    A_scales_row: Tensor,
    B_scales_col: Tensor,
    verbose: bool = True,
    prod_precision_list: Optional[List[Tuple[int, int]]] = None,
    acc_precision_list: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2

    TM, TN, TK = TILE_M, TILE_N, TILE_K
    Gk = (K + GROUP - 1) // GROUP
    assert A_scales_row.shape == (M, Gk)
    assert B_scales_col.shape == (Gk, N)

    C_out = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)
    tile_count = 0

    for k0 in range(0, K, TK):
        for n0 in range(0, N, TN):
            for m0 in range(0, M, TM):
                tile_count += 1
                A_tile = A_in[m0:m0+TM, k0:k0+TK]
                B_tile = B_in[k0:k0+TK, n0:n0+TN]

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"TILE {tile_count}: m0={m0}, n0={n0}, k0={k0} (group={k0//GROUP})")
                    print(f"{'='*60}")
                    print_matrix("A_tile", A_tile, INPUT_SPEC)
                    print_matrix("B_tile", B_tile, INPUT_SPEC)

                C_tile = matmul_outer_quantized_hwlike(A_tile, B_tile,
                    prod_precision_list=prod_precision_list,
                    acc_precision_list=acc_precision_list)

                if verbose:
                    print_matrix("C_tile (pre-scale, bf16)", C_tile, "bf16")

                S_tile = compute_tile_scale_matrix(A_scales_row, B_scales_col, m0, n0, k0, TM, TN)

                if verbose:
                    print_matrix("S_tile (scale factors)", S_tile, "bf16")

                C_tile_scaled = q_bf16_rne(C_tile * S_tile)

                if verbose:
                    print_matrix("C_tile_scaled (post-scale, bf16)", C_tile_scaled, "bf16")

                C_out[m0:m0+TM, n0:n0+TN] = bf16_accum_add(
                    C_out[m0:m0+TM, n0:n0+TN], C_tile_scaled
                )

                if verbose:
                    print_matrix("C_accumulated (running sum, bf16)",
                                 C_out[m0:m0+TM, n0:n0+TN], "bf16")

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL OUTPUT C_out (bf16)")
        print(f"{'='*60}")
        print_matrix("C_out", C_out, "bf16")

    return C_out

# ── Output requantization ──────────────────────────────────────────────────────
# Uses _fp_emax so log2_pmax is correct for FP4 E2M1 (emax=2, max value=6.0)
# vs FP8 E4M3 (emax=8, max value=448). fp8_matmul_model's shortcut formula
# (1<<(e_bits-1)) accidentally equals bias+1 for E4M3 only.

# ── Hardware-accurate BF16 → E2M1 quantization ────────────────────────────────
# Matches BF16ScaleRoundToTiny.scala (fp4 path):
#   roundToMx(scaled_bf16, ..., format_fp4=MxFType.E3M1, pack=E3M1Tofp4)
#
# Step 1: RoundAnyRawFNToRecFN rounds BF16 → E3M1 (1+3+1 bit, bias=3) with RNE.
# Step 2: E3M1Tofp4 maps each 5-bit E3M1 code deterministically to a 4-bit E2M1.
#
# Full E3M1-magnitude → E2M1-magnitude mapping (sign preserved throughout):
#   0, 0.125, 0.25  → 0      mapToZero:    biased_exp<2, except (exp=1,sig=1)
#   0.375, 0.5      → 0.5    mapToSubnorm: FP4Min = 0b001
#   0.75            → 1.0    mapToMinNorm: sign ## 0b010
#   1.0..6.0        → same   normal:       exp_adj = exp_e3m1 − biasDiff(2)
#   ≥8.0, Inf, NaN  → 6.0   mapToMax:     exp>5 or isSpecial; FP4Max = 0b111

def _e3m1_to_e2m1(x: Tensor) -> Tensor:
    """Deterministic E3M1 → E2M1 float mapping, matching E3M1Tofp4 hardware."""
    sign = x.sign()
    ax   = x.abs()

    mapToZero    = (ax == 0.0) | (ax == 0.125) | (ax == 0.25)
    mapToSubnorm = (ax == 0.375) | (ax == 0.5)
    mapToMinNorm = (ax == 0.75)
    mapToMax     = (ax >= 8.0) | ~torch.isfinite(ax)  # exp>5 or Inf/NaN

    out = x.clone()
    out = torch.where(mapToZero,    torch.zeros_like(x), out)
    out = torch.where(mapToMax,     sign * 6.0,          out)  # checked before suborm/minNorm
    out = torch.where(mapToSubnorm, sign * 0.5,          out)
    out = torch.where(mapToMinNorm, sign * 1.0,          out)
    return out


def _bf16_to_e3m1_rne(x: Tensor) -> Tensor:
    """Round BF16 tensor to E3M1 (1+3+1, bias=3) using RNE.

    Bit-accurate match to hardfloat RoundAnyRawFNToRecFN(8,8,3,2,options=0)
    with round_near_even / tininess_afterRounding.

    Works directly on BF16 bit patterns, handling all boundary cases:
      e ∈ [-2,3]  → E3M1 normal: keep 1 mantissa bit, RNE on remaining 6 bits
      e = -3      → BF16 in [0.125, 0.25):  M≥64 → 0.25, M<64 → 0.125
      e = -4      → BF16 in [0.0625, 0.125): M>0  → 0.125, M=0 → 0
      e ≤ -5      → 0
      BF16 subnormal (E=0) → 0
      E=255       → E3M1 Inf / NaN
    """
    bf16 = x.to(torch.bfloat16)
    bits = bf16.view(torch.int16).to(torch.int32) & 0xFFFF

    sign_bit = (bits >> 15) & 1
    E = (bits >> 7) & 0xFF   # 8-bit biased BF16 exponent
    M = bits & 0x7F           # 7-bit fractional BF16 mantissa

    sign_f = torch.where(sign_bit == 0, torch.ones_like(x), -torch.ones_like(x))
    e = E - 127  # unbiased exponent (valid for normal BF16, E in [1,254])

    # ── Normal BF16 → E3M1 normal (general formula, valid for e ∈ [-2, 3]) ──
    # Full significand: S = {1, M[6], ..., M[0]} = 128 + M  (8-bit integer)
    # Round to 2-bit significand {1, q}:
    #   kept bit q = S[6] = M[6]
    #   round bit r = S[5] = M[5]
    #   sticky    s = S[4:0] = M[4:0] != 0
    # RNE: round_up = r & (s | q)
    S = 128 + M
    q = (S >> 6) & 1
    r = (S >> 5) & 1
    sticky = (S & 0x1F).bool()
    round_up = r.bool() & (sticky | q.bool())

    sig2 = q + round_up.to(torch.int32)   # 0 or 1; becomes 2 on carry
    carry = sig2 >= 2
    mant_out = torch.where(carry, torch.zeros_like(sig2), sig2)
    exp_out  = torch.where(carry, e + 1, e)            # unbiased E3M1 exp

    is_overflow = exp_out > 3
    safe_exp = exp_out.float().clamp(-10, 10)
    val_normal = sign_f * (1.0 + mant_out.float() * 0.5) * torch.pow(
        torch.full_like(x, 2.0), safe_exp)
    val_normal = torch.where(is_overflow, sign_f * float('inf'), val_normal)

    # ── Special boundary cases below E3M1 emin=-2 ────────────────────────────
    # e = -3: BF16 in [0.125, 0.25)
    #   midpoint 0.1875 = (128+64)/1024 → M=64
    #   M≥64: round to 0.25 (E3M1 normal emin, even code=2), M<64: 0.125 (subnorm)
    is_e_neg3 = (e == -3) & (E >= 1)
    val_e_neg3 = sign_f * torch.where(
        M >= 64, torch.full_like(x, 0.25), torch.full_like(x, 0.125))

    # e = -4: BF16 in [0.0625, 0.125)
    #   midpoint 0.0625 = 128/2048 → M=0; at tie, 0 (code=0 even) wins over 0.125 (code=1 odd)
    is_e_neg4 = (e == -4) & (E >= 1)
    val_e_neg4 = sign_f * torch.where(
        M > 0, torch.full_like(x, 0.125), torch.zeros_like(x))

    # ── Assemble ─────────────────────────────────────────────────────────────
    result = val_normal
    result = torch.where(is_e_neg3, val_e_neg3, result)
    result = torch.where(is_e_neg4, val_e_neg4, result)
    result = torch.where(((e <= -5) & (E >= 1)) | (E == 0),
                         torch.zeros_like(x), result)

    # BF16 Inf / NaN → E3M1 Inf / NaN
    is_nan = (E == 255) & (M != 0)
    is_inf = (E == 255) & (M == 0)
    result = torch.where(is_inf, sign_f * float('inf'), result)
    result = torch.where(is_nan, torch.full_like(x, float('nan')), result)

    return result


def hw_bf16_to_e2m1(x: Tensor) -> Tensor:
    """Hardware-accurate BF16 → E2M1 matching BF16ScaleRoundToTiny (fp4 path).

    1. Manual RNE round of BF16 bit pattern to E3M1 (1+3+1, bias=3).
    2. Apply deterministic E3M1Tofp4 map to obtain the E2M1 float value.
    """
    e3m1 = _bf16_to_e3m1_rne(x)
    return _e3m1_to_e2m1(e3m1)


def matrix_mx_requantize(matrix: Tensor, quant_spec: str = INPUT_SPEC):
    M, N = matrix.shape
    nblocks = N // GROUP
    e_bits, m_bits = parse_fp_spec(quant_spec)
    # Requant block-scale floor: p=0 for all formats -> output normalizes to [1,2) (chained-matmul
    # fix). Matches the RTL/spike log2_pmax=0. (Previously subtracted emax=_fp_emax(e_bits,m_bits).)
    log2_pmax = 0

    scale_q_fn = make_fp_quantizer(SCALE_SPEC, "nearest")
    C_quantized = torch.zeros_like(matrix)
    C_scales    = torch.zeros(M, nblocks)

    for bi in range(nblocks):
        block     = matrix[:, bi*GROUP:(bi+1)*GROUP]
        block_max = block.abs().amax(dim=1, keepdim=True)
        max_exp   = torch.floor(torch.log2(block_max.clamp(min=1e-45)))
        scale_exp = max_exp - log2_pmax
        scale     = torch.pow(2.0, scale_exp)
        scale     = scale_q_fn(scale)
        # Scale is a power-of-2 → division is an exact BF16 exponent shift.
        # Apply the hardware two-stage quantization: RNE→E3M1, then E3M1→E2M1.
        scaled = q_bf16_rne(block / scale)
        C_quantized[:, bi*GROUP:(bi+1)*GROUP] = hw_bf16_to_e2m1(scaled)
        C_scales[:, bi] = scale.squeeze(1)

    return C_quantized, C_scales

# ── Header writing ─────────────────────────────────────────────────────────────

def bf16_tensor_to_recfn_hex(t: Tensor, hw_zero: int = 0x01600) -> list:
    flat = t.to(torch.bfloat16).reshape(-1)
    rows = []
    for val in flat:
        buf = val.view(torch.int16).item() & 0xFFFF
        if (buf & 0x7FFF) == 0:
            rec = hw_zero
        else:
            rec = ieee_to_recfn(buf, exp_bits=8, mant_bits_total=8)
        rows.append(f"{rec:05x}")
    shape = t.shape
    result, idx = [], 0
    for i in range(shape[0]):
        result.append([rows[idx + j] for j in range(shape[1])])
        idx += shape[1]
    return result

def write_c_header_fp4_direct(
    path: str,
    M: int, K: int, N: int,
    group: int,
    input_spec: str,
    scale_spec: str,
    A_codes: torch.Tensor,
    B_codes: torch.Tensor,
    A_scales_row_q: torch.Tensor,
    B_scales_col_q: torch.Tensor,
    C_out_bf16: torch.Tensor,
    C_out_quantized: torch.Tensor = None,
    C_out_scales: torch.Tensor = None,
    a_tile_m: int = 32,
    k_tile: int = 16,
):
    assert M % a_tile_m == 0 and K % k_tile == 0

    A_hw_codes = _a_indices_to_hw_layout(A_codes, a_tile_m, k_tile)
    A_hw_hex   = codes_to_hex_rows(A_hw_codes, 8)

    B_list   = B_codes.tolist()
    B_packed = [[(B_list[k][n + 1] << 4) | B_list[k][n] for n in range(0, N, 2)] for k in range(K)]
    B_hex    = codes_to_hex_rows(B_packed, 8)

    As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), scale_spec)
    Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, scale_spec)
    As_bits -= 1
    Bs_bits -= 1
    As_hex = codes_to_hex_rows(As_codes, As_bits)
    Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)

    C_codes, C_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")
    C_hex = codes_to_hex_rows(C_codes, C_bits)

    if C_out_quantized is not None:
        Cq_packed = _a_indices_to_hw_layout(C_out_quantized, a_tile_m, k_tile)
        Cq_hex    = codes_to_hex_rows(Cq_packed, 8)

    if C_out_scales is not None:
        Cqs_codes, Cqs_bits = tensor_to_custom_fp_codes(C_out_scales.transpose(0, 1), scale_spec)
        Cqs_bits -= 1
        Cqs_hex = codes_to_hex_rows(Cqs_codes, Cqs_bits)

    guard = path.upper()
    for ch in [".", "/", "\\", "-"]:
        guard = guard.replace(ch, "_")

    def fmt2d(hex_rows):
        lines = ["    { " + ", ".join(f"0x{h}" for h in row) + " }" for row in hex_rows]
        return ",\n".join(lines)

    Gk   = K // group
    M_hw = M // 2

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
        f.write(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // group}\n")
        f.write(f"#define A_TILE_M   {a_tile_m}\n#define K_TILE     {k_tile}\n\n")
        f.write(f"// Input precision: {input_spec} (direct 4-bit FP4 codes, no LUT)\n")
        f.write("// A stored in HW tiled layout: per (A_TILE_M x K_TILE) block,\n")
        f.write("//   pairs of m-rows interleaved per k, then adjacent column pairs nibble-packed.\n")
        f.write(f"//   Dimensions: [M/2][K] = [{M_hw}][{K}]\n")
        f.write(f"//   byte layout: bits[7:4]=fp4(row 2r+1, k), bits[3:0]=fp4(row 2r, k)\n")
        f.write(f"static const uint8_t A_in_hw[{M_hw}][{K}] = {{\n{fmt2d(A_hw_hex)}\n}};\n\n")
        f.write("// B stored in standard layout: odd-col nibble in high bits, even-col in low bits\n")
        f.write(f"//   Dimensions: [K][N/2] = [{K}][{N // 2}]\n")
        f.write(f"static const uint8_t B_in[{K}][{N // 2}] = {{\n{fmt2d(B_hex)}\n}};\n\n")
        f.write(f"// Per-row per-{group}-K-group A scales in {scale_spec}\n")
        f.write(f"static const uint8_t A_scales_row[{Gk}][{M}] = {{\n{fmt2d(As_hex)}\n}};\n\n")
        f.write(f"// Per-col per-{group}-K-group B scales in {scale_spec}\n")
        f.write(f"static const uint8_t B_scales_col[{Gk}][{N}] = {{\n{fmt2d(Bs_hex)}\n}};\n\n")
        if C_out_quantized is not None:
            f.write(f"// Final output requantized to {input_spec} (direct 4-bit codes)\n")
            f.write(f"// HW tiled layout: [M/2][N] = [{M // 2}][{N}]\n")
            f.write(f"//   byte layout: bits[7:4]=fp4(row 2r+1, n), bits[3:0]=fp4(row 2r, n)\n")
            f.write(f"static const uint8_t C_out[{M // 2}][{N}] = {{\n{fmt2d(Cq_hex)}\n}};\n\n")
        if C_out_scales is not None:
            Gk_out = C_out_scales.shape[1]
            f.write(f"// Per-row per-{group}-N-group C output scales in {scale_spec}\n")
            f.write(f"static const uint8_t C_scales_row[{Gk_out}][{M}] = {{\n{fmt2d(Cqs_hex)}\n}};\n\n")
        f.write("// Final output (scaled+accumulated), bf16\n")
        f.write(f"static const uint16_t C_out_bf16[{M}][{N}] = {{\n{fmt2d(C_hex)}\n}};\n\n")
        f.write(f"#endif // {guard}\n")

def write_tensor_bins(
    base_dir: str,
    A_in: Tensor,
    B_in: Tensor,
    C_out_quantized: Tensor,
    C_out_bf16: Tensor,
):
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A_codes, A_bits = tensor_to_custom_fp_codes(A_in, INPUT_SPEC)
    B_codes, B_bits = tensor_to_custom_fp_codes(B_in, INPUT_SPEC)
    Cq_codes, Cq_bits = tensor_to_custom_fp_codes(C_out_quantized, INPUT_SPEC)
    C_bf16_codes, C_bf16_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")

    A_hw_codes = _a_indices_to_hw_layout(A_codes, TILE_M, TILE_K)
    B_packed = [[(B_codes[k][n + 1] << 4) | B_codes[k][n] for n in range(0, B_in.shape[1], 2)]
                for k in range(B_in.shape[0])]
    Cq_packed = _a_indices_to_hw_layout(Cq_codes, TILE_M, TILE_K)

    write_code_bin(str(out_dir / "A_in.bin"), A_hw_codes, 8)
    write_code_bin(str(out_dir / "B_in.bin"), B_packed, 8)
    write_code_bin(str(out_dir / "C_out_quant.bin"), Cq_packed, 8)
    write_code_bin(str(out_dir / "C_out_bf16.bin"), C_bf16_codes, C_bf16_bits)

# ── Entry point ────────────────────────────────────────────────────────────────

def run(M: int, K: int, N: int, header_path: str = "matmul_data_fp4.h", verbose: bool = True,
        prod_precision_list: Optional[List[Tuple[int, int]]] = None,
        acc_precision_list: Optional[List[Tuple[int, int]]] = None):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cpu")

    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0
    assert GROUP % TILE_K == 0

    A = torch.randn(M, K, device=dev, dtype=torch.float32)
    B = torch.randn(K, N, device=dev, dtype=torch.float32)

    in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
    A_in = in_q(A)
    B_in = in_q(B)

    Gk = (K + GROUP - 1) // GROUP
    torch.manual_seed(SEED + 123)
    A_scale_exp = torch.randint(-3, 4, size=(M, Gk), device=dev).float()
    B_scale_exp = torch.randint(-3, 4, size=(Gk, N), device=dev).float()
    A_scales_row = torch.pow(2.0, A_scale_exp.to(torch.float32))
    B_scales_col = torch.pow(2.0, B_scale_exp.to(torch.float32))
    A_scales_row_q = make_fp_quantizer(SCALE_SPEC, "nearest")(A_scales_row)
    B_scales_col_q = make_fp_quantizer(SCALE_SPEC, "nearest")(B_scales_col)

    C_out_bf16 = tiled_matmul_hwlike(A_in, B_in, A_scales_row_q, B_scales_col_q, verbose=verbose,
        prod_precision_list=prod_precision_list, acc_precision_list=acc_precision_list)

    C_out_quantized, C_out_scales = matrix_mx_requantize(C_out_bf16)

    # Get raw 4-bit codes for A, B, C for header
    A_codes_raw, _ = tensor_to_custom_fp_codes(A_in, INPUT_SPEC)
    B_codes_raw, _ = tensor_to_custom_fp_codes(B_in, INPUT_SPEC)
    A_codes = torch.tensor(A_codes_raw, dtype=torch.int32)
    B_codes = torch.tensor(B_codes_raw, dtype=torch.int32)
    C_req_codes_raw, _ = tensor_to_custom_fp_codes(C_out_quantized, INPUT_SPEC)
    C_req_codes = torch.tensor(C_req_codes_raw, dtype=torch.int32)

    write_c_header_fp4_direct(
        path           = header_path,
        M=M, K=K, N=N,
        group          = GROUP,
        input_spec     = INPUT_SPEC,
        scale_spec     = SCALE_SPEC,
        A_codes        = A_codes,
        B_codes        = B_codes,
        A_scales_row_q = A_scales_row_q,
        B_scales_col_q = B_scales_col_q,
        C_out_bf16     = C_out_bf16,
        C_out_quantized= C_req_codes,
        C_out_scales   = C_out_scales,
        a_tile_m       = TILE_M,
        k_tile         = TILE_K,
    )
    write_tensor_bins(Path(header_path).resolve().parent, A_in, B_in, C_out_quantized, C_out_bf16)
    print(f"\nHeader written to: {header_path}")
    print(f"Binary tensors written to: {Path(header_path).resolve().parent}")

    return C_out_bf16


# ── Debug / standalone execution ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=M)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--header-path", type=str, default="matmul_data_fp4.h")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Configure precision lists to match hardware mesh settings.
    # prod_precision_list: (exp_bits, frac_bits) per k-step within a tile.
    # acc_precision_list:  (exp_bits, man_bits) per k-step within a tile.
    # Set to None to use default (PROD_MANT_BITS truncation + bf16 accumulation).
    prod_precision_list = [(4,3)] * 16
    acc_precision_list = [(4,4)] * 8 + [(4,5)] * 2 + [(4,6)] * 5 + [(8,7)] * 1

    C = run(args.M, args.K, args.N, header_path=args.header_path, verbose=not args.quiet,
            prod_precision_list=prod_precision_list, acc_precision_list=acc_precision_list)

    # ── Per-row group breakdown (first 2 rows, first group) ───────────────────
    C_bf16 = q_bf16_rne(C)
    C_requantized, C_req_scales = matrix_mx_requantize(C_bf16)
    scale_codes, scale_bits = tensor_to_custom_fp_codes(C_req_scales, SCALE_SPEC)

    print(f"\n--- C per-row group breakdown (group_size={GROUP}, showing first 2 rows) ---")
    for m in range(min(2, args.M)):
        print(f"  row {m:3d}:")
        bf16_row   = C_bf16[m]
        for g in range(1):
            col_start  = g * GROUP
            col_end    = col_start + GROUP
            golden_grp = bf16_row[col_start:col_end]
            quant_grp  = C_requantized[m, col_start:col_end]
            scale_val  = C_req_scales[m, g].item()
            scale_hex  = f"{scale_codes[m][g]:02x}"
            abs_grp    = golden_grp.abs()
            max_mag    = abs_grp.max().item()
            max_bf16   = abs_grp.max().to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
            q_codes, q_bits = tensor_to_custom_fp_codes(quant_grp.unsqueeze(0), INPUT_SPEC)
            q_hex      = codes_to_hex_rows(q_codes, q_bits)[0]
            golden_hex = " ".join(
                f"{b.item() & 0xFFFF:04x}" for b in golden_grp.to(torch.bfloat16).view(torch.int16)
            )
            print(f"    group {g} (cols {col_start:3d}-{col_end-1:3d})\n"
                  f"    max_magnitude={max_mag:.6g} (bf16=0x{max_bf16:04x})\n"
                  f"    scale={scale_val:.6g} (e8m0=0x{scale_hex})\n"
                  f"    golden_bf16=[{golden_hex}]\n"
                  f"    quantized={q_hex}")

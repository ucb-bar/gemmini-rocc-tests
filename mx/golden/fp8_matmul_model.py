#!/usr/bin/env python3
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Optional, Dict, Tuple, List

import torch

Tensor = torch.Tensor
QuantFn = Optional[Callable[[Tensor], Tensor]]

# Hardcoded parameters matching hardware
INPUT_SPEC = "fp8:e4m3"
PROD_MANT_BITS = 7
TILE = 16
GROUP = 32
SCALE_SPEC = "fpe8m0"
SEED = 0

_FP_PRESETS = {
    "fp16":      dict(exp=5, man=10),
    "bf16":      dict(exp=8, man=7),
    "fp8:e4m3":  dict(exp=4, man=3),
    "fp8:e5m2":  dict(exp=5, man=2),
    "fp6:e3m2":  dict(exp=3, man=2),
    "fp4:e4m1":  dict(exp=2, man=1),
    "fp26":      dict(exp=7, man=18),
    "fp20":      dict(exp=6, man=13),
    "fp14":      dict(exp=5, man=8),
    "fpe8m0":    dict(exp=8, man=0),
}

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

def trunc_product_mantissa(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
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
    out_nz = torch.where(out_nz.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(out_nz), out_nz)
    out[nz] = out_nz
    return out

def q_bf16_rne(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)

def _rne_e8(x: Tensor, man_bits: int) -> Tensor:
    """RNE-round float32 mantissa to man_bits bits, preserving float32 exponent range.
    For man_bits==7 this is equivalent to q_bf16_rne (validated by native BF16 path).
    Uses IEEE 754 sign-magnitude bit manipulation so rounding applies to the magnitude."""
    if man_bits >= 23:
        return x
    drop = 23 - man_bits
    xi   = x.view(torch.int32)
    mag  = xi & 0x7FFFFFFF                       # magnitude bits (exp+mantissa), sign=0
    sign = xi ^ mag                               # sign bit only
    keep_mask = ~((1 << drop) - 1)               # clear dropped mantissa bits
    round_bit = ((mag >> (drop - 1)) & 1).bool()
    sticky    = (mag & ((1 << (drop - 1)) - 1)).ne(0)
    lsb       = ((mag >> drop) & 1).bool()
    round_up  = round_bit & (sticky | lsb)
    mag_r = (mag & keep_mask) + round_up.int() * (1 << drop)
    return (sign | mag_r).view(torch.float32)

def _rne_general(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """RNE quantize float32 to an IEEE-like FP format.
    This matches the HardFloat resize/adder path used for the accumulator:
    normals + subnormals are preserved, overflow becomes inf, and NaN/inf
    propagate instead of saturating."""
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

    # Exact decomposition via frexp: m in [0.5, 1.0), ax = m * 2^e
    m, e = torch.frexp(ax_f)
    E = (e - 1).float()                       # unbiased exponent
    frac = 2.0 * m - 1.0                      # mantissa fraction in [0, 1)

    # --- Normal: emin <= E <= emax ---
    normal = (E >= emin) & (E <= emax)
    if normal.any():
        frac_n = frac[normal]
        E_n = E[normal]
        frac_scaled = frac_n * scale           # exact (frac is exact, scale is power of 2)
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

    # --- Subnormal: E < emin ---
    subnormal = E < emin
    if subnormal.any():
        ax_s = ax_f[subnormal]
        quantum = 2.0 ** (emin - man_bits)     # smallest subnormal
        k_exact = ax_s / quantum               # exact (quantum is power of 2)
        fl = torch.floor(k_exact)
        d  = k_exact - fl
        up = (d > 0.5) | ((d == 0.5) & ((fl % 2) == 1))
        k  = fl + up.float()
        sub_val = k * quantum                  # k=0 → zero, k=2^man → smallest normal
        sub_val = torch.where(k >= scale, torch.full_like(sub_val, 2.0 ** emin), sub_val)
        result[subnormal] = sub_val

    # --- Overflow: E > emax ---
    overflow = E > emax
    if overflow.any():
        result[overflow] = torch.full_like(result[overflow], float("inf"))

    out[finite_nz] = sign_all[finite_nz] * result
    return out

#: Set truthy to force the scalar reference paths for fp_quantize_rne / fp_add_exact. The vectorized
#: implementations below are gated against these element-for-element by test_vector_primitives.py;
#: this switch exists so that test can obtain the reference, and as an escape hatch.
#: A plain module attribute, not an env lookup: this region is EXTRACTED verbatim into
#: npu-exploration/app/mxmesh/fp8.py, whose preamble does not import os.
_FORCE_SCALAR_PRIMITIVES = False

#: Below this many elements the scalar path wins: torch's per-op dispatch dominates, and these
#: primitives issue ~20 ops. THIS THRESHOLD IS NOT COSMETIC -- the two consumers sit on opposite
#: sides of it. MEASURED single-threaded at (e4,m4), vectorized vs scalar throughput:
#:
#:     n=256  1.02x | n=1024  3.19x | n=4096  8.46x | n=16384+  11-14x
#:
#: `tiled_matmul_hwlike` (the kernel goldens) works per DIM-tile, so n = 16*16 = 256 and it lands
#: exactly on the neutral point -- switching it wholesale measured SLOWER end to end (_run_mesh
#: 1.73s vs 1.55s). `MXLinearSim` (rtl_exact, the perplexity path) works on the full [M][N] and is
#: the case this vectorization exists for. Dispatching on size gives each what it wants.
_VEC_MIN_ELEMS = 1024


def _rne_shift_vec(n: Tensor, sh) -> Tensor:
    """RNE(n / 2**sh) on integer tensors -- the vector form of _round_div_pow2_rne_int.

    Preserves `n`'s dtype (int32 suffices for the quantizer, where |n| < 2**24; the adder needs
    int64 for its aligned significands). `sh` may be an int or an integer tensor, and may be
    negative (a left shift, which is exact). Right shifts are clamped to the dtype width: every
    caller passes |n| < 2**24, so any shift past 25 yields q = 0 with rem = n < half and therefore
    0, exactly as the unclamped scalar does -- the clamp only avoids integer shift UB.
    """
    lim = 30 if n.dtype == torch.int32 else 62
    if not torch.is_tensor(sh):
        sh = torch.full_like(n, int(sh))
    sh = sh.to(n.dtype)
    left = sh <= 0
    q_left = n << (-sh).clamp(min=0, max=lim)
    s = sh.clamp(min=1, max=lim)
    q = n >> s
    one = torch.ones_like(n)
    rem = n & ((one << s) - 1)
    half = one << (s - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) != 0))
    return torch.where(left, q_left, q + up.to(n.dtype))


def _rne_vec(x: Tensor, exp_bits: int, man_bits: int,
             overflow_to_inf: bool = True) -> Tensor:
    """Vectorized `fp_quantize_rne_scalar`: IEEE-like RNE to (exp_bits, man_bits).

    Subnormals preserved, overflow to +-inf, NaN/inf propagated, signed zero preserved. Mirrors
    `_round_dyadic_to_scalar` branch for branch, on int64 significands: a float32 is itself a dyadic
    rational, so `frexp` gives the exact (num, exp2) that `as_integer_ratio` gives up to trailing
    zeros -- and RNE of an exact value does not depend on that normalization.

    NOT the same function as the (unused) `_rne_general` above, which rounds differently.
    """
    x = x.float()
    sign = torch.signbit(x)
    ax = x.abs()
    nan = torch.isnan(x)
    inf = torch.isinf(x)
    zero = ax == 0
    special = nan | inf | zero

    bias = (1 << (exp_bits - 1)) - 1
    emin, emax = 1 - bias, bias

    # frexp is undefined on nan/inf; feed it a harmless value there and discard the result.
    safe = torch.where(special, torch.ones_like(ax), ax)
    m_f, e_i = torch.frexp(safe)
    # INT32 THROUGHOUT, and the scaling is exact in float32: m_f has 24 mantissa bits and 2**24 is
    # a power of two, so the product is an integer in [2**23, 2**24) that float32 represents
    # exactly. Using int64/float64 here doubles every temporary and, at [2048,2048], walks off the
    # cache -- measured 173 Melem/s at 1M elements against 7.6 at 4M before this.
    num = (m_f * float(1 << 24)).to(torch.int32)
    exp2 = e_i.to(torch.int32) - 24
    P = 23
    E = exp2 + P

    # --- normal: E >= emin ---
    tot_n = _rne_shift_vec(num, P - man_bits)
    carry_n = tot_n >= (1 << (man_bits + 1))
    tot_n = torch.where(carry_n, tot_n >> 1, tot_n)
    E_n = torch.where(carry_n, E + 1, E)

    # --- subnormal: E < emin. sh > 0 here, so no left shift arises on the taken branch. ---
    sh_s = (emin - man_bits) - exp2
    sub = _rne_shift_vec(num, sh_s)
    promoted = sub >= (1 << man_bits)            # rounding lifted it into the normal range
    tot_p = torch.where(sub >= (1 << (man_bits + 1)), sub >> 1, sub)
    E_p = torch.where(sub >= (1 << (man_bits + 1)),
                      torch.full_like(E, emin + 1), torch.full_like(E, emin))

    is_sub = E < emin
    sig = torch.where(is_sub, torch.where(promoted, tot_p, sub), tot_n)
    ex = torch.where(is_sub,
                     torch.where(promoted, E_p - man_bits,
                                 torch.full_like(E, emin - man_bits)),
                     E_n - man_bits)
    overflow = torch.where(is_sub, promoted & (E_p > emax), E_n > emax)

    val = torch.ldexp(sig.to(torch.float32), ex.to(torch.int32))
    if overflow_to_inf:
        val = torch.where(overflow, torch.full_like(val, float("inf")), val)
    out = torch.where(sign, -val, val)

    # UNDERFLOW IS UNSIGNED. `_round_dyadic_to_scalar` returns a literal 0.0 when the rounded
    # subnormal significand is zero, discarding the sign -- so a negative value that underflows
    # becomes +0.0, NOT -0.0. This must be applied AFTER the sign, and it is deliberately not the
    # same as an input of -0.0, which the `x == 0.0: return x` branch passes through with its sign.
    out = torch.where(is_sub & (sub == 0), torch.zeros_like(out), out)

    out = torch.where(zero, x, out)                                   # preserves -0.0
    out = torch.where(inf, x, out)
    return torch.where(nan, torch.full_like(out, float("nan")), out)


def fp_quantize_rne(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Unified FP quantizer with round-to-nearest-even.
    For (8, 7) exactly matches q_bf16_rne (native torch.bfloat16).
    For exp=8 uses bit manipulation on float32 (same exponent range).
    For exp<8 uses IEEE-like rounding with subnormal + inf/NaN support.
    Used for accumulator quantization and resize steps."""
    x = x.float()
    if exp_bits == 8 and man_bits == 7:
        return x.to(torch.bfloat16).to(torch.float32)
    if exp_bits == 8:
        return _rne_e8(x, man_bits)
    if _FORCE_SCALAR_PRIMITIVES or x.numel() < _VEC_MIN_ELEMS:
        flat = x.detach().cpu().reshape(-1).tolist()
        out_flat = [fp_quantize_rne_scalar(float(v), exp_bits, man_bits) for v in flat]
        return torch.tensor(out_flat, dtype=torch.float32, device=x.device).view_as(x)
    return _rne_vec(x, exp_bits, man_bits)

def mx_product_saturate(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Clamp product overflow to match MxPEOutToRaw saturation (MxFPMul.scala:374-377).
    In hardfloat convention, biased exponent 2^exp-1 is 'special', so the hardware
    saturates to satFrac=(2^man-2) at satExp corresponding to unbiased (bias+1).
    For MX FP8 this equals the format max (448). For other formats it exceeds the
    format max — the resize step then converts to the accumulator format.
    Underflow passes through unchanged (resize handles it)."""
    x = x.float()
    bias = (1 << (exp_bits - 1)) - 1
    is_mx_fp8 = (exp_bits == 4 and man_bits == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    scale = float(2 ** man_bits)
    # Max normal value in the product format
    max_mant = (2 ** man_bits - 2) if is_mx_fp8 else (2 ** man_bits - 1)
    max_normal = (2.0 ** emax) * (1.0 + max_mant / scale)
    # Hardware saturation value: satFrac = 2^man - 2, at exponent bias+1
    sat_man = 2 ** man_bits - 2
    sat_val = (1.0 + sat_man / scale) * (2.0 ** (bias + 1))
    sat_val = min(sat_val, torch.finfo(torch.float32).max)  # BF16 products (exp 8): 2^128 overflows fp32
    sign = torch.sign(x)
    ax = x.abs()
    return sign * torch.where(ax > max_normal, torch.full_like(ax, sat_val), ax)

def mx_product_quantize_trunc(x: Tensor, exp_bits: int, frac_bits: int) -> Tensor:
    """
    Match the MxFPMul product path before the accumulator resize.
    The PE product keeps `frac_bits` fractional bits with truncation.
    Because of the corrected MxPEOutToRaw logic, NO subnormal grid is applied
    at this stage. The fraction remains perfectly normalized with full precision
    regardless of how small the exponent is.
    """
    x = x.float()
    out = x.clone()
    finite = torch.isfinite(x)
    ax = x.abs()
    nz = finite & (ax != 0)

    if not nz.any():
        return mx_product_saturate(out, exp_bits, frac_bits)

    ax_nz = ax[nz]

    # frexp gives m in [0.5, 1.0) and e such that x = m * 2^e
    m, e = torch.frexp(ax_nz)
    E = e - 1

    # The hardware multiplier simply normalizes and truncates to frac_bits.
    # We apply this universally without checking for 'emin'.
    scale = float(1 << frac_bits)
    frac = 2.0 * m - 1.0
    frac_q = torch.floor(frac * scale) / scale

    # Reconstruct the value
    val_nz = (1.0 + frac_q) * torch.ldexp(torch.ones_like(frac_q), E)
    # MxFPMul PROD_FLOOR: products below the product path's wrap point flush to zero (2^-16 for E4M3)
    bias = (1 << (exp_bits - 1)) - 1
    val_nz = torch.where(E < 2 - (frac_bits + 1) - 2 * bias, torch.zeros_like(val_nz), val_nz)

    out[nz] = torch.sign(x[nz]) * val_nz
    out[finite & (ax == 0)] = x[finite & (ax == 0)]

    return mx_product_saturate(out, exp_bits, frac_bits)

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

def _mx_product_quantize_trunc_scalar(x: float, exp_bits: int, man_bits: int) -> float:
    if math.isnan(x):
        return float("nan")
    if x == 0.0:
        return x

    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    is_mx_fp8 = (exp_bits == 4 and man_bits == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    scale = 1 << man_bits
    max_mant = (scale - 2) if is_mx_fp8 else (scale - 1)
    max_normal = (2.0 ** emax) * (1.0 + max_mant / scale)
    sat_man = scale - 2
    sat_val = (1.0 + sat_man / scale) * (2.0 ** (bias + 1))
    sat_val = min(sat_val, torch.finfo(torch.float32).max)  # BF16 products (exp 8): 2^128 overflows fp32

    if math.isinf(x):
        return -sat_val if math.copysign(1.0, x) < 0 else sat_val

    sign, num, exp2 = _scalar_to_dyadic(x)
    p = num.bit_length() - 1
    E = exp2 + p

    if E < emin:
        shift = (emin - man_bits) - exp2
        sub_sig = num >> shift if shift >= 0 else num << (-shift)
        val = math.ldexp(float(sub_sig), emin - man_bits)
    else:
        shift = p - man_bits
        total_sig = num >> shift if shift >= 0 else num << (-shift)
        if total_sig >= (1 << (man_bits + 1)):
            total_sig >>= 1
            E += 1
        val = math.ldexp(float(total_sig), E - man_bits)

    if val > max_normal:
        val = sat_val
    return -val if sign else val

def fp_quantize_rne_scalar(x: float, exp_bits: int, man_bits: int) -> float:
    if math.isnan(x):
        return float("nan")
    if math.isinf(x):
        return x
    if x == 0.0:
        return x
    sign, num, exp2 = _scalar_to_dyadic(x)
    return _round_dyadic_to_scalar(sign, num, exp2, exp_bits, man_bits)

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

def _add_exact_vec(x: Tensor, y: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Vectorized `fp_add_exact_scalar`: encode both operands into (exp_bits, man_bits), add the
    significands EXACTLY, then round the sum back with RNE.

    The scalar version encodes through `_encode_exact_scalar_to_fields`, which quantizes its input;
    `_rne_vec` does the same thing, so the operands here are in-format and the alignment shift is
    bounded by the format's exponent span (about 2*bias + man_bits, ~20 bits at exp_bits=4). That
    bound is what keeps the exact sum inside int64 -- it is a precondition, not an approximation,
    and it is why the inputs are quantized first rather than taken as given.
    """
    # `_encode_exact_scalar_to_fields` rounds to (exp_bits, man_bits) WITHOUT an overflow check --
    # it keeps an out-of-range exponent field rather than saturating. Using fp_quantize_rne here
    # would turn an over-range operand into inf and an inf-inf into NaN, which the scalar never
    # does (measured: 150/4119 elements at (4,4), e.g. 512 + -294.727 -> 224, not NaN).
    x_raw, y_raw = x.float(), y.float()
    x = _rne_vec(x_raw, exp_bits, man_bits, overflow_to_inf=False)
    y = _rne_vec(y_raw, exp_bits, man_bits, overflow_to_inf=False)

    nan = torch.isnan(x) | torch.isnan(y)
    ix, iy = torch.isinf(x), torch.isinf(y)
    opposite_inf = ix & iy & (torch.signbit(x) != torch.signbit(y))
    any_inf = ix | iy

    zx, zy = (x_raw == 0), (y_raw == 0)

    def _dyadic(v):
        av = v.abs()
        bad = torch.isnan(v) | torch.isinf(v) | (av == 0)
        safe = torch.where(bad, torch.ones_like(av), av)
        m_f, e_i = torch.frexp(safe)
        num = (m_f.to(torch.float64) * float(1 << 24)).to(torch.int64)
        num = torch.where(bad, torch.zeros_like(num), num)
        return torch.signbit(v), num, e_i.to(torch.int64) - 24

    sx, nx, ex = _dyadic(x)
    sy, ny, ey = _dyadic(y)

    e_min = torch.minimum(ex, ey)
    lhs = nx << (ex - e_min).clamp(min=0, max=62)
    rhs = ny << (ey - e_min).clamp(min=0, max=62)
    total = torch.where(sx, -lhs, lhs) + torch.where(sy, -rhs, rhs)

    neg = total < 0
    mag = total.abs()

    # Reuse the quantizer's rounding by rebuilding the value: mag * 2^e_min is exact in float64
    # for the magnitudes reachable here (mag < 2^44 at exp_bits=4), so rounding it with _rne_vec is
    # the same operation _round_dyadic_to_scalar performs on (mag, e_min).
    val = torch.ldexp(mag.to(torch.float64), e_min.to(torch.int32)).to(torch.float32)
    out = _rne_vec(val, exp_bits, man_bits)
    out = torch.where(neg, -out, out)

    out = torch.where(total == 0, torch.zeros_like(out), out)
    # The scalar returns the OTHER OPERAND VERBATIM when one side is exactly zero -- unencoded,
    # so the raw value, not its rounding. `kind == "zero"` tests the raw input, so a value that
    # merely UNDERFLOWS to zero does not take this path.
    out = torch.where(zx, y_raw, out)
    out = torch.where(zy, x_raw, out)
    out = torch.where(any_inf, torch.where(ix, x, y), out)
    out = torch.where(opposite_inf | nan, torch.full_like(out, float("nan")), out)
    return out


def _add_exact_vec_is_exact(exp_bits: int, man_bits: int) -> bool:
    """Can `_add_exact_vec` hold the EXACT sum for this format in int64?

    The aligned significands span (24 bits from frexp) + (the format's exponent range,
    2*bias + man_bits - 1), plus one bit for the carry and one for the sign. Past 62 the int64
    shift silently truncates and the sum stops being exact -- so the scalar path is used instead
    rather than returning a wrong answer. The RTL's accumulator lanes are exp_bits=4 (bias 7), which
    needs ~44 bits and is comfortably inside; exp_bits=8 never reaches here (it has its own path).
    """
    bias = (1 << (exp_bits - 1)) - 1
    return 24 + (2 * bias + man_bits - 1) + 2 <= 62


def fp_add_exact(x: Tensor, y: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    assert x.shape == y.shape
    if exp_bits == 8 and man_bits == 7 and not _FORCE_SCALAR_PRIMITIVES:
        # BF16 ADD, THE CROSS-TILE STEP. `bf16_accum_add` routes here, so it is on the hot path of
        # every rtl_exact run -- and (8,7) is far too wide for the int64 path below, which would
        # otherwise send it to the scalar loop at 0.74 Melem/s.
        #
        # fp32 add then round is EXACT here, not an approximation: double rounding from p1 = 24
        # (float32) down to p2 = 8 (bf16) is innocuous whenever p1 >= 2*p2 + 2, and 24 >= 18. So
        # the float32 sum cannot land where a second rounding disagrees with rounding the exact
        # sum once. test_vector_primitives.py checks it against the scalar reference regardless.
        # Quantize the operands FIRST. The scalar path encodes each into the format before
        # adding (`_encode_exact_scalar_to_fields`), so adding raw values and rounding once is a
        # DIFFERENT function whenever an operand is not already bf16 -- caught by
        # tests/selftest_extracted.py at 685/4119 elements.
        xr, yr = x.float(), y.float()
        r = q_bf16_rne(q_bf16_rne(xr) + q_bf16_rne(yr))
        # ... and pass the other operand through RAW on exact zero, as the scalar does.
        return torch.where(xr == 0, yr, torch.where(yr == 0, xr, r))
    if (not _FORCE_SCALAR_PRIMITIVES and x.numel() >= _VEC_MIN_ELEMS
            and _add_exact_vec_is_exact(exp_bits, man_bits)):
        # `_add_exact_vec_is_exact` bounds the alignment shift using the FORMAT's exponent span,
        # which only holds while the operands are in-format. They need not be: the encode step
        # deliberately does not saturate (see _add_exact_vec), so a wildly out-of-range input --
        # 1e38 into (4,4) -- would need a ~136-bit shift and wrap int64. Check it and fall back.
        # NaN fails the comparison and falls back too, which is correct if slower.
        bias = (1 << (exp_bits - 1)) - 1
        fmax = (2.0 - 2.0 ** -man_bits) * (2.0 ** bias)
        in_range = bool((x.abs() <= fmax).all()) and bool((y.abs() <= fmax).all())
        if in_range:
            return _add_exact_vec(x, y, exp_bits, man_bits)
    x_flat = x.detach().cpu().reshape(-1).tolist()
    y_flat = y.detach().cpu().reshape(-1).tolist()
    out_flat = [fp_add_exact_scalar(float(a), float(b), exp_bits, man_bits)
                for a, b in zip(x_flat, y_flat)]
    return torch.tensor(out_flat, dtype=torch.float32, device=x.device).view_as(x)

def float_quantize_trunc(x: Tensor, exp: int, man: int) -> Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    out = torch.zeros_like(x)
    is_zero = (x == 0)
    if torch.all(is_zero):
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
    # MX FP8 E4M3: biased_exp goes up to 15 (unbiased=8), NaN=0x7F only → pmax=448
    is_mx_fp8 = (exp == 4 and man == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    underflow_mask = E < emin
    overflow_mask = E > emax
    normal_mask = (~underflow_mask) & (~overflow_mask)
    ax_q = torch.zeros_like(ax_nz)
    ax_q[underflow_mask] = 0.0
    if overflow_mask.any():
        E_max = float(emax)
        base_max = torch.pow(torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device), E_max)
        delta_max = base_max / (2 ** man)
        # MX FP8: biased_exp=15 + mant=7 is NaN, so max normal mant=6
        max_mant = (2 ** man - 2) if is_mx_fp8 else (2 ** man - 1)
        max_val = base_max + max_mant * delta_max
        ax_q[overflow_mask] = max_val
    if normal_mask.any():
        E_norm = E[normal_mask]
        x_norm = ax_nz[normal_mask]
        base = torch.pow(torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device), E_norm)
        delta = base / (2 ** man)
        t = (x_norm - base) / delta
        # MX FP8: when E=8 (biased_exp=15), mant=7 would be NaN; clamp to 6
        if is_mx_fp8:
            at_emax = (E_norm == emax).to(ax_nz.device)
            clamp_hi = torch.where(at_emax,
                torch.full_like(t, 2 ** man - 2 - 1e-7),
                torch.full_like(t, 2 ** man - 1 - 1e-7))
            k = torch.floor(torch.clamp(t, torch.zeros_like(t), clamp_hi))
        else:
            k = torch.floor(torch.clamp(t, 0, 2 ** man - 1 - 1e-7))
        ax_q[normal_mask] = base + k * delta
    out[nz_mask] = sign[nz_mask] * ax_q
    return out

def make_fp_quantizer(spec: str, rounding: str = "nearest") -> QuantFn:
    s = spec.strip().lower()
    if s in ("", "none", "identity"):
        return None
    if s == "fp32":
        return None
    e, m = parse_fp_spec(spec)
    rounding = rounding.lower()
    # E2M3 uses MxQuant's grid (emax=2, max_norm 7.5), which qtorch's float_quantize does NOT match
    # (qtorch treats exp=2 as emax=1). Route ALL rounding modes for E2M3 through MxQuant so the codebook
    # floats, operands (incl. the RTZ re-quant of gridded values) and requant all sit on the grid the
    # hardware decodes. On already-gridded values this is a no-op; MxQuant nearest is the canonical grid.
    if s == "fp6:e2m3" and rounding in ("nearest", "nearest_even", "zero", "toward_zero", "trunc", "rtz"):
        return _mxquant_e2m3_quantizer()
    if rounding in ("nearest", "nearest_even", "stochastic"):
        from qtorch.quant import float_quantize
        mode = "nearest" if rounding in ("nearest", "nearest_even") else "stochastic"
        return lambda x: float_quantize(x, exp=e, man=m, rounding=mode)
    if rounding in ("zero", "toward_zero", "trunc", "rtz"):
        return lambda x: float_quantize_trunc(x, exp=e, man=m)
    raise ValueError(f"Unsupported rounding mode: {rounding}")


def _mxquant_e2m3_quantizer():
    """fp6_e2m3 grid (emax=2, max_norm 7.5, subnormals, round half away from zero, saturate)."""
    from elemwise import quantize_elemwise
    return lambda x: quantize_elemwise(x, 'fp6_e2m3', round='nearest')

def tensor_to_custom_fp_codes(t: Tensor, spec: str) -> Tuple[List[List[int]], int]:
    e_bits, m_bits = parse_fp_spec(spec)
    total_bits = 1 + e_bits + m_bits
    s = spec.strip().lower()
    arr = t.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    if arr.ndim != 2:
        raise ValueError("Expected 1D or 2D tensor for hex dump.")

    if s == "bf16":
        bf16_bits = arr.to(torch.bfloat16).view(torch.int16)
        out_codes = [[int(bf16_bits[r, c].item()) & 0xFFFF for c in range(arr.shape[1])]
                     for r in range(arr.shape[0])]
        return out_codes, total_bits

    bias = (1 << (e_bits - 1)) - 1
    emin = 1 - bias
    # MX FP8 E4M3: biased_exp goes up to 15 (unbiased=8), NaN=0x7F only → pmax=448
    is_mx_fp8 = (e_bits == 4 and m_bits == 3)
    # MxQuant fp6_e2m3 uses emax = 2^(ebits-1) = 2 (max_norm 7.5), not bias=1. Match MxQuant
    # (npu-exploration/MXQuant microxcaling formats.py) so E2M3 codes use exp field up to 3.
    is_e2m3 = (e_bits == 2 and m_bits == 3)
    emax = bias + 1 if (is_mx_fp8 or is_e2m3) else bias
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
                    # Subnormal range: quantum = 2^(emin - m_bits)
                    quantum = 2.0 ** (emin - m_bits)
                    k = int(round(av / quantum))
                    if k <= 0:
                        code = 0
                    elif k >= 2**m_bits:
                        # Rounds up to min normal (biased_exp=1, mant=0)
                        code = (s << (e_bits + m_bits)) | (1 << m_bits)
                    else:
                        code = (s << (e_bits + m_bits)) | k  # subnormal: biased_exp=0
                else:
                    if E > emax:
                        E_used = emax
                        # MX FP8: biased_exp=15 + mant=7 is NaN, so max normal mant=6
                        mant = (2 ** m_bits) - 2 if is_mx_fp8 else (2 ** m_bits) - 1
                    else:
                        E_used = E
                        base = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        tpos = (av - base) / delta
                        mant = int(round(tpos))
                        if mant >= 2 ** m_bits:
                            # Rounding carry: banker's rounding pushed mant over the top (e.g. 7.5→8).
                            # Increment exponent and reset mantissa instead of clamping.
                            E_used += 1
                            mant = 0
                            if E_used > emax:
                                # Carry pushed past pmax: clip to max representable
                                E_used = emax
                                mant = (2 ** m_bits) - 2 if is_mx_fp8 else (2 ** m_bits) - 1
                        else:
                            # MX FP8: at biased_exp=15 (E=8), mant=7 would be NaN; clamp to 6
                            max_mant = (2 ** m_bits) - 2 if (is_mx_fp8 and E_used == emax) else (2 ** m_bits) - 1
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

def zext_codes(codes: List[List[int]], target_bits: int) -> List[List[int]]:
    mask = (1 << target_bits) - 1
    return [[(code & mask) for code in row] for row in codes]

def c_type_for_bits(total_bits: int) -> str:
    if total_bits <= 8: return "uint8_t"
    elif total_bits <= 16: return "uint16_t"
    elif total_bits <= 32: return "uint32_t"
    else: return "uint64_t"

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

# --- Hardware-matching MAC ---

def prod_quant(x: Tensor) -> Tensor:
    return trunc_product_mantissa(x, frac_bits=PROD_MANT_BITS)

def matmul_outer_quantized_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    prod_precision_list: Optional[List[Tuple[int, int]]] = None,
    acc_precision_list: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    """
    prod_precision_list: list of (exp_bits, frac_bits) per k-step within a tile.
                         Indexed by k % TILE. When None, uses PROD_MANT_BITS + bf16
                         (existing behaviour).
    acc_precision_list:  list of (exp_bits, frac_bits) per k-step within a tile.
                         Indexed by k % TILE. When None, uses bf16 (existing behaviour).
    """
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2
    if prod_precision_list is not None:
        assert len(prod_precision_list) == TILE
    if acc_precision_list is not None:
        assert len(acc_precision_list) == TILE
    C = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)
    for k in range(K):
        k_idx = k % TILE
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
            C = fp_add_exact(q_bf16_rne(C), q_bf16_rne(outer), 8, 7)
    return C

def compute_tile_scale_matrix(A_scales_row, B_scales_col, m0, n0, k0, TM, TN):
    g = k0 // GROUP
    sA = A_scales_row[m0:m0+TM, g].to(torch.float32)
    sB = B_scales_col[g, n0:n0+TN].to(torch.float32)
    S = torch.outer(sA, sB)
    S_q = make_fp_quantizer(SCALE_SPEC, "nearest")(S)
    return S_q

def bf16_accum_add(x: Tensor, y: Tensor) -> Tensor:
    return fp_add_exact(q_bf16_rne(x), q_bf16_rne(y), 8, 7)

# --- Printing helpers ---

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

# --- Main tiled matmul ---

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

    TM = TN = TK = TILE
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

# --- Header writing ---

def write_c_header_tiled(
    path: str,
    M: int, K: int, N: int,
    A_in: Tensor,
    B_in: Tensor,
    A_scales_row_q: Tensor,
    B_scales_col_q: Tensor,
    C_out_bf16: Tensor,
    C_out_quantized: Tensor,
    C_out_scales: Tensor,
):
    input_spec = INPUT_SPEC
    scale_spec = SCALE_SPEC
    group = GROUP

    A_codes, A_bits = tensor_to_custom_fp_codes(A_in, input_spec)
    B_codes, B_bits = tensor_to_custom_fp_codes(B_in, input_spec)
    Cq_codes, Cq_bits = tensor_to_custom_fp_codes(C_out_quantized, input_spec)
    As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), scale_spec)
    Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, scale_spec)
    C_codes, C_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")
    Cqs_codes, Cqs_bits = tensor_to_custom_fp_codes(C_out_scales.transpose(0, 1), scale_spec)
    As_bits -= 1
    Bs_bits -= 1
    Cqs_bits -= 1

    guard = path.upper()
    for ch in [".", "/", "\\", "-"]:
        guard = guard.replace(ch, "_")

    A_hex = codes_to_hex_rows(A_codes, A_bits)
    B_hex = codes_to_hex_rows(B_codes, B_bits)
    As_hex = codes_to_hex_rows(As_codes, As_bits)
    Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)
    C_hex = codes_to_hex_rows(C_codes, C_bits)
    Cq_hex = codes_to_hex_rows(Cq_codes, Cq_bits)
    Cqs_hex = codes_to_hex_rows(Cqs_codes, Cqs_bits)

    Gk = K // group

    def fmt(hex_rows):
        lines = []
        for row in hex_rows:
            lines.append("    { " + ", ".join(f"0x{h}" for h in row) + " }")
        return ",\n".join(lines)

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M {M}\n#define MATMUL_K {K}\n#define MATMUL_N {N}\n")
        f.write(f"#define MATMUL_GK {Gk}\n#define MATMUL_GN {N // group}\n\n")
        f.write(f"// Input precision: {input_spec}\n")
        f.write(f"static const {c_type_for_bits(A_bits)} A_in[MATMUL_M][MATMUL_K] = {{\n{fmt(A_hex)}\n}};\n\n")
        f.write(f"static const {c_type_for_bits(B_bits)} B_in[MATMUL_K][MATMUL_N] = {{\n{fmt(B_hex)}\n}};\n\n")
        f.write(f"// Per-row per-{group}-K-group scales in {scale_spec}\n")
        f.write(f"static const {c_type_for_bits(As_bits)} A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt(As_hex)}\n}};\n\n")
        f.write(f"// Per-col per-{group}-K-group scales in {scale_spec}\n")
        f.write(f"static const {c_type_for_bits(Bs_bits)} B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt(Bs_hex)}\n}};\n\n")
        f.write(f"// Final output requantized to {input_spec}\n")
        f.write(f"static const {c_type_for_bits(Cq_bits)} C_out[MATMUL_M][MATMUL_N] = {{\n{fmt(Cq_hex)}\n}};\n\n")
        f.write(f"// Per-row per-{group}-K-group output scales in {scale_spec}\n")
        f.write(f"static const {c_type_for_bits(Cqs_bits)} C_scales_row[MATMUL_GN][MATMUL_M] = {{\n{fmt(Cqs_hex)}\n}};\n\n")
        f.write(f"// Final output (already scaled+accumulated), bf16\n")
        f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt(C_hex)}\n}};\n\n")
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

    write_code_bin(str(out_dir / "A_in.bin"), A_codes, A_bits)
    write_code_bin(str(out_dir / "B_in.bin"), B_codes, B_bits)
    write_code_bin(str(out_dir / "C_out_quant.bin"), Cq_codes, Cq_bits)
    write_code_bin(str(out_dir / "C_out_bf16.bin"), C_bf16_codes, C_bf16_bits)

def array_mx_requantize(array: Tensor, quant_spec: str) -> Tuple[Tensor, float]:
    e_bits, m_bits = parse_fp_spec(quant_spec)
    max_val = array.abs().to(torch.bfloat16).view(torch.int16).max()
    max_exp = max_val.log2().floor()
    exp = max_exp - (e_bits + m_bits + 1)
    scale_factor = make_fp_quantizer(SCALE_SPEC, "nearest")(torch.pow(2.0, exp)).item()
    return array / scale_factor, scale_factor

def _po2(x: Tensor) -> Tensor:
    """Round to nearest power of 2 (toward +inf in exponent)."""
    x = x.to(torch.float32)
    # For each element, find floor(log2(x)) then return 2^that
    nz = x > 0
    out = torch.ones_like(x)
    if nz.any():
        log2_x = torch.log2(x[nz])
        exp = torch.floor(log2_x)
        out[nz] = torch.pow(2.0, exp)
        # If the value isn't exactly a power of 2, round up
        too_small = out[nz] < x[nz]
        exp[too_small] = exp[too_small] + 1
        out[nz] = torch.pow(2.0, exp)
    return out

def matrix_mx_requantize(matrix, quant_spec=INPUT_SPEC):
    M, N = matrix.shape
    nblocks = N // GROUP
    e_bits, m_bits = parse_fp_spec(quant_spec)
    # Requant block-scale floor: p=0 for all formats -> output normalizes to [1,2) (chained-matmul
    # fix). Matches the RTL/spike log2_pmax=0. (Previously subtracted emax=(1<<(e_bits-1)).)
    log2_pmax = 0

    scale_q_fn = make_fp_quantizer(SCALE_SPEC, "nearest")

    C_quantized = torch.zeros_like(matrix)
    C_scales = torch.zeros(M, nblocks)

    for bi in range(nblocks):
        block = matrix[:, bi*GROUP:(bi+1)*GROUP]
        block_max = block.abs().amax(dim=1, keepdim=True)
        # Match HW: extract exponent of block_max, subtract log2_pmax
        max_exp = torch.floor(torch.log2(block_max.clamp(min=1e-45)))
        scale_exp = max_exp - log2_pmax
        scale = torch.pow(2.0, scale_exp)
        scale = scale_q_fn(scale)

        # Store scaled BF16 values; tensor_to_custom_fp_codes handles MX FP8 rounding/encoding
        C_quantized[:, bi*GROUP:(bi+1)*GROUP] = block / scale
        C_scales[:, bi] = scale.squeeze(1)

    return C_quantized, C_scales


# --- Entry point ---

def run(M: int, K: int, N: int, header_path: str = "matmul_data.h", verbose: bool = True,
        prod_precision_list: Optional[List[Tuple[int, int]]] = None,
        acc_precision_list: Optional[List[Tuple[int, int]]] = None):
    torch.manual_seed(SEED)
    dev = torch.device("cpu")

    assert M % TILE == 0 and N % TILE == 0 and K % TILE == 0
    assert GROUP % TILE == 0

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

    write_c_header_tiled(
        path=header_path,
        M=M, K=K, N=N,
        A_in=A_in,
        B_in=B_in,
        A_scales_row_q=A_scales_row_q,
        B_scales_col_q=B_scales_col_q,
        C_out_bf16=C_out_bf16,
        C_out_quantized=C_out_quantized,
        C_out_scales=C_out_scales,
    )
    print(f"\nHeader written to: {header_path}")
    if os.environ.get("MXGEMMINI_BIN_DIR"):   # optional raw tensor dumps (debug only)
        write_tensor_bins(os.environ["MXGEMMINI_BIN_DIR"], A_in, B_in, C_out_quantized, C_out_bf16)

    return C_out_bf16

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--header-path", type=str, default="matmul_data.h")
    parser.add_argument("--quiet", action="store_true")
    import precision
    precision.add_arg(parser)
    args = parser.parse_args()
    precision.load(args.params)
    prod_precision_list, acc_precision_list = precision.for_dim(TILE)   # per mesh row (exp, frac)

    run(args.M, args.K, args.N, header_path=args.header_path, verbose=not args.quiet, acc_precision_list=acc_precision_list, prod_precision_list=prod_precision_list)

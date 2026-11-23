"""MX wire format: decode the bytes the datapath reads/writes and encode requant outputs as the device
rounds them. Transcribed from software/libgemmini/mx_fp_math.h (line refs are to that file)."""
from __future__ import annotations

import math

import numpy as np

BLOCK = 32       # MX block size
E8M0_BIAS = 127  # E8M0 scale bias

__all__ = ["BLOCK", "E8M0_BIAS", "fp8_e4m3_decode", "fp8_e5m2_decode",
           "fp6_e3m2_decode", "fp6_e2m3_decode", "fp4_e2m1_decode", "e8m0_decode",
           "bf16_bits_to_float", "float_to_bf16_bits", "DECODERS", "ENCODERS",
           "encode_requant"]


def fp8_e4m3_decode(code: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:232``. 1 sign | 4 exp | 3 mantissa, bias 7; ``e == 0`` is subnormal."""
    code = np.asarray(code, dtype=np.uint8)
    s = (code >> 7) & 1
    e = (code >> 3) & 0xF
    m = code & 0x7
    val = np.where(e == 0,
                   (m / 8.0) * 2.0 ** (1 - 7),
                   (1.0 + m / 8.0) * 2.0 ** (e.astype(np.int32) - 7))
    return np.where(s == 1, -val, val).astype(np.float32)


def fp4_e2m1_decode(code: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:256``. 1 sign | 2 exp | 1 mantissa, bias 1; ``e == 0`` is subnormal.

    Only the low nibble is read: FP4 operands are nibble-packed on the wire, and this decodes a
    single 4-bit code, not the byte that carries two of them.
    """
    code = np.asarray(code, dtype=np.uint8) & 0xF
    s = (code >> 3) & 1
    e = (code >> 1) & 0x3
    m = code & 0x1
    val = np.where(e == 0,
                   m / 2.0,
                   (1.0 + m / 2.0) * 2.0 ** (e.astype(np.int32) - 1))
    return np.where(s == 1, -val, val).astype(np.float32)


def fp8_e5m2_decode(code: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:416``. 1 sign | 5 exp | 2 mantissa, bias 15; ``e == 0x1F`` is Inf/NaN."""
    code = np.asarray(code, dtype=np.uint8)
    s = (code >> 7) & 1
    e = ((code >> 2) & 0x1F).astype(np.int32)
    m = (code & 0x3).astype(np.float32)
    val = np.where(e == 0, m * 2.0 ** -16, (1.0 + m * 0.25) * 2.0 ** (e - 15))
    val = np.where(e == 0x1F, np.where(m != 0, np.nan, np.inf), val)
    return np.where(s == 1, -val, val).astype(np.float32)


def fp6_e3m2_decode(code: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:274``. 1 sign | 3 exp | 2 mantissa, bias 3. Six bits, no Inf/NaN."""
    code = np.asarray(code, dtype=np.uint8) & 0x3F
    s = (code >> 5) & 1
    e = ((code >> 2) & 0x7).astype(np.int32)
    m = (code & 0x3).astype(np.float32)
    val = np.where(e == 0, m * 0.0625, (1.0 + m * 0.25) * 2.0 ** (e - 3))
    return np.where(s == 1, -val, val).astype(np.float32)


def fp6_e2m3_decode(code: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:285``. 1 sign | 2 exp | 3 mantissa, bias 1. Six bits, no Inf/NaN."""
    code = np.asarray(code, dtype=np.uint8) & 0x3F
    s = (code >> 5) & 1
    e = ((code >> 3) & 0x3).astype(np.int32)
    m = (code & 0x7).astype(np.float32)
    val = np.where(e == 0, m * 0.125, (1.0 + m * 0.125) * 2.0 ** (e - 1))
    return np.where(s == 1, -val, val).astype(np.float32)


def e8m0_decode(code: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:254``. Code ``0xFF`` is NaN; otherwise ``2**(code - 127)``."""
    code = np.asarray(code, dtype=np.uint8).astype(np.int32)
    return np.where(code == 0xFF, np.nan, 2.0 ** (code - E8M0_BIAS)).astype(np.float32)


def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    """Decode the bf16 bit patterns the device reports over ``OUT`` back to float32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def float_to_bf16_bits(v: np.ndarray) -> np.ndarray:
    """``mx_fp_math.h:9`` (``f32_to_bf16_rne``), including its zero canonicalization.

    Not ``torch.bfloat16``: this must agree with the device bit for bit, and the device folds -0.0
    (and anything that rounds to zero magnitude) to +0.0, which a cast does not.
    """
    v = np.ascontiguousarray(v, dtype=np.float32)
    b = v.view(np.uint32).astype(np.uint64)
    exp_all1 = ((b >> 23) & 0xFF) == 0xFF
    lsb = (b >> 16) & 1
    out = ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)
    nan = ((b >> 16) | np.where((b & 0x7FFFFF) != 0, 0x40, 0).astype(np.uint64)).astype(np.uint16)
    out = np.where(exp_all1, nan, out)
    return np.where((out & 0x7FFF) == 0, np.uint16(0), out).astype(np.uint16)


#: format name -> its element decoder. Exact operand encoders are built by
#: inverting these over the whole code space, so a decoder is the ONLY place an element format's
#: semantics are written down.
DECODERS = {
    "fp8_e4m3": fp8_e4m3_decode,
    "fp8_e4m3_quad": fp8_e4m3_decode,      # same element format, 4-bit indices into it
    "fp8_e5m2": fp8_e5m2_decode,
    "fp6_e3m2": fp6_e3m2_decode,
    "fp6_e2m3": fp6_e2m3_decode,
    "fp4_e2m1": fp4_e2m1_decode,
}


# ---------------------------------------------------------------------------------------------
# Requant-output encoders: bf16 bits -> element code, AS THE HARDWARE ROUNDS
#
# These are NOT the inverse of the decoders above, and that is the whole point. An operand is
# encoded on the host from a value the reference already produced on the format's grid, so
# `operands._exact_encoder` is a lookup and any miss is a bug worth raising. A REQUANT OUTPUT is
# different: the device rounds an arbitrary accumulator value, and each format rounds it its own
# way. Diffed against a C oracle built from `mx_fp_math.h` (scratchpad `requant_oracle.cc`), a
# generic "round to the nearest grid point" model disagrees on 2-9% of elements, three ways:
#
#   * SIGNED ZERO -- a negative value that rounds to zero keeps its sign (code 0x20, not 0). Only a
#     zero INPUT is canonicalized. Costs ~4% of elements on its own.
#   * ROUNDING IS RNE (round half to even) in all four, matching the OCP MX reference
#     (`mx._quantize_elemwise(round='even')`). This became true on 2026-09-10: E4M3 and E5M2 used
#     to round half AWAY from zero and E3M2 used to double-round through an E4M2 intermediate, so
#     no single MXQuant configuration could reproduce all four. See the plan, section 4.18.
#   * saturation, subnormal handling and the zero cases still differ per format, which is why
#     these stay four separate transcriptions rather than one parameterized rounder.
#
# Scalar, mirroring the C statement for statement: these run once per requantized element, and a
# vectorized rewrite would be a second place for the conventions above to drift.

def fp8_e4m3_encode(bits: int) -> int:
    """``mx_fp_math.h:196`` (``fp8_e4m3_to_code``), fed the bf16-rounded value. RNE."""
    return fp8_e4m3_encode_f32(float(bf16_bits_to_float(np.array([bits], dtype=np.uint16))[0]))


def fp8_e4m3_encode_f32(v: float) -> int:
    """``mx_fp_math.h:196`` (``fp8_e4m3_to_code``) taking a FLOAT, which is how the DIRECT 8-bit
    requant path calls it -- that path does not round to bf16 first, unlike the codebook path.
    """
    if math.isnan(v):
        return 0x7F
    if math.isinf(v):
        return (0x80 if v < 0 else 0x00) | 0x7F
    if v == 0.0:
        return 0
    s = 1 if math.copysign(1.0, v) < 0 else 0
    av, bias, emin, emax = abs(v), 7, -6, 8
    E = math.floor(math.log2(av))
    if E < emin:
        k = _rint_even(av / 2.0 ** (emin - 3))
        if k <= 0:
            return s << 7                      # underflow KEEPS ITS SIGN
        if k >= 8:
            return (s << 7) | (1 << 3)
        return (s << 7) | k
    if E > emax:
        E_used, mant = emax, 6
    else:
        E_used = E
        base = 2.0 ** E_used
        k = _rint_even((av - base) / (base / 8.0))
        if k >= 8:
            E_used, k = E_used + 1, 0
            if E_used > emax:
                E_used, k = emax, 6
        else:
            hi = 6 if E_used == emax else 7
            k = min(max(k, 0), hi)
        mant = k
    return (s << 7) | (((E_used + bias) & 0xF) << 3) | (mant & 0x7)


def fp8_e5m2_encode(bits: int) -> int:
    """``mx_fp_math.h:456`` (``bf16_bits_to_e5m2_code``). RNE, in integer form."""
    bits = int(bits) & 0xFFFF
    sign, E, M = (bits >> 15) & 1, (bits >> 7) & 0xFF, bits & 0x7F
    s7 = sign << 7
    if E == 0xFF:
        return s7 | (0x7D if M else 0x7C)
    if E == 0:
        return s7                              # zero / bf16 subnormal -> SIGNED zero
    e = E - 127
    if -14 <= e <= 15:
        q = (M >> 5) & 0x3
        sig = q + (((M >> 4) & 1) & (int((M & 0xF) != 0) | (q & 1)))   # RNE
        carry = sig >> 2
        mant = 0 if carry else sig
        exp_out = e + carry
        if exp_out > 15:
            return s7 | 0x7B
        return s7 | (((exp_out + 15) & 0x1F) << 2) | (mant & 0x3)
    if e > 15:
        return s7 | 0x7B
    sig8, shift = 0x80 | M, -(e + 9)
    if shift >= 9:
        k = 0
    else:
        k = sig8 >> shift
        rem, half = sig8 & ((1 << shift) - 1), 1 << (shift - 1)
        if rem > half or (rem == half and (k & 1)):          # RNE
            k += 1
    if k <= 0:
        return s7
    if k >= 4:
        return s7 | (1 << 2)
    return s7 | (k & 0x3)


def fp6_e2m3_encode(bits: int) -> int:
    """``mx_fp_math.h:498`` (``bf16_bits_to_fp6_e2m3_code``). RNE. Unchanged by the 2026-09-10
    header revision -- this format was already the consistent one."""
    bits = int(bits) & 0xFFFF
    sign, E8, M = (bits >> 15) & 1, (bits >> 7) & 0xFF, bits & 0x7F
    s5 = sign << 5
    if E8 == 0 or E8 == 0xFF:
        return s5
    m_bits, bias, emax = 3, 1, 2
    av = math.ldexp(1.0 + M / 128.0, E8 - 127)
    Efl = math.floor(math.log2(av))
    if Efl < 0:
        k = _rint_even(av / 2.0 ** -m_bits)
        if k <= 0:
            return s5
        if k >= (1 << m_bits):
            return s5 | (1 << m_bits)
        return s5 | k
    E_used = min(Efl, emax)
    base = 2.0 ** E_used
    m = _rint_even((av - base) / (base / (1 << m_bits)))
    if m >= (1 << m_bits):
        E_used, m = E_used + 1, 0
        if E_used > emax:
            E_used, m = emax, (1 << m_bits) - 1
    else:
        m = min(max(m, 0), (1 << m_bits) - 1)
    return s5 | (((E_used + bias) & 0x3) << m_bits) | (m & 0x7)


def _rint_even(x: float) -> int:
    """C's ``nearbyintf`` under the default rounding mode: half to even."""
    return int(np.rint(np.float32(x)))


def fp6_e3m2_encode(bits: int) -> int:
    """``mx_fp_math.h:286`` (``bf16_bits_to_fp6_e3m2_code``). RNE, straight to the E3M2 grid.

    Until the 2026-09-10 header revision this went BF16 -> E4M2 -> a deterministic E4M2->FP6 map,
    which double-rounded: it flushed ``|v|`` in ``[0.0315, 0.0583]`` to zero even though E3M2
    represents 0.0625, and sent 0.0859375 to 0.125 when true-nearest is 0.0625. That is now gone and
    the format rounds like its three siblings.
    """
    bits = int(bits) & 0xFFFF
    sign, E8, M = (bits >> 15) & 1, (bits >> 7) & 0xFF, bits & 0x7F
    s5 = sign << 5
    if E8 == 0 or E8 == 0xFF:
        return s5
    m_bits, bias, emin, emax = 2, 3, -2, 4
    av = math.ldexp(1.0 + M / 128.0, E8 - 127)
    Efl = math.floor(math.log2(av))
    if Efl < emin:
        k = _rint_even(av / 2.0 ** (emin - m_bits))            # quantum 2^-4
        if k <= 0:
            return s5
        if k >= (1 << m_bits):
            return s5 | (1 << m_bits)
        return s5 | k
    E_used = min(Efl, emax)
    base = 2.0 ** E_used
    m = _rint_even((av - base) / (base / (1 << m_bits)))
    if m >= (1 << m_bits):
        E_used, m = E_used + 1, 0
        if E_used > emax:
            E_used, m = emax, (1 << m_bits) - 1
    else:
        m = min(max(m, 0), (1 << m_bits) - 1)
    return s5 | (((E_used + bias) & 0x7) << m_bits) | (m & 0x3)


#: format name -> the encoder the REQUANTIZER applies, taking bf16 BITS. Every codebook format has
#: one; the direct formats do not appear because their chained output never passes through here
#: (E4M3-single is written as an 8-bit code by MXQuant's own convention, FP4 by the FP4 kernel).
ENCODERS = {
    "fp8_e4m3_quad": fp8_e4m3_encode,
    "fp8_e5m2": fp8_e5m2_encode,
    "fp6_e3m2": fp6_e3m2_encode,
    "fp6_e2m3": fp6_e2m3_encode,
}


def encode_requant(values: np.ndarray, *, dtype: str) -> np.ndarray:
    """Element codes for an already-block-normalized array, rounded as the device rounds.

    Goes through bf16 first because the device does: the requantizer's divide-by-scale result is
    rounded to bf16 before the encoder ever sees it, so a value exactly between two grid points can
    be moved off the tie by that rounding.
    """
    enc = ENCODERS.get(dtype)
    if enc is None:
        raise KeyError(
            f"no requant encoder for {dtype!r}; add one to wire.ENCODERS, transcribed from "
            "mx_fp_math.h. A format that reaches the requantizer without one would otherwise be "
            "rounded by a generic rule, which disagrees with the hardware on 2-9% of elements.")
    bits = float_to_bf16_bits(values)
    flat = bits.ravel().tolist()
    return np.array([enc(b) for b in flat], dtype=np.uint8).reshape(bits.shape)


E8M0_NAN = 0xFF


def e8m0_encode_exact(X: np.ndarray) -> np.ndarray:
    """Encode exact powers of two to E8M0 codes. Non-finite scales become the E8M0 NaN code."""
    x = np.asarray(X, dtype=np.float32)
    out = np.full(x.shape, E8M0_NAN, dtype=np.uint8)
    finite = np.isfinite(x) & (x > 0)
    if finite.any():
        exp = np.log2(x[finite].astype(np.float64))
        if not np.all(exp == np.floor(exp)):
            raise ValueError(f"block scale(s) are not powers of two: log2 = {exp[exp != np.floor(exp)][:8]}")
        code = exp + E8M0_BIAS
        if np.any((code < 0) | (code > 254)):
            raise ValueError(f"block scale exponent(s) outside E8M0 range [0, 254]: {(float(code.min()), float(code.max()))}")
        out[finite] = code.astype(np.uint8)
    return out

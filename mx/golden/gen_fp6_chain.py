#!/usr/bin/env python3
"""Generate a CHAINED fp6 (e3m2, LUT-indexed) matmul header: C2 = (A1 @ B1) @ B2.

Operands are 4-bit LUT indices (16-entry LUT per 2^G rows/cols) with random LUTs, indices and E8M0
scales. MM1's requantized output C1 stays RESIDENT in the scratchpad as MM2's operand A, so MM2's
activation LUT is MM1's output LUT (k-means over the requantized values) and its A-scales are MM1's
output scales.

  python3 gen_fp6_chain.py 64     # -> include/matmul_fp6_64x64_chain.h
  python3 gen_fp6_chain.py 128
"""
from __future__ import annotations

import precision
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import math

from fp8_matmul_model import tiled_matmul_hwlike, matrix_mx_requantize, make_fp_quantizer  # noqa: E402
from lut_golden_model import (make_lut, quantize_lut_indices, tensor_to_custom_fp_codes,             # noqa: E402
                              pack_lut_hw_words, _a_indices_to_hw_layout, q_bf16_rne)
from wire import e8m0_decode                                                              # noqa: E402


# --- fp6:e3m2 helpers, vendored VERBATIM from lut_mapping_demo.py (imported as functions here to
#     avoid that module's script-level side effects, which regenerate matmul_fp6_128x128x512.h). ---

def _bf16_to_e4m2_rne(x: torch.Tensor) -> torch.Tensor:
    bf16 = x.to(torch.bfloat16)
    bits = bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    sign_bit = (bits >> 15) & 1
    E = (bits >> 7) & 0xFF
    M = bits & 0x7F
    sign_f = torch.where(sign_bit == 0, torch.ones_like(x), -torch.ones_like(x))
    e = E - 127
    q = (M >> 5) & 3
    r = ((M >> 4) & 1).bool()
    sticky = (M & 0xF).ne(0)
    lsb = ((M >> 5) & 1).bool()
    round_up = r & (sticky | lsb)
    sig_rounded = q + round_up.to(torch.int32)
    carry = sig_rounded >= 4
    mant_out = torch.where(carry, torch.zeros_like(sig_rounded), sig_rounded)
    exp_out = torch.where(carry, e + 1, e)
    is_overflow = exp_out > 7
    safe_exp = exp_out.float().clamp(-10, 10)
    val_normal = sign_f * (1.0 + mant_out.float() * 0.25) * torch.pow(torch.full_like(x, 2.0), safe_exp)
    val_normal = torch.where(is_overflow, sign_f * float('inf'), val_normal)
    quantum = 2.0 ** -8
    is_e_neg7 = (e == -7) & (E >= 1)
    k7 = torch.where(M <= 32, torch.full_like(M, 2),
         torch.where(M <= 95, torch.full_like(M, 3), torch.full_like(M, 4)))
    val_e7 = sign_f * k7.float() * quantum
    is_e_neg8 = (e == -8) & (E >= 1)
    k8 = torch.where(M < 64, torch.ones_like(M), torch.full_like(M, 2))
    val_e8 = sign_f * k8.float() * quantum
    is_e_neg9 = (e == -9) & (E >= 1)
    k9 = torch.where(M == 0, torch.zeros_like(M), torch.ones_like(M))
    val_e9 = sign_f * k9.float() * quantum
    result = val_normal
    result = torch.where(is_e_neg7, val_e7, result)
    result = torch.where(is_e_neg8, val_e8, result)
    result = torch.where(is_e_neg9, val_e9, result)
    result = torch.where(((e <= -10) & (E >= 1)) | (E == 0), torch.zeros_like(x), result)
    is_nan = (E == 255) & (M != 0)
    is_inf = (E == 255) & (M == 0)
    result = torch.where(is_inf, sign_f * float('inf'), result)
    result = torch.where(is_nan, torch.full_like(x, float('nan')), result)
    return result


def _e4m2_to_fp6(x: torch.Tensor) -> torch.Tensor:
    sign = x.sign()
    ax = x.abs()
    mapToZero = (ax <= 0.0546875)
    mapToMax = (ax >= 32.0) | ~torch.isfinite(ax)
    mapToSubnorm = (ax >= 0.0625) & (ax <= 0.21875)
    sub_val = torch.where(ax <= 0.078125, torch.full_like(ax, 0.0625),
              torch.where(ax <= 0.15625, torch.full_like(ax, 0.125), torch.full_like(ax, 0.1875)))
    out = x.clone()
    out = torch.where(mapToZero, torch.zeros_like(x), out)
    out = torch.where(mapToMax, sign * 28.0, out)
    out = torch.where(mapToSubnorm, sign * sub_val, out)
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
    # Round STRAIGHT to the E3M2 grid, RNE (BF16 acc -> E3M2), replacing the old BF16->E4M2->fp6
    # double-rounding path. The BF16 cast first matches the hardware's BF16 accumulator.
    return _e3m2_quantizer()(x.to(torch.bfloat16).to(torch.float32))


def _fp6_value_to_code(v: float) -> int:
    if v == 0.0:
        return 0
    s = 1 if v < 0 else 0
    av = abs(v)
    e_bits, m_bits, bias = 3, 2, 3
    emin = 1 - bias
    if av < 2.0 ** emin:
        quantum = 2.0 ** (emin - m_bits)
        mant = int(round(av / quantum))
        return (s << (e_bits + m_bits)) | max(0, min(mant, (1 << m_bits) - 1))
    E = int(math.floor(math.log2(av)))
    base = 2.0 ** E
    mant = int(round((av - base) / (base / 4)))
    if mant >= 4:
        mant = 0
        E += 1
    biased_exp = min(E + bias, (1 << e_bits) - 1)
    mant = min(mant, (1 << m_bits) - 1)
    return (s << (e_bits + m_bits)) | (biased_exp << m_bits) | mant


def _fp6_tensor_to_codes(t: torch.Tensor) -> list:
    arr = t.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    rows, cols = arr.shape
    return [[_fp6_value_to_code(float(arr[r, c].item())) for c in range(cols)] for r in range(rows)]


def _raw_fp6e3m2_from_bits(val: int) -> dict:
    val = val & 0x3F
    sign = (val >> 5) & 1
    exp = (val >> 2) & 0b111
    mant = val & 0b011
    is_zero = (exp == 0) and (mant == 0)
    s_exp = -2 if exp == 0 else (exp - 3)
    implicit_bit = 0 if exp == 0 else 1
    sig = (implicit_bit << 2) | mant
    return {'sign': sign, 'is_zero': is_zero, 's_exp': s_exp, 'sig': sig}


def fp6_to_fixed_point(val: int) -> int:
    raw = _raw_fp6e3m2_from_bits(val)
    shift_amt = (raw['s_exp'] + 2) & 0b111
    shifted = (raw['sig'] << shift_amt) & 0x1FF
    magnitude = shifted & 0xFF
    signed = -magnitude if raw['sign'] else magnitude
    return 0 if raw['is_zero'] else signed


def fp6e3m2_nearest_finder(in_fp6: int, in_lut: list) -> int:
    assert len(in_lut) == 16, "LUT must have exactly 16 entries"
    fixed_in = fp6_to_fixed_point(in_fp6)
    fixed_lut = [fp6_to_fixed_point(v) for v in in_lut]
    diffs = [abs(fixed_in - f) & 0x1FF for f in fixed_lut]
    min_idx = 0
    for i in range(1, 16):
        if diffs[i] < diffs[min_idx]:
            min_idx = i
    return min_idx

INPUT_SPEC = "fp6:e3m2"
SCALE_SPEC = "fpe8m0"
GROUP      = 32      # K-group for scales AND the LUT-group is 2^G rows
G          = 1       # QUANT_LUT_UPDATE_GRANULARITY: one LUT per 2 rows/cols
LUT_SIZE   = 16
A_TILE_M   = 32
K_TILE     = 16
DEV        = torch.device("cpu")

_in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")


# --- output-LUT construction (k-means over the requantized fp6 values) ---

def _kmeans_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Deterministic weighted 1-D k-means over the distinct values of ``values``.

    Returns up to ``k`` centroids. The support is tiny (FP6 has 64 codes), so this collapses to a
    weighted Lloyd over distinct values and converges in a handful of passes.
    """
    uniq, counts = np.unique(values, return_counts=True)
    if uniq.size <= k:
        return uniq
    # Quantile init over the weighted distribution: spreads seeds by mass, not by range, so a
    # cluster of near-zero values (which is most of a real activation block) is not under-served.
    cdf = np.cumsum(counts) / counts.sum()
    probes = (np.arange(k) + 0.5) / k
    centers = np.unique(uniq[np.searchsorted(cdf, probes).clip(0, uniq.size - 1)])
    for _ in range(50):
        lab = np.abs(uniq[:, None] - centers[None, :]).argmin(axis=1)
        new = np.array([
            (uniq[lab == c] * counts[lab == c]).sum() / counts[lab == c].sum()
            if np.any(lab == c) else centers[c]
            for c in range(centers.size)])
        new = np.unique(new)
        if new.size == centers.size and np.allclose(new, centers):
            break
        centers = new
    return centers


def build_luts(P: torch.Tensor, *, axis: str, G: int, lut_size: int,
               codebook: torch.Tensor) -> list[torch.Tensor]:
    """One ``lut_size``-entry LUT per group of ``2**G`` rows (axis="row") or columns (axis="col").

    ``P`` is already MX-quantized, so its values are exact FP6 codebook entries; the k-means
    centroids are snapped back onto ``codebook`` so every LUT entry stays a representable code.
    Entries are padded to exactly ``lut_size`` with unused codes nearest the group's range, so no
    slot is a duplicate that could make the index assignment ambiguous.
    """
    p = P.numpy()
    n_groups = (p.shape[0] if axis == "row" else p.shape[1]) >> G
    cb = codebook.numpy()
    luts = []
    for g in range(n_groups):
        sl = slice(g << G, (g + 1) << G)
        vals = (p[sl, :] if axis == "row" else p[:, sl]).ravel()
        centers = _kmeans_1d(vals, lut_size)
        snapped = cb[np.abs(centers[:, None] - cb[None, :]).argmin(axis=1)]
        entries = list(dict.fromkeys(snapped.tolist()))          # order-preserving dedupe
        if len(entries) < lut_size:                              # pad with unused codes
            for c in sorted(cb.tolist(), key=lambda v: abs(v)):
                if c not in entries:
                    entries.append(c)
                if len(entries) == lut_size:
                    break
        luts.append(torch.tensor(sorted(entries[:lut_size]), dtype=torch.float32))
    return luts


def fp6_codebook() -> torch.Tensor:
    """Every finite FP6 E3M2 value, decoded from its 6-bit code (mx_fp_math.h:274)."""
    vals = []
    for code in range(64):
        s, e, m = (code >> 5) & 1, (code >> 2) & 0x7, code & 0x3
        v = (m * 0.0625) if e == 0 else (1.0 + m * 0.25) * 2.0 ** (e - 3)
        vals.append(-v if s else v)
    return torch.tensor(sorted(set(vals)), dtype=torch.float32)



def _project(indices: torch.Tensor, luts_t: torch.Tensor, axis: str, K: int) -> torch.Tensor:
    """indices [rows][cols] + per-group LUT -> fp6 values (LUT lookup)."""
    R = indices.shape[0] if axis == "row" else indices.shape[1]
    if axis == "row":
        gi = (torch.arange(indices.shape[0]) >> G).unsqueeze(1).expand(-1, indices.shape[1])
    else:
        gi = (torch.arange(indices.shape[1]) >> G).unsqueeze(0).expand(indices.shape[0], -1)
    return luts_t[gi, indices]


def _lut_codes(luts_t: torch.Tensor) -> list[list[int]]:
    """Each group's 16 fp6 6-bit codes (for nearest-index projection + header packing)."""
    codes, _ = tensor_to_custom_fp_codes(luts_t, INPUT_SPEC)   # [n_groups][16]
    return codes


def run_mm(A_idx, A_luts_t, A_scale_dec, B_idx, B_luts_t, B_scale_dec, M, K, N):
    """One fp6 LUT matmul. Returns C1 as (indices [M][N], out-LUTs [M>>G][16], e8m0 scales [M][GN],
    proj_hw [M/2][N], requant scale-values [M][GN] for reuse, bf16 output)."""
    A_fp = _project(A_idx, A_luts_t, "row", K)                 # [M][K]
    B_fp = _project(B_idx, B_luts_t, "col", K)                 # [K][N]
    A_in, B_in = _in_q(A_fp), _in_q(B_fp)
    C_bf16 = tiled_matmul_hwlike(A_in, B_in, A_scale_dec, B_scale_dec, verbose=False,
                                 prod_precision_list=precision.for_dim(K_TILE)[0], acc_precision_list=precision.for_dim(K_TILE)[1])
    C_req, C_scale_vals = matrix_mx_requantize(C_bf16, quant_spec=INPUT_SPEC)   # values, [M][N/32]
    C_req6 = hw_bf16_to_fp6(q_bf16_rne(C_req)).float().view(M, N)
    # Data-derived OUTPUT LUTs (k-means on the requantized fp6 values), 16 entries per 2 rows.
    C_luts = build_luts(C_req6, axis="row", G=G, lut_size=LUT_SIZE, codebook=fp6_codebook())
    C_luts_t = torch.stack(C_luts)                             # [M>>G][16]
    C_lut_codes = _lut_codes(C_luts_t)
    C_req_codes = _fp6_tensor_to_codes(C_req6)                 # [M][N] 6-bit fp6 codes
    C_proj = torch.zeros(M, N, dtype=torch.int32)
    for m in range(M):
        lc = C_lut_codes[m >> G]
        for n in range(N):
            C_proj[m, n] = fp6e3m2_nearest_finder(C_req_codes[m][n], lc)
    C_proj_hw = np.array(_a_indices_to_hw_layout(C_proj, A_TILE_M, K_TILE), dtype=np.uint8)  # [M/2][N]
    C_scale_codes = np.array(tensor_to_custom_fp_codes(C_scale_vals, SCALE_SPEC)[0], dtype=np.uint8)  # [M][GN]
    return dict(idx=C_proj, luts_t=C_luts_t, scale_vals=C_scale_vals, scale_codes=C_scale_codes,
                proj_hw=C_proj_hw, bf16=C_bf16)


def _random_lut_operand(rows, cols, axis, g):
    """Random LUTs (one per 2^G rows/cols), randn values -> nearest LUT index, E8M0 scales 2^[-4,4)."""
    n = rows if axis == "row" else cols
    luts = torch.stack([make_lut(INPUT_SPEC, 4, DEV) for _ in range(n >> G)])
    V = torch.randn(rows, cols, generator=g)
    if axis == "row":
        idx = torch.stack([quantize_lut_indices(luts[i >> G], V[i]) for i in range(rows)])
        sc = 127 + torch.randint(-4, 4, (rows, cols // 32), generator=g)
    else:
        idx = torch.stack([quantize_lut_indices(luts[j >> G], V[:, j]) for j in range(cols)], dim=1)
        sc = 127 + torch.randint(-4, 4, (rows // 32, cols), generator=g)
    sc = sc.numpy().astype(np.uint8)
    return idx, luts, sc, torch.from_numpy(e8m0_decode(sc).astype(np.float32))


def build_chain(S: int):
    M = K = N = S
    torch.manual_seed(S)                      # make_lut draws from the global RNG
    g = torch.Generator().manual_seed(S)
    A1_idx, A1_luts, A1_sc, A1_sd = _random_lut_operand(M, K, "row", g)
    B1_idx, B1_luts, B1_sc, B1_sd = _random_lut_operand(K, N, "col", g)

    C1 = run_mm(A1_idx, A1_luts, A1_sd, B1_idx, B1_luts, B1_sd, M, K, N)

    B2_idx, B2_luts, B2_sc, B2_sd = _random_lut_operand(N, N, "col", g)

    # MM2: A2 = C1 (indices), A2 LUT = C1 output LUT, A2 scales = C1 output scales.
    C2 = run_mm(C1["idx"], C1["luts_t"], C1["scale_vals"], B2_idx, B2_luts, B2_sd, M, K, N)

    return dict(M=M, K=K, N=N,
                A1_idx=A1_idx.numpy().astype(np.uint8), A1_sc=A1_sc, A1_luts=A1_luts,
                B1_idx=B1_idx.numpy().astype(np.uint8), B1_sc=B1_sc, B1_luts=B1_luts,
                B2_idx=B2_idx.numpy().astype(np.uint8), B2_sc=B2_sc, B2_luts=B2_luts,
                C1=C1, C2=C2)


def _pack_lut_rows(luts_t):
    codes = _lut_codes(luts_t)
    return [pack_lut_hw_words([int(c) for c in row]) for row in codes]   # [n_groups][3]


def _hw_a(idx, K):                                       # [M][K] indices -> [M/2][K] HW-tiled bytes
    return np.array(_a_indices_to_hw_layout(torch.from_numpy(idx), A_TILE_M, K_TILE), dtype=np.uint8)


def _pack_b(idx):                                        # [K][N] indices -> [K][N/2] nibble-packed
    return ((idx[:, 1::2].astype(np.uint16) << 4) | idx[:, 0::2]).astype(np.uint8)


def _rows(a, w):
    fmt = f"0x%0{w}x"
    return ",\n".join("    { " + ", ".join(fmt % int(v) for v in row) + " }" for row in a)


def _rows_lut(packed):
    return ",\n".join("    { " + ", ".join(f"0x{w:08x}" for w in row) + " }" for row in packed)


def emit(d):
    M, K, N = d["M"], d["K"], d["N"]
    GK, GN = K // 32, N // 32
    stem = f"matmul_fp6_{M}x{N}_chain"
    guard = f"INCLUDE_{stem.upper()}_H"
    A_hw = _hw_a(d["A1_idx"], K)
    B_pk, B2_pk = _pack_b(d["B1_idx"]), _pack_b(d["B2_idx"])
    A_lut, B_lut = _pack_lut_rows(d["A1_luts"]), _pack_lut_rows(d["B1_luts"])
    B2_lut = _pack_lut_rows(d["B2_luts"])
    C1, C2 = d["C1"], d["C2"]
    C1_lut, C2_lut = _pack_lut_rows(C1["luts_t"]), _pack_lut_rows(C2["luts_t"])
    nAg, nBg = M >> G, N >> G
    path = Path(os.environ.get("MX_DATA_DIR", HERE.parent.parent / "include")) / f"{stem}.h"
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by mx/golden/gen_fp6_chain.py {M} -- do not edit by hand.
//
// CHAINED fp6:e3m2 (LUT-indexed) matmul: C2 = (A1 @ B1) @ B2. MM1's requantized output C1 stays
// RESIDENT (block-tiled operand layout) and is reused as MM2's operand A; MM2's activation-in LUT =
// MM1's OUTPUT LUT (C1_lut), and MM1's output block-scales are reused as MM2's input A-scales.
// Random operands: 4-bit LUT indices (A in HW tiled [M/2][K], B packed [K][N/2]); LUTs are 16 fp6 codes / 2 rows.
#ifndef {guard}
#define {guard}
#include <stdint.h>

#define MATMUL_M   {M}
#define MATMUL_K   {K}
#define MATMUL_N   {N}
#define MATMUL_GK  {GK}
#define MATMUL_GN  {GN}
#define A_TILE_M   {A_TILE_M}
#define K_TILE     {K_TILE}
#define LUT_GROUPS_A {nAg}
#define LUT_GROUPS_B {nBg}

// ---- MM1 operands (A1 @ B1) ----
static const uint8_t A_in_hw[{M // 2}][{K}] = {{
{_rows(A_hw, 2)}
}};
static const uint8_t B_in[{K}][{N // 2}] = {{
{_rows(B_pk, 2)}
}};
static const uint8_t A_scales_row[{GK}][{M}] = {{
{_rows(d['A1_sc'].T, 2)}
}};
static const uint8_t B_scales_col[{GK}][{N}] = {{
{_rows(d['B1_sc'], 2)}
}};
static const uint32_t A_lut[{nAg}][3] = {{
{_rows_lut(A_lut)}
}};
static const uint32_t B_lut[{nBg}][3] = {{
{_rows_lut(B_lut)}
}};

// ---- MM2 weight operand (C1 @ B2) ----
static const uint8_t B2_in[{K}][{N // 2}] = {{
{_rows(B2_pk, 2)}
}};
static const uint8_t B2_scales_col[{GK}][{N}] = {{
{_rows(d['B2_sc'], 2)}
}};
static const uint32_t B2_lut[{nBg}][3] = {{
{_rows_lut(B2_lut)}
}};

// ---- MM1 output C1 = requant(A1 @ B1): RESIDENT operand + residency check (HW tiled [M/2][N]) ----
static const uint8_t C1_out[{M // 2}][{N}] = {{
{_rows(C1['proj_hw'], 2)}
}};
static const uint8_t C1_scales_out[{M}][{GN}] = {{
{_rows(C1['scale_codes'], 2)}
}};
// C1 output LUT -- reused as MM2's activation-in LUT.
static const uint32_t C1_lut[{nAg}][3] = {{
{_rows_lut(C1_lut)}
}};

// ---- MM2 output C2 = requant(C1 @ B2): final chained result ----
static const uint8_t C2_out[{M // 2}][{N}] = {{
{_rows(C2['proj_hw'], 2)}
}};
static const uint8_t C2_scales_out[{M}][{GN}] = {{
{_rows(C2['scale_codes'], 2)}
}};
static const uint32_t C2_lut[{nAg}][3] = {{
{_rows_lut(C2_lut)}
}};

#endif // {guard}
""")
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="fp6 chained-matmul data header (DIM 16).")
    ap.add_argument("size", type=int, nargs="?", default=64)
    precision.add_arg(ap)
    a = ap.parse_args()
    precision.load(a.params)
    S = a.size
    print(f"fp6 chain {S}x{S}x{S} ...")
    d = build_chain(S)
    p = emit(d)
    mask = 0x3F
    print(f"  C1 idx max {int(d['C1']['idx'].max())} scales {int(d['C1']['scale_codes'].min())}.."
          f"{int(d['C1']['scale_codes'].max())}")
    print(f"  C2 idx max {int(d['C2']['idx'].max())} scales {int(d['C2']['scale_codes'].min())}.."
          f"{int(d['C2']['scale_codes'].max())}")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()

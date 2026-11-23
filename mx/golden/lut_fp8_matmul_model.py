#!/usr/bin/env python3
import math
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
    if rounding in ("nearest", "nearest_even", "stochastic"):
        from qtorch.quant import float_quantize
        mode = "nearest" if rounding in ("nearest", "nearest_even") else "stochastic"
        return lambda x: float_quantize(x, exp=e, man=m, rounding=mode)
    if rounding in ("zero", "toward_zero", "trunc", "rtz"):
        return lambda x: float_quantize_trunc(x, exp=e, man=m)
    raise ValueError(f"Unsupported rounding mode: {rounding}")

def tensor_to_custom_fp_codes(t: Tensor, spec: str) -> Tuple[List[List[int]], int]:
    e_bits, m_bits = parse_fp_spec(spec)
    total_bits = 1 + e_bits + m_bits
    bias = (1 << (e_bits - 1)) - 1
    emin = 1 - bias
    # MX FP8 E4M3: biased_exp goes up to 15 (unbiased=8), NaN=0x7F only → pmax=448
    is_mx_fp8 = (e_bits == 4 and m_bits == 3)
    emax = bias + 1 if is_mx_fp8 else bias
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
                    code = 0
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

# --- Hardware-matching MAC ---

def prod_quant(x: Tensor) -> Tensor:
    return trunc_product_mantissa(x, frac_bits=PROD_MANT_BITS)

def matmul_outer_quantized_hwlike(A_in: Tensor, B_in: Tensor) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2
    C = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)
    for k in range(K):
        outer = torch.outer(A_in[:, k], B_in[k, :])
        outer = prod_quant(outer)
        outer = q_bf16_rne(outer)
        C = q_bf16_rne(q_bf16_rne(C) + q_bf16_rne(outer))
    return C

def compute_tile_scale_matrix(A_scales_row, B_scales_col, m0, n0, k0, TM, TN):
    g = k0 // GROUP
    sA = A_scales_row[m0:m0+TM, g].to(torch.float32)
    sB = B_scales_col[g, n0:n0+TN].to(torch.float32)
    S = torch.outer(sA, sB)
    S_q = make_fp_quantizer(SCALE_SPEC, "nearest")(S)
    return S_q

def bf16_accum_add(x: Tensor, y: Tensor) -> Tensor:
    return q_bf16_rne(q_bf16_rne(x) + q_bf16_rne(y))

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

                C_tile = matmul_outer_quantized_hwlike(A_tile, B_tile)

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
    # max_representable = 2^emax * (2 - 2^-m) = 2^8 * 1.75 = 448
    emax = (1 << (e_bits - 1))  # = 8 for e4m3
    log2_pmax = emax  # floor(log2(448)) = 8

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

def run(M: int, K: int, N: int, header_path: str = "matmul_data.h", verbose: bool = True):
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
    A_scale_exp = torch.zeros(size=(M, Gk), device=dev)
    B_scale_exp = torch.zeros(size=(Gk, N), device=dev)
    A_scales_row = torch.pow(2.0, A_scale_exp.to(torch.float32))
    B_scales_col = torch.pow(2.0, B_scale_exp.to(torch.float32))
    A_scales_row_q = make_fp_quantizer(SCALE_SPEC, "nearest")(A_scales_row)
    B_scales_col_q = make_fp_quantizer(SCALE_SPEC, "nearest")(B_scales_col)

    C_out_bf16 = tiled_matmul_hwlike(A_in, B_in, A_scales_row_q, B_scales_col_q, verbose=verbose)

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

    return C_out_bf16

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--header-path", type=str, default="matmul_data.h")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args.M, args.K, args.N, header_path=args.header_path, verbose=not args.quiet)


#!/usr/bin/env python3
import math
import re
import time
from typing import Callable, Optional, Dict, Tuple, List

import torch

Tensor = torch.Tensor
QuantFn = Optional[Callable[[Tensor], Tensor]]

# ----------------------------------------------------------------------
# FP format presets
# ----------------------------------------------------------------------

_FP_PRESETS = {
    "fp16":      dict(exp=5, man=10),
    "bf16":      dict(exp=8, man=7),
    "fp8:e4m3":  dict(exp=4, man=3),
    "fp8:e5m2":  dict(exp=5, man=2),
    "fp6:e3m2":  dict(exp=3, man=2),
    "fp6:e2m3":  dict(exp=2, man=3),
    "fp4:e4m1":  dict(exp=2, man=1),

    # Experimental / custom
    "fp26":      dict(exp=7, man=18),
    "fp20":      dict(exp=6, man=13),
    "fp14":      dict(exp=5, man=8),

    # Scaling format: fpE8M0 (1 sign, 8 exp, 0 mant)
    "fpe8m0":    dict(exp=8, man=0),
}

# ----------------------------------------------------------------------
# Spec parsing / quantization helpers
# ----------------------------------------------------------------------

def trunc_product_mantissa(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """
    Truncate mantissa (fraction) bits of x to frac_bits, with RTZ/chop behavior.

    Models: x = sign * (1.fraction) * 2^e  (normalized)
      -> fraction chopped to frac_bits

    Notes:
    - Keeps exponent basically FP32-wide (no exponent quantization).
    - Flushes subnormals to 0 for simplicity (often matches hardware if you don't support denorms).
    - Preserves inf/nan.
    """
    x = x.to(torch.float32)

    # Handle specials
    is_finite = torch.isfinite(x)
    ax = x.abs()
    out = x.clone()

    # zero stays zero
    nz = (ax != 0) & is_finite
    if not torch.any(nz):
        return out

    xn = x[nz]

    # Decompose: xn = m * 2^e with m in [0.5, 1)
    m, e = torch.frexp(xn)  # m in (-1, -0.5] U [0.5, 1)
    sign = torch.sign(m)
    m = m.abs()

    # Convert to [1,2): m2 = m*2, e2 = e-1
    m2 = m * 2.0
    e2 = e - 1

    # Chop fraction bits: m2 = 1 + frac, frac in [0,1)
    frac = m2 - 1.0
    scale = float(1 << frac_bits)
    frac_q = torch.floor(frac * scale) / scale   # RTZ/chop
    m2_q = 1.0 + frac_q

    # Recompose
    out_nz = (sign * m2_q) * torch.ldexp(torch.ones_like(m2_q), e2)

    # Flush subnormals to zero (optional, but often what hardware does)
    # Smallest normal float32 is ~1.175e-38; anything smaller becomes subnormal.
    # If your hardware *does* keep subnormals, remove this block.
    out_nz = torch.where(out_nz.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(out_nz), out_nz)

    out[nz] = out_nz
    return out

def q_bf16_rne(x: torch.Tensor) -> torch.Tensor:
    # PyTorch bf16 cast is round-to-nearest-even; converting back keeps values as fp32
    return x.to(torch.bfloat16).to(torch.float32)

def hw_add_bf16(raw_in: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    raw_q = q_bf16_rne(raw_in)
    c_q   = q_bf16_rne(c)
    return q_bf16_rne(raw_q + c_q)

def zext_codes(codes: List[List[int]], target_bits: int) -> List[List[int]]:
    mask = (1 << target_bits) - 1
    return [[(code & mask) for code in row] for row in codes]

def parse_fp_spec(spec: str) -> Tuple[int, int]:
    """
    Parse something like:
      - 'bf16'       -> (8, 7)
      - 'fp8:e4m3'   -> (4, 3)
      - 'fp32'       -> (8, 23)
      - 'fpe8m0'     -> (8, 0)
    using _FP_PRESETS or the regex 'fp\\d+:(e(\\d+)m(\\d+))'.
    """
    s = spec.strip().lower()
    if s in _FP_PRESETS:
        return _FP_PRESETS[s]["exp"], _FP_PRESETS[s]["man"]
    if s == "fp32":
        return 8, 23
    m = re.fullmatch(r"fp\d+:(e(\d+)m(\d+))", s)
    if m:
        return int(m.group(2)), int(m.group(3))
    raise ValueError(f"Unrecognized floating-point spec: {spec}")


def float_quantize_trunc(x: Tensor, exp: int, man: int) -> Tensor:
    """
    Quantize to a floating-point format with `exp` exponent bits and `man` mantissa bits,
    using round-toward-zero (truncation).
    - Normalized numbers only (subnormals flushed to zero).
    - Overflow saturates to max finite.
    """
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

    bias = (1 << (exp - 1)) - 1   # 2^(exp-1) - 1
    emin = 1 - bias
    emax = bias

    underflow_mask = E < emin
    overflow_mask = E > emax
    normal_mask = (~underflow_mask) & (~overflow_mask)

    ax_q = torch.zeros_like(ax_nz)

    # Underflow -> 0
    ax_q[underflow_mask] = 0.0

    # Overflow -> max finite
    if overflow_mask.any():
        E_max = float(emax)
        base_max = torch.pow(
            torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device),
            E_max,
        )
        delta_max = base_max / (2 ** man)
        max_val = base_max + (2 ** man - 1) * delta_max
        ax_q[overflow_mask] = max_val

    # Normal
    if normal_mask.any():
        E_norm = E[normal_mask]
        x_norm = ax_nz[normal_mask]

        base = torch.pow(
            torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device),
            E_norm,
        )
        delta = base / (2 ** man)
        t = (x_norm - base) / delta
        # Round-toward-zero in magnitude: floor
        k = torch.floor(torch.clamp(t, 0, 2 ** man - 1 - 1e-7))

        q_norm = base + k * delta
        ax_q[normal_mask] = q_norm

    out_nz = sign[nz_mask] * ax_q
    out[nz_mask] = out_nz
    return out


def make_fp_quantizer(spec: str, rounding: str = "nearest") -> QuantFn:
    """
    Create a quantizer for a given FP format.
    rounding:
      - 'nearest' or 'nearest_even' or 'stochastic' -> use qtorch.float_quantize
      - 'zero' / 'toward_zero' / 'trunc' / 'rtz'    -> custom RTZ
    """
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


def make_lut(spec: str, index_bits: int, dev: torch.device, attempts: int = 1024) -> Tensor:
    elem_count = 1 << index_bits
    fp_quantizer = make_fp_quantizer(spec)
    vals = torch.randn(attempts, dtype=torch.float32)
    qvals = fp_quantizer(vals)
    codes, _ = tensor_to_custom_fp_codes(qvals.unsqueeze(-1), spec)
    lut = [0.0] # Always include zero
    for v, code in zip(qvals.cpu().detach().tolist(), codes):
        if v not in lut and code[0] != 0:
            lut.append(v)
        if len(lut) == elem_count:
            break
    if len(lut) != elem_count:
        lut.extend([0.0] for i in range(elem_count - len(lut)))
    return torch.tensor(lut, device=dev, dtype=torch.float32)


def quantize_lut_indices(lut: Tensor, t: Tensor) -> Tensor:
    difference = t.unsqueeze(-1) - lut
    return difference.abs().argmin(dim=-1)


def lut_lookup(lut: Tensor, i: Tensor) -> Tensor:
    return lut[i]


def tensor_to_custom_fp_codes(t: Tensor, spec: str) -> Tuple[List[List[int]], int]:
    """
    Encode a *quantized* tensor (values assumed representable in the given FP format)
    into sign/exponent/mantissa bits.

    Format:
      - 1 sign bit
      - e_bits exponent bits with bias
      - m_bits mantissa bits
      - no subnormals (all denormals flushed to zero)
      - overflow assumed already saturated

    Returns:
      codes: List of rows, each row is a list of integer codes.
      total_bits: 1 + e_bits + m_bits
    """
    e_bits, m_bits = parse_fp_spec(spec)
    total_bits = 1 + e_bits + m_bits

    bias = (1 << (e_bits - 1)) - 1
    emin = 1 - bias
    emax = bias

    arr = t.detach().cpu()
    if arr.ndim != 2:
        raise ValueError("Expected 2D tensor for hex dump.")
    rows, cols = arr.shape

    out_codes: List[List[int]] = []
    for r in range(rows):
        row_codes: List[int] = []
        for c in range(cols):
            v = float(arr[r, c].item())
            if v == 0.0 or not math.isfinite(v):
                code = 0
            else:
                sign = 1 if v < 0 else 0
                av = abs(v)
                E = math.floor(math.log2(av))

                if E < emin:
                    code = 0
                else:
                    if E > emax:
                        E_used = emax
                        base = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        mant = (2 ** m_bits) - 1
                    else:
                        E_used = E
                        base = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        tpos = (av - base) / delta
                        mant = int(round(tpos))
                        mant = max(0, min(mant, 2 ** m_bits - 1))

                    exp_bits_val = int(E_used + bias)
                    code = ((sign & 0x1) << (e_bits + m_bits)) | \
                           ((exp_bits_val & ((1 << e_bits) - 1)) << m_bits) | \
                           (mant & ((1 << m_bits) - 1))

            row_codes.append(code)
        out_codes.append(row_codes)
    return out_codes, total_bits


def codes_to_hex_rows(codes: List[List[int]], total_bits: int) -> List[List[str]]:
    width = (total_bits + 3) // 4  # hex digits
    return [[f"{code:0{width}x}" for code in row] for row in codes]


def c_type_for_bits(total_bits: int) -> str:
    if total_bits <= 8:
        return "uint8_t"
    elif total_bits <= 16:
        return "uint16_t"
    elif total_bits <= 32:
        return "uint32_t"
    else:
        return "uint64_t"

# ----------------------------------------------------------------------
# Matmul and error metrics
# ----------------------------------------------------------------------

def matmul_outer(A: Tensor, B: Tensor) -> Tensor:
    """
    Reference outer-product matmul (full precision).
    C = sum_k A[:,k] * B[k,:]
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)
    for k in range(K):
        C += torch.outer(A[:, k], B[k, :])
    return C


def matmul_outer_quantized_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    prod_quant: QuantFn = None,   # whatever you already use for mul output rounding/trunc
    acc_each_add: bool = True,    # true matches “round after add each cycle”
) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2

    # Accumulator stored as fp32 values, but forced onto BF16 grid after each add
    C = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)

    for k in range(K):
        outer = torch.outer(A_in[:, k], B_in[k, :])        # mul
        if prod_quant:
            outer = prod_quant(outer)                      # round/trunc after mul (your rule)

        # IMPORTANT: match your resize(... -> cType) before the add
        outer = q_bf16_rne(outer)

        # Add in BF16 and round result to BF16
        C = q_bf16_rne(q_bf16_rne(C) + q_bf16_rne(outer))   # BF16 add + RNE

        if not acc_each_add:
            pass  # keep as-is; if you want “only round at end”, move rounding outside loop

    return C

def compute_tile_scale_matrix_fpe8m0(
    A_scales_row: Tensor,     # shape [M * (K // 32), Gk]  (per-row, per-32 K-group)
    B_scales_col: Tensor,     # shape [Gk, N * (K // 32)]  (per-col, per-32 K-group)
    m0: int, n0: int, k0: int,
    TM: int, TN: int,
    M: int, N: int, K: int,
    group: int,
    scale_spec: str = "fpe8m0",
) -> Tensor:
    """
    Build S_tile (TM x TN) for this tile using outer product:
      sA_vec[i] = A_scales_row[m0+i, g]
      sB_vec[j] = B_scales_col[g, n0+j]
      S_tile[i,j] = sA_vec[i] * sB_vec[j]
    where g = k0 // group.

    Returned S_tile is quantized to scale_spec (e.g., fpe8m0).
    """
    g = k0 // group
    sA = A_scales_row[m0:m0+TM, g].to(torch.float32)   # [TM]
    sB = B_scales_col[g, n0:n0+TN].to(torch.float32)   # [TN]
    S = torch.outer(sA, sB)                            # [TM, TN]

    S_q = make_fp_quantizer(scale_spec, "nearest")(S)  # e8m0 in your case
    return S_q


def bf16_accum_add(x: Tensor, y: Tensor) -> Tensor:
    # BF16 add with rounding to BF16
    return q_bf16_rne(q_bf16_rne(x) + q_bf16_rne(y))


def tiled_matmul_scaled_accum_hwlike(
    A_in: Tensor, B_in: Tensor,
    *,
    prod_quant: QuantFn,
    A_scales_row: Tensor,     # [M, Gk]
    B_scales_col: Tensor,     # [Gk, N]
    tile: int = 16,
    group: int = 32,
    log_file: str = "/dev/null",
    scale_spec: str = "fpe8m0",
    trace_tiles: Optional[List[Tuple[int,int,int]]] = None,  # list of (m0,n0,k0)
    trace_max: int = 8,
) -> Tuple[Tensor, Dict]:
    """
    Full tiled GEMM:
      for m0,n0,k0 in tiles:
        C_tile_bf16 = systolic-like outer-product MAC (bf16 rounding each add)
        S_tile_q    = outer(row_scales, col_scales) quantized to scale_spec
        C_tile_scaled_bf16 = bf16( C_tile_bf16 * S_tile_q )
        C_out = bf16( C_out + C_tile_scaled_bf16 )

    Returns:
      C_out (bf16 grid in fp32 tensor),
      debug dict containing optional per-tile dumps.
    """
    log = open(log_file, "w")
    def format_2d_array(hex_rows: List[List[str]]) -> str:
        lines = []
        for row in hex_rows:
            line = ", ".join(f"0x{h}" for h in row)
            lines.append("    { " + line + " }")
        return ",\n".join(lines)
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2

    TM = TN = TK = tile
    assert M % TM == 0 and N % TN == 0 and K % TK == 0, \
        "For now require M,N,K multiples of tile=16 (easy to relax later)."
    assert group % TK == 0, "group (32) should be multiple of TK (16) for your reuse rule."

    Gk = (K + group - 1) // group
    assert A_scales_row.shape == (M, Gk)
    assert B_scales_col.shape == (Gk, N)

    C_out = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)

    debug = {"tiles": []}
    traced = 0

    def should_trace(m0,n0,k0):
        nonlocal traced
        if trace_tiles is not None:
            return (m0,n0,k0) in trace_tiles
        return traced < trace_max

    for m0 in range(0, M, TM):
        for n0 in range(0, N, TN):
            log.write(f"\n\n\n\n== M: {m0}, N: {n0} ==\n\n")
            C_out_tile = torch.zeros((TM, TN), dtype=torch.float32, device=A_in.device)
            for k0 in range(0, K, TK):
                log.write(f"\n\n= K: {k0} =\n\n")
                A_tile = A_in[m0:m0+TM, k0:k0+TK]  # [16,16]
                B_tile = B_in[k0:k0+TK, n0:n0+TN]  # [16,16]

                # 1) tile MAC (bf16 after each add inside)
                C_tile = matmul_outer_quantized_hwlike(
                    A_tile, B_tile,
                    prod_quant=prod_quant,
                    acc_each_add=True
                )  # returns fp32 tensor on bf16 grid

                log.write("Pre-scaled:\n")
                C_codes, C_bits = tensor_to_custom_fp_codes(C_tile, "bf16")
                C_hex  = codes_to_hex_rows(C_codes, C_bits)
                log.write(format_2d_array(C_hex))

                # 2) scale tile (outer product scales, quantized to scale_spec)
                S_tile_q = compute_tile_scale_matrix_fpe8m0(
                    A_scales_row, B_scales_col,
                    m0=m0, n0=n0, k0=k0,
                    TM=TM, TN=TN,
                    M=M, N=N, K=K,
                    group=group,
                    scale_spec=scale_spec
                )

                # 3) scale down and keep bf16
                C_tile_scaled = q_bf16_rne(C_tile * S_tile_q)

                log.write("\nScaled:\n")
                C_codes, C_bits = tensor_to_custom_fp_codes(C_tile_scaled, "bf16")
                C_hex  = codes_to_hex_rows(C_codes, C_bits)
                log.write(format_2d_array(C_hex))

                # 4) accumulate into output in bf16 each tile-add
                C_prev = C_out_tile
                C_out_tile = bf16_accum_add(C_out_tile, C_tile_scaled)

                log.write("\nAccumulated:\n")
                C_codes, C_bits = tensor_to_custom_fp_codes(C_out_tile, "bf16")
                C_hex  = codes_to_hex_rows(C_codes, C_bits)
                log.write(format_2d_array(C_hex))

                # optional tracing
                # if should_trace(m0,n0,k0):
                #     traced += 1
                #     print(f"\n=== TILE (m0={m0}, n0={n0}, k0={k0}) g={k0//group} ===")
                #     print("A_tile:")
                #     print(A_tile.detach().cpu().numpy())
                #     print("B_tile:")
                #     print(B_tile.detach().cpu().numpy())
                #     print("C_tile (bf16 grid):")
                #     print(C_tile.detach().cpu().numpy())
                #     print(f"S_tile_q ({scale_spec}):")
                #     print(S_tile_q.detach().cpu().numpy())
                #     print("C_tile_scaled (bf16):")
                #     print(C_tile_scaled.detach().cpu().numpy())
                #     print("C_out before add (bf16 grid):")
                #     print(q_bf16_rne(C_prev).detach().cpu().numpy())
                #     print("C_out after add (bf16 grid):")
                #     print(C_out.detach().cpu().numpy())

                #     debug["tiles"].append({
                #         "m0": m0, "n0": n0, "k0": k0, "g": k0//group,
                #         "A_tile": A_tile.detach().cpu(),
                #         "B_tile": B_tile.detach().cpu(),
                #         "C_tile": C_tile.detach().cpu(),
                #         "S_tile_q": S_tile_q.detach().cpu(),
                #         "C_tile_scaled": C_tile_scaled.detach().cpu(),
                #         "C_out": C_out.detach().cpu(),
                #     })
            C_out[m0:m0+TM, n0:n0+TN] = C_out_tile

    return C_out, debug


def array_mx_requantize(array: Tensor, scale_spec: str, quant_spec: str) -> Tuple[Tensor, float]:
    e_bits, m_bits = parse_fp_spec(quant_spec)
    max_val = array.abs().to(torch.bfloat16).view(torch.int16).max()
    max_exp = max_val.log2().floor() # We'll ignore the edge case where all elements are zero, it probably won't happen. Change the seed if it does
    exp = max_exp - (e_bits + m_bits + 1)
    scale_factor = make_fp_quantizer(scale_spec)(torch.pow(2.0, exp)).item()
    return array / scale_factor, scale_factor


def matrix_mx_requantize(matrix: Tensor, scale_spec: str, quant_spec: str, group_size: int, Gk: int) -> Tuple[Tensor, Tensor]:
    device = matrix.device
    M, N = matrix.shape
    matrix_scale_groups = matrix.view(M * N // group_size, group_size).detach().cpu()
    q_groups = []
    q_scales = []
    for row in matrix_scale_groups:
        group, scale = array_mx_requantize(row, scale_spec, quant_spec)
        q_groups.append(group)
        q_scales.append(scale)
    return torch.stack(q_groups).to(device=device).view(M, N), torch.tensor(q_scales, dtype=torch.float32, device=device).view(len(q_scales) // Gk, Gk)


def reverse_lut_quantize(matrix: Tensor, lut: Tensor) -> Tensor:
    diff = (matrix.unsqueeze(-1) - lut).abs()
    return diff.argmin(dim=-1)


def matmul_loss(C_ref: Tensor, C_quant: Tensor) -> Dict[str, float]:
    diff = (C_quant - C_ref).detach()
    mse = float(torch.mean(diff.pow(2)).item())
    mae = float(torch.mean(diff.abs()).item())
    max_abs = float(torch.max(diff.abs()).item())
    fro_ref = float(torch.linalg.norm(C_ref).item())
    fro_diff = float(torch.linalg.norm(diff).item())
    rel_fro = (fro_diff / (fro_ref + 1e-12)) if fro_ref != 0.0 else float("inf")
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_fro": rel_fro,
    }

# ----------------------------------------------------------------------
# Header file writer
# ----------------------------------------------------------------------

def write_c_header(
    path: str,
    M: int,
    K: int,
    N: int,
    input_spec: str,
    acc_spec: str,
    scale_spec: str,
    scaled_spec: str,
    A_codes: List[List[int]],
    A_bits: int,
    B_codes: List[List[int]],
    B_bits: int,
    C_codes: List[List[int]],
    C_bits: int,
    S_codes: List[List[int]],
    S_bits: int,
    C_scaled_codes: List[List[int]],
    C_scaled_bits: int,
):
    # Derive a crude include guard from file name
    guard = path.upper()
    for ch in [".", "/", "\\", "-"]:
        guard = guard.replace(ch, "_")

    # Force storage types
    in_type_A = "uint16_t"
    in_type_B = "uint32_t"

    # Zero-extend element codes into those storage widths
    A_store_bits = 16
    B_store_bits = 32
    A_codes_store = zext_codes(A_codes, A_store_bits)
    B_codes_store = zext_codes(B_codes, B_store_bits)

    A_hex = codes_to_hex_rows(A_codes_store, A_store_bits)
    B_hex = codes_to_hex_rows(B_codes_store, B_store_bits)

    out_type  = c_type_for_bits(C_bits)
    scale_type = c_type_for_bits(S_bits)
    scaled_type = c_type_for_bits(C_scaled_bits)

    C_hex = codes_to_hex_rows(C_codes, C_bits)
    S_hex = codes_to_hex_rows(S_codes, S_bits)
    C_scaled_hex = codes_to_hex_rows(C_scaled_codes, C_scaled_bits)

    def format_2d_array(hex_rows: List[List[str]]) -> str:
        lines = []
        for row in hex_rows:
            line = ", ".join(f"0x{h}" for h in row)
            lines.append("    { " + line + " }")
        return ",\n".join(lines)

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M {M}\n")
        f.write(f"#define MATMUL_K {K}\n")
        f.write(f"#define MATMUL_N {N}\n\n")

        f.write(f"// Input precision: {input_spec}\n")
        f.write(f"static const {in_type_A} A_in[MATMUL_M][MATMUL_K] = {{\n")
        f.write(format_2d_array(A_hex))
        f.write("\n};\n\n")

        f.write(f"static const {in_type_B} B_in[MATMUL_K][MATMUL_N] = {{\n")
        f.write(format_2d_array(B_hex))
        f.write("\n};\n\n")

        f.write(f"// Output precision (pre-scale): {acc_spec}\n")
        f.write(f"static const {out_type} C_out[MATMUL_M][MATMUL_N] = {{\n")
        f.write(format_2d_array(C_hex))
        f.write("\n};\n\n")

        f.write(f"// Scaling factors (elementwise) in {scale_spec}\n")
        f.write(f"static const {scale_type} C_scale[MATMUL_M][MATMUL_N] = {{\n")
        f.write(format_2d_array(S_hex))
        f.write("\n};\n\n")

        f.write(f"// Scaled outputs, quantized to {scaled_spec}\n")
        f.write(f"static const {scaled_type} C_scaled[MATMUL_M][MATMUL_N] = {{\n")
        f.write(format_2d_array(C_scaled_hex))
        f.write("\n};\n\n")

        f.write(f"#endif // {guard}\n")

def write_c_header_tiled(
    path: str,
    M: int, K: int, N: int,
    group: int,
    input_spec: str,
    acc_spec: str,
    scale_spec: str,
    lut_index_bits: int,
    A_in: Tensor,
    B_in: Tensor,
    A_scales_row_q: Tensor,   # [M, Gk]
    B_scales_col_q: Tensor,   # [Gk, N]
    C_out_bf16: Tensor,       # [M, N] (bf16 grid stored as fp32)
    C_out_quantized: Tensor,
    C_out_scales: Tensor,
    A_lut: Tensor | None = None,
    B_lut: Tensor | None = None,
    C_lut: Tensor | None = None,
):
    # encode A/B in input_spec, scales in scale_spec, output in bf16
    if lut_index_bits >= 0:
        A_codes = A_in.tolist()
        B_codes = B_in.tolist()
        Cq_codes = C_out_quantized.tolist()
        A_bits = lut_index_bits
        B_bits = lut_index_bits
        Cq_bits = lut_index_bits
        A_lut_codes, A_lut_bits = tensor_to_custom_fp_codes(A_lut.unsqueeze(-1), input_spec)
        B_lut_codes, B_lut_bits = tensor_to_custom_fp_codes(B_lut.unsqueeze(-1), input_spec)
        C_lut_codes, C_lut_bits = tensor_to_custom_fp_codes(C_lut.unsqueeze(-1), input_spec)
        A_lut_hex = codes_to_hex_rows(A_lut_codes, A_lut_bits)
        B_lut_hex = codes_to_hex_rows(B_lut_codes, B_lut_bits)
        C_lut_hex = codes_to_hex_rows(C_lut_codes, C_lut_bits)
    else:
        A_codes, A_bits = tensor_to_custom_fp_codes(A_in, input_spec)
        B_codes, B_bits = tensor_to_custom_fp_codes(B_in, input_spec)
        Cq_codes, Cq_bits = tensor_to_custom_fp_codes(C_out_quantized, input_spec)
        
    As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), scale_spec)
    Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, scale_spec)
    C_codes, C_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")
    Cqs_codes, Cqs_bits = tensor_to_custom_fp_codes(C_out_scales.transpose(0, 1), scale_spec)
    # Scale factors are unsigned
    As_bits -= 1
    Bs_bits -= 1
    Cqs_bits -= 1

    guard = path.upper()
    for ch in [".", "/", "\\", "-"]:
        guard = guard.replace(ch, "_")

    if A_bits <= 4:
        A_codes = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in A_codes]
    if B_bits <= 4:
        B_codes = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in B_codes]
    if Cq_bits <= 4:
        Cq_codes = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in Cq_codes]
    
    A_hex = codes_to_hex_rows(A_codes, A_bits)
    B_hex = codes_to_hex_rows(B_codes, B_bits)
    As_hex = codes_to_hex_rows(As_codes, As_bits)
    Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)
    C_hex  = codes_to_hex_rows(C_codes, C_bits)
    Cq_hex  = codes_to_hex_rows(Cq_codes, Cq_bits)
    Cqs_hex  = codes_to_hex_rows(Cqs_codes, Cqs_bits)

    def format_2d_array(hex_rows: List[List[str]]) -> str:
        lines = []
        for row in hex_rows:
            line = ", ".join(f"0x{h}" for h in row)
            lines.append("    { " + line + " }")
        return ",\n".join(lines)

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M {M}\n#define MATMUL_K {K}\n#define MATMUL_N {N}\n")
        f.write(f"#define MATMUL_GK {K // group}\n")
        f.write(f"#define MATMUL_GN {N // group}\n\n")

        f.write(f"// Input precision: {input_spec}")
        if lut_index_bits >= 0:
            f.write(f" ({lut_index_bits}-bit int LUT indices)")
        f.write("\n")
        f.write(f"static const {c_type_for_bits(A_bits)} A_in[MATMUL_M][MATMUL_K{' / 2' if A_bits <= 4 else ''}] = {{\n{format_2d_array(A_hex)}\n}};\n\n")
        f.write(f"static const {c_type_for_bits(B_bits)} B_in[MATMUL_K][MATMUL_N{' / 2' if B_bits <= 4 else ''}] = {{\n{format_2d_array(B_hex)}\n}};\n\n")

        if lut_index_bits >= 0:
            f.write("// Lookup tables\n")
            f.write(f"static const {c_type_for_bits(A_lut_bits)} A_lut[{1 << lut_index_bits}] = {{\n    {(', '.join([f'0x{e[0]}' for e in A_lut_hex]))}\n}};\n\n")
            f.write(f"static const {c_type_for_bits(B_lut_bits)} B_lut[{1 << lut_index_bits}] = {{\n    {(', '.join([f'0x{e[0]}' for e in B_lut_hex]))}\n}};\n\n")
            f.write(f"static const {c_type_for_bits(C_lut_bits)} C_lut[{1 << lut_index_bits}] = {{\n    {(', '.join([f'0x{e[0]}' for e in C_lut_hex]))}\n}};\n\n")

        f.write(f"// Per-row per-32-K-group scales in {scale_spec}\n")
        f.write(f"static const {c_type_for_bits(As_bits)} A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{format_2d_array(As_hex)}\n}};\n\n")

        f.write(f"// Per-col per-32-K-group scales in {scale_spec}\n")
        f.write(f"static const {c_type_for_bits(Bs_bits)} B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{format_2d_array(Bs_hex)}\n}};\n\n")

        f.write(f"// Final output requantized to {input_spec}\n")
        f.write(f"static const {c_type_for_bits(Cq_bits)} C_out[MATMUL_M][MATMUL_N{' / 2' if Cq_bits <= 4 else ''}] = {{\n{format_2d_array(Cq_hex)}\n}};\n\n")

        f.write(f"// Per-row per-32-K-group output scales in {scale_spec}\n")
        f.write(f"static const {c_type_for_bits(Cqs_bits)} C_scales_row[MATMUL_GN][MATMUL_M] = {{\n{format_2d_array(Cqs_hex)}\n}};\n\n")

        f.write("// Final output (already scaled+accumulated), bf16\n")
        f.write(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{format_2d_array(C_hex)}\n}};\n\n")
        f.write(f"#endif // {guard}\n")

# ----------------------------------------------------------------------
# Single experiment: generate A,B, quantize, run matmul, print everything
# ----------------------------------------------------------------------

def run_experiment(
    M: int,
    K: int,
    N: int,
    input_spec: str,
    prod_spec: str,
    acc_spec: str,
    scaled_spec: Optional[str] = None,          # ignored (kept for signature compatibility)
    input_rounding: str = "nearest",
    prod_rounding: str = "nearest",
    acc_rounding: str = "q_bf16_rne",           # you want bf16 grid behavior
    scale_spec: str = "fpe8m0",
    scale_exp: int = 0,                         # optional global multiplier (applied to scale vectors)
    header_path: str = "matmul_data.h",
    seed: int = 0,
    device: str = "cpu",
    *,
    tile: int = 16,
    group: int = 32,
    trace_tiles: Optional[List[Tuple[int,int,int]]] = None,
    trace_max: int = 4,
    print_inputs_fp32: bool = False,
    print_inputs_quant: bool = True,
    print_hex_inputs: bool = True,
    lut_index_bits: int = -1,
    log_file: str = "/dev/null",
):
    """
    Tiled-only golden model:

    - Inputs: A,B generated FP32 -> quantized to input_spec (fp8)
    - Scales:
        A_scales_row_q: [M, Gk]   per-row per-32-K-group (fpE8M0)
        B_scales_col_q: [Gk, N]   per-col per-32-K-group (fpE8M0)
      Tile scale matrix S_tile = outer(A_scales_row_q[:,g], B_scales_col_q[g,:])
    - For each (m0,n0,k0) tile:
        C_tile_bf16 = matmul_outer_quantized_hwlike(A_tile,B_tile,prod_quant, bf16 each add)
        S_tile_q    = outer-product scales (quantized to scale_spec)
        C_tile_scaled_bf16 = bf16(C_tile_bf16 * S_tile_q)
        C_out = bf16(C_out + C_tile_scaled_bf16)
    - Writes one header using write_c_header_tiled().

    Notes:
    - Requires M,N,K multiples of 16 (tile).
    - group=32 means tiles at k0=0 and k0=16 share the same scale group.
    """

    # ------------------------------------------------------------------
    # setup / inputs
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    assert M % tile == 0 and N % tile == 0 and K % tile == 0, \
        f"Require M,N,K multiples of tile={tile} for now."
    assert group % tile == 0, f"Require group ({group}) multiple of tile ({tile})."

    A = torch.randn(M, K, device=dev, dtype=torch.float32)
    B = torch.randn(K, N, device=dev, dtype=torch.float32)

    use_lut = lut_index_bits >= 0
    if use_lut:
        A_lut = make_lut(input_spec, lut_index_bits, dev)
        B_lut = make_lut(input_spec, lut_index_bits, dev)
        C_lut = make_lut(input_spec, lut_index_bits, dev)
        A_indices = quantize_lut_indices(A_lut, A)
        B_indices = quantize_lut_indices(B_lut, B)

    print("=== Configuration (TILED MODE) ===")
    print(f"M={M}, K={K}, N={N}, tile={tile}, group={group}")
    print(f"input_spec={input_spec}, input_rounding={input_rounding}")
    print(f"prod_spec={prod_spec}, prod_rounding={prod_rounding}")
    print(f"acc_spec={acc_spec}, acc_rounding={acc_rounding}")
    print(f"scale_spec={scale_spec}, scale_exp(global_mult)={scale_exp}")
    print(f"header_path={header_path}")
    print(f"device={dev}, seed={seed}")
    print(f"lut_index_bits={lut_index_bits}")
    print()

    if print_inputs_fp32:
        print("=== Full-precision Inputs (FP32) ===")
        print("A_fp32:")
        print(A.detach().cpu().numpy())
        print("\nB_fp32:")
        print(B.detach().cpu().numpy())
        print()

    # ------------------------------------------------------------------
    # quantizers
    # ------------------------------------------------------------------
    in_q = make_fp_quantizer(input_spec, rounding=input_rounding)

    # Product quantization:
    # - If you are using --prod-mant-bits, keep the global 'args' usage.
    #   Otherwise use prod_spec quantizer.
    if "args" in globals() and getattr(args, "prod_mant_bits", None) is not None:
        prod_q = lambda t: trunc_product_mantissa(t, frac_bits=args.prod_mant_bits)
        print(f"Product quantization: mantissa chopped to {args.prod_mant_bits} fraction bits (RTZ/chop)")
    else:
        prod_q = make_fp_quantizer(prod_spec, rounding=prod_rounding)

    # Accumulator quantization: you want BF16 grid after each add
    if acc_rounding == "q_bf16_rne":
        acc_q = q_bf16_rne
    else:
        # if you ever want a different acc_spec model
        acc_q = make_fp_quantizer(acc_spec, rounding=acc_rounding)

    # Quantize inputs to storage/compute format
    if use_lut:
        A_in = lut_lookup(A_lut, A_indices)
        B_in = lut_lookup(B_lut, B_indices)
    else:
        A_in = in_q(A) if in_q is not None else A
        B_in = in_q(B) if in_q is not None else B

    # if print_inputs_quant:
    #     print("=== Inputs quantized to input precision ===")
    #     print(f"A_in (float, {input_spec}, rounding={input_rounding}):")
    #     print(A_in.detach().cpu().numpy())
    #     print()
    #     print(f"B_in (float, {input_spec}, rounding={input_rounding}):")
    #     print(B_in.detach().cpu().numpy())
    #     print()

    # Optional hex dump of input encodings
    A_codes = B_codes = None
    A_bits = B_bits = None
    if print_hex_inputs:
        try:
            A_codes, A_bits = tensor_to_custom_fp_codes(A_in, input_spec)
            B_codes, B_bits = tensor_to_custom_fp_codes(B_in, input_spec)
            A_hex = codes_to_hex_rows(A_codes, A_bits)
            B_hex = codes_to_hex_rows(B_codes, B_bits)

            print(f"=== A_in hex encoding ({input_spec}) ===")
            for row in A_hex[:min(8, len(A_hex))]:
                print(" ".join(row))
            if len(A_hex) > 8:
                print("... (truncated)")
            print()

            print(f"=== B_in hex encoding ({input_spec}) ===")
            for row in B_hex[:min(8, len(B_hex))]:
                print(" ".join(row))
            if len(B_hex) > 8:
                print("... (truncated)")
            print()
        except ValueError as e:
            print(f"[WARN] Could not hex-encode inputs for {input_spec}: {e}")
            print()

    # ------------------------------------------------------------------
    # build scale vectors (placeholder; swap with real extraction later)
    # ------------------------------------------------------------------
    Gk = (K + group - 1) // group

    # deterministic but different from A/B
    torch.manual_seed(seed + 123)

    # Make power-of-two exponents for fpE8M0 friendliness
    A_scale_exp = torch.randint(low=-4, high=4, size=(M, Gk), device=dev)
    B_scale_exp = torch.randint(low=-4, high=4, size=(Gk, N), device=dev)

    A_scales_row = torch.pow(2.0, A_scale_exp.to(torch.float32))
    B_scales_col = torch.pow(2.0, B_scale_exp.to(torch.float32))

    # Optional global multiplier (also power-of-two)
    if scale_exp != 0:
        global_mult = float(2.0 ** scale_exp)
        A_scales_row = A_scales_row * global_mult
        # (or distribute across A/B however you want; this keeps behavior simple)

    # Quantize to scale_spec (fpe8m0)
    A_scales_row_q = make_fp_quantizer(scale_spec, "nearest")(A_scales_row)
    B_scales_col_q = make_fp_quantizer(scale_spec, "nearest")(B_scales_col)

    print("=== A_scales_row_q (per-row per-32-K-group) ===")
    print(A_scales_row_q.detach().cpu().numpy())
    print()
    print("=== B_scales_col_q (per-col per-32-K-group) ===")
    print(B_scales_col_q.detach().cpu().numpy())
    print()

    # ------------------------------------------------------------------
    # references (optional but useful): full FP32 and input-quantized FP32
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    C_ref_full = matmul_outer(A, B)
    t1 = time.perf_counter()
    # print("=== Reference: FP32 matmul, FP32 inputs ===")
    # print(f"(time: {t1 - t0:.6f} s)")
    # print()

    t2 = time.perf_counter()
    C_ref_in = matmul_outer(A_in, B_in)
    t3 = time.perf_counter()
    # print(f"=== Reference: FP32 matmul, input-quantized inputs ({input_spec}) ===")
    # print(f"(time: {t3 - t2:.6f} s)")
    # print()

    # ------------------------------------------------------------------
    # tiled quantized matmul + tile scale + bf16 accumulation
    # ------------------------------------------------------------------
    t4 = time.perf_counter()
    C_out_bf16, debug = tiled_matmul_scaled_accum_hwlike(
        A_in, B_in,
        prod_quant=prod_q,
        A_scales_row=A_scales_row_q,
        B_scales_col=B_scales_col_q,
        tile=tile,
        group=group,
        log_file=log_file,
        scale_spec=scale_spec,
        trace_tiles=trace_tiles,
        trace_max=trace_max,
    )
    C_out_quantized, C_out_scales = matrix_mx_requantize(
        C_out_bf16,
        scale_spec=scale_spec,
        quant_spec=input_spec,
        group_size=group,
        Gk=Gk,
    )
    if use_lut:
        C_out_quantized = reverse_lut_quantize(
            C_out_quantized,
            lut=C_lut
        )
    t5 = time.perf_counter()

    print("=== Output: TILED quantized, scaled per-tile, accumulated in bf16 ===")
    print("C_out_bf16:")
    print(C_out_bf16.detach().cpu().numpy())
    print(f"(time: {t5 - t4:.6f} s)")
    print()

    # ------------------------------------------------------------------
    # error metrics (against useful references)
    # ------------------------------------------------------------------
    metrics_vs_full = matmul_loss(C_ref_full, C_out_bf16)
    metrics_vs_in   = matmul_loss(C_ref_in,   C_out_bf16)

    # print("=== Error metrics: C_out_bf16 vs C_ref_full (FP32 inputs) ===")
    # for k, v in metrics_vs_full.items():
    #     print(f"{k}: {v:.6e}")
    # print()

    # print(f"=== Error metrics: C_out_bf16 vs C_ref_in (input-quantized, ideal MAC) ===")
    # for k, v in metrics_vs_in.items():
    #     print(f"{k}: {v:.6e}")
    # print()

    # ------------------------------------------------------------------
    # output header (single writer; do NOT call the old write_c_header)
    # ------------------------------------------------------------------
    try:
        write_c_header_tiled(
            path=header_path,
            M=M, K=K, N=N,
            group=group,
            input_spec=input_spec,
            acc_spec=acc_spec,
            scale_spec=scale_spec,
            lut_index_bits=lut_index_bits,
            A_in=A_indices if use_lut else A_in,
            B_in=B_indices if use_lut else B_in,
            A_scales_row_q=A_scales_row_q,
            B_scales_col_q=B_scales_col_q,
            C_out_bf16=C_out_bf16,
            C_out_quantized=C_out_quantized,
            C_out_scales=C_out_scales,
            A_lut=A_lut if use_lut else None,
            B_lut=B_lut if use_lut else None,
            C_lut=C_lut if use_lut else None,
        )
        print(f"Header written to: {header_path}")
    except Exception as e:
        print(f"[WARN] Failed to write tiled header: {e}")

def trace_dot(
    A_in: Tensor,
    B_in: Tensor,
    i: int,
    j: int,
    *,
    prod_quant: QuantFn = None,
    acc_quant: QuantFn = None,
    cast_each_add: bool = True,   # if True: acc_quant after every add; else only at end
    cast_prod: bool = True,       # if True: apply prod_quant to each product
    verbose: bool = True,
) -> Tensor:
    """
    Trace C[i,j] = sum_k A[i,k]*B[k,j] with optional quant at product and/or accumulator.

    Returns final scalar tensor.
    """
    K = A_in.shape[1]
    acc = torch.zeros((), dtype=torch.float32, device=A_in.device)

    if verbose:
        print(f"Tracing C[{i},{j}] over K={K}")
        print(f"  cast_prod={cast_prod}, cast_each_add={cast_each_add}")
        print(f"  prod_quant={'yes' if prod_quant else 'no'}, acc_quant={'yes' if acc_quant else 'no'}")
        print()

    for k in range(K):
        a = A_in[i, k]
        b = B_in[k, j]
        p = a * b

        p_q = prod_quant(p) if (cast_prod and prod_quant) else p
        acc_next = acc + p_q
        acc_q = acc_quant(acc_next) if (cast_each_add and acc_quant) else acc_next

        if verbose:
            # .item() is fine for scalars
            print(
                f"k={k:2d}  a={a.item(): .10g}  b={b.item(): .10g}  "
                f"p={p.item(): .10g}  p_q={p_q.item(): .10g}  "
                f"acc-> {acc_next.item(): .10g}  acc_q={acc_q.item(): .10g}"
            )

        acc = acc_q

    if (not cast_each_add) and acc_quant:
        acc = acc_quant(acc)
        if verbose:
            print("\nfinal acc quant:", acc.item())

    return acc

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="fp8:e4m3", help="Input storage precision spec")
    parser.add_argument("--prod",  type=str, default="fp8:e4m3", help="Product precision spec")
    parser.add_argument("--acc",   type=str, default="bf16",     help="Accumulator precision spec")

    parser.add_argument(
        "--scaled-spec",
        type=str,
        default=None,
        help="Post-scale FP8 spec (defaults to --prod if not set)."
    )

    parser.add_argument(
        "--input-rounding",
        type=str,
        default="zero",
        choices=["nearest", "nearest_even", "stochastic", "zero", "toward_zero", "trunc", "rtz"],
        help="Rounding mode for input quantization."
    )
    parser.add_argument(
        "--prod-rounding",
        type=str,
        default="zero",
        choices=["nearest", "nearest_even", "stochastic", "zero", "toward_zero", "trunc", "rtz"],
        help="Rounding mode for product."
    )
    parser.add_argument(
        "--acc-rounding",
        type=str,
        default="nearest",
        choices=["nearest", "nearest_even", "stochastic", "zero", "toward_zero", "trunc", "rtz", "q_bf16_rne"],
        help="Rounding mode for accumulator."
    )

    parser.add_argument(
        "--scale-spec",
        type=str,
        default="fpe8m0",
        help="Scaling factor FP spec (e.g., fpe8m0 for fpE8M0)."
    )
    parser.add_argument(
        "--scale-exp",
        type=int,
        default=0,
        help="Exponent for scaling factor: scale = 2**scale_exp."
    )

    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--header-path",
        type=str,
        default="matmul_data.h",
        help="Output C header file path."
    )

    parser.add_argument(
        "--prod-mant-bits",
        type=int,
        default=None,
        help="If set, truncate product mantissa to this many fraction bits (RTZ)."
    )

    parser.add_argument(
        "--lut-index-bits",
        type=int,
        default=-1,
        help="If set and not equal to -1, use inputs quantized to n-bit integers and generate lookup tables to convert to floating point."
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=16,
        help="Matrix multiplication tile size."
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="/dev/null",
        help="Log of intermediate tile values."
    )

    args = parser.parse_args()

    run_experiment(
        M=args.M,
        K=args.K,
        N=args.N,
        input_spec=args.input,
        prod_spec=args.prod,
        acc_spec=args.acc,
        scaled_spec=args.scaled_spec,
        input_rounding=args.input_rounding,
        prod_rounding=args.prod_rounding,
        acc_rounding=args.acc_rounding,
        scale_spec=args.scale_spec,
        scale_exp=args.scale_exp,
        header_path=args.header_path,
        seed=args.seed,
        device=args.device,
        tile=args.tile,
        lut_index_bits=args.lut_index_bits,
        log_file=args.log_file,
    )

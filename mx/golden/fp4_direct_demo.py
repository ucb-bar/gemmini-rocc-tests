#!/usr/bin/env python3
"""
FP4 E2M1 Direct Matmul Demo
============================
Same flow as lut_mapping_demo.py but A_in and B_in are direct FP4 E2M1
values (1s + 2e + 1m = 4 bits total), used directly for computation
without any LUT projection step.

FP4 E2M1 format ("fp4:e4m1" in golden_model.py, exp=2, man=1, bias=1):
  - 1 sign bit
  - 2 exponent bits (bias=1)
  - 1 mantissa bit
  Total: 4 bits, 16 distinct values

Packing convention (same as LUT demo):
  A: HW tiled layout - per (A_TILE_M x K_TILE) block, pairs of m-rows
     interleaved per k, then nibble-packed (bits[7:4]=row 2r+1, bits[3:0]=row 2r)
  B: standard layout - bits[7:4]=odd col, bits[3:0]=even col
"""

import sys
import struct
import random
import numpy as np
import torch

sys.path.insert(0, ".")
from fp_decoder import ieee_to_recfn
import fp8_matmul_model
from fp8_matmul_model import (
    matmul_outer_quantized_hwlike,
    compute_tile_scale_matrix,
    tiled_matmul_hwlike,
    matrix_mx_requantize,
    tensor_to_custom_fp_codes_fp4,
)
from golden_model import (
    tensor_to_custom_fp_codes,
    codes_to_hex_rows,
    make_fp_quantizer,
    q_bf16_rne,
    hw_add_bf16,
    parse_fp_spec,
    compute_tile_scale_matrix_fpe8m0,
    _a_indices_to_hw_layout,
)


def bf16_tensor_to_recfn_hex(t: torch.Tensor, hw_zero: int = 0x01600) -> list:
    """Convert float32 tensor (on BF16 grid) to HardFloat recFN hex strings (17-bit, 5 hex digits).

    Any zero value (positive or negative BF16 zero) is encoded as hw_zero (default 0x01600),
    matching the hardware's non-canonical zero representation where top3 exp bits = 000
    but lower exponent bits may be non-zero.
    """
    flat = t.to(torch.bfloat16).reshape(-1)
    rows = []
    for val in flat:
        buf = val.view(torch.int16).item() & 0xFFFF
        if (buf & 0x7FFF) == 0:
            # zero (positive or negative): use hardware's zero encoding
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
    A_codes: torch.Tensor,      # [M, K]  4-bit FP4 raw codes (int)
    B_codes: torch.Tensor,      # [K, N]  4-bit FP4 raw codes (int)
    A_scales_row_q: torch.Tensor,   # [M, Gk]
    B_scales_col_q: torch.Tensor,   # [Gk, N]
    C_out_bf16: torch.Tensor,       # [M, N]
    C_out_quantized: torch.Tensor | None = None,  # [M, N] 4-bit FP4 codes
    C_out_scales: torch.Tensor | None = None,     # [M, Gk_out]
    a_tile_m: int = 32,
    k_tile: int = 16,
):
    """
    Write a C header for the direct FP4 matmul (no LUT).

    A layout: HW tiled - per (a_tile_m x k_tile) block, pairs of m-rows interleaved
              per k, then nibble-packed → [M/2][K], bits[7:4]=row 2r+1, bits[3:0]=row 2r
    B layout: standard nibble packing → [K][N/2], bits[7:4]=odd col, bits[3:0]=even col
    """
    assert M % a_tile_m == 0 and K % k_tile == 0, \
        f"M ({M}) must be divisible by a_tile_m ({a_tile_m}), K ({K}) by k_tile ({k_tile})"

    # ── A: HW tiled layout (same function as LUT demo, works for any 4-bit codes) ──
    A_hw_codes = _a_indices_to_hw_layout(A_codes, a_tile_m, k_tile)   # [M//2, K] list-of-lists
    A_hw_hex   = codes_to_hex_rows(A_hw_codes, 8)

    # ── B: standard nibble packing (odd col in high nibble, even col in low nibble) ──
    B_list = B_codes.tolist()
    B_packed = [[(B_list[k][n + 1] << 4) | B_list[k][n] for n in range(0, N, 2)] for k in range(K)]
    B_hex = codes_to_hex_rows(B_packed, 8)

    # ── Scales ────────────────────────────────────────────────────────────────────
    As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), scale_spec)
    Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, scale_spec)
    As_bits -= 1   # scale factors are unsigned (e8m0 has no sign bit)
    Bs_bits -= 1
    As_hex = codes_to_hex_rows(As_codes, As_bits)
    Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)

    # ── C output ─────────────────────────────────────────────────────────────────
    C_codes, C_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")
    C_hex = codes_to_hex_rows(C_codes, C_bits)

    if C_out_quantized is not None:
        Cq_packed = _a_indices_to_hw_layout(C_out_quantized, a_tile_m, k_tile)  # [M//2, N]
        Cq_hex = codes_to_hex_rows(Cq_packed, 8)
    
    if C_out_quantized is not None:
        Cq_hex_full = codes_to_hex_rows(C_out_quantized, 8)

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

    Gk = K // group
    M_hw = M // 2

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M   {M}\n")
        f.write(f"#define MATMUL_K   {K}\n")
        f.write(f"#define MATMUL_N   {N}\n")
        f.write(f"#define MATMUL_GK  {Gk}\n")
        f.write(f"#define MATMUL_GN  {N // group}\n")
        f.write(f"#define A_TILE_M   {a_tile_m}\n")
        f.write(f"#define K_TILE     {k_tile}\n\n")

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
            f.write(f"// HW tiled layout (same as A): per ({a_tile_m} x {k_tile}) block,\n")
            f.write(f"//   pairs of m-rows interleaved per n, then nibble-packed.\n")
            f.write(f"//   Dimensions: [M/2][N] = [{M // 2}][{N}]\n")
            f.write(f"//   byte layout: bits[7:4]=fp4(row 2r+1, n), bits[3:0]=fp4(row 2r, n)\n")
            f.write(f"static const uint8_t C_out[{M // 2}][{N}] = {{\n{fmt2d(Cq_hex)}\n}};\n\n")

        if C_out_scales is not None:
            Gk_out = C_out_scales.shape[1]
            f.write(f"// Per-row per-{group}-N-group C output scales in {scale_spec}\n")
            f.write(f"static const uint8_t C_scales_row[{Gk_out}][{M}] = {{\n{fmt2d(Cqs_hex)}\n}};\n\n")

        f.write("// Final output (scaled+accumulated), bf16\n")
        f.write(f"static const uint16_t C_out_bf16[{M}][{N}] = {{\n{fmt2d(C_hex)}\n}};\n\n")

        f.write(f"#endif // {guard}\n")


# ── Parameters ────────────────────────────────────────────────────────────────
SEED           = 0
M, K, N        = 128, 128, 128
INPUT_SPEC     = "fp4:e2m1"     # FP4 E2M1: 1 sign + 2 exp (bias=1) + 1 mant = 4 bits
SCALE_SPEC     = "fpe8m0"
GROUP          = 32
TILE           = 16
Gk             = K // GROUP
DEBUG_R        = 8
A_TILE_M       = 32
K_TILE         = 16
B_TILE_N       = 32
DEV            = torch.device("cpu")
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
A = torch.randn(M, K, device=DEV, dtype=torch.float32)
B = torch.randn(K, N, device=DEV, dtype=torch.float32)


print("[Step 1]: Quantize A and B directly to FP4 E2M1")
#fp4_quantizer = make_fp_quantizer(INPUT_SPEC, "nearest")
in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
A_quant = in_q(A)   # [M, K] float32, values snapped to FP4 E2M1 grid
B_quant = in_q(B)   # [K, N] float32, values snapped to FP4 E2M1 grid

# Get raw 4-bit codes for A and B
A_codes_raw, A_code_bits = tensor_to_custom_fp_codes_fp4(A_quant)   # list[list[int]], 4 bits
B_codes_raw, B_code_bits = tensor_to_custom_fp_codes_fp4(B_quant)   # list[list[int]], 4 bits
A_codes = torch.tensor(A_codes_raw, dtype=torch.int32)  # [M, K]
B_codes = torch.tensor(B_codes_raw, dtype=torch.int32)  # [K, N]

print(f"  A_quant[0,:4] float : {A_quant[0, :4].tolist()}")
print(f"  A_codes[0,:4] hex   : {[hex(A_codes[0, j].item()) for j in range(4)]}")
print(f"  B_quant[0,:4] float : {B_quant[0, :4].tolist()}")
print(f"  B_codes[0,:4] hex   : {[hex(B_codes[0, j].item()) for j in range(4)]}")


print("\n[Step 2]: Generate E8M0 scales")
A_scale_exp = torch.randint(low=-4, high=4, size=(M, Gk), device=DEV)
B_scale_exp = torch.randint(low=-4, high=4, size=(Gk, N), device=DEV)
A_scales_row = torch.pow(2.0, A_scale_exp.to(torch.float32))
B_scales_col = torch.pow(2.0, B_scale_exp.to(torch.float32))

A_scales_row_q = make_fp_quantizer(SCALE_SPEC, "nearest")(A_scales_row)  # [M, Gk]
B_scales_col_q = make_fp_quantizer(SCALE_SPEC, "nearest")(B_scales_col)  # [Gk, N]


print("\n[Step 3]: Build full scaled matrices and compute C_out HW-like")
k_group_idx   = torch.arange(K, device=DEV) // GROUP   # (K,)
A_fp_scaled   = A_quant * A_scales_row_q[:, k_group_idx]   # [M, K]
B_fp_scaled   = B_quant * B_scales_col_q[k_group_idx, :]   # [K, N]

fp8_matmul_model.prod_quant = make_fp_quantizer(INPUT_SPEC, "nearest")
C_out = matmul_outer_quantized_hwlike(A_fp_scaled, B_fp_scaled)


print("\n[Step 4]: Tile-by-tile trace (first tile only for brevity)")
for k_base in range(0, K, K_TILE):
    k_group = k_base // GROUP
    for n_base in range(0, N, B_TILE_N):
        # ── B tile: get FP4 codes and float values ────────────────────────────
        b_codes_tile = B_codes[k_base:k_base+K_TILE, n_base:n_base+B_TILE_N]   # [K_TILE, B_TILE_N]
        b_packed     = (b_codes_tile[:, 1::2] << 4) | b_codes_tile[:, 0::2]    # [K_TILE, B_TILE_N//2]
        B_hex_tile   = codes_to_hex_rows(b_packed.tolist(), 8)
        B_in_tile    = B_quant[k_base:k_base+K_TILE, n_base:n_base+B_TILE_N]   # [K_TILE, B_TILE_N]
        b_scale      = B_scales_col_q[k_group, n_base:n_base+B_TILE_N]         # [B_TILE_N]
        B_in_scaled  = B_in_tile * b_scale.unsqueeze(0)                         # [K_TILE, B_TILE_N]

        for m_base in range(0, M, A_TILE_M):
            # ── A tile: get FP4 codes and float values ────────────────────────
            a_codes_tile = A_codes[m_base:m_base+A_TILE_M, k_base:k_base+K_TILE]  # [A_TILE_M, K_TILE]
            # HW layout: interleave m-row pairs per k, then nibble-pack
            a_hw = a_codes_tile.reshape(A_TILE_M // 2, 2, K_TILE).permute(0, 2, 1).reshape(A_TILE_M // 2, K_TILE * 2)
            a_packed = (a_hw[:, 1::2] << 4) | a_hw[:, 0::2]                    # [A_TILE_M//2, K_TILE]
            A_hex_tile = codes_to_hex_rows(a_packed.tolist(), 8)

            A_in_tile    = A_quant[m_base:m_base+A_TILE_M, k_base:k_base+K_TILE]  # [A_TILE_M, K_TILE]
            a_scale      = A_scales_row_q[m_base:m_base+A_TILE_M, k_group]        # [A_TILE_M]
            A_in_scaled  = A_in_tile * a_scale.unsqueeze(1)                        # [A_TILE_M, K_TILE]

            if m_base == 0 and n_base == 0 and k_base == 0:
                # ── Per-element debug for first tile ─────────────────────────
                r = 0
                print(f"\n--- A a_packed[{r}] nibble breakdown (HW layout, k_base=0) ---")
                print(f"{'k':>3}  {'byte':>4}  {'bits[7:4]=row{2*r+1}':>22}  {'bits[3:0]=row{2*r}':>20}")
                for k_pos in range(K_TILE):
                    msb_code = a_codes_tile[2*r+1, k_pos].item()
                    lsb_code = a_codes_tile[2*r,   k_pos].item()
                    byte_hex = A_hex_tile[r][k_pos]
                    print(f"{k_pos:>3}  {byte_hex:>4}  row{2*r+1},k{k_pos}:0x{msb_code:x}  row{2*r},k{k_pos}:0x{lsb_code:x}")

                print(f"\n--- A_in_tile float values (first 2 rows, first {DEBUG_R} k) ---")
                for row in range(2):
                    vals = [f"{A_in_tile[row, kp].item():>8.4f}" for kp in range(DEBUG_R)]
                    codes_hex = [f"0x{a_codes_tile[row, kp].item():x}" for kp in range(DEBUG_R)]
                    print(f"  row {row}: float={vals}  fp4={codes_hex}")

                sA = A_scales_row_q[m_base:m_base+A_TILE_M, k_group]
                sB = B_scales_col_q[k_group, n_base:n_base+B_TILE_N]
                sA_codes, sA_bits = tensor_to_custom_fp_codes(sA[:DEBUG_R].unsqueeze(1), SCALE_SPEC)
                sB_codes, sB_bits = tensor_to_custom_fp_codes(sB[:DEBUG_R].unsqueeze(1), SCALE_SPEC)
                sA_hex = codes_to_hex_rows(sA_codes, sA_bits)
                sB_hex = codes_to_hex_rows(sB_codes, sB_bits)
                print(f"\n--- A scales[:{DEBUG_R}] (float | {SCALE_SPEC} hex) ---")
                print([f"{v.item():.4f} ({h[0]})" for v, h in zip(sA[:DEBUG_R], sA_hex)])
                print(f"\n--- B scales[:{DEBUG_R}] (float | {SCALE_SPEC} hex) ---")
                print([f"{v.item():.4f} ({h[0]})" for v, h in zip(sB[:DEBUG_R], sB_hex)])

                S_joint = compute_tile_scale_matrix_fpe8m0(
                    A_scales_row_q, B_scales_col_q,
                    m0=m_base, n0=n_base, k0=k_base,
                    TM=A_TILE_M, TN=B_TILE_N,
                    M=M, N=N, K=K, group=GROUP, scale_spec=SCALE_SPEC,
                )
                S_codes, S_bits = tensor_to_custom_fp_codes(S_joint[:DEBUG_R, :DEBUG_R], SCALE_SPEC)
                S_hex = codes_to_hex_rows(S_codes, S_bits)
                print(f"\n--- Joint scale S_tile[:{DEBUG_R},:{DEBUG_R}] ({SCALE_SPEC} hex | float) ---")
                for r in range(DEBUG_R):
                    float_row = [f"{S_joint[r, c].item():>8.4f}" for c in range(DEBUG_R)]
                    print(f"  row{r}: {S_hex[r]}  |  {float_row}")

                # Hardware-faithful inner-product trace for k=0
                C_before_hw = None
                for k_pos in range(K_TILE):
                    prod = A_in_tile[:, k_pos:k_pos+1] * B_in_tile[k_pos:k_pos+1, :]
                    if C_before_hw is None:
                        C_before_hw = q_bf16_rne(prod)
                    else:
                        C_before_hw = hw_add_bf16(prod, C_before_hw)

                    if k_pos == 0:
                        a_in_k = A_in_tile[:DEBUG_R, 0]
                        b_in_k = B_in_tile[0, :DEBUG_R]
                        a_codes_k, a_bits_k = tensor_to_custom_fp_codes_fp4(a_in_k.unsqueeze(1))
                        b_codes_k, b_bits_k = tensor_to_custom_fp_codes_fp4(b_in_k.unsqueeze(1))
                        c_codes_k, c_bits_k = tensor_to_custom_fp_codes(C_before_hw[:DEBUG_R, :DEBUG_R], "bf16")
                        a_hex = [r[0] for r in codes_to_hex_rows(a_codes_k, a_bits_k)]
                        b_hex = [r[0] for r in codes_to_hex_rows(b_codes_k, b_bits_k)]
                        c_hex = codes_to_hex_rows(c_codes_k, c_bits_k)
                        print(f"\n--- k=0 ---")
                        print(f"  A_in[:{DEBUG_R}], fp4 hex={a_hex}")
                        print(f"  B_in[:{DEBUG_R}], fp4 hex={b_hex}")
                        print(f"  C[:{DEBUG_R},:{DEBUG_R}], bf16:")
                        for row in c_hex:
                            print(f"    {row}")
                        c_rec = bf16_tensor_to_recfn_hex(C_before_hw[:DEBUG_R, :DEBUG_R])
                        print(f"  C[:{DEBUG_R},:{DEBUG_R}] recfn:")
                        for row in c_rec:
                            print(f"    {row}")

                # Per-K_TILE cumulative accumulation trace
                _pq_all = make_fp_quantizer(INPUT_SPEC, "zero")
                C_cumulative = None
                for t, kb in enumerate(range(0, K, K_TILE)):
                    c_tile = None
                    for kp in range(kb, kb + K_TILE):
                        prod = A_quant[:, kp:kp+1] * B_quant[kp:kp+1, :]
                        if c_tile is None:
                            c_tile = q_bf16_rne(prod)
                        else:
                            c_tile = hw_add_bf16(prod, c_tile)
                    c_tile_r = c_tile[:DEBUG_R, :DEBUG_R]
                    St = compute_tile_scale_matrix(
                        A_scales_row_q, B_scales_col_q,
                        m0=m_base, n0=n_base, k0=kb, TM=DEBUG_R, TN=DEBUG_R
                    )
                    c_tile_scaled = q_bf16_rne(c_tile_r * St)
                    C_cumulative = c_tile_scaled if C_cumulative is None else hw_add_bf16(C_cumulative, c_tile_scaled)
                    tile_codes,   tile_bits   = tensor_to_custom_fp_codes(c_tile_r,      "bf16")
                    scaled_codes, scaled_bits = tensor_to_custom_fp_codes(c_tile_scaled, "bf16")
                    cum_codes,    cum_bits    = tensor_to_custom_fp_codes(C_cumulative,  "bf16")
                    print(f"\n=== K_TILE {t+1} (k={kb}..{kb+K_TILE-1}) ===")
                    print(f"  tile (no scale):")
                    for row in codes_to_hex_rows(tile_codes, tile_bits):
                        print(f"    {row}")
                    print(f"  tile (scaled):")
                    for row in codes_to_hex_rows(scaled_codes, scaled_bits):
                        print(f"    {row}")
                    print(f"  cumulative tiles 1..{t+1}:")
                    for row in codes_to_hex_rows(cum_codes, cum_bits):
                        print(f"    {row}")


print("\n[Step 5]: Compare C_out with tiled_matmul_hwlike golden")
fp8_matmul_model.prod_quant = lambda x: x   # no product re-quantization — matches hardware
C_golden = tiled_matmul_hwlike(
    A_quant, B_quant,
    A_scales_row_q, B_scales_col_q,
    verbose=False,
)
C_out_bf16   = q_bf16_rne(C_out)
C_golden_bf16 = q_bf16_rne(C_golden)
C_golden_r = C_golden_bf16[:DEBUG_R, :DEBUG_R]
C_golden_r_codes, C_golden_r_bits = tensor_to_custom_fp_codes(C_golden_r, "bf16")
print(f"  C_golden[:{DEBUG_R},:{DEBUG_R}], bf16:")
for row in codes_to_hex_rows(C_golden_r_codes, C_golden_r_bits):
    print(f"    {row}")
C_golden_r_rec = bf16_tensor_to_recfn_hex(C_golden_r)
print(f"  C_golden[:{DEBUG_R},:{DEBUG_R}] recfn (ieee_to_recfn, 17-bit BF16):")
for row in C_golden_r_rec:
    print(f"    {row}")


# print("\n[Step 6]: Requantize C_golden_bf16 to FP4 E2M1 via matrix_mx_requantize")
Gk_out = N // GROUP
C_requantized, C_req_scales = matrix_mx_requantize(
        C_golden_bf16,
        quant_spec=INPUT_SPEC)
print(f"  quant_spec: {INPUT_SPEC}  scale_spec: {SCALE_SPEC}")
Cq_r = C_requantized[:DEBUG_R, :DEBUG_R]
print(Cq_r)
Cq_r_codes, Cq_r_bits = tensor_to_custom_fp_codes_fp4(Cq_r)
print(f"  C_requantized[:{DEBUG_R},:{DEBUG_R}], fp4:")
for row in codes_to_hex_rows(Cq_r_codes, Cq_r_bits):
    print(f"    {row}")
scale_codes, scale_bits = tensor_to_custom_fp_codes(C_req_scales, SCALE_SPEC)

# Per-row group breakdown (first 2 rows)
from fp8_matmul_model import quantize_single_bf16_ieee
print(f"\n--- C per-row group breakdown (group_size={GROUP}, showing first 2 rows) ---")
for m in range(min(2, M)):
    print(f"  row {m:3d}:")
    bf16_row   = C_golden_bf16[m]
    for g in range(1):
        col_start     = g * GROUP
        col_end       = col_start + GROUP
        golden_group  = bf16_row[col_start:col_end]
        quant_group   = C_requantized[m, col_start:col_end]
        scale_val     = C_req_scales[m, g].item()
        scale_e8m0_hex = f"{scale_codes[m][g]:02x}"
        # print(col_start, col_end)
        # print(golden_group)
        # print(scale_val)
        # print(scale_e8m0_hex)
        # exit()
        #quantized_data = quantize_single_bf16_ieee(quant_group[4], scale_e8m0_hex)
        print(quant_group.unsqueeze(0)) 
        abs_group     = golden_group.abs()
        max_mag_val   = abs_group.max().item()
        max_mag_bf16  = abs_group.max().to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
        q_codes, q_bits = tensor_to_custom_fp_codes_fp4(quant_group.unsqueeze(0))
        q_hex = codes_to_hex_rows(q_codes, q_bits)[0]
        golden_bits  = golden_group.to(torch.bfloat16).view(torch.int16)
        golden_hex   = " ".join(f"{b.item() & 0xFFFF:04x}" for b in golden_bits)
        print(f"    group {g} (cols {col_start:3d}-{col_end-1:3d})\n"
              f"    max_magnitude={max_mag_val:.6g} (bf16=0x{max_mag_bf16:04x})\n"
              f"    scale={scale_val:.6g} (e8m0=0x{scale_e8m0_hex})\n"
              f"    golden_bf16=[{golden_hex}]\n"
              f"    quantized={q_hex}")
    
# Get FP4 codes for quantized C (for header)
C_req_codes_raw, _ = tensor_to_custom_fp_codes_fp4(C_requantized.float().view(M, N))
C_req_codes = torch.tensor(C_req_codes_raw, dtype=torch.int32)  # [M, N]


# print("\n[Step 7]: Pack to HW layout and print flat FP4 codes")
# print("\n--- A_codes flat [M, K] (4-bit fp4) ---")
# for m in range(M):
#     flat_hex = " ".join(f"{A_codes[m, k].item():x}" for k in range(K))
#     print(f"  row {m:3d}: {flat_hex}")

# print("\n--- B_codes flat [K, N] (4-bit fp4) ---")
# for k in range(K):
#     flat_hex = " ".join(f"{B_codes[k, n].item():x}" for n in range(N))
#     print(f"  row {k:3d}: {flat_hex}")

# print("\n--- A_in HW layout [M//2, K] (bits[3:0]=even row, bits[7:4]=odd row per col) ---")
# A_hw_layout = _a_indices_to_hw_layout(A_codes, a_tile_m=A_TILE_M, k_tile=K_TILE)  # [M//2, K]
# for r in range(M // 2):
#     hw_hex = " ".join(f"{A_hw_layout[r][k]:02x}" for k in range(K))
#     print(f"  hw row {r:3d}: {hw_hex}")

# print("\n--- C_req flat [M, N] (4-bit fp4 indices) ---")
# for m in range(M):
#     flat_hex = " ".join(f"{C_req_codes[m, n].item():x}" for n in range(N))
#     print(f"  row {m:3d}: {flat_hex}")


print("\n[Step 7]: Write C header (direct FP4, no LUT)")
C_out_bf16_final = q_bf16_rne(C_golden)
write_c_header_fp4_direct(
    path           = "./include/matmul_data_fp4_direct.h",
    M=M, K=K, N=N,
    group          = GROUP,
    input_spec     = INPUT_SPEC,
    scale_spec     = SCALE_SPEC,
    A_codes        = A_codes,
    B_codes        = B_codes,
    A_scales_row_q = A_scales_row_q,
    B_scales_col_q = B_scales_col_q,
    C_out_bf16     = C_out_bf16_final,
    C_out_quantized= C_req_codes,
    C_out_scales   = C_req_scales,
    a_tile_m       = A_TILE_M,
    k_tile         = K_TILE,
)
print("Header written to ./include/matmul_data_fp4_direct.h")

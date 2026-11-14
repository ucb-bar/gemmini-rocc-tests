#!/usr/bin/env python3
"""Unified generator for the ASYMMETRIC MX matmul test-data headers.

Merges the ~30 gen_asym_<act>_<wei>.py generators into ONE code path + a per-combo table.
All asym combos are non-requant: output is BF16 (C_out_bf16), golden via
fp8_matmul_model.tiled_matmul_hwlike. This generator adds DIM-awareness: --dim 16 reproduces
the committed headers byte-identical; --dim 32 keeps operands identical and recomputes the golden
with precision_for_dim(32) (emits ..._dim32.h).

Usage:
    ./gen_asym.py --act <spec> --wei <spec> [--dim 16|32] [--M 64 --K 64 --N 64 --seed 0]

The math (encoders/quantizers/deproject/mesh model) is imported verbatim from lut_golden_model.py
and fp8_matmul_model.py -- nothing is re-implemented here. The only per-combo variation is:
  * A/B operand path: fp4 direct (HW-tiled or nibble), fp8:e4m3 single direct, or LUT-deprojected
  * LUT entry/load bit-widths and which LUTs are real vs placeholder
  * header comment layout (reproduced byte-for-byte, including a couple of stale copy-paste labels)
"""
import argparse
import os
import sys

# --- precision_for_dim: replicated from gen_matmul_llama.py:66-74 / gen_mx_lut.py -------------
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1


def precision_for_dim(dim: int):
    prod = [(4, 3)] * dim
    if dim == 8:
        acc = [(8, 7)] * 8   # DIM=8: ALL rows bf16 (design assumption -- no reduced-precision ramp)
    elif dim <= 16:
        acc = ACC_PRECISION[:dim]
    else:
        acc = ACC_PRECISION + [(8, 7)] * (dim - 16)
    return prod, acc


# --- Combo registry ---------------------------------------------------------------------------
# Keyed by (act_key, wei_key) where key = "fp4" for any fp4 format (spec-string irrelevant to the
# emitted bytes), else the exact spec. Value: (basename, l2_template, overrides). l2_template
# selects the L2 codebook-comment style (only e3m2_e5m2 uses "Y"; every other dual-LUT combo uses
# "X"). basename is stored explicitly because the committed filenames are inconsistent (e3m2 is
# tagged "fp6" in fp4_fp6 / fp6_fp4 but "e3m2" everywhere else). overrides reproduces stale
# copy-paste comment labels in a couple of committed headers (e.g. e5m2_e4m3s' A-comment says
# FP6_E3M2 though the activation is actually FP8_E5M2).
COMBOS = {
    ("fp6:e2m3", "fp6:e3m2"): ("e2m3_e3m2", "X", {}),
    ("fp6:e2m3", "fp8:e4m3"): ("e2m3_e4m3", "X", {}),
    ("fp6:e2m3", "fp8:e5m2"): ("e2m3_e5m2", "X", {}),
    ("fp6:e2m3", "fp8:e4m3s"):("e2m3_e4m3s", None, {}),
    ("fp6:e2m3", "fp4"):      ("e2m3_fp4",  None, {}),
    ("fp6:e3m2", "fp6:e2m3"): ("e3m2_e2m3", "X", {}),
    ("fp6:e3m2", "fp8:e4m3"): ("e3m2_e4m3", "X", {}),
    ("fp6:e3m2", "fp8:e4m3s"):("e3m2_e4m3s", None, {}),
    ("fp6:e3m2", "fp8:e5m2"): ("e3m2_e5m2", "Y", {}),
    ("fp8:e4m3", "fp6:e2m3"): ("e4m3_e2m3", "X", {}),
    ("fp8:e4m3", "fp6:e3m2"): ("e4m3_e3m2", "X", {}),
    ("fp8:e4m3", "fp8:e5m2"): ("e4m3_e5m2", "X", {}),
    ("fp8:e4m3", "fp8:e4m3s"):("e4m3_e4m3s", None, {}),
    ("fp8:e4m3", "fp4"):      ("e4m3_fp4",  None, {}),
    ("fp8:e4m3s", "fp6:e2m3"):("e4m3s_e2m3", None, {}),
    ("fp8:e4m3s", "fp8:e4m3"):("e4m3s_e4m3", None, {}),
    ("fp8:e4m3s", "fp6:e3m2"):("e4m3s_e3m2", None, {}),
    ("fp8:e4m3s", "fp8:e5m2"):("e4m3s_e5m2", None, {}),
    ("fp8:e4m3s", "fp4"):     ("e4m3s_fp4", None, {}),
    ("fp8:e5m2", "fp6:e2m3"): ("e5m2_e2m3", "X", {}),
    ("fp8:e5m2", "fp6:e3m2"): ("e5m2_e3m2", "X", {}),
    ("fp8:e5m2", "fp8:e4m3"): ("e5m2_e4m3", "X", {}),
    ("fp8:e5m2", "fp8:e4m3s"):("e5m2_e4m3s", None, {"a_in_label": "FP6_E3M2"}),
    ("fp8:e5m2", "fp4"):      ("e5m2_fp4",  None, {}),
    ("fp4", "fp6:e2m3"):      ("fp4_e2m3", None, {}),
    ("fp4", "fp8:e4m3"):      ("fp4_e4m3", None, {}),
    ("fp4", "fp8:e4m3s"):     ("fp4_e4m3s", None, {}),
    ("fp4", "fp8:e5m2"):      ("fp4_e5m2", None, {}),
    ("fp4", "fp6:e3m2"):      ("fp4_fp6", None, {}),
    ("fp6:e3m2", "fp4"):      ("fp6_fp4", None, {}),
}


def combo_key(spec: str) -> str:
    return "fp4" if spec.startswith("fp4") else spec


# --- Operand descriptor -----------------------------------------------------------------------
def classify(spec: str, parse_fp_spec):
    """Return a descriptor dict for an operand spec string."""
    prefix, sub = spec.split(":")
    if prefix == "fp4":
        return dict(kind="fp4", label="FP4 E2M1")
    if sub.endswith("s"):                       # e.g. "e4m3s" -> single-throughput direct fp8
        base = sub[:-1]
        return dict(kind="e4m3_single", label=f"{prefix.upper()}_{base.upper()}",
                    subname=base.upper(), golden_spec=f"{prefix}:{base}", native_bits=8)
    native = 1 + sum(parse_fp_spec(spec))
    return dict(kind="lut", label=f"{prefix.upper()}_{sub.upper()}",
                subname=sub.upper(), golden_spec=spec, native_bits=native)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", required=True, help="Activation format spec (e.g. fp4:e4m1, fp6:e3m2, "
                                                 "fp8:e5m2, fp8:e4m3, fp8:e4m3s).")
    ap.add_argument("--wei", required=True, help="Weight format spec (e.g. fp4:e2m1, fp6:e2m3, "
                                                 "fp8:e4m3, fp8:e4m3s).")
    ap.add_argument("--dim", type=int, default=16, choices=(8, 16, 32),
                    help="Mesh DIM (per-tile accumulation depth). 16 reproduces committed headers.")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requant", choices=("fp4",), default=None,
                    help="Also emit the requantized output (C_out HW-tiled codes + C_scales_out) in this format.")
    ap.add_argument("--lut-g", type=int, default=1,
                    help="Codebook granularity: one LUT per 2^G rows (act) / cols (wei). 1 reproduces committed headers.")
    ap.add_argument("--bf16-arith", action="store_true",
                    help="Full BF16 PE products and adds (MxE4M3LutBf16Dim* configs); header suffix _bf16.")
    ap.add_argument("--header-path", type=str, default=None,
                    help="Output path (default ./include/matmul_data_asym_<combo>[_dim32].h).")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if here not in sys.path:
        sys.path.insert(0, here)

    import torch
    from lut_golden_model import (
        make_lut, quantize_lut_indices, make_fp_quantizer,
        tensor_to_custom_fp_codes, codes_to_hex_rows, pack_lut_hw_words,
        _a_indices_to_hw_layout, parse_fp_spec,
    )
    import fp8_matmul_model
    from fp8_matmul_model import tiled_matmul_hwlike

    a = classify(args.act, parse_fp_spec)
    b = classify(args.wei, parse_fp_spec)
    key = (combo_key(args.act), combo_key(args.wei))
    if key not in COMBOS:
        ap.error(f"unknown combo {key}; supported:\n  " +
                 "\n  ".join(f"{k[0]} x {k[1]} -> {v[0]}" for k, v in COMBOS.items()))
    basename, l2_template, overrides = COMBOS[key]

    M, K, N = args.M, args.K, args.N
    SEED = args.seed
    SCALE_SPEC = "fpe8m0"
    LUT_INDEX_BITS = 4
    GROUP = 32
    G = args.lut_g
    A_TILE_M = 32
    K_TILE = 16
    DEV = torch.device("cpu")
    Gk = K // GROUP

    dim = args.dim
    dim_suffix = "" if dim == 16 else f"_dim{dim}"
    if args.lut_g != 1:
        dim_suffix += f"_g{args.lut_g}"
    if args.bf16_arith:
        dim_suffix += "_bf16"
    # Shape tag: empty for the default 64x64 (preserves committed names/guards), else _<M>x<N>x<K>.
    shape_tag = "" if (M == 64 and N == 64 and K == 64) else f"_{M}x{N}x{K}"
    rq_tag = f"_rq{args.requant}" if args.requant else ""
    default_path = f"./include/matmul_data_asym_{basename}{shape_tag}{rq_tag}{dim_suffix}.h"
    HEADER = args.header_path if args.header_path is not None else default_path

    # ---- Inputs (identical for every combo at a given seed/shape) --------------------------
    torch.manual_seed(SEED)
    A = torch.randn(M, K, device=DEV, dtype=torch.float32)
    B = torch.randn(K, N, device=DEV, dtype=torch.float32)

    # ---- FP grids (exact spike decode) ----------------------------------------------------
    def _fp4_grid():
        def dec(code):
            s = (code >> 3) & 1
            e = (code >> 1) & 0x3
            m = code & 0x1
            v = (m / 2.0) if e == 0 else (1.0 + m / 2.0) * (2.0 ** (e - 1))
            return -v if s else v
        return torch.tensor([dec(c) for c in range(16)], dtype=torch.float32)

    def _e4m3_grid():
        def dec(code):
            if (code & 0x7F) == 0x7F:
                return float("nan")
            s = (code >> 7) & 1
            e = (code >> 3) & 0xF
            m = code & 0x7
            bias = 7
            val = (m / 8.0) * (2.0 ** (1 - bias)) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - bias))
            return -val if s else val
        codes = [c for c in range(256) if (c & 0x7F) != 0x7F]
        grid = torch.tensor([dec(c) for c in codes], dtype=torch.float32)
        return codes, grid

    # ---- Encode A (must run BEFORE B: make_lut consumes torch RNG in original order) -------
    A_luts_t = None
    A_in_hw = A_in_row = None
    if a["kind"] == "fp4":
        grid = _fp4_grid()
        code_t = (A.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.int64)
        A_fp = grid[code_t]
        A_in_hw = _a_indices_to_hw_layout(code_t.tolist(), A_TILE_M, K_TILE)
    elif a["kind"] == "e4m3_single":
        codes, grid = _e4m3_grid()
        idx = (A.unsqueeze(-1) - grid).abs().argmin(dim=-1)
        A_fp = grid[idx]
        A_in_row = [[codes[j] for j in row] for row in idx.tolist()]
    else:  # lut
        spec = a["golden_spec"]
        A_luts = [make_lut(spec, LUT_INDEX_BITS, DEV) for _ in range(M >> G)]
        A_luts_t = torch.stack(A_luts)
        A_indices = torch.stack([quantize_lut_indices(A_luts[i >> G], A[i, :]) for i in range(M)])
        mi = (torch.arange(M, device=DEV) >> G).unsqueeze(1).expand(-1, K)
        A_fp = A_luts_t[mi, A_indices]
        A_in_hw = _a_indices_to_hw_layout(A_indices.tolist(), A_TILE_M, K_TILE)

    # ---- Encode B -------------------------------------------------------------------------
    B_luts_t = None
    B_nibble = B_in_row = None
    if b["kind"] == "fp4":
        grid = _fp4_grid()
        code_t = (B.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.int64)
        B_fp = grid[code_t]
        rows = code_t.tolist()
        B_nibble = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in rows]
    elif b["kind"] == "e4m3_single":
        codes, grid = _e4m3_grid()
        idx = (B.unsqueeze(-1) - grid).abs().argmin(dim=-1)
        B_fp = grid[idx]
        B_in_row = [[codes[j] for j in row] for row in idx.tolist()]
    else:  # lut
        spec = b["golden_spec"]
        B_luts = [make_lut(spec, LUT_INDEX_BITS, DEV) for _ in range(N >> G)]
        B_luts_t = torch.stack(B_luts)
        B_indices = torch.stack(
            [quantize_lut_indices(B_luts[j >> G], B[:, j]) for j in range(N)], dim=1)
        ni = (torch.arange(N, device=DEV) >> G).unsqueeze(0).expand(K, -1)
        B_fp = B_luts_t[ni, B_indices]
        rows = B_indices.tolist()
        B_nibble = [[((r[i + 1] << 4) | r[i]) for i in range(0, len(r), 2)] for r in rows]

    # ---- Scales (power-of-two, fpe8m0) ----------------------------------------------------
    torch.manual_seed(SEED + 123)
    A_scale_exp = torch.randint(-4, 4, (M, Gk), device=DEV)
    B_scale_exp = torch.randint(-4, 4, (Gk, N), device=DEV)
    sq = make_fp_quantizer(SCALE_SPEC, "nearest")
    A_scales_row_q = sq(torch.pow(2.0, A_scale_exp.to(torch.float32)))
    B_scales_col_q = sq(torch.pow(2.0, B_scale_exp.to(torch.float32)))

    # ---- Golden (DIM-aware) ---------------------------------------------------------------
    fp8_matmul_model.INPUT_SPEC = b.get("golden_spec", a.get("golden_spec", args.act))
    fp8_matmul_model.TILE = dim
    for attr in ("TILE_K", "TILE_M", "TILE_N"):
        if hasattr(fp8_matmul_model, attr):
            setattr(fp8_matmul_model, attr,
                    {"TILE_K": dim, "TILE_M": 2 * dim, "TILE_N": 2 * dim}[attr])
    prod_prec, acc_prec = precision_for_dim(dim)
    if args.bf16_arith:
        prod_prec, acc_prec = [(8, 7)] * dim, [(8, 7)] * dim
    C_out_bf16 = tiled_matmul_hwlike(
        A_fp, B_fp, A_scales_row_q, B_scales_col_q,
        verbose=False, prod_precision_list=prod_prec, acc_precision_list=acc_prec,
    )

    # ---- Encode scale + golden arrays -----------------------------------------------------
    As_codes, As_bits = tensor_to_custom_fp_codes(A_scales_row_q.transpose(0, 1), SCALE_SPEC)
    Bs_codes, Bs_bits = tensor_to_custom_fp_codes(B_scales_col_q, SCALE_SPEC)
    As_bits -= 1
    Bs_bits -= 1
    As_hex = codes_to_hex_rows(As_codes, As_bits)
    Bs_hex = codes_to_hex_rows(Bs_codes, Bs_bits)
    C_codes, C_bits = tensor_to_custom_fp_codes(C_out_bf16, "bf16")
    C_hex = codes_to_hex_rows(C_codes, C_bits)

    # ---- Formatters -----------------------------------------------------------------------
    def fmt2d_u8(rows):
        return ",\n".join("    { " + ", ".join(f"0x{v:02x}" for v in row) + " }" for row in rows)

    def fmt2d_hex(hex_rows):
        return ",\n".join("    { " + ", ".join(f"0x{h}" for h in row) + " }" for row in hex_rows)

    def fmt_lut_packed(lut_codes, entry_bits):
        return ",\n".join(
            "    { " + ", ".join(f"0x{x:08x}" for x in pack_lut_hw_words(grp, entry_bits)) + " }"
            for grp in lut_codes)

    guard = HEADER.upper()
    for ch in [".", "/", "\\", "-"]:
        guard = guard.replace(ch, "_")

    a_hw_tiled = a["kind"] in ("fp4", "lut")
    dual_thru = (a["kind"] == "e4m3_single") or (b["kind"] == "e4m3_single")

    out = []
    out.append(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
    out.append(f"#define MATMUL_M   {M}\n#define MATMUL_K   {K}\n#define MATMUL_N   {N}\n")
    if G != 1:
        out.append(f"#define MATMUL_LUT_G {G}\n")
    if a_hw_tiled:
        out.append(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n")
        out.append(f"#define A_TILE_M   {A_TILE_M}\n#define K_TILE     {K_TILE}\n\n")
    else:
        out.append(f"#define MATMUL_GK  {Gk}\n#define MATMUL_GN  {N // GROUP}\n\n")

    # ---- A section ----
    if a["kind"] == "e4m3_single":
        out.append("// A (activation) = FP8_E4M3 DIRECT 8-bit codes, row-major [M][K] "
                   "(single throughput, no LUT).\n")
        out.append(f"static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{\n{fmt2d_u8(A_in_row)}\n}};\n\n")
    else:
        if a["kind"] == "fp4":
            head = "FP4 E2M1 DIRECT 4-bit codes" if not dual_thru else "FP4 E2M1 4-bit codes"
        else:
            head = f"{overrides.get('a_in_label', a['label'])} 4-bit LUT indices"
        if dual_thru:
            extra = " (direct, quad 2 rows/lane)" if a["kind"] == "fp4" else " (quad 2 rows/lane)"
            out.append(f"// A (activation) = {head}, HW-tiled [M/2][K]{extra}.\n")
        else:
            out.append(f"// A (activation) = {head}, HW-tiled [M/2][K].\n")
            out.append("//   byte layout: bits[7:4]=a(2r+1,k), bits[3:0]=a(2r,k) for hw-row r\n")
        out.append(f"static const uint8_t A_in_hw[{M // 2}][{K}] = {{\n{fmt2d_u8(A_in_hw)}\n}};\n\n")

    # ---- B section ----
    if b["kind"] == "e4m3_single":
        out.append("// B (weight) = FP8_E4M3 DIRECT 8-bit codes, row-major [K][N] "
                   "(single, 1 col/lane, no LUT).\n")
        out.append(f"static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{\n{fmt2d_u8(B_in_row)}\n}};\n\n")
    else:
        if b["kind"] == "fp4":
            bhead = "FP4 E2M1 DIRECT 4-bit codes"
        else:
            bhead = f"{b['label']} 4-bit LUT indices"
        out.append(f"// B (weight) = {bhead}, nibble-packed [K][N/2] (odd col high nibble)\n")
        out.append(f"static const uint8_t B_in[MATMUL_K][MATMUL_N / 2] = {{\n{fmt2d_u8(B_nibble)}\n}};\n\n")

    # ---- LUT section ----
    def zeroext_phrase(native, slot):
        return (f"{native}-bit codes in {slot}-bit slots" if native == slot
                else f"{native}-bit codes zero-extended to {slot}-bit slots")

    if a["kind"] == "lut" and b["kind"] == "lut":
        # L2 dual-LUT: both loaded at the wider native width.
        load_bits = max(a["native_bits"], b["native_bits"])
        words = (16 * load_bits + 31) // 32
        n_a, n_b = M >> G, N >> G
        A_lut_codes, _ = tensor_to_custom_fp_codes(A_luts_t, a["golden_spec"])
        B_lut_codes, _ = tensor_to_custom_fp_codes(B_luts_t, b["golden_spec"])
        A_packed = fmt_lut_packed(A_lut_codes, load_bits)
        B_packed = fmt_lut_packed(B_lut_codes, load_bits)
        if l2_template == "Y":
            out.append(f"// A_lut = E3M2 codebook, 6-bit codes zero-extended to {load_bits}-bit slots "
                       f"(RTL slices low 6) -> {words}x uint32. Loaded (sel=1).\n")
            out.append(f"static const uint32_t A_lut[{n_a}][{words}] = {{\n{A_packed}\n}};\n\n")
            out.append(f"// B_lut = E5M2 codebook (16x{load_bits}-bit -> {words}x uint32). Loaded (sel=0).\n")
            out.append(f"static const uint32_t B_lut[{n_b}][{words}] = {{\n{B_packed}\n}};\n\n")
        else:  # Template X (all other dual-LUT combos)
            out.append(f"// A_lut = E5M2 codebook (16x{load_bits}-bit -> {words}x uint32). Loaded (sel=1).\n")
            out.append(f"static const uint32_t A_lut[{n_a}][{words}] = {{\n{A_packed}\n}};\n\n")
            out.append(f"// B_lut = E3M2 codebook, 6-bit codes zero-extended to {load_bits}-bit slots "
                       f"(RTL slices low 6) -> {words}x uint32. Loaded (sel=0).\n")
            out.append(f"static const uint32_t B_lut[{n_b}][{words}] = {{\n{B_packed}\n}};\n\n")
        out.append("// C_lut placeholder (output is bf16, never read); 8-bit like A_lut.\n")
        out.append(f"static const uint32_t C_lut[{n_a}][{words}] = {{\n{A_packed}\n}};\n\n")

    elif a["kind"] == "e4m3_single" and b["kind"] == "lut":
        # L3a: B_lut only, loaded at 8-bit.
        slot = 8
        words = (16 * slot + 31) // 32
        n_b = N >> G
        B_lut_codes, _ = tensor_to_custom_fp_codes(B_luts_t, b["golden_spec"])
        B_packed = fmt_lut_packed(B_lut_codes, slot)
        out.append(f"// B_lut = {b['subname']} codebook, {zeroext_phrase(b['native_bits'], slot)} -> "
                   f"{words}x uint32. Loaded (sel=0). No A_lut/C_lut (act direct, output bf16).\n")
        out.append(f"static const uint32_t B_lut[{n_b}][{words}] = {{\n{B_packed}\n}};\n\n")

    elif a["kind"] == "lut" and b["kind"] == "e4m3_single":
        # L4: A_lut only, loaded at 8-bit.
        slot = 8
        words = (16 * slot + 31) // 32
        n_a = M >> G
        A_lut_codes, _ = tensor_to_custom_fp_codes(A_luts_t, a["golden_spec"])
        A_packed = fmt_lut_packed(A_lut_codes, slot)
        out.append(f"// A_lut = {a['subname']} codebook, {zeroext_phrase(a['native_bits'], slot)} -> "
                   f"{words}x uint32. Loaded (sel=1). No B_lut/C_lut (weight direct, output bf16).\n")
        out.append(f"static const uint32_t A_lut[{n_a}][{words}] = {{\n{A_packed}\n}};\n\n")

    elif a["kind"] == "lut" or b["kind"] == "lut":
        # L1: single real LUT (native width), other two are placeholders reusing the real one.
        if b["kind"] == "lut":                              # L1a: A fp4 direct, B lut real
            entry = b["native_bits"]
            words = (16 * entry + 31) // 32
            n_lut = N >> G
            codes, _ = tensor_to_custom_fp_codes(B_luts_t, b["golden_spec"])
            real = fmt_lut_packed(codes, entry)
            A_body, B_body, C_body = real, real, real  # A/C placeholders = real B
            which, deproj = "B_lut", "weight deproject"
            plc = "A_lut/C_lut"
        else:                                               # L1b: A lut real, B fp4 direct
            entry = a["native_bits"]
            words = (16 * entry + 31) // 32
            n_lut = M >> G
            codes, _ = tensor_to_custom_fp_codes(A_luts_t, a["golden_spec"])
            real = fmt_lut_packed(codes, entry)
            A_body, B_body, C_body = real, real, real
            which, deproj = "A_lut", "activation deproject"
            plc = "B_lut/C_lut"
        out.append(f"// LUTs: 16x{entry}-bit entries -> {words}x uint32 per group. Only {which} is used\n")
        out.append(f"// ({deproj}); {plc} are placeholders so both build paths compile.\n")
        out.append(f"static const uint32_t A_lut[{n_lut}][{words}] = {{\n{A_body}\n}};\n\n")
        out.append(f"static const uint32_t B_lut[{n_lut}][{words}] = {{\n{B_body}\n}};\n\n")
        out.append(f"static const uint32_t C_lut[{n_lut}][{words}] = {{\n{C_body}\n}};\n\n")
    # else: no LUT section (L5 fp4 x e4m3_single, or e4m3_single x fp4)

    # ---- Scales + golden ----
    out.append(f"// Per-row per-{GROUP}-K-group activation scales in {SCALE_SPEC}\n")
    out.append(f"static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{\n{fmt2d_hex(As_hex)}\n}};\n\n")
    out.append(f"// Per-col per-{GROUP}-K-group weight scales in {SCALE_SPEC}\n")
    out.append(f"static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{\n{fmt2d_hex(Bs_hex)}\n}};\n\n")
    out.append("// Golden output (scaled + bf16-accumulated), bf16\n")
    out.append(f"static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{\n{fmt2d_hex(C_hex)}\n}};\n\n")
    if args.requant:
        # the FP4 requantizer model the fp4 requant goldens use (gen_matmul_llama._requant), on the BF16 golden
        import numpy as np
        from gen_matmul_llama import _requant, FORMATS
        rf = FORMATS[args.requant]
        rq_codes, rq_scales, _ = _requant(C_out_bf16.numpy().astype(np.float32), rf)
        rq_hw = _a_indices_to_hw_layout(rq_codes.tolist(), rf.tile_m, rf.tile_k)
        out.append(f"// Requant output: {rf.name} codes, HW-tiled [M/2][N] (bits[3:0] = row 2r, bits[7:4] = row 2r+1)\n")
        out.append(f"static const uint8_t C_out[MATMUL_M / 2][MATMUL_N] = {{\n{fmt2d_u8(rq_hw)}\n}};\n\n")
        out.append("// Requant output E8M0 block scales [M][N/32]\n")
        out.append(f"static const uint8_t C_scales_out[MATMUL_M][MATMUL_N / 32] = {{\n{fmt2d_u8(rq_scales.tolist())}\n}};\n\n")
    out.append(f"#endif // {guard}\n")

    with open(HEADER, "w") as f:
        f.write("".join(out))
    print(f"Wrote {HEADER}  (combo={basename}, M={M} K={K} N={N}, Gk={Gk}, dim={dim})")


if __name__ == "__main__":
    main()

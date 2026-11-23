#!/usr/bin/env python3
"""Unified MX-LUT 64x64 test-data generator for the 4 symmetric LUT formats.

Formats: e4m3 / e5m2 (fp8, 8-bit codebook, 4-bit indices) and e2m3 / e3m2 (fp6,
6-bit codebook, 4-bit indices). Generalizes gen_lut_dim.py (e4m3-only) to all 4
with the same DIM-aware golden: operands (A_in_hw/B_in/luts/scales) come byte-for-byte
from lut_mapping_demo (imported with the exact env that produced each committed header),
and only the goldens (C_out / C_out_bf16 / C_scales_row / C_proj_hw) are recomputed with
the per-lane precision model (fp8_matmul_model.TILE=dim, precision_for_dim(dim)) so they
match Spike / the RTL at the chosen mesh DIM.

Per-format requant snap, requant-code encode, and nearest-LUT finder are REUSED verbatim
from lut_mapping_demo (nothing is rewritten) -- this file only selects the right ones per
--format, mirroring lut_mapping_demo Steps 7-9.

Run:
  python3 gen_mx_lut.py --format e5m2 --dim 32
"""
import argparse
import contextlib
import os
import sys
import tempfile

# ---- precision_for_dim: replicated from gen_matmul_llama.py:66-74 (as in gen_lut_dim.py) ----
# (exp_bits, frac_bits); frac = Scala sigWidth - 1. dim=16 reproduces the original ramp.
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1


def precision_for_dim(dim: int):
    prod = [(4, 3)] * dim
    if dim <= 8:
        acc = [(8, 7)] * dim   # DIM<=8: ALL rows bf16 (design assumption -- no reduced-precision ramp)
    elif dim <= 16:
        acc = ACC_PRECISION[:dim]
    else:
        acc = ACC_PRECISION + [(8, 7)] * (dim - 16)
    return prod, acc


# Per-format constants: env INPUT_SPEC + the committed dim16 header's include-guard path prefix.
# The guard is derived from the exact path string passed to write_c_header_tiled_hw; the committed
# e2m3 header was emitted with a "./include/..." path (guard "__INCLUDE..."), the others with
# "include/..." (guard "INCLUDE..."). We must match this for byte-identical dim16 reproduction.
FORMAT_SPEC = {
    "e4m3": "fp8:e4m3",
    "e5m2": "fp8:e5m2",
    "e2m3": "fp6:e2m3",
    "e3m2": "fp6:e3m2",
}
DIM16_GUARD_PREFIX = {"e4m3": "", "e5m2": "", "e2m3": "./", "e3m2": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=sorted(FORMAT_SPEC),
                    help="LUT format.")
    ap.add_argument("--dim", type=int, default=16, help="Mesh DIM (per-tile accumulation depth).")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16-arith", action="store_true",
                    help="Full BF16 PE products and adds (MxE4M3LutBf16Dim* configs); header suffix _bf16.")
    ap.add_argument("--header-path", type=str, default=None,
                    help="Output header path (default include/matmul_data_mx_lut_<fmt>_64x64[_dim32].h).")
    args = ap.parse_args()
    fmt = args.format
    dim = args.dim
    input_spec = FORMAT_SPEC[fmt]

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if here not in sys.path:
        sys.path.insert(0, here)

    # Default output filename: no suffix for dim16, _dim32 for dim32 (matches committed naming).
    # Shape tag: <M>x<N> for square (M==N==K, e.g. 64x64), else <M>x<N>x<K> (e.g. 128x128x256).
    dim_suffix = "" if dim == 16 else f"_dim{dim}"
    shape_tag = f"{args.M}x{args.N}" if (args.M == args.N == args.K) else f"{args.M}x{args.N}x{args.K}"
    if args.bf16_arith:
        dim_suffix += "_bf16"
    default_name = f"include/matmul_data_mx_lut_{fmt}_{shape_tag}{dim_suffix}.h"
    # Guard-path prefix: reproduce the committed dim16 guard exactly; new dim!=16 headers use "include/".
    guard_prefix = DIM16_GUARD_PREFIX[fmt] if dim == 16 else ""
    header_path = args.header_path if args.header_path is not None else (guard_prefix + default_name)

    # --- Reproduce the exact operands of the committed <fmt> 64x64 header ------------------
    os.environ["MXGEMMINI_SEED"] = str(args.seed)
    os.environ["MXGEMMINI_M"] = str(args.M)
    os.environ["MXGEMMINI_K"] = str(args.K)
    os.environ["MXGEMMINI_N"] = str(args.N)
    os.environ["MXGEMMINI_INPUT_SPEC"] = input_spec
    os.environ["MXGEMMINI_LLAMA"] = "0"
    scratch = tempfile.NamedTemporaryFile(suffix=".h", delete=False)
    scratch.close()
    os.environ["MXGEMMINI_HEADER_PATH"] = scratch.name

    import torch
    import fp8_matmul_model
    from fp8_matmul_model import (
        tiled_matmul_hwlike, matrix_mx_requantize, make_fp_quantizer,
    )
    from lut_golden_model import write_c_header_tiled_hw, _a_indices_to_hw_layout

    # Import (and run) lut_mapping_demo with stdout muted -- it builds all operands and exposes
    # the per-format requant encoders/finders we reuse verbatim.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        import lut_mapping_demo as lmd
    try:
        os.unlink(scratch.name)
    except OSError:
        pass

    M, K, N = lmd.M, lmd.K, lmd.N
    GROUP = lmd.GROUP
    G = lmd.G
    INPUT_SPEC = lmd.INPUT_SPEC
    SCALE_SPEC = lmd.SCALE_SPEC
    A_TILE_M, K_TILE = lmd.A_TILE_M, lmd.K_TILE
    assert INPUT_SPEC == input_spec, f"lmd INPUT_SPEC {INPUT_SPEC} != requested {input_spec}"
    IS_E4M3 = lmd.IS_E4M3
    IS_E5M2 = lmd.IS_E5M2
    IS_E2M3 = lmd.IS_E2M3
    IS_E3M2 = not (IS_E4M3 or IS_E5M2 or IS_E2M3)

    # --- Step 5 (DIM-aware): C_out_bf16 = tiled_matmul_hwlike with precision_for_dim(dim) -----
    fp8_matmul_model.TILE = dim
    for attr in ("TILE_K", "TILE_M", "TILE_N"):
        if hasattr(fp8_matmul_model, attr):
            setattr(fp8_matmul_model, attr, {"TILE_K": dim, "TILE_M": 2 * dim, "TILE_N": 2 * dim}[attr])
    prod_prec, acc_prec = precision_for_dim(dim)
    if args.bf16_arith:
        prod_prec, acc_prec = [(8, 7)] * dim, [(8, 7)] * dim

    in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
    A_in = in_q(lmd.A_fp)
    B_in = in_q(lmd.B_fp)
    C_out_bf16 = tiled_matmul_hwlike(
        A_in, B_in, lmd.A_scales_row_q, lmd.B_scales_col_q,
        verbose=False, prod_precision_list=prod_prec, acc_precision_list=acc_prec,
    )
    assert torch.isfinite(C_out_bf16).all(), "mesh model produced non-finite output"

    # --- Step 7: requantize + per-format BF16/fp6 snap (mirrors lut_mapping_demo:806-823) ----
    C_requantized, C_req_scales = matrix_mx_requantize(C_out_bf16, quant_spec=INPUT_SPEC)
    if IS_E5M2 or IS_E4M3 or IS_E2M3:
        C_requantized = lmd.q_bf16_rne(C_requantized)          # encode happens in Step 8
    else:  # e3m2
        C_requantized = lmd.hw_bf16_to_fp6(lmd.q_bf16_rne(C_requantized))

    # --- Step 8: encode C_req to raw codes (mirrors lut_mapping_demo:870-895) -----------------
    if IS_E4M3:
        Cf = C_requantized.float().view(M, N)
        C_req_codes_raw = [[lmd.bf16f_to_e4m3_code_rha(float(Cf[m, n].item())) for n in range(N)]
                           for m in range(M)]
    elif IS_E5M2:
        import numpy as _np
        from wire import encode_requant as _encode_requant
        _Cf = C_requantized.float().view(M, N).detach().cpu().numpy().astype(_np.float32)
        C_req_codes_raw = _encode_requant(_Cf, dtype="fp8_e5m2").astype(_np.uint8).tolist()
    elif IS_E2M3:
        C_req_codes_raw, _ = lmd.tensor_to_custom_fp_codes(C_requantized.float().view(M, N), INPUT_SPEC)
    else:  # e3m2
        C_req_codes_raw = lmd._fp6_tensor_to_codes(C_requantized.float().view(M, N))

    # --- nearest-LUT finder per format (mirrors lut_mapping_demo:905-909) ----------------------
    finder = (lmd.fp8_e5m2_nearest_finder if IS_E5M2 else
              lmd.fp8_e4m3_nearest_finder if IS_E4M3 else
              lmd.fp6e2m3_nearest_finder  if IS_E2M3 else
              lmd.fp6e3m2_nearest_finder)
    C_proj = torch.zeros(M, N, dtype=torch.int32)
    for m in range(M):
        lut_codes = lmd.C_luts_codes_raw[m >> G]
        for n in range(N):
            C_proj[m, n] = finder(C_req_codes_raw[m][n], lut_codes)
    C_proj_hw = _a_indices_to_hw_layout(C_proj, a_tile_m=A_TILE_M, k_tile=K_TILE)

    # --- Step 9: write header (operands from lmd, goldens dim-aware) ------------------------
    write_c_header_tiled_hw(
        path=header_path,
        M=M, K=K, N=N,
        group=GROUP,
        input_spec=INPUT_SPEC,
        acc_spec="bf16",
        scale_spec=SCALE_SPEC,
        lut_index_bits=lmd.LUT_INDEX_BITS,
        A_in=lmd.A_indices,
        B_in=lmd.B_indices,
        A_scales_row_q=lmd.A_scales_row_q,
        B_scales_col_q=lmd.B_scales_col_q,
        C_out_bf16=C_out_bf16,
        C_out_quantized=C_proj,
        C_out_scales=C_req_scales,
        A_lut=lmd.A_luts_t,
        B_lut=lmd.B_luts_t,
        C_lut=lmd.C_luts_t,
        a_tile_m=A_TILE_M,
        k_tile=K_TILE,
    )

    # Append C_proj_hw section (mirrors lut_mapping_demo:980-999 / gen_lut_dim.py:147-165).
    guard = header_path.upper()
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
    with open(header_path, "r") as f:
        content = f.read()
    content = content.replace(f"#endif // {guard}\n", c_proj_hw_section)
    with open(header_path, "w") as f:
        f.write(content)

    print(f"[gen_mx_lut] format={fmt} ({INPUT_SPEC})  dim={dim}  "
          f"prod_len={len(prod_prec)}  acc_len={len(acc_prec)}")
    print(f"[gen_mx_lut] acc_precision_list={acc_prec}")
    print(f"[gen_mx_lut] Header written to {header_path}")


if __name__ == "__main__":
    main()

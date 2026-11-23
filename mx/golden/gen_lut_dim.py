#!/usr/bin/env python3
"""Generate the E4M3-quad-LUT 64x64 test-data header with a DIM-AWARE golden.

The operands (A_in_hw / B_in / A_lut / B_lut / C_lut / A_scales_row / B_scales_col)
are byte-identical to include/matmul_data_mx_lut_e4m3_64x64.h -- they are produced by
importing lut_mapping_demo.py with the exact env that generated that header
(SEED=0, M=K=N=64, INPUT_SPEC=fp8:e4m3, G=1). Only the goldens
(C_out_bf16 / C_out=nibble-packed C_proj / C_scales_row / C_proj_hw) are recomputed
with the DIM-aware per-lane precision model (fp8_matmul_model driven exactly as
gen_matmul_llama.build(): mm.TILE=dim, prod/acc = precision_for_dim(dim)) so the
golden matches Spike (gemmini.cc acc ramp) and the RTL at DIM=32.

Emission / deproject / operand helpers are REUSED from lut_golden_model and
lut_mapping_demo -- nothing is rewritten. precision_for_dim is replicated inline
(source: gen_matmul_llama.py:66-74) to avoid pulling app.mxq_golden.

Run:
  PATH=../../npu-exploration/.venv/bin:$PATH \
  ../../npu-exploration/.venv/bin/python3 gen_lut_dim.py --dim 32
"""
import argparse
import contextlib
import os
import sys
import tempfile

# ---- precision_for_dim: replicated from gen_matmul_llama.py:66-74 -------------------------
# (exp_bits, frac_bits); frac = Scala sigWidth - 1. dim=16 reproduces the original ramp.
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1


def precision_for_dim(dim: int):
    prod = [(4, 3)] * dim
    acc = ACC_PRECISION[:dim] if dim <= 16 else ACC_PRECISION + [(8, 7)] * (dim - 16)
    return prod, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=32, help="Mesh DIM (per-tile accumulation depth).")
    ap.add_argument("--header-path", type=str, default=None,
                    help="Output header path (default include/matmul_data_mx_lut_e4m3_64x64_dim<DIM>.h).")
    args = ap.parse_args()
    dim = args.dim

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if here not in sys.path:
        sys.path.insert(0, here)

    header_path = args.header_path or f"include/matmul_data_mx_lut_e4m3_64x64_dim{dim}.h"

    # --- Reproduce the exact operands of the committed e4m3 64x64 header --------------------
    # lut_mapping_demo.py reads these from the environment at import time and, running top to
    # bottom, builds every operand (same seed/params => byte-identical A/B/LUT/scales). Its own
    # DIM=16 header emit is redirected to a scratch file we discard.
    os.environ["MXGEMMINI_SEED"] = "0"
    os.environ["MXGEMMINI_M"] = "64"
    os.environ["MXGEMMINI_K"] = "64"
    os.environ["MXGEMMINI_N"] = "64"
    os.environ["MXGEMMINI_INPUT_SPEC"] = "fp8:e4m3"
    os.environ["MXGEMMINI_LLAMA"] = "0"
    scratch = tempfile.NamedTemporaryFile(suffix=".h", delete=False)
    scratch.close()
    os.environ["MXGEMMINI_HEADER_PATH"] = scratch.name

    import torch
    import fp8_matmul_model
    from fp8_matmul_model import (
        tiled_matmul_hwlike, matrix_mx_requantize, make_fp_quantizer,
    )
    from lut_golden_model import (
        write_c_header_tiled_hw, _a_indices_to_hw_layout, parse_fp_spec, q_bf16_rne,
    )

    # Import (and run) lut_mapping_demo with stdout muted -- it builds all operands and exposes
    # the e4m3-specific requant encoders/finders we reuse verbatim.
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
    assert INPUT_SPEC == "fp8:e4m3" and lmd.IS_E4M3, "gen_lut_dim currently supports e4m3 only"

    # --- Step 5 (DIM-aware): C_out_bf16 = tiled_matmul_hwlike with precision_for_dim(dim) -----
    fp8_matmul_model.TILE = dim                 # per-tile accumulation depth = mesh DIM
    for attr in ("TILE_K", "TILE_M", "TILE_N"):  # nibble models only; harmless if absent
        if hasattr(fp8_matmul_model, attr):
            setattr(fp8_matmul_model, attr, {"TILE_K": dim, "TILE_M": 2 * dim, "TILE_N": 2 * dim}[attr])
    prod_prec, acc_prec = precision_for_dim(dim)

    in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
    A_in = in_q(lmd.A_fp)
    B_in = in_q(lmd.B_fp)
    C_out_bf16 = tiled_matmul_hwlike(
        A_in, B_in, lmd.A_scales_row_q, lmd.B_scales_col_q,
        verbose=False, prod_precision_list=prod_prec, acc_precision_list=acc_prec,
    )
    assert torch.isfinite(C_out_bf16).all(), "mesh model produced non-finite output"

    # --- Step 7: requantize (fp8_matmul_model.matrix_mx_requantize) + BF16 snap (E4M3 branch) --
    C_requantized, C_req_scales = matrix_mx_requantize(C_out_bf16, quant_spec=INPUT_SPEC)
    C_requantized = q_bf16_rne(C_requantized)   # E4M3: BF16 snap; encode happens next

    # --- Step 8: E4M3 encode (round-half-away) then nearest-LUT index (reused from lut_mapping_demo)
    Cf = C_requantized.float().view(M, N)
    C_req_codes_raw = [[lmd.bf16f_to_e4m3_code_rha(float(Cf[m, n].item())) for n in range(N)]
                       for m in range(M)]
    C_proj = torch.zeros(M, N, dtype=torch.int32)
    for m in range(M):
        lut_codes = lmd.C_luts_codes_raw[m >> G]
        for n in range(N):
            C_proj[m, n] = lmd.fp8_e4m3_nearest_finder(C_req_codes_raw[m][n], lut_codes)
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

    # Append C_proj_hw section (mirrors lut_mapping_demo.py:980-999).
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

    print(f"[gen_lut_dim] dim={dim}  prod_len={len(prod_prec)}  acc_len={len(acc_prec)}")
    print(f"[gen_lut_dim] acc_precision_list={acc_prec}")
    print(f"[gen_lut_dim] Header written to {header_path}")


if __name__ == "__main__":
    main()

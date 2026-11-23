"""MX element formats used by the generators, and the hardware requantizer for fp4/fp6 outputs."""
from dataclasses import dataclass

import numpy as np
import torch

from wire import e8m0_encode_exact, encode_requant, fp8_e4m3_decode


@dataclass(frozen=True)
class Format:
    """One MX element format, as both oracles implement it."""
    name: str
    mxq: str          #: the alias quantize_mx_block32 understands
    bits: int         #: element width; 4 and 6 are nibble-packed / LUT-indexed in the headers
    #: `log2_pmax_floor` the requantizer subtracts when it commits OUTPUT blocks. 0 is MXQuant's
    #: e2e convention (FP8, after chain_seam_hw_notes.md section 8); FP6/FP4 still use the OCP
    #: `- emax` shift in `gemmini.cc`, so their goldens must reproduce it.
    out_pmax: int
    #: mesh tile geometry. FP8 marches 16x16; the nibble formats pack 32x32 (fp4_matmul_model.py:28,
    #: gemmini.cc:1303 `const int TM = 32, TN = 32`).
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    #: which model module carries the matching mesh + code encoder
    model: str = "fp8_matmul_model"
    #: who requantizes the OUTPUT. "mxquant" is MXQuant's `_quantize_elemwise`, which FP8's
    #: requantizer was migrated to. FP4/FP6 were not: `gemmini.cc:1508` requantizes with
    #: `q_bf16_rne(v/scale)` then `hw_bf16_to_e2m1`, a two-stage RNE->E3M1->E2M1 that rounds ties
    #: differently from MXQuant (`mx_fp_math.h:402-404` says so outright). Their goldens must use
    #: the hardware's quantizer or they disagree on ~14% of elements in the mantissa LSB.
    out_requant: str = "mxquant"

FORMATS = {
    # out_pmax=0 for every format now: the requant output normalizes to [1,2) (chained-matmul fix),
    # matching the RTL/spike log2_pmax=0. FP8 was already 0 (MXQuant e2e); FP6/FP4 dropped 4/2.
    "fp8": Format("fp8:e4m3", "MXFP8_E4M3", 8, out_pmax=0),
    "fp6": Format("fp6:e3m2", "MXFP6_E3M2", 6, out_pmax=0,
                  tile_m=32, tile_n=32, model="fp4_matmul_model", out_requant="model"),
    "fp4": Format("fp4:e2m1", "MXFP4", 4, out_pmax=0,
                  tile_m=32, tile_n=32, model="fp4_matmul_model", out_requant="model"),
}


def requant_model(C_bf16: np.ndarray, f: Format):
    """Requantize a BF16 tile with the hardware quantizer (fp4/fp6). Returns (codes, scales, P values)."""
    mm = __import__(f.model)
    C_q, C_sc = mm.matrix_mx_requantize(torch.from_numpy(C_bf16), f.name)
    codes = np.array(mm.tensor_to_custom_fp_codes(C_q, f.name)[0], dtype=np.uint8)
    scales = e8m0_encode_exact(C_sc.numpy().astype(np.float32))
    return codes, scales, C_q.numpy().astype(np.float32)


def requant(C_bf16: np.ndarray, f: Format):
    """Hardware requantizer for any format: scale = 2^floor(log2 block amax) per 32 columns,
    elements encoded as the device rounds them. Returns (codes, scales [M][N/32], P values)."""
    if f.bits != 8:
        return requant_model(C_bf16, f)
    import fp8_matmul_model as fm
    C_q, C_sc = fm.matrix_mx_requantize(torch.from_numpy(C_bf16), f.name)
    codes = encode_requant(C_q.numpy().astype(np.float32), dtype="fp8_e4m3_quad")
    return codes, e8m0_encode_exact(C_sc.numpy().astype(np.float32)), fp8_e4m3_decode(codes)

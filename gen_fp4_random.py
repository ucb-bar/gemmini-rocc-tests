#!/usr/bin/env python3
"""FP4 goldens without the TinyLlama capture (npu-exploration/MXQuant/end_to_end_linear absent): runs gen_matmul_llama
unchanged except for two hooks -- seeded random operands instead of load_pair, and a local MX block quantizer (block
scale 2^floor(log2 amax) over 32 values, FP4 E2M1 codes) instead of MXQuant's quantize_mx_block32. The mesh model, the
output requant (fp4_matmul_model.matrix_mx_requantize) and the header writer are gen_matmul_llama's, so the header
layout matches the existing fp4 tests. Self-consistent: the codes/scales written are the ones the golden C uses.

  python3 gen_fp4_random.py fp4_64x64_dim4
  python3 gen_fp4_random.py fp8_64x64_dim4 fp8_128x128_dim4   (E4M3 codes, same block scale)
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)

import numpy as np
import torch
import gen_matmul_llama as g


def load_pair(shape):
    rng = np.random.default_rng(0)
    return (rng.standard_normal((shape.M, shape.K)).astype(np.float32),
            rng.standard_normal((shape.K, shape.N)).astype(np.float32))


def quantize(V, *, axis, f, pmax_shift=0):
    assert f.bits in (4, 8) and pmax_shift == 0, "gen_fp4_random: FP4 / FP8 E4M3 operands only"
    mm = __import__(f.model)
    Vb = V if axis == "row" else V.T                     # 32-value blocks along the last axis
    R, C = Vb.shape
    blk = Vb.reshape(R, C // 32, 32)
    amax = np.maximum(np.abs(blk).max(axis=2), np.finfo(np.float32).eps)
    X = np.exp2(np.floor(np.log2(amax))).astype(np.float32)          # [R][C/32]
    Y = (blk / X[:, :, None]).reshape(R, C).astype(np.float32)
    if f.bits == 8:   # E4M3 round-to-nearest-even (the requantizer's; qtorch "nearest" breaks BF16 ties upward)
        e = np.maximum(np.floor(np.log2(np.maximum(np.abs(Y), 2.0 ** -9))), -6)
        ulp = np.exp2(e - 3)
        P = np.clip(np.round(Y / ulp) * ulp, -448, 448).astype(np.float32)
    else:
        q = mm.make_fp_quantizer(f.name, rounding="nearest")
        P = q(torch.from_numpy(Y)).numpy().astype(np.float32)
    if axis == "col":
        P, X = P.T, X.T
    P = np.ascontiguousarray(P)
    raw, _ = mm.tensor_to_custom_fp_codes(torch.from_numpy(P), f.name)
    raw = np.array(raw, dtype=np.uint8)
    if f.bits == 8:   # a negative value that rounds to zero keeps its sign (-0 = 0x80), as the requantizer
        Ys = Y if axis == "row" else Y.T
        raw[(np.ascontiguousarray(Ys) < 0) & (P == 0)] |= 0x80
    return raw, g.e8m0_encode_exact(np.ascontiguousarray(X)), P


g.load_pair = load_pair
g.quantize = quantize
sys.exit(g.main())

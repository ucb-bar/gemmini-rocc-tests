#!/usr/bin/env python3
"""Generate the plain-format (fp8/fp4) matmul data headers include/matmul_fp*.h.

Operands are seeded random element codes with random E8M0 block scales, so this needs nothing outside
this repo. Goldens: C_out_bf16 from the bit-exact mesh model (fp8_matmul_model / fp4_matmul_model),
C_out / C_scales_out from the hardware requantizer (formats.requant).

  python3 gen_plain.py                     # all headers
  python3 gen_plain.py fp8_64x64           # one header
  python3 gen_plain.py chain_fp8_64x64     # chained-matmul header
  python3 gen_plain.py fp8_64x64 --operands ops.npz   # use given A/B codes + scales (e.g. real model data)
"""
from __future__ import annotations

import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from wire import e8m0_decode, e8m0_encode_exact  # noqa: E402

BLOCK = 32
#: where headers are written (MX_DATA_DIR overrides; default: the repo include/)
OUT_DIR = Path(os.environ.get("MX_DATA_DIR", HERE.parent.parent / "include"))

#: Mesh precision, from ConfigsFP.scala's meshProdPrecisionList / meshAccPrecisionList. Written
#: (exp_bits, frac_bits); the same schedule appears in gemmini.cc as acc_e[]/acc_m[].
PROD_PRECISION = [(4, 3)] * 16
ACC_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1

def precision_for_dim(dim: int):
    """Per-DIM (prod, acc) precision lists mirroring ConfigsFP dim{16,32}MxFPConfig
    (Python frac = Scala sigWidth - 1): prod = MxFloat(4,4,4) uniform -> (4,3); acc = the DIM=16 ramp
    for rows 0-15, extra rows (DIM>16) bf16 MxFloat(8,8,4) -> (8,7). dim=16 reproduces the originals."""
    prod = [(4, 3)] * dim
    if dim <= 8:
        return prod, [(8, 7)] * dim   # DIM=4/8: all acc rows bf16 (dim4MxFPConfig, dim8MxFPConfig)
    acc = ACC_PRECISION[:dim] if dim <= 16 else ACC_PRECISION + [(8, 7)] * (dim - 16)
    return prod, acc


from formats import Format, FORMATS, requant  # noqa: E402


# --- the shape table ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """One generated header: its dimensions, format, and where its operands come from."""
    header: str                 #: file under include/
    M: int
    K: int
    N: int
    fmt: str
    layer: str = "layer0"
    proj: str = "mlp.gate_proj"
    guard: str = ""
    dim: int = 16               #: mesh DIM = per-tile accumulation depth (TILE); golden matches this DIM
    #: tests that consume this header, for the report
    tests: tuple[str, ...] = field(default_factory=tuple)

    @property
    def GK(self) -> int: return self.K // BLOCK

    @property
    def GN(self) -> int: return self.N // BLOCK


SHAPES = [
    Shape("matmul_fp8_32x32x32.h", 32, 32, 32, "fp8",
          guard="INCLUDE_MATMUL_FP8_32X32X32_H",
          tests=("matmul_tiled_fp8_32x32x32", "matmul_tiled_fp8_32x32x32_requant")),
    Shape("matmul_fp8_64x64.h", 64, 64, 64, "fp8",
          guard="INCLUDE_MATMUL_FP8_64X64_H",
          tests=("matmul_tiled_fp8_64x64", "matmul_tiled_fp8_64x64_requant",
                 "matmul_tiled_fp8_64x64_DRAMMvout")),
    Shape("matmul_fp8_64x64_dim32.h", 64, 64, 64, "fp8", dim=32,
          guard="INCLUDE_MATMUL_FP8_64X64_DIM32_H",
          tests=("matmul_tiled_fp8_64x64_requant_dim32",)),
    # E4M3 single (mode8) on the 4x4 (DIM=4) mesh: 4 col-tiles per widened acc row.
    Shape("matmul_fp8_64x64_dim4.h", 64, 64, 64, "fp8", dim=4,
          guard="INCLUDE_MATMUL_FP8_64X64_DIM4_H",
          tests=("matmul_tiled_fp8_64x64_requant_dim4", "matmul_tiled_fp8_64x64_single_dim4")),
    Shape("matmul_fp8_64x64_dim8.h", 64, 64, 64, "fp8", dim=8,
          guard="INCLUDE_MATMUL_FP8_64X64_DIM8_H",
          tests=("matmul_tiled_fp8_64x64_requant_dim8", "matmul_tiled_fp8_64x64_single_dim8")),
    Shape("matmul_fp8_128x128_dim8.h", 128, 128, 128, "fp8", dim=8,
          guard="INCLUDE_MATMUL_FP8_128X128_DIM8_H",
          tests=("matmul_tiled_fp8_128x128_requant_dim8", "matmul_tiled_fp8_128x128_single_dim8")),
    Shape("matmul_fp8_128x128_dim4.h", 128, 128, 128, "fp8", proj="mlp.down_proj", dim=4,
          guard="INCLUDE_MATMUL_FP8_128X128_DIM4_H",
          tests=("matmul_tiled_fp8_128x128_requant_dim4", "matmul_tiled_fp8_128x128_single_dim4")),
    Shape("matmul_fp8_96x32x32.h", 96, 32, 32, "fp8", proj="mlp.up_proj",
          guard="INCLUDE_MATMUL_FP8_96X32X32_H",
          tests=("matmul_tiled_fp8_96x32x32", "matmul_tiled_fp8_96x32x32_requant")),
    Shape("matmul_fp8_64x96x64.h", 64, 64, 96, "fp8", proj="attn.q_proj",
          guard="INCLUDE_MATMUL_FP8_64X96X64_H",
          tests=("matmul_tiled_fp8_64x96x64", "matmul_tiled_fp8_64x96x64_requant")),
    # DIM=32 partial j-group (N=96 = 3 col-tiles: 2 acc chunks); operands re-used via --operands matmul_fp8_64x96x64.h
    Shape("matmul_fp8_64x96x64_dim32.h", 64, 64, 96, "fp8", proj="attn.q_proj", dim=32,
          guard="INCLUDE_MATMUL_FP8_64X96X64_DIM32_H",
          tests=("matmul_tiled_fp8_64x96x64_dramloop_dim32",)),
    Shape("matmul_fp8_64x96x64_dim8.h", 64, 64, 96, "fp8", proj="attn.q_proj", dim=8,
          guard="INCLUDE_MATMUL_FP8_64X96X64_DIM8_H",
          tests=("matmul_tiled_fp8_64x96x64_dramloop_dim8", "matmul_tiled_fp8_64x96x64_dim8")),
    Shape("matmul_fp8_64x96x64_dim4.h", 64, 64, 96, "fp8", proj="attn.q_proj", dim=4,
          guard="INCLUDE_MATMUL_FP8_64X96X64_DIM4_H",
          tests=("matmul_tiled_fp8_64x96x64_dramloop_dim4", "matmul_tiled_fp8_64x96x64_dim4")),
    Shape("matmul_fp8_96x96x64.h", 96, 64, 96, "fp8", proj="attn.o_proj",
          guard="INCLUDE_MATMUL_FP8_96X96X64_H",
          tests=("matmul_tiled_fp8_96x96x64", "matmul_tiled_fp8_96x96x64_requant")),
    Shape("matmul_fp8_128x128.h", 128, 128, 128, "fp8", proj="mlp.down_proj",
          guard="INCLUDE_MATMUL_FP8_128X128_H",
          tests=("matmul_tiled_fp8_128x128", "matmul_tiled_fp8_128x128_requant")),
    Shape("matmul_fp8_128x128x256.h", 128, 256, 128, "fp8", layer="layer1",
          guard="INCLUDE_MATMUL_FP8_128X128X256_H",
          tests=("matmul_tiled_fp8_128x128x256", "matmul_tiled_fp8_128x128x256_DRAMMvout")),
    Shape("matmul_fp8_128x128x256_dim32.h", 128, 256, 128, "fp8", dim=32, layer="layer1",
          guard="INCLUDE_MATMUL_FP8_128X128X256_DIM32_H",
          tests=("matmul_tiled_fp8_128x128x256_requant_dim32",)),
    # Wide-N stress: N=256 = 8 col-tiles = 2 DIM=32 acc rows per output row (offsets 0/8/16/24 x2 rows).
    Shape("matmul_fp8_64x256x64_dim32.h", 64, 64, 256, "fp8", dim=32, layer="layer1",
          guard="INCLUDE_MATMUL_FP8_64X256X64_DIM32_H",
          tests=("matmul_tiled_fp8_64x256x64_requant_dim32",)),

    Shape("matmul_fp4_64x64.h", 64, 64, 64, "fp4", layer="layer2",
          guard="MATMUL_DATA_FP4_H",
          tests=("matmul_tiled_fp4_64x64", "matmul_tiled_fp4_64x64_requant")),
    # FP4 on the 32x32 (DIM=32) mesh -- first wide-mesh nibble/quad-throughput requant test.
    Shape("matmul_fp4_64x64_dim32.h", 64, 64, 64, "fp4", dim=32, layer="layer2",
          guard="INCLUDE_MATMUL_FP4_64X64_DIM32_H",
          tests=("matmul_tiled_fp4_64x64_requant_dim32",)),
    # FP4 on the 8x8 (DIM=8) widen mesh -- quad nibble requant, DIM8 sibling of the dim32 fp4 test.
    Shape("matmul_fp4_64x64_dim8.h", 64, 64, 64, "fp4", dim=8, layer="layer2",
          guard="INCLUDE_MATMUL_FP4_64X64_DIM8_H",
          tests=("matmul_tiled_fp4_64x64_requant_dim8",)),
    # FP4 on the 4x4 (DIM=4) mesh: W=4 acc widen (one acc row = 4 col-tiles = a 32-col block per output row).
    Shape("matmul_fp4_64x64_dim4.h", 64, 64, 64, "fp4", dim=4, layer="layer2",
          guard="INCLUDE_MATMUL_FP4_64X64_DIM4_H",
          tests=("matmul_tiled_fp4_64x64_requant_dim4",)),
    # FP4 DIM=32, larger: 2 i-tiles x 2 j-tiles (GN=4, loop_bound_j=2), deeper K -- stresses multi-tile.
    Shape("matmul_fp4_128x128_dim32.h", 128, 128, 128, "fp4", dim=32, layer="layer2",
          guard="INCLUDE_MATMUL_FP4_128X128_DIM32_H",
          tests=("matmul_tiled_fp4_128x128_requant_dim32",)),
    # Same 2x2 output-tile shape but K=64 -> only 2 K-tiles (vs 4). Isolates the cross-K-tile accumulate
    # as the fp4 128x128 residual trigger: if this passes on RTL, the bug scales with K-tile count.
    Shape("matmul_fp4_128x128x64_dim32.h", 128, 64, 128, "fp4", dim=32, layer="layer2",
          guard="INCLUDE_MATMUL_FP4_128X128X64_DIM32_H",
          tests=("matmul_tiled_fp4_128x128x64_requant_dim32",)),
    Shape("matmul_fp4_128x128.h", 128, 128, 128, "fp4", layer="layer2", proj="mlp.up_proj",
          guard="INCLUDE_MATMUL_FP4_128X128_H",
          tests=("matmul_tiled_fp4_128x128", "matmul_tiled_fp4_128x128_requant")),
    Shape("matmul_fp4_128x128x512.h", 128, 512, 128, "fp4", layer="layer3", proj="mlp.down_proj",
          guard="INCLUDE_MATMUL_FP4_128X128X512_H",
          tests=("matmul_tiled_fp4_128x128x512", "matmul_tiled_fp4_128x128x512_requant")),
]

BY_NAME = {s.header.removeprefix("matmul_").removesuffix(".h"): s for s in SHAPES}
#: chained-matmul headers generated by default (chain_<name>)
CHAINS = ["fp8_64x64", "fp8_128x128", "fp8_64x64_dim8", "fp8_64x64_dim32", "fp4_64x64", "fp4_128x128"]


# --- operand sourcing --------------------------------------------------------------------------

def _seed(name: str) -> int:
    return zlib.crc32(name.encode())


def random_operand(rows: int, cols: int, f: Format, axis: str, seed: int):
    """``(codes, scales, P)``: randn rounded to the element format, E8M0 block scales 2^[-4,4).

    axis="row": blocks of 32 along each row (scales [rows][cols/32]); "col": along each column.
    """
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(rows, cols, generator=g)
    mm = __import__(f.model)
    raw, _ = mm.tensor_to_custom_fp_codes(V, f.name)
    codes = np.array(raw, dtype=np.uint8)
    e_bits, m_bits = mm.parse_fp_spec(f.name)
    P = _decode_codes(codes, e_bits, m_bits)
    sshape = (rows, cols // BLOCK) if axis == "row" else (rows // BLOCK, cols)
    scales = (127 + torch.randint(-4, 4, sshape, generator=g)).numpy().astype(np.uint8)
    return codes, scales, P


# --- emission ----------------------------------------------------------------------------------

def _decode_codes(codes: np.ndarray, e_bits: int, m_bits: int) -> np.ndarray:
    """Decode raw (1+e+m)-bit float codes. Mirrors mx_fp_math.h's decoders, format-generically."""
    c = np.asarray(codes, dtype=np.uint32)
    bias = (1 << (e_bits - 1)) - 1
    s = (c >> (e_bits + m_bits)) & 1
    e = (c >> m_bits) & ((1 << e_bits) - 1)
    m = c & ((1 << m_bits) - 1)
    scale = float(1 << m_bits)
    val = np.where(e == 0,
                   (m / scale) * 2.0 ** (1 - bias),
                   (1.0 + m / scale) * 2.0 ** (e.astype(np.int64) - bias))
    return np.where(s == 1, -val, val).astype(np.float32)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> BF16 bit pattern, RNE, matching mx_fp_math.h::f32_to_bf16_rne."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    nan = ((u >> 23) & 0xFF) == 0xFF
    lsb = (u >> 16) & 1
    r = (u + 0x7FFF + lsb) >> 16
    r = np.where(nan, u >> 16, r).astype(np.uint32)
    out = (r & 0xFFFF).astype(np.uint16)
    return np.where(x == 0, np.uint16(0), out).astype(np.uint16)


def _rows(a: np.ndarray, width: int) -> str:
    """Format a 2-D array as C initializer rows of 0x-prefixed values."""
    fmt = f"0x%0{width}x"
    return ",\n".join("    { " + ", ".join(fmt % int(v) for v in row) + " }" for row in a)


def build(shape: Shape, verbose: bool = True, operands: dict | None = None) -> dict[str, np.ndarray]:
    """Quantize, run the mesh model, requantize. Returns everything the header needs.
    operands: A/B codes + scales to use instead of random ones (from --operands)."""
    f = FORMATS[shape.fmt]
    if operands is None:
        A_codes, A_scales, A_P = random_operand(shape.M, shape.K, f, "row", _seed(shape.header))      # [M][GK]
        B_codes, B_scales, B_P = random_operand(shape.K, shape.N, f, "col", _seed(shape.header) + 1)  # [GK][N]
    else:   # given codes/scales; the mesh multiplies the decoded code values
        e_bits, m_bits = __import__(f.model).parse_fp_spec(f.name)
        A_codes, A_scales, B_codes, B_scales = (operands[k] for k in ("A_codes", "A_scales", "B_codes", "B_scales"))
        A_P, B_P = _decode_codes(A_codes, e_bits, m_bits), _decode_codes(B_codes, e_bits, m_bits)

    # DEBUG (env-gated): keep the REAL (varied) operand codes but force all block scales to unit (E8M0 127),
    # so the acc = raw code products (no scale applied). Isolates whether the scale path is involved.
    if os.environ.get("GEMMINI_NOSCALE"):
        A_scales = np.full_like(A_scales, 127)
        B_scales = np.full_like(B_scales, 127)
        if verbose: print("  [GEMMINI_NOSCALE] real codes, block scales forced to unit (E8M0 127)")
    elif os.environ.get("GEMMINI_AROWENC"):
        # real codes; A (row/act) scale encodes row%8 (distinct within each 8-row group), B scale unit.
        # A row0/row1 scale swap or period-8 mis-association shows as a distinct wrong value + which row.
        for i in range(A_scales.shape[0]):
            A_scales[i, :] = 120 + (i % 8)
        B_scales = np.full_like(B_scales, 127)
        if verbose: print("  [GEMMINI_AROWENC] A-scale=120+(row%8), B-scale unit")
    elif os.environ.get("GEMMINI_BCOLENC"):
        # real codes; B (col/weight) scale encodes col%8, A scale unit. Isolates a column-scale mis-index.
        for j in range(B_scales.shape[1]):
            B_scales[:, j] = 120 + (j % 8)
        A_scales = np.full_like(A_scales, 127)
        if verbose: print("  [GEMMINI_BCOLENC] B-scale=120+(col%8), A-scale unit")
    elif os.environ.get("GEMMINI_UNIFSCALE"):
        # real codes, but ALL block scales the same non-unit value (E8M0 128 = 2.0). Distinguishes a
        # scale-index/ordering bug (uniform scale -> wrong index reads same value -> passes) from a
        # magnitude/exponent bug (still fails).
        A_scales = np.full_like(A_scales, 128)
        B_scales = np.full_like(B_scales, 128)
        if verbose: print("  [GEMMINI_UNIFSCALE] real codes, all block scales = E8M0 128 (2.0), uniform")

    # The mesh model takes the code VALUES and the block scales separately, exactly as the
    # datapath does (raw code products accumulate first, scales apply afterwards).
    mm = __import__(f.model)
    mm.TILE = shape.dim                                   # per-tile accumulation depth = mesh DIM (fp8 model)
    # Nibble models (fp4) tile the quad output 2*DIM x 2*DIM with K-depth = DIM. The module ships the
    # DIM=16 constants (32/32/16); rescale to the mesh DIM so dim=32 uses 64x64x32 (matches the
    # prod/acc precision-list length = DIM asserted in matmul_outer_quantized_hwlike).
    if hasattr(mm, "TILE_K"):
        mm.TILE_K = shape.dim
        mm.TILE_M = 2 * shape.dim
        mm.TILE_N = 2 * shape.dim
    prod_prec, acc_prec = precision_for_dim(shape.dim)
    C_bf16 = mm.tiled_matmul_hwlike(
        torch.from_numpy(A_P), torch.from_numpy(B_P),
        torch.from_numpy(e8m0_decode(A_scales).astype(np.float32)),       # [M][GK]
        torch.from_numpy(e8m0_decode(B_scales).astype(np.float32)),       # [GK][N]
        verbose=False, prod_precision_list=prod_prec,
        acc_precision_list=acc_prec,
    ).numpy().astype(np.float32)

    if not np.isfinite(C_bf16).all():
        raise SystemExit(f"{shape.header}: mesh model produced non-finite output -- operands "
                         "exceed the accumulator bound, pick a different (layer, proj)")

    # Requantize that exact BF16 tile. axis="row" is the requantizer's own layout: one E8M0 byte
    # per row per 32 output columns, i.e. [M][N/32].
    C_codes, C_scales, _ = requant(C_bf16, f)
    if verbose:
        mask = (1 << (f.bits - 1)) - 1
        print(f"  C_bf16 |max|={np.abs(C_bf16).max():.6g}   requant peak code "
              f"0x{int(np.max(C_codes & mask)):02X}, E8M0 {int(C_scales.min())}.."
              f"{int(C_scales.max())}")

    return dict(A_codes=A_codes, B_codes=B_codes,
                A_scales=A_scales, B_scales=B_scales,
                C_bf16=C_bf16, C_codes=C_codes, C_scales=C_scales)


def emit(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    f = FORMATS[shape.fmt]
    path = OUT_DIR / shape.header
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by mx/golden/gen_plain.py -- do not edit by hand.
//
// Random {f.name} operands (block 32, E8M0 scales). C_out_bf16 = bit-exact mesh model;
// C_out / C_scales_out = hardware requantizer on that BF16 tile, scales in its [M][N/32] layout.
#ifndef {shape.guard}
#define {shape.guard}

#include <stdint.h>

#define MATMUL_M {shape.M}
#define MATMUL_K {shape.K}
#define MATMUL_N {shape.N}
#define MATMUL_GK {shape.GK}
#define MATMUL_GN {shape.GN}

// Input precision: {f.name}
static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{
{_rows(d['A_codes'], 2)}
}};

static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{
{_rows(d['B_codes'], 2)}
}};

// A's scale is per row per K-group: a_off = group * M + row
static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{
{_rows(d['A_scales'].T, 2)}
}};

// B's scale is per column per K-group: b_off = group * N + col
static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(d['B_scales'], 2)}
}};

// Non-requant output: BF16 bit patterns.
static const uint16_t C_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(d['C_bf16']), 4)}
}};

// Requant output: {f.name} codes, and the E8M0 block scales in the layout the requantizer WRITES
// (one byte per row per 32 output columns).
static const uint8_t C_out[MATMUL_M][MATMUL_N] = {{
{_rows(d['C_codes'], 2)}
}};

static const uint8_t C_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(d['C_scales'], 2)}
}};

// Transpose of C_scales_out, kept for tests written against the older [GN][M] declaration.
static const uint8_t C_scales_row[MATMUL_GN][MATMUL_M] = {{
{_rows(d['C_scales'].T, 2)}
}};

#endif // {shape.guard}
""")
    return path


def emit_fp4(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    """FP4 header: nibble-packed operands in the layouts the FP4 datapath reads.

    A and C go into the HW-tiled layout (`lut_golden_model._a_indices_to_hw_layout`): within each
    32x16 block, two m-rows share a byte, low nibble = even row. B is packed along N, low nibble =
    even column. Same packers `fp4_matmul_model.write_c_header_fp4_direct` used, so only the DATA
    changes.
    """
    from lut_golden_model import _a_indices_to_hw_layout
    f = FORMATS[shape.fmt]
    M, K, N = shape.M, shape.K, shape.N

    A_hw = np.array(_a_indices_to_hw_layout(d["A_codes"].tolist(), f.tile_m, f.tile_k),
                    dtype=np.uint8)                                    # [M/2][K]
    B = d["B_codes"]
    B_packed = ((B[:, 1::2].astype(np.uint16) << 4) | B[:, 0::2]).astype(np.uint8)   # [K][N/2]
    C_hw = np.array(_a_indices_to_hw_layout(d["C_codes"].tolist(), f.tile_m, f.tile_k),
                    dtype=np.uint8)                                    # [M/2][N]

    path = OUT_DIR / shape.header
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by mx/golden/gen_plain.py -- do not edit by hand.
//
// Random {f.name} operands (block 32, E8M0 scales). C_out_bf16 = bit-exact mesh model;
// C_out / C_scales_out = hardware requantizer on that BF16 tile.
#ifndef {shape.guard}
#define {shape.guard}

#include <stdint.h>

#define MATMUL_M   {M}
#define MATMUL_K   {K}
#define MATMUL_N   {N}
#define MATMUL_GK  {shape.GK}
#define MATMUL_GN  {shape.GN}
#define A_TILE_M   {f.tile_m}
#define K_TILE     {f.tile_k}

// Input precision: {f.name} (direct 4-bit FP4 codes, no LUT)
// A stored in HW tiled layout: per (A_TILE_M x K_TILE) block,
//   pairs of m-rows interleaved per k, then adjacent column pairs nibble-packed.
//   Dimensions: [M/2][K] = [{M // 2}][{K}]
//   byte layout: bits[7:4]=fp4(row 2r+1, k), bits[3:0]=fp4(row 2r, k)
static const uint8_t A_in_hw[{M // 2}][{K}] = {{
{_rows(A_hw, 2)}
}};

// B stored in standard layout: odd-col nibble in high bits, even-col in low bits
//   Dimensions: [K][N/2] = [{K}][{N // 2}]
static const uint8_t B_in[{K}][{N // 2}] = {{
{_rows(B_packed, 2)}
}};

// Per-row per-32-K-group A scales in fpe8m0
static const uint8_t A_scales_row[{shape.GK}][{M}] = {{
{_rows(d['A_scales'].T, 2)}
}};

// Per-col per-32-K-group B scales in fpe8m0
static const uint8_t B_scales_col[{shape.GK}][{N}] = {{
{_rows(d['B_scales'], 2)}
}};

// Final output requantized to {f.name} (direct 4-bit codes)
// HW tiled layout: [M/2][N] = [{M // 2}][{N}]
//   byte layout: bits[7:4]=fp4(row 2r+1, n), bits[3:0]=fp4(row 2r, n)
static const uint8_t C_out[{M // 2}][{N}] = {{
{_rows(C_hw, 2)}
}};

// Per-row per-32-N-group C output scales in fpe8m0, as the requantizer WRITES them [M][GN]
static const uint8_t C_scales_out[{M}][{shape.GN}] = {{
{_rows(d['C_scales'], 2)}
}};

// Transpose of C_scales_out, kept for tests written against the older [GN][M] declaration.
static const uint8_t C_scales_row[{shape.GN}][{M}] = {{
{_rows(d['C_scales'].T, 2)}
}};

// Final output (scaled+accumulated), bf16
static const uint16_t C_out_bf16[{M}][{N}] = {{
{_rows(bf16_bits(d['C_bf16']), 4)}
}};

#endif // {shape.guard}
""")
    return path


EMITTERS = {"fp8": emit, "fp4": emit_fp4}


# --- chained (back-to-back) matmul --------------------------------------------------------------
#
# C2 = (A1 @ B1) @ B2, all fp8. MM1's requantized fp8 output C1 stays RESIDENT in the scratchpad
# and is fed to MM2 as operand A -- exactly as the hardware reads it: the fp8 code VALUES multiply
# in the mesh and C1's block scales apply afterwards. MM1's output block-scales are reused as MM2's
# input A-scales. This is the golden for matmul_tiled_fp8_64x64_chain.c (Step 2 of the standalone
# plan): C1_out/C1_scales_out check residency, C2_out/C2_scales_out check the final chained result.


def _run_mesh(A_P: np.ndarray, A_scales: np.ndarray,
              B_P: np.ndarray, B_scales: np.ndarray, f: Format, dim: int = 16) -> np.ndarray:
    """A_P @ B_P through the bit-exact mesh model at mesh DIM, block scales applied after the raw products."""
    mm = __import__(f.model)
    mm.TILE = dim
    if hasattr(mm, "TILE_K"):
        mm.TILE_K, mm.TILE_M, mm.TILE_N = dim, 2 * dim, 2 * dim
    prod_prec, acc_prec = precision_for_dim(dim)
    C = mm.tiled_matmul_hwlike(
        torch.from_numpy(A_P), torch.from_numpy(B_P),
        torch.from_numpy(e8m0_decode(A_scales).astype(np.float32)),
        torch.from_numpy(e8m0_decode(B_scales).astype(np.float32)),
        verbose=False, prod_precision_list=prod_prec, acc_precision_list=acc_prec,
    ).numpy().astype(np.float32)
    if not np.isfinite(C).all():
        raise SystemExit("chain: mesh model produced non-finite output -- operands exceed the "
                         "accumulator bound, pick a different (layer, proj) or B2 slice")
    return C


def chain_operands_from_header(path: Path) -> dict[str, np.ndarray]:
    """fp8 chain operands (codes + scales) parsed from an existing chain header, to re-golden at another DIM."""
    import re
    txt = Path(path).read_text()
    def arr(name):
        m = re.search(rf"\b{name}\[[^\]]*\]\[[^\]]*\](?: __attribute__\(\(aligned\(\d+\)\)\))? = \{{(.*?)\n\}};", txt, re.S)
        if not m: raise SystemExit(f"{path}: no array {name}")
        return np.array([[int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", row)]
                         for row in re.findall(r"\{([^{}]*)\}", m.group(1))], dtype=np.uint8)
    d = dict(A_codes=arr("A_in"), A_scales=arr("A_scales_row").T.copy(), B_codes=arr("B_in"),
             B_scales=arr("B_scales_col"))
    if "B2_in[" in txt:
        d.update(B2_codes=arr("B2_in"), B2_scales=arr("B2_scales_col"))
    return d


def operand_values(f: Format, codes: np.ndarray, scales: np.ndarray, axis: str) -> np.ndarray:
    """Real values of header operands: decoded code * 2^(E8M0 - 127), scales per 32-block along K."""
    e_bits, m_bits = __import__(f.model).parse_fp_spec(f.name)
    s = (2.0 ** (scales.astype(np.float64) - 127)).astype(np.float32)
    s = np.repeat(s, BLOCK, axis=1 if axis == "row" else 0)
    return _decode_codes(codes, e_bits, m_bits) * s


def build_chain(shape: Shape, verbose: bool = True, operands: dict | None = None) -> dict[str, np.ndarray]:
    """MM1 = A1@B1 -> C1 (resident); MM2 = C1@B2 -> C2. Returns all header arrays. Any MX format.
    operands: A/B/B2 codes + scales to use instead of random ones (from --operands)."""
    f = FORMATS[shape.fmt]

    if operands is not None:
        e_bits, m_bits = __import__(f.model).parse_fp_spec(f.name)
        A1_codes, A1_scales, B1_codes, B1_scales = (operands[k] for k in ("A_codes", "A_scales", "B_codes", "B_scales"))
        B2_codes, B2_scales = operands["B2_codes"], operands["B2_scales"]
        A1_P, B1_P, B2_P = (_decode_codes(c, e_bits, m_bits) for c in (A1_codes, B1_codes, B2_codes))
        if verbose: print(f"  operands from header: A1{A1_codes.shape} B1{B1_codes.shape} B2{B2_codes.shape}")
    else:
        sd = _seed("chain_" + shape.header)
        A1_codes, A1_scales, A1_P = random_operand(shape.M, shape.K, f, "row", sd)       # [M][GK]
        B1_codes, B1_scales, B1_P = random_operand(shape.K, shape.N, f, "col", sd + 1)   # [GK][N]
        B2_codes, B2_scales, B2_P = random_operand(shape.N, shape.N, f, "col", sd + 2)   # MM2 K = N

    # MM1 -------------------------------------------------------------------------------------
    C1_bf16 = _run_mesh(A1_P, A1_scales, B1_P, B1_scales, f, shape.dim)
    C1_codes, C1_scales, C1_P = requant(C1_bf16, f)           # C1_scales [M][N/32]

    # MM2: C1 (resident) @ B2. A2 fed exactly as the mesh reads C1 ---------------------------
    # A2_P = C1's code values; A2_scales = C1's output block scales ([M][N1/32] == [M][K2/32]).
    C2_bf16 = _run_mesh(C1_P, C1_scales, B2_P, B2_scales, f, shape.dim)
    C2_codes, C2_scales, C2_P = requant(C2_bf16, f)

    if verbose:
        mask = (1 << (f.bits - 1)) - 1
        print(f"  C1 |max|={np.abs(C1_bf16).max():.6g} peak 0x{int(np.max(C1_codes & mask)):02X} "
              f"E8M0 {int(C1_scales.min())}..{int(C1_scales.max())}")
        print(f"  C2 |max|={np.abs(C2_bf16).max():.6g} peak 0x{int(np.max(C2_codes & mask)):02X} "
              f"E8M0 {int(C2_scales.min())}..{int(C2_scales.max())}")

    return dict(A_codes=A1_codes, A_scales=A1_scales, B_codes=B1_codes, B_scales=B1_scales,
                B2_codes=B2_codes, B2_scales=B2_scales,
                C1_codes=C1_codes, C1_scales=C1_scales, C1_bf16=C1_bf16,
                C2_codes=C2_codes, C2_scales=C2_scales, C2_bf16=C2_bf16)


def emit_chain(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    """Emit the chained-matmul header, dispatching on element width (fp8 byte vs fp4/fp6 nibble)."""
    return _emit_chain_fp8(shape, d) if FORMATS[shape.fmt].bits == 8 else _emit_chain_nibble(shape, d)


def _emit_chain_fp8(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    """Emit include/matmul_fp8_<M>x<N>_chain.h -- MM1 operands, B2, and both C1 and C2 goldens."""
    f = FORMATS[shape.fmt]
    stem = f"matmul_fp8_{shape.M}x{shape.N}_chain" + ("" if shape.dim == 16 else f"_dim{shape.dim}")
    guard = f"INCLUDE_{stem.upper()}_H"
    path = OUT_DIR / f"{stem}.h"
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by mx/golden/gen_plain.py chain_fp8_{shape.M}x{shape.N}{"" if shape.dim == 16 else f"_dim{shape.dim}"} -- do not edit by hand.
//
// CHAINED fp8 matmul: C2 = (A1 @ B1) @ B2, all {f.name}, random operands. MM1's requantized output C1
// stays RESIDENT in the scratchpad as MM2's operand A; its block scales are MM2's A-scales.
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define MATMUL_M {shape.M}
#define MATMUL_K {shape.K}
#define MATMUL_N {shape.N}
#define MATMUL_GK {shape.GK}
#define MATMUL_GN {shape.GN}

// ---- MM1 operands (A1 @ B1) ----
// Input precision: {f.name}
static const uint8_t A_in[MATMUL_M][MATMUL_K] = {{
{_rows(d['A_codes'], 2)}
}};

static const uint8_t B_in[MATMUL_K][MATMUL_N] = {{
{_rows(d['B_codes'], 2)}
}};

// A's scale is per row per K-group: a_off = group * M + row
static const uint8_t A_scales_row[MATMUL_GK][MATMUL_M] = {{
{_rows(d['A_scales'].T, 2)}
}};

// B's scale is per column per K-group: b_off = group * N + col
static const uint8_t B_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(d['B_scales'], 2)}
}};

// ---- MM2 weight operand (C1 @ B2); C1 comes from the scratchpad, not DRAM ----
static const uint8_t B2_in[MATMUL_K][MATMUL_N] = {{
{_rows(d['B2_codes'], 2)}
}};

static const uint8_t B2_scales_col[MATMUL_GK][MATMUL_N] = {{
{_rows(d['B2_scales'], 2)}
}};

// ---- MM1 output C1 = requant(A1 @ B1): the RESIDENT operand + residency check ----
// C1 fp8 codes (what mvout of the scratchpad must reproduce).
static const uint8_t C1_out[MATMUL_M][MATMUL_N] = {{
{_rows(d['C1_codes'], 2)}
}};

// C1 block scales in the requantizer's [M][N/32] output layout. These ARE MM2's A-scales:
// C1_scales_out[m][b] == A2's per-row per-K-group scale (N1/32 == K2/32).
static const uint8_t C1_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(d['C1_scales'], 2)}
}};

static const uint16_t C1_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(d['C1_bf16']), 4)}
}};

// ---- MM2 output C2 = requant(C1 @ B2): the final chained result ----
static const uint8_t C2_out[MATMUL_M][MATMUL_N] = {{
{_rows(d['C2_codes'], 2)}
}};

static const uint8_t C2_scales_out[MATMUL_M][MATMUL_GN] = {{
{_rows(d['C2_scales'], 2)}
}};

static const uint16_t C2_out_bf16[MATMUL_M][MATMUL_N] = {{
{_rows(bf16_bits(d['C2_bf16']), 4)}
}};

#endif // {guard}
""")
    return path


def _emit_chain_nibble(shape: Shape, d: dict[str, np.ndarray]) -> Path:
    """Emit include/matmul_<fp4|fp6>_<M>x<N>_chain.h -- nibble-packed chained matmul.

    A and C go into the HW-tiled layout (`_a_indices_to_hw_layout`, 2 m-rows per byte, low nibble =
    even row); B and B2 are packed along N (low nibble = even column). Same packers as emit_fp4, so
    only the data differs. C1_out is the RESIDENT operand + residency check; C2_out the final result.
    Reusing BOTH C1 codes and C1 scales preserves the value, so the pmax convention is irrelevant.
    """
    from lut_golden_model import _a_indices_to_hw_layout
    f = FORMATS[shape.fmt]
    M, K, N = shape.M, shape.K, shape.N
    fmt = f.name.split(":")[0]                       # fp4 / fp6
    stem = f"matmul_{fmt}_{M}x{N}_chain"
    guard = f"INCLUDE_{stem.upper()}_H"

    def hw(codes):
        return np.array(_a_indices_to_hw_layout(codes.tolist(), f.tile_m, f.tile_k), dtype=np.uint8)

    def pack_b(codes):                               # [K][N] indices -> [K][N/2] nibble-packed
        return ((codes[:, 1::2].astype(np.uint16) << 4) | codes[:, 0::2]).astype(np.uint8)

    A_hw   = hw(d["A_codes"])                         # [M/2][K]
    B_pk   = pack_b(d["B_codes"])                     # [K][N/2]
    B2_pk  = pack_b(d["B2_codes"])                    # [K][N/2]
    C1_hw  = hw(d["C1_codes"])                        # [M/2][N]
    C2_hw  = hw(d["C2_codes"])                        # [M/2][N]

    path = OUT_DIR / f"{stem}.h"
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by mx/golden/gen_plain.py chain_{fmt}_{M}x{N} -- do not edit by hand.
//
// CHAINED {f.name} matmul: C2 = (A1 @ B1) @ B2, random operands. MM1's requantized output C1 stays
// RESIDENT in the scratchpad (block-tiled) as MM2's operand A; its block scales are MM2's A-scales.
// A and C are HW-tiled (2 m-rows/byte, low nibble = even row); B is packed along N.
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define MATMUL_M   {M}
#define MATMUL_K   {K}
#define MATMUL_N   {N}
#define MATMUL_GK  {shape.GK}
#define MATMUL_GN  {shape.GN}
#define A_TILE_M   {f.tile_m}
#define K_TILE     {f.tile_k}

// ---- MM1 operands (A1 @ B1) ----
// A in HW tiled layout [M/2][K]: bits[7:4]=code(row 2r+1,k), bits[3:0]=code(row 2r,k)
static const uint8_t A_in_hw[{M // 2}][{K}] = {{
{_rows(A_hw, 2)}
}};

// B packed [K][N/2]: odd-col nibble high, even-col low
static const uint8_t B_in[{K}][{N // 2}] = {{
{_rows(B_pk, 2)}
}};

static const uint8_t A_scales_row[{shape.GK}][{M}] = {{
{_rows(d['A_scales'].T, 2)}
}};

static const uint8_t B_scales_col[{shape.GK}][{N}] = {{
{_rows(d['B_scales'], 2)}
}};

// ---- MM2 weight operand (C1 @ B2); C1 comes from the scratchpad, not DRAM ----
static const uint8_t B2_in[{K}][{N // 2}] = {{
{_rows(B2_pk, 2)}
}};

static const uint8_t B2_scales_col[{shape.GK}][{N}] = {{
{_rows(d['B2_scales'], 2)}
}};

// ---- MM1 output C1 = requant(A1 @ B1): the RESIDENT operand + residency check (HW tiled [M/2][N]) ----
static const uint8_t C1_out[{M // 2}][{N}] = {{
{_rows(C1_hw, 2)}
}};

// C1 block scales [M][GN] (as the requantizer writes them). These ARE MM2's A-scales.
static const uint8_t C1_scales_out[{M}][{shape.GN}] = {{
{_rows(d['C1_scales'], 2)}
}};

static const uint16_t C1_out_bf16[{M}][{N}] = {{
{_rows(bf16_bits(d['C1_bf16']), 4)}
}};

// ---- MM2 output C2 = requant(C1 @ B2): the final chained result (HW tiled [M/2][N]) ----
static const uint8_t C2_out[{M // 2}][{N}] = {{
{_rows(C2_hw, 2)}
}};

static const uint8_t C2_scales_out[{M}][{shape.GN}] = {{
{_rows(d['C2_scales'], 2)}
}};

static const uint16_t C2_out_bf16[{M}][{N}] = {{
{_rows(bf16_bits(d['C2_bf16']), 4)}
}};

#endif // {guard}
""")
    return path


def load_operands(path: Path) -> dict[str, np.ndarray]:
    """A/B(/B2) codes + scales from an .npz (keys A_codes, A_scales, B_codes, B_scales[, B2_*]) or a chain header."""
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    return chain_operands_from_header(path)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Generate plain-format matmul data headers.")
    ap.add_argument("names", nargs="*", help=f"shapes ({', '.join(sorted(BY_NAME))}) or chain_<shape>; default: all")
    ap.add_argument("--operands", type=Path, help="use these codes + scales instead of random ones")
    a = ap.parse_args()
    operands = load_operands(a.operands) if a.operands else None
    names = a.names or list(BY_NAME) + [f"chain_{c}" for c in CHAINS]
    for name in names:
        key = name.removeprefix("chain_")
        if key not in BY_NAME:
            raise SystemExit(f"unknown shape {key}; known: {sorted(BY_NAME)}")
        shape, t0 = BY_NAME[key], time.time()
        if name.startswith("chain_"):
            p = emit_chain(shape, build_chain(shape, verbose=False, operands=operands))
        else:
            p = EMITTERS[shape.fmt](shape, build(shape, verbose=False, operands=operands))
        print(f"wrote {p}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

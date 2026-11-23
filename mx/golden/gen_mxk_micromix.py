#!/usr/bin/env python3
"""Data + golden for src/mxk_micromix.c: a MicroMix linear layer (Liu et al., ICLR 2026) on MxGemmini.

  C[M][N] (BF16) = sum over channel groups g of A_g[M][K_g] @ W_g[K_g][N]
The input channels are already permuted into contiguous groups, each one MX format on BOTH operands:
    FP4: MXFP4 E2M1 x E2M1 (mode0; A row pairs and W column pairs packed in nibbles, as mxn_matmul_fp4)
    FP6: MXFP6 E3M2 x E3M2, direct (uselut = 0: one 6-bit code per byte, run as E4M3-single on the shared acc layout)
    FP8: MXFP8 E4M3 x E4M3 (mode8 on the shared acc layout)
All groups accumulate into one accumulator in the order FP4, FP6, FP8 (the kernel's issue order); scales are E8M0 per
32 channels: AS [K/32][M], BS [K/32][N]. --groups gives K-blocks (32 channels) per format, e.g. "4:4,6:2,8:2".

    python3 gen_mxk_micromix.py --m 64 --k 256 --n 64 --groups 4:4,6:2,8:2 --tag tiny468
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from gen_lutk_proj import DATA, GROUP, FM, e4m3_grid, fp4_grid, golden, nearest, nib
from gen_mxk_kblock import e3m2_grid

ORDER = (4, 6, 8)


def scale_codes(fmt, base, shape, gen):
    """Random block-scale codes in fmt (0/1 = E8M0, X = 2..7 = eXmY above base) and their values."""
    if fmt <= 1:
        e = torch.randint(-4, 4, shape, generator=gen)
        return (e + 127).numpy().astype(np.uint8), torch.pow(2.0, e.float())
    y = 8 - fmt
    mid = 1 << (fmt - 1)
    e = torch.clamp(torch.randint(mid - 4, mid + 4, shape, generator=gen), 0, (1 << fmt) - 1)
    m = torch.randint(0, 1 << y, shape, generator=gen)
    vals = torch.pow(2.0, (base + e).float()) * (1 + m.float() / (1 << y))
    return ((e << y) | m).numpy().astype(np.uint8), vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--groups", required=True, help='K-blocks per format, "fmt:nkb,..." with fmt in 4/6/8')
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dims", default="32")
    ap.add_argument("--amp", type=float, default=1.0, help="multiply A and W before quantizing (large-value tests)")
    ap.add_argument("--afmt", type=int, default=0, help="A scale format: 0 = E8M0, X = 2..7 = eXmY (MX_FLEX_SCALES)")
    ap.add_argument("--abase", type=int, default=0)
    ap.add_argument("--wfmt", type=int, default=0, help="W scale format, as --afmt")
    ap.add_argument("--wbase", type=int, default=0)
    a = ap.parse_args()
    flex = a.afmt > 1 or a.wfmt > 1
    M, N = a.m, a.n
    grp = {f: 0 for f in ORDER}
    for item in a.groups.split(","):
        f, n = (int(x) for x in item.split(":"))
        assert f in grp, f"format {f}: want one of {ORDER}"
        grp[f] += n
    NKB = sum(grp.values())
    K = NKB * GROUP
    assert N % 64 == 0 and M % 64 == 0 and NKB > 0
    dims = [int(d) for d in a.dims.split(",")]

    torch.manual_seed(a.seed)
    Af = torch.randn(M, K) * a.amp
    Wf = torch.randn(K, N) * a.amp
    grids = {4: (None, fp4_grid()), 6: e3m2_grid(), 8: e4m3_grid()}
    A_fp, B_fp = torch.zeros(M, K), torch.zeros(K, N)
    parts, h_groups, kb = [], [], 0
    for f in ORDER:
        if not grp[f]:
            continue
        k0, k1 = kb * GROUP, (kb + grp[f]) * GROUP
        codes, vals = grids[f]
        ai, wi = nearest(Af[:, k0:k1], vals), nearest(Wf[k0:k1], vals)
        A_fp[:, k0:k1], B_fp[k0:k1] = vals[ai], vals[wi]
        if f == 4:   # A [M/2][Kg]: byte (r, k) = row 2r (low nibble) | row 2r+1 (high); W [Kg][N/2] as lutk's nib()
            an = ai.numpy().astype(np.uint8)
            A_g = (an[0::2] | (an[1::2] << 4)).astype(np.uint8)
            B_g = nib(wi.numpy())
        else:
            A_g, B_g = codes[ai.numpy()], codes[wi.numpy()]
        parts += [(f"A{f}", A_g), (f"B{f}", B_g)]
        h_groups.append((f, kb, grp[f]))
        kb += grp[f]
    torch.manual_seed(a.seed + 123)
    if not flex:
        As_e = torch.randint(-4, 4, (M, NKB))
        Bs_e = torch.randint(-4, 4, (NKB, N))
        As, Bs = torch.pow(2.0, As_e.float()), torch.pow(2.0, Bs_e.float())
        parts += [("AS", (As_e.T + 127).numpy().astype(np.uint8)), ("BS", (Bs_e + 127).numpy().astype(np.uint8))]
    else:
        g = torch.Generator().manual_seed(a.seed + 321)
        As_c, As = scale_codes(a.afmt, a.abase, (M, NKB), g)
        Bs_c, Bs = scale_codes(a.wfmt, a.wbase, (NKB, N), g)
        parts += [("AS", np.ascontiguousarray(As_c.T)), ("BS", Bs_c)]
        # sA * sW stays exact (7b x 7b mantissas), as in the RTL: no E8M0 rounding of the combined scale
        FM.compute_tile_scale_matrix = lambda Ar, Bc, m0, n0, k0, TM, TN: torch.outer(
            Ar[m0:m0 + TM, k0 // GROUP].float(), Bc[k0 // GROUP, n0:n0 + TN].float())
    for d in dims:   # one BF16 accumulation over the K-blocks in issue order (FP4, FP6, FP8 = K order here)
        parts.append((f"C_D{d}", golden(A_fp, B_fp, As, Bs, list(range(NKB)), d)))

    buf, offs = bytearray(), {}
    for n, x in parts:
        buf += bytes((-len(buf)) % 64)
        offs[n] = len(buf)
        buf += np.ascontiguousarray(x).tobytes()
    base = f"mxk_micromix_{a.tag}"
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{base}.bin").write_bytes(bytes(buf))
    sym = f"_binary_{base}_bin_start"
    h = [f"// generated by gen/gen_mxk_micromix.py --m {M} --n {N} --groups {a.groups} --tag {a.tag} --seed {a.seed}",
         f"#ifndef MXK_MICROMIX_{a.tag.upper()}_H", f"#define MXK_MICROMIX_{a.tag.upper()}_H", "#include <stdint.h>",
         f"#define MM_M {M}", f"#define MM_K {K}", f"#define MM_N {N}", f"#define MM_NKB {NKB}",
         f"extern const uint8_t {sym}[];", f"#define MM_BLOB {sym}"]
    if flex:
        h += [f"#define MM_SFMT_A {a.afmt}", f"#define MM_SBASE_A {a.abase}", f"#define MM_SFMT_W {a.wfmt}",
              f"#define MM_SBASE_W {a.wbase}"]
    for f in ORDER:   # MM_KB<f>: first K-block of the group, MM_NB<f>: its K-blocks (0 = format absent)
        g = next((x for x in h_groups if x[0] == f), (f, 0, 0))
        h += [f"#define MM_KB{f} {g[1]}", f"#define MM_NB{f} {g[2]}"]
    for n, o in offs.items():
        h.append(f"#define MM_OFF_{n} {o}")
    h += ["#endif", ""]
    (DATA / f"{base}.h").write_text("\n".join(h))
    print(f"wrote data/{base}.bin ({len(buf)} B) and data/{base}.h; groups "
          + ", ".join(f"FP{f} x{n} K-blocks @kb{k}" for f, k, n in h_groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

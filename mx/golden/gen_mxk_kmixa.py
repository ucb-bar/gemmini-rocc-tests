#!/usr/bin/env python3
"""Data + golden for src/mxk_kmixa.c: ONE mixed-K LOOP_WS whose ACTIVATION format changes per K-block
(LOOP_WS_CONFIG_KMIX_A). --pattern gives each K-block (32 channels): 'L' = the loop's own mode, 'P' = plain.

  --base 44: loop = MXFP4 x MXFP4 (mode0); plain = E4M3 x E4M3 (or E3M2 direct x E3M2 direct with --plain 6)
             -> KMIX (B) + KMIX_A (A) both switch
  --base 48: loop = MXFP4 A x E4M3-single W (mode2); plain = E4M3 x E4M3 (W the same array: only A switches)
Arrays are full-K and indexed by the global K-block (unused blocks zero):
  A4 [M/2][K] nibbles (byte (r, k) = row 2r low | row 2r+1 high), AP [M][K] bytes (plain A),
  base 44: B4 [K][N/2] nibbles + BP [K][N] (plain B);  base 48: B8 [K][N] (every K-block).
Golden = one BF16 accumulation over the K-blocks in K order (the loop's issue order).

    python3 gen_mxk_kmixa.py --base 44 --pattern LPLPPLLP --tag a44
"""
from __future__ import annotations

import argparse

import precision

import numpy as np
import torch

from gen_lutk_proj import DATA, GROUP, FM, e4m3_grid, fp4_grid, golden, nearest, nib
from gen_mxk_kblock import e3m2_grid
from gen_mxk_micromix import scale_codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--base", type=int, choices=(44, 48), required=True)
    ap.add_argument("--plain", type=int, choices=(6, 8), default=8, help="plain K-block format (6 = E3M2 direct)")
    ap.add_argument("--pattern", required=True, help="per K-block: L = loop mode, P = plain")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dims", default="16,32")
    ap.add_argument("--afmt", type=int, default=0, help="A scale format (0 = E8M0, X = 2..7 eXmY)")
    ap.add_argument("--abase", type=int, default=0)
    ap.add_argument("--wfmt", type=int, default=0)
    ap.add_argument("--wbase", type=int, default=0)
    precision.add_arg(ap, many=True)
    a = ap.parse_args()
    precision.load(a.params)
    assert a.base == 44 or a.plain == 8, "base 48: plain W is the loop's E4M3 W"
    M, N, pat = a.m, a.n, a.pattern.upper()
    assert set(pat) <= {"L", "P"} and len(pat) <= 128 and M % 64 == 0 and N % 64 == 0
    NKB = len(pat)
    K = NKB * GROUP
    flex = a.afmt > 1 or a.wfmt > 1

    torch.manual_seed(a.seed)
    Af, Wf = torch.randn(M, K), torch.randn(K, N)
    v4 = fp4_grid()
    c8, v8 = e4m3_grid()
    cp, vp = e3m2_grid() if a.plain == 6 else (c8, v8)
    A_fp, B_fp = torch.zeros(M, K), torch.zeros(K, N)
    A4 = np.zeros((M // 2, K), np.uint8)
    AP = np.zeros((M, K), np.uint8)
    B4 = np.zeros((K, N // 2), np.uint8)
    BP = np.zeros((K, N), np.uint8)
    for kb, t in enumerate(pat):
        ks = slice(kb * GROUP, (kb + 1) * GROUP)
        if t == "L":
            ai = nearest(Af[:, ks], v4)
            an = ai.numpy().astype(np.uint8)
            A4[:, ks] = an[0::2] | (an[1::2] << 4)
            A_fp[:, ks] = v4[ai]
            if a.base == 44:
                wi = nearest(Wf[ks], v4)
                B4[ks] = nib(wi.numpy())
                B_fp[ks] = v4[wi]
            else:
                wi = nearest(Wf[ks], v8)
                BP[ks] = c8[wi.numpy()]
                B_fp[ks] = v8[wi]
        else:
            ai, wi = nearest(Af[:, ks], vp), nearest(Wf[ks], vp)
            AP[:, ks], BP[ks] = cp[ai.numpy()], cp[wi.numpy()]
            A_fp[:, ks], B_fp[ks] = vp[ai], vp[wi]
    parts = [("A4", A4), ("AP", AP)] + ([("B4", B4), ("BP", BP)] if a.base == 44 else [("BP", BP)])

    torch.manual_seed(a.seed + 123)
    if not flex:
        As_e, Bs_e = torch.randint(-4, 4, (M, NKB)), torch.randint(-4, 4, (NKB, N))
        As, Bs = torch.pow(2.0, As_e.float()), torch.pow(2.0, Bs_e.float())
        parts += [("AS", (As_e.T + 127).numpy().astype(np.uint8)), ("BS", (Bs_e + 127).numpy().astype(np.uint8))]
    else:
        g = torch.Generator().manual_seed(a.seed + 321)
        As_c, As = scale_codes(a.afmt, a.abase, (M, NKB), g)
        Bs_c, Bs = scale_codes(a.wfmt, a.wbase, (NKB, N), g)
        parts += [("AS", np.ascontiguousarray(As_c.T)), ("BS", Bs_c)]
        FM.compute_tile_scale_matrix = lambda Ar, Bc, m0, n0, k0, TM, TN: torch.outer(
            Ar[m0:m0 + TM, k0 // GROUP].float(), Bc[k0 // GROUP, n0:n0 + TN].float())
    for d in (int(x) for x in a.dims.split(",")):
        parts.append((f"C_D{d}", golden(A_fp, B_fp, As, Bs, list(range(NKB)), d)))

    buf, offs = bytearray(), {}
    for n, x in parts:
        buf += bytes((-len(buf)) % 64)
        offs[n] = len(buf)
        buf += np.ascontiguousarray(x).tobytes()
    base = f"mxk_kmixa_{a.tag}"
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{base}.bin").write_bytes(bytes(buf))
    sym = f"_binary_{base}_bin_start"
    mlo = sum(1 << i for i, t in enumerate(pat[:64]) if t == "L")
    mhi = sum(1 << i for i, t in enumerate(pat[64:]) if t == "L")
    h = [f"// generated by gen/gen_mxk_kmixa.py --base {a.base} --plain {a.plain} --pattern {pat} --tag {a.tag}",
         f"#ifndef MXK_KMIXA_{a.tag.upper()}_H", f"#define MXK_KMIXA_{a.tag.upper()}_H", "#include <stdint.h>",
         f"#define KA_M {M}", f"#define KA_K {K}", f"#define KA_N {N}", f"#define KA_NKB {NKB}",
         f"#define KA_BASE {a.base}", f"#define KA_PLAIN {a.plain}",
         f"#define KA_MLO 0x{mlo:x}ULL", f"#define KA_MHI 0x{mhi:x}ULL", f"#define KA_NL {pat.count('L')}",
         f"extern const uint8_t {sym}[];", f"#define KA_BLOB {sym}"]
    if flex:
        h += [f"#define KA_SFMT_A {a.afmt}", f"#define KA_SBASE_A {a.abase}", f"#define KA_SFMT_W {a.wfmt}",
              f"#define KA_SBASE_W {a.wbase}"]
    h += [f"#define KA_OFF_{n} {o}" for n, o in offs.items()] + ["#endif", ""]
    (DATA / f"{base}.h").write_text("\n".join(h))
    print(f"wrote data/{base}.bin ({len(buf)} B) and data/{base}.h; base {a.base}, plain FP{a.plain}, {pat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

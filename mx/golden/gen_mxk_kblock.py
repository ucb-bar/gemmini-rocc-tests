#!/usr/bin/env python3
"""Data + golden for src/mxk_kblock.c: the per-K-block weight LUT linear layer on an MXFP6 E3M2 base.

  C[M][N] (BF16) = A[M][K] (E3M2, direct: one 6-bit code per byte) @ W[K][N]
  Each 32-channel K-block of W is either plain MXFP6 E3M2 (direct, mode8 after the E4M3 expansion) or low precision:
    lut: 4-bit indices into a 16-entry codebook of E3M2 VALUES, one per 2 output columns per K-block. The entries are
         stored E4M3-encoded (every E3M2 value is exact in E4M3), so it runs on the E4M3 LUT path (mode13).
    fp4: plain MXFP4 E2M1 on the same K-blocks (mode6).
Same layout, offsets and golden tags as gen_lutk_proj.py (whose helpers it imports): B8 / A hold E3M2 codes here.

    python3 gen_mxk_kblock.py --m 32 --k 128 --n 64 --tag tiny
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gen_lutk_proj import (DATA, DEFAULT_SHARES, GROUP, bf16_add, fp4_grid, golden, make_lut, nearest, nib,
                           tensor_to_custom_fp_codes)


def e3m2_grid():   # all 64 E3M2 codes (bias 3, subnormals at e = 0); FP6 has no Inf/NaN
    codes, vals = [], []
    for c in range(64):
        s, e, m = (c >> 5) & 1, (c >> 2) & 7, c & 3
        v = (m / 4.0) * 2.0 ** -2 if e == 0 else (1.0 + m / 4.0) * 2.0 ** (e - 3)
        codes.append(c); vals.append(-v if s else v)
    return np.array(codes, np.uint8), torch.tensor(vals, dtype=torch.float32)


def e4m3_code(v: float) -> int:   # exact E4M3 encoding of an E3M2 value
    if v == 0.0:
        return 0x80 if np.signbit(v) else 0
    s, a = (0x80 if v < 0 else 0), abs(v)
    e = int(np.floor(np.log2(a)))
    if e < -6:   # E4M3 subnormal (never reached by E3M2, whose smallest magnitude is 2^-4)
        return s | int(round(a / 2.0 ** -9))
    m = a / 2.0 ** e - 1.0
    mi = int(round(m * 8))
    assert abs(mi / 8 - m) < 1e-9, f"{v} not exact in E4M3"
    return s | ((e + 7) << 3) | mi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-file", help='JSON {"name": [0/1 per K-block]}; default: random masks at --shares')
    ap.add_argument("--dims", default="16,32")
    a = ap.parse_args()
    M, K, N = a.m, a.k, a.n
    NKB = K // GROUP
    assert K % GROUP == 0 and N % 64 == 0 and M % 32 == 0
    dims = [int(d) for d in a.dims.split(",")]

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    e6c, e6v = e3m2_grid()
    Af = torch.randn(M, K)
    Wf = torch.randn(K, N)
    ai, wi = nearest(Af, e6v), nearest(Wf, e6v)
    A_fp, A_codes = e6v[ai], e6c[ai.numpy()]
    B8_fp, B8_codes = e6v[wi], e6c[wi.numpy()]
    g4 = fp4_grid()
    w4 = nearest(Wf, g4)
    B4_fp, B4_codes = g4[w4], w4.numpy()
    # per-K-block codebooks of E3M2 values, one per 2 columns: [NKB][N/2][16], stored as E4M3 codes
    luts = torch.stack([torch.stack([make_lut("fp6:e3m2", 4, torch.device("cpu")) for _ in range(N // 2)])
                        for _ in range(NKB)])
    on_grid = (luts.reshape(-1, 1) - e6v).abs().min(dim=1).values
    assert float(on_grid.max()) == 0.0, "codebook entry off the E3M2 grid"
    kb_of = torch.arange(K) // GROUP
    cp_of = torch.arange(N) // 2
    tab = luts[kb_of][:, cp_of]                                   # [K][N][16]
    BL_idx = (Wf.unsqueeze(-1) - tab).abs().argmin(dim=-1)        # [K][N]
    BL_fp = torch.gather(tab, 2, BL_idx.unsqueeze(-1)).squeeze(-1)
    lut_codes = np.array([e4m3_code(float(v)) for v in luts.reshape(-1)], np.uint8).reshape(NKB, N // 2, 16)
    chk = np.array(tensor_to_custom_fp_codes(luts.reshape(-1, 16), "fp8:e4m3")[0], np.uint8).reshape(NKB, N // 2, 16)
    assert np.array_equal(lut_codes & 0x7F, chk & 0x7F), "E4M3 codebook encoding disagrees with lut_golden_model"
    torch.manual_seed(a.seed + 123)
    As_e = torch.randint(-4, 4, (M, NKB))
    Bs_e = torch.randint(-4, 4, (NKB, N))
    As, Bs = torch.pow(2.0, As_e.float()), torch.pow(2.0, Bs_e.float())

    if a.mask_file:
        masks = {k: [int(x) for x in v] for k, v in json.loads(Path(a.mask_file).read_text()).items()}
    else:
        masks = {}
        for name, s in DEFAULT_SHARES.items():
            m = np.zeros(NKB, np.uint8)
            m[rng.choice(NKB, int(round(s * NKB)), replace=False)] = 1
            masks[name] = m.tolist()
    for name, m in masks.items():
        assert len(m) == NKB, f"mask {name}: {len(m)} entries, want {NKB}"

    parts = [("A", A_codes), ("AS", (As_e.T + 127).numpy().astype(np.uint8)), ("B8", B8_codes),
             ("BL", nib(BL_idx.numpy())), ("BLUT", lut_codes), ("B4", nib(B4_codes)),
             ("BS", (Bs_e + 127).numpy().astype(np.uint8))]
    for vi, m in enumerate(masks.values()):   # codebooks of mask vi's LUT K-blocks in K order (LK_MIX layout, one chunk)
        parts.append((f"BLUTC{vi}", lut_codes[[i for i in range(NKB) if m[i]]]))
    for d in dims:
        allk = list(range(NKB))
        parts.append((f"FP8_D{d}", golden(A_fp, B8_fp, As, Bs, allk, d)))   # all plain MXFP6 (tag kept from lutk)
        for vi, (name, m) in enumerate(masks.items()):
            lo = [i for i in range(NKB) if m[i]]
            hi = [i for i in range(NKB) if not m[i]]
            p_hi = golden(A_fp, B8_fp, As, Bs, hi, d)
            p_l = golden(A_fp, BL_fp, As, Bs, lo, d)
            p_4 = golden(A_fp, B4_fp, As, Bs, lo, d)
            ks_lo = np.concatenate([np.arange(kb * GROUP, (kb + 1) * GROUP) for kb in lo]) if lo else np.zeros(0, int)
            bm_l, bm_4 = B8_fp.clone(), B8_fp.clone()
            bm_l[ks_lo], bm_4[ks_lo] = BL_fp[ks_lo], B4_fp[ks_lo]
            c_sl = golden(A_fp, bm_l, As, Bs, lo + hi, d)   # shared acc: low K-blocks, then MXFP6 ones
            c_s4 = golden(A_fp, bm_4, As, Bs, lo + hi, d)
            c_ml = golden(A_fp, bm_l, As, Bs, allk, d)   # mixed-K loops: one accumulation in K order
            c_m4 = golden(A_fp, bm_4, As, Bs, allk, d)
            for tag, arr in (("PHI", p_hi), ("PLUT", p_l), ("PFP4", p_4),
                             ("CLUT", bf16_add(p_l, p_hi)), ("CFP4", bf16_add(p_4, p_hi)),
                             ("CSL", c_sl), ("CS4", c_s4), ("CML", c_ml), ("CM4", c_m4)):
                parts.append((f"{tag}{vi}_D{d}", arr))
            print(f"dim{d} {name}: {len(lo)}/{NKB} K-blocks low precision")

    buf, offs = bytearray(), {}
    for n, x in parts:
        buf += bytes((-len(buf)) % 64)
        offs[n] = len(buf)
        buf += np.ascontiguousarray(x).tobytes()
    base = f"mxk_kblock6_{a.tag}"
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{base}.bin").write_bytes(bytes(buf))
    sym = f"_binary_{base}_bin_start"
    h = [f"// generated by gen/gen_mxk_kblock.py --m {M} --k {K} --n {N} --tag {a.tag} --seed {a.seed}"
         + (f" --mask-file {a.mask_file}" if a.mask_file else ""),
         f"#ifndef MXK_KBLOCK6_{a.tag.upper()}_H", f"#define MXK_KBLOCK6_{a.tag.upper()}_H", "#include <stdint.h>",
         "#define MK_DATA_FP6 1", f"#define LK_M {M}", f"#define LK_K {K}", f"#define LK_N {N}",
         f"#define LK_NKB {NKB}", f"#define LK_NVAR {len(masks)}", f"extern const uint8_t {sym}[];",
         f"#define LK_BLOB {sym}"]
    for n, o in offs.items():
        h.append(f"#define LK_OFF_{n} {o}")
    h.append(f"static const char *const lk_var_name[LK_NVAR] = {{{', '.join(json.dumps(k) for k in masks)}}};")
    h.append("static const uint8_t lk_mask[LK_NVAR][LK_NKB] = {")
    for m in masks.values():
        h.append("  {" + ", ".join(str(x) for x in m) + "},")
    h += ["};", "#endif", ""]
    (DATA / f"{base}.h").write_text("\n".join(h))
    print(f"wrote data/{base}.bin ({len(buf)} B) and data/{base}.h; variants {list(masks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

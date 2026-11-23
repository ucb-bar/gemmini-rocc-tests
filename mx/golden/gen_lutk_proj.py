#!/usr/bin/env python3
"""Data + golden for src/lutk_proj.c: a weights-only per-K-block LUT linear layer (MXQuant "32xN usage, kblock tables").

  C[M][N] (BF16) = A[M][K] (E4M3 single, E8M0 per 32) @ W[K][N]
  Each 32-channel K-block of W is either MXFP8 E4M3 (mode8) or low precision for the whole row of N:
    lut: 4-bit indices into a 16-entry E4M3 codebook, one codebook per 2 output columns per K-block (mode13)
    fp4: plain MXFP4 E2M1 on the same K-blocks (mode6)
  The two layouts cannot share an accumulator, so the kernel computes P_lo (low-precision K-blocks) and P_hi (MXFP8
  K-blocks) as separate BF16 partials and adds them on the CPU: C = bf16(f32(P_lo) + f32(P_hi)), RNE.

Variants (one mask each; all share A, W and the codebooks): the LUT shares of --shares (default: MXQuant TinyLlama
weights-only budgets 5/6/7 and 1.0), or --mask-file (JSON {"name": [0/1 per K-block], ...}) for real allocations.

Writes data/lutk_<tag>.bin (operands + per-dim goldens) and data/lutk_<tag>.h (shape, offsets, masks).

    python3 gen_lutk_proj.py --m 32 --k 2048 --n 128 --tag gate
    python3 gen_lutk_proj.py --m 32 --k 128 --n 64 --tag tiny      # bring-up
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
#: output dir for <tag>.bin / <tag>.h (MX_DATA_DIR overrides; default mx/data)
DATA = Path(os.environ.get("MX_DATA_DIR", HERE.parent / "data"))
sys.path.insert(0, str(HERE))
import fp8_matmul_model as FM  # noqa: E402
from lut_golden_model import make_lut, tensor_to_custom_fp_codes  # noqa: E402

GROUP = 32
DEFAULT_SHARES = {"b5": 0.6607, "b6": 0.4405, "b7": 0.2204, "all": 1.0}   # MXQuant TinyLlama w-only LUT shares


def e4m3_grid():
    codes, vals = [], []
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            continue
        s, e, m = (c >> 7) & 1, (c >> 3) & 0xF, c & 7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
        codes.append(c); vals.append(-v if s else v)
    return np.array(codes, np.uint8), torch.tensor(vals, dtype=torch.float32)


def fp4_grid():
    vals = []
    for c in range(16):
        s, e, m = (c >> 3) & 1, (c >> 1) & 3, c & 1
        v = m / 2.0 if e == 0 else (1.0 + m / 2.0) * 2.0 ** (e - 1)
        vals.append(-v if s else v)
    return torch.tensor(vals, dtype=torch.float32)


def nearest(x, grid):
    return (x.unsqueeze(-1) - grid).abs().argmin(dim=-1)


def nib(codes):   # [K][N] 4-bit -> [K][N/2] bytes, low nibble = even column
    c = codes.astype(np.uint8)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)


def bf16_bits(t):   # float32 tensor already on the BF16 grid -> uint16
    return (t.contiguous().view(torch.int32) >> 16).to(torch.int32).numpy().astype(np.uint16)


def bf16_add(a, b):   # exact sum rounded once to BF16 (RNE): the VPU's VPU_ADD (vpu_ref.h vpu_add)
    d = ((a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
         + (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64))
    m, e = np.frexp(d)                          # d = m * 2^e, 0.5 <= |m| < 1; 8 significant bits, np.rint = RNE
    r = np.ldexp(np.rint(np.ldexp(m, 8)), e - 8).astype(np.float32)
    return (r.view(np.uint32) >> 16).astype(np.uint16)


def golden(A, B, As, Bs, kbs, dim):
    """BF16 mesh golden (bf16 products and accumulation) over the K-blocks kbs, in order."""
    if not kbs:
        return np.zeros((A.shape[0], B.shape[1]), np.uint16)
    ks = np.concatenate([np.arange(kb * GROUP, (kb + 1) * GROUP) for kb in kbs])
    FM.TILE = dim
    for attr, v in (("TILE_K", dim), ("TILE_M", 2 * dim), ("TILE_N", 2 * dim)):
        if hasattr(FM, attr):
            setattr(FM, attr, v)
    prec = [(8, 7)] * dim
    C = FM.tiled_matmul_hwlike(A[:, ks], B[ks, :], As[:, kbs], Bs[kbs, :], verbose=False,
                               prod_precision_list=prec, acc_precision_list=prec)
    return bf16_bits(C)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=128)
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
    e4c, e4v = e4m3_grid()
    Af = torch.randn(M, K)
    Wf = torch.randn(K, N)
    ai, wi = nearest(Af, e4v), nearest(Wf, e4v)
    A_fp, A_codes = e4v[ai], e4c[ai.numpy()]
    B8_fp, B8_codes = e4v[wi], e4c[wi.numpy()]
    g4 = fp4_grid()
    w4 = nearest(Wf, g4)
    B4_fp, B4_codes = g4[w4], w4.numpy()
    # per-K-block codebooks, one per 2 columns: [NKB][N/2][16]
    luts = torch.stack([torch.stack([make_lut("fp8:e4m3", 4, torch.device("cpu")) for _ in range(N // 2)])
                        for _ in range(NKB)])
    kb_of = torch.arange(K) // GROUP
    cp_of = torch.arange(N) // 2
    tab = luts[kb_of][:, cp_of]                                   # [K][N][16]
    BL_idx = (Wf.unsqueeze(-1) - tab).abs().argmin(dim=-1)        # [K][N]
    BL_fp = torch.gather(tab, 2, BL_idx.unsqueeze(-1)).squeeze(-1)
    lut_codes = np.array(tensor_to_custom_fp_codes(luts.reshape(-1, 16), "fp8:e4m3")[0], np.uint8).reshape(NKB, N // 2, 16)
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
    gold = {}
    for d in dims:
        allk = list(range(NKB))
        parts.append((f"FP8_D{d}", golden(A_fp, B8_fp, As, Bs, allk, d)))
        for vi, (name, m) in enumerate(masks.items()):
            lo = [i for i in range(NKB) if m[i]]
            hi = [i for i in range(NKB) if not m[i]]
            p_hi = golden(A_fp, B8_fp, As, Bs, hi, d)
            p_l = golden(A_fp, BL_fp, As, Bs, lo, d)
            p_4 = golden(A_fp, B4_fp, As, Bs, lo, d)
            # shared accumulator: one BF16 accumulation, low-precision K-blocks then MXFP8 ones (kernel issue order)
            ks_lo = np.concatenate([np.arange(kb * GROUP, (kb + 1) * GROUP) for kb in lo]) if lo else np.zeros(0, int)
            bm_l, bm_4 = B8_fp.clone(), B8_fp.clone()
            bm_l[ks_lo], bm_4[ks_lo] = BL_fp[ks_lo], B4_fp[ks_lo]
            c_sl = golden(A_fp, bm_l, As, Bs, lo + hi, d)
            c_s4 = golden(A_fp, bm_4, As, Bs, lo + hi, d)
            # mixed-K loops: one BF16 accumulation over the K-blocks in K order (per-K-block LUT/FP4 or MXFP8)
            c_ml = golden(A_fp, bm_l, As, Bs, allk, d)
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
    base = f"lutk_{a.tag}"
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{base}.bin").write_bytes(bytes(buf))
    sym = f"_binary_{base}_bin_start"
    h = [f"// generated by gen/gen_lutk_proj.py --m {M} --k {K} --n {N} --tag {a.tag} --seed {a.seed}"
         + (f" --mask-file {a.mask_file}" if a.mask_file else ""),
         f"#ifndef LUTK_{a.tag.upper()}_H", f"#define LUTK_{a.tag.upper()}_H", "#include <stdint.h>",
         f"#define LK_M {M}", f"#define LK_K {K}", f"#define LK_N {N}", f"#define LK_NKB {NKB}",
         f"#define LK_NVAR {len(masks)}", f"extern const uint8_t {sym}[];", f"#define LK_BLOB {sym}"]
    for n, o in offs.items():
        h.append(f"#define LK_OFF_{n} {o}")
    h.append(f"static const char *const lk_var_name[LK_NVAR] = {{{', '.join(json.dumps(k) for k in masks)}}};")
    h.append(f"static const uint8_t lk_mask[LK_NVAR][LK_NKB] = {{")
    for m in masks.values():
        h.append("  {" + ", ".join(str(x) for x in m) + "},")
    h += ["};", "#endif", ""]
    (DATA / f"{base}.h").write_text("\n".join(h))
    print(f"wrote data/{base}.bin ({len(buf)} B) and data/{base}.h; variants {list(masks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

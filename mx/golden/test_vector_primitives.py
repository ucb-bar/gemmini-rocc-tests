#!/usr/bin/env python3
"""The gate on the vectorized `fp_quantize_rne` / `fp_add_exact`.

Those two primitives were scalar Python loops over `.cpu().tolist()` -- 2.02 and 0.74 Melem/s -- and
they are the inner loop of the hardware datapath model, so they set the cost of every golden this
repo generates and made a perplexity sweep with `rtl_exact` impossible (measured projection: 1483 h
per 2048-token window). Vectorizing them is only safe if the result is IDENTICAL, because the
goldens they produce are compared to hardware bit for bit.

This compares the vectorized path against the scalar reference ELEMENT FOR ELEMENT, on the bit
pattern rather than a tolerance, over values chosen to hit every branch: normals, subnormals, the
subnormal/normal boundary, rounding ties, overflow to inf, signed zeros, NaN and inf.

    python3 test_vector_primitives.py
    python3 -c "import fp8_matmul_model as M; M._FORCE_SCALAR_PRIMITIVES = True; ..."
"""
from __future__ import annotations

import math
import sys
import time

import torch

import fp8_matmul_model as M

#: The accumulator schedule the RTL actually uses, plus neighbours that exercise other widths.
#: (8,7) is bf16 -- `bf16_accum_add` and so the cross-tile step of every rtl_exact run. It takes
#: a dedicated fp32-add-then-round path justified by double rounding being innocuous at
#: p1=24 >= 2*p2+2=18; that argument is checked here rather than trusted.
CASES = [(4, 4), (4, 5), (4, 6), (5, 2), (5, 5), (3, 3), (6, 7), (2, 1), (8, 7)]


def probe_values(exp_bits: int, man_bits: int, n: int = 40000) -> torch.Tensor:
    """Values that land on every branch of _round_dyadic_to_scalar, not just the generic one."""
    bias = (1 << (exp_bits - 1)) - 1
    emin, emax = 1 - bias, bias
    g = torch.Generator().manual_seed(1234)

    lo, hi = emin - man_bits - 3, emax + 3
    e = torch.randint(lo, hi + 1, (n,), generator=g).float()
    mant = 1.0 + torch.rand(n, generator=g)
    sgn = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    rand = sgn * mant * torch.exp2(e)

    # Exact ties: a representable value plus exactly half an ulp, where RNE's even rule decides.
    steps = torch.arange(0, 1 << (man_bits + 1))
    ties = []
    for E in range(emin, emax + 1):
        base = torch.ldexp(steps.float(), torch.tensor(E - man_bits))
        ties.append(base + torch.ldexp(torch.tensor(1.0), torch.tensor(E - man_bits - 1)))
    ties = torch.cat(ties)

    # Subnormal grid, the boundary itself, overflow, and the specials.
    sub = torch.ldexp(torch.arange(0, 1 << man_bits).float(),
                      torch.tensor(emin - man_bits)) * 1.5
    edge = torch.tensor([
        math.ldexp(1.0, emin), math.ldexp(1.0, emin - man_bits), math.ldexp(1.0, emax),
        math.ldexp(2.0 - 2.0 ** -man_bits, emax), math.ldexp(1.0, emax + 1),
        0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0,
    ])
    return torch.cat([rand, ties, -ties, sub, -sub, edge]).float()


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def same(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int]:
    """Bit equality, with all NaNs treated as equal (payload is not specified by either path)."""
    na, nb = torch.isnan(a), torch.isnan(b)
    ok = (na & nb) | (bits(a) == bits(b))
    return bool(ok.all()), int((~ok).sum())


def scalar_quantize(x, e, m):
    flat = x.reshape(-1).tolist()
    return torch.tensor([M.fp_quantize_rne_scalar(float(v), e, m) for v in flat],
                        dtype=torch.float32).view_as(x)


def scalar_add(x, y, e, m):
    xf, yf = x.reshape(-1).tolist(), y.reshape(-1).tolist()
    return torch.tensor([M.fp_add_exact_scalar(float(a), float(b), e, m) for a, b in zip(xf, yf)],
                        dtype=torch.float32).view_as(x)


def main() -> int:
    fails = 0
    print(f"{'case':>10}  {'fp_quantize_rne':>28}  {'fp_add_exact':>28}")
    for e, m in CASES:
        x = probe_values(e, m)
        t0 = time.time(); ref_q = scalar_quantize(x, e, m); t_sq = time.time() - t0
        t0 = time.time(); vec_q = M._rne_vec(x, e, m); t_vq = time.time() - t0
        ok_q, bad_q = same(ref_q, vec_q)

        # Operands for the adder are in-format, as `accumulate` always supplies them.
        xa = M._rne_vec(x, e, m)
        ya = M._rne_vec(x.flip(0), e, m)
        t0 = time.time(); ref_a = scalar_add(xa, ya, e, m); t_sa = time.time() - t0
        t0 = time.time(); vec_a = M.fp_add_exact(xa, ya, e, m); t_va = time.time() - t0
        ok_a, bad_a = same(ref_a, vec_a)
        # Wide formats cannot hold the exact sum in int64 and fall back to the scalar path on
        # purpose; the check still runs, it just cannot fail there.
        if (e, m) == (8, 7):
            note = " (bf16 fast path)"
        elif not M._add_exact_vec_is_exact(e, m):
            note = " (scalar: too wide for int64)"
        else:
            note = ""

        fails += (not ok_q) + (not ok_a)
        print(f"  (e{e},m{m}) n={x.numel():6d}  "
              f"{'OK' if ok_q else f'FAIL {bad_q}':>10} {t_sq / max(t_vq, 1e-9):9.0f}x  "
              f"{'OK' if ok_a else f'FAIL {bad_a}':>10} {t_sa / max(t_va, 1e-9):9.0f}x{note}")

    print()
    if fails:
        print(f"FAILED -- {fails} case(s) differ from the scalar reference. The vectorized "
              f"primitives must NOT be used; every golden would inherit the difference.")
        return 1
    print("PASSED -- vectorized primitives are bit-identical to the scalar reference on every "
          "case, including ties, subnormals, the subnormal/normal boundary, overflow and specials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

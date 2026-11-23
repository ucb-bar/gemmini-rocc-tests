#!/usr/bin/env python3
"""Generate mnist_mxgemmini_params.h for the MX-FP8 MNIST inference demo.

Trains a tiny linear classifier on 10 hand-drawn 16x16 digit templates
(no external dataset / network required), quantizes inputs and weights
to MX FP8 (E4M3 codes + per-32-element E8M0 row/col scales), and emits
a C header in the same layout used by the existing matmul_tiled_fp8
tests, plus the labels of each batch row.

Run from this directory:
  ../../.conda-env/bin/python mnist_mxgemmini_gen.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT.parent
sys.path.insert(0, str(TESTS_DIR / "mx" / "golden"))

import numpy as np
import torch

import precision  # per mesh row (exp, frac): ConfigsFP default for TILE
from fp8_matmul_model import (
    INPUT_SPEC, SCALE_SPEC, GROUP, TILE,
    make_fp_quantizer, tiled_matmul_hwlike, write_c_header_tiled,
    matrix_mx_requantize,
)

# M and N are padded out to a 4x4 tile grid (tiles_I = tiles_J = 4). The RTL MX
# store path computes its scratchpad stride as (loop_bound_j / 2) * 2, which
# collapses to 0 for tiles_J = 1 (N = 16) and corrupts the store DMA's command
# tracking (DMACommandTracker assert(cmds(cmd_id).valid)). Every passing
# matmul_tiled RTL test uses a square tile grid with tiles_J >= 4, so we match
# them. Rows beyond NUM_CLASSES are dummy images and columns beyond NUM_CLASSES
# are zero; classification only reads the first NUM_CLASSES rows/cols, so the
# padding does not affect the result.
M = 64
K = 256
N = 64
NUM_CLASSES = 10
IMG_H = IMG_W = 16
assert K == IMG_H * IMG_W
assert K % GROUP == 0 and M % TILE == 0 and N % TILE == 0

TEMPLATES_TXT = [
[
    "................",
    "................",
    "....######......",
    "...########.....",
    "..####....##....",
    "..###......##...",
    "..##.......##...",
    "..##.......##...",
    "..##.......##...",
    "..##.......##...",
    "..##......###...",
    "...##....###....",
    "...########.....",
    "....######......",
    "................",
    "................",
],
[
    "................",
    "................",
    ".......##.......",
    "......###.......",
    ".....####.......",
    "....##.##.......",
    "...##..##.......",
    "..##...##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    "...########.....",
    "................",
    "................",
],
[
    "................",
    "................",
    "....######......",
    "..##########....",
    "..##......##....",
    "..........##....",
    ".........##.....",
    "........##......",
    ".......##.......",
    "......##........",
    ".....##.........",
    "....##..........",
    "..##............",
    "..##########....",
    "..##########....",
    "................",
],
[
    "................",
    "................",
    "...########.....",
    "..##########....",
    ".........###....",
    ".........##.....",
    "........##......",
    "....######......",
    "....######......",
    "........###.....",
    ".........##.....",
    ".........##.....",
    ".........##.....",
    "..#########.....",
    "..########......",
    "................",
],
[
    "................",
    "................",
    "........##......",
    ".......###......",
    "......####......",
    ".....##.##......",
    "....##..##......",
    "...##...##......",
    "..##....##......",
    "..##########....",
    "..##########....",
    "........##......",
    "........##......",
    "........##......",
    "........##......",
    "................",
],
[
    "................",
    "................",
    "..#########.....",
    "..#########.....",
    "..##............",
    "..##............",
    "..##............",
    "..########......",
    "..##########....",
    ".........###....",
    "..........##....",
    "..........##....",
    "..........##....",
    "..########......",
    "..#######.......",
    "................",
],
[
    "................",
    "................",
    ".....######.....",
    "....########....",
    "...###....##....",
    "..##............",
    "..##............",
    "..##########....",
    "..###########...",
    "..##.......##...",
    "..##.......##...",
    "..##.......##...",
    "...##.....##....",
    "....#######.....",
    "................",
    "................",
],
[
    "................",
    "................",
    "..##########....",
    "..##########....",
    "..........##....",
    "..........##....",
    ".........##.....",
    "........##......",
    ".......##.......",
    "......##........",
    "......##........",
    ".....##.........",
    ".....##.........",
    "....##..........",
    "....##..........",
    "................",
],
[
    "................",
    "................",
    "....######......",
    "...########.....",
    "..##......##....",
    "..##......##....",
    "...##....##.....",
    "....######......",
    "....######......",
    "..##......##....",
    "..##......##....",
    "..##......##....",
    "..##......##....",
    "...########.....",
    "....######......",
    "................",
],
[
    "................",
    "................",
    "....######......",
    "...########.....",
    "..##......##....",
    "..##......##....",
    "..##......##....",
    "..##......##....",
    "...#########....",
    "....########....",
    "..........##....",
    "..........##....",
    ".........##.....",
    "...########.....",
    "...######.......",
    "................",
],
]
assert len(TEMPLATES_TXT) == NUM_CLASSES

def rasterize(template):
    img = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for r, row in enumerate(template):
        for c, ch in enumerate(row):
            img[r, c] = 1.0 if ch == "#" else 0.0
    return img

templates = np.stack([rasterize(t) for t in TEMPLATES_TXT], axis=0)
assert templates.shape == (NUM_CLASSES, IMG_H, IMG_W)

def make_noisy_dataset(rng, n_per_class=64, sigma=0.15, jitter=1):
    Xs, ys = [], []
    for c in range(NUM_CLASSES):
        base = templates[c]
        for _ in range(n_per_class):
            img = base.copy()
            dy = rng.integers(-jitter, jitter + 1)
            dx = rng.integers(-jitter, jitter + 1)
            img = np.roll(img, shift=(dy, dx), axis=(0, 1))
            img = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
            Xs.append(img.reshape(-1))
            ys.append(c)
    return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64)

def train_linear(rng, epochs=150, lr=0.1, weight_decay=5e-3):
    X, y = make_noisy_dataset(rng, n_per_class=64, sigma=0.18)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    W = torch.zeros(K, NUM_CLASSES, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(NUM_CLASSES, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([W, b], lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        logits = Xt @ W + b
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (Xt @ W + b).argmax(dim=1)
        acc = (preds == yt).float().mean().item()
    print(f"[trainer] noisy train accuracy = {acc*100:.2f}%  |W|_max = {W.abs().max().item():.3f}")
    return W.detach(), b.detach()

def main():
    seed = 0xCAFE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    W, b = train_linear(rng)

    A_fp32 = torch.zeros(M, K, dtype=torch.float32)
    labels = [0] * M
    for c in range(NUM_CLASSES):
        A_fp32[c] = torch.from_numpy(templates[c].reshape(-1))
        labels[c] = c
    for p in range(NUM_CLASSES, M):
        A_fp32[p] = torch.from_numpy(templates[0].reshape(-1))
        labels[p] = 0

    fp32_logits = A_fp32 @ W
    fp32_preds = fp32_logits[:NUM_CLASSES, :NUM_CLASSES].argmax(dim=1).tolist()
    print(f"[fp32] argmax over first 10 rows / first 10 cols: {fp32_preds}")
    print(f"[fp32] expected                                : {list(range(NUM_CLASSES))}")

    B_fp32 = torch.zeros(K, N, dtype=torch.float32)
    B_fp32[:, :NUM_CLASSES] = W

    Gk = K // GROUP
    A_blk = A_fp32.reshape(M, Gk, GROUP)
    B_blk = B_fp32.reshape(Gk, GROUP, N)

    log2_pmax = 2.0

    def block_scale(block_amax):
        e = torch.floor(torch.log2(torch.clamp(block_amax, min=1e-30)))
        scale = torch.pow(2.0, e - log2_pmax)
        scale = torch.where(block_amax > 0, scale, torch.ones_like(scale))
        return scale

    A_scales_row_q = block_scale(A_blk.abs().amax(dim=2))
    B_scales_col_q = block_scale(B_blk.abs().amax(dim=1))

    in_q = make_fp_quantizer(INPUT_SPEC, rounding="zero")
    A_norm = (A_blk / A_scales_row_q.unsqueeze(-1)).reshape(M, K)
    B_norm = (B_blk / B_scales_col_q.unsqueeze(1)).reshape(K, N)
    A_in = in_q(A_norm)
    B_in = in_q(B_norm)

    C_out_bf16 = tiled_matmul_hwlike(
        A_in, B_in, A_scales_row_q, B_scales_col_q,
        verbose=False,
        prod_precision_list=precision.for_dim(TILE)[0],
        acc_precision_list=precision.for_dim(TILE)[1],
    )

    mx_logits = C_out_bf16[:NUM_CLASSES, :NUM_CLASSES]
    print(f"[mx-fp8 ref] dtype={C_out_bf16.dtype} sample row 0: {mx_logits[0].tolist()}")
    mx_preds = mx_logits.argmax(dim=1).tolist()
    print(f"[mx-fp8 ref] argmax over first 10 rows / first 10 cols: {mx_preds}")
    print(f"[mx-fp8 ref] matches FP32                              : {mx_preds == list(range(NUM_CLASSES))}")

    C_out_q, C_out_scales = matrix_mx_requantize(C_out_bf16)

    header_path = ROOT / "mnist_mxgemmini_params.h"
    write_c_header_tiled(
        path=str(header_path),
        M=M, K=K, N=N,
        A_in=A_in,
        B_in=B_in,
        A_scales_row_q=A_scales_row_q,
        B_scales_col_q=B_scales_col_q,
        C_out_bf16=C_out_bf16,
        C_out_quantized=C_out_q,
        C_out_scales=C_out_scales,
    )

    with open(header_path, "r") as f:
        body = f.read()
    extra = []
    extra.append(f"#define MNIST_NUM_CLASSES {NUM_CLASSES}")
    extra.append(f"#define MNIST_BATCH       {M}")
    extra.append("")
    extra.append(f"static const uint8_t mnist_labels[{M}] = {{")
    extra.append("  " + ", ".join(str(int(l)) for l in labels))
    extra.append("};")
    extra.append("")
    body = body.replace(
        "#include <stdint.h>\n\n",
        "#include <stdint.h>\n\n" + "\n".join(extra) + "\n",
        1,
    )
    with open(header_path, "w") as f:
        f.write(body)
    print(f"[ok] wrote {header_path}")

if __name__ == "__main__":
    main()

"""Only key out the background colour. Nothing else is touched.

Every other pixel keeps its exact RGB value — no requantisation, no palette
change, no speck removal. Use this when the frames are already final and you
just want the background gone.

    python tools/key_out.py <frame-dir> --out <dir> [--bg 249,5,251]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="")
    ap.add_argument("--bg", default="",
                    help="background RGB, e.g. 249,5,251; default = most common colour")
    ap.add_argument("--tol", type=int, default=150,
                    help="L1 distance from bg still counted as background")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    fs = sorted(src.glob("f*.png"))
    if not fs:
        raise SystemExit(f"no frames in {src}")

    dst = Path(args.out) if args.out else src.parent / f"{src.name}_keyed"
    if not dst.is_absolute():
        dst = ROOT / dst
    dst.mkdir(parents=True, exist_ok=True)

    if args.bg:
        bg = np.array([int(x) for x in args.bg.split(",")], int)
    else:
        a0 = np.asarray(Image.open(fs[0]).convert("RGB"))
        q = (a0 // 24 * 24).reshape(-1, 3)
        v, c = np.unique(q, axis=0, return_counts=True)
        bg = v[c.argmax()].astype(int)

    cut = 0
    for i, p in enumerate(fs):
        im = Image.open(p)
        rgb = np.asarray(im.convert("RGB"))
        keyed = np.abs(rgb.astype(int) - bg).sum(2) <= args.tol
        if im.mode == "RGBA":
            alpha = np.asarray(im)[..., 3].copy()
        else:
            alpha = np.full(rgb.shape[:2], 255, np.uint8)
        alpha[keyed] = 0
        cut += int(keyed.sum())
        Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(
            dst / f"f{i:02d}.png")

    print(f"背景 {tuple(int(x) for x in bg)}  抠掉 {cut} px")
    print(f"  {len(fs)} 帧 -> {dst}")
    print("  其余像素 RGB 一个未改")


if __name__ == "__main__":
    main()

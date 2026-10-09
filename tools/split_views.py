"""Split a multi-view reference sheet into one image per view.

Panels are separated by full-height background columns, so the split points are
found by scanning for column runs that contain no foreground at all — no manual
coordinates, no guessing.

    python tools/split_views.py assets/two_bear.png --names front,side
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def bg_color(a: np.ndarray) -> np.ndarray:
    q = (a // 24 * 24).reshape(-1, 3)
    v, c = np.unique(q, axis=0, return_counts=True)
    return v[c.argmax()].astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--names", default="",
                    help="comma-separated names, left to right")
    ap.add_argument("--out", default="")
    ap.add_argument("--tol", type=int, default=90)
    ap.add_argument("--pad", type=int, default=6,
                    help="background margin kept around each panel")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    a = np.asarray(Image.open(src).convert("RGB"))
    bg = bg_color(a)
    fg = np.abs(a.astype(int) - bg).sum(2) > args.tol

    cols = fg.any(0)
    # 找前景列的连续段，每段就是一个视图
    spans, start = [], None
    for i, v in enumerate(cols):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(cols)))
    # 丢掉极窄的碎片（描边毛刺之类）
    spans = [(s, e) for s, e in spans if e - s > a.shape[1] * 0.05]

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    out = Path(args.out) if args.out else src.parent
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    print(f"{src.name}  {a.shape[1]}x{a.shape[0]}  背景 "
          f"{tuple(int(x) for x in bg)}  找到 {len(spans)} 个视图")
    for i, (s, e) in enumerate(spans):
        rows = np.where(fg[:, s:e].any(1))[0]
        y0 = max(0, rows[0] - args.pad)
        y1 = min(a.shape[0], rows[-1] + 1 + args.pad)
        x0 = max(0, s - args.pad)
        x1 = min(a.shape[1], e + args.pad)
        panel = a[y0:y1, x0:x1]
        name = names[i] if i < len(names) else f"view{i}"
        dst = out / f"{src.stem}_{name}.png"
        Image.fromarray(panel).save(dst)
        print(f"   {name:<8} x[{s},{e})  ->  {panel.shape[1]}x{panel.shape[0]}"
              f"  {dst.name}")


if __name__ == "__main__":
    main()

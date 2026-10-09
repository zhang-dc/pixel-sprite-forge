"""Synthetic "turn in place" clip with known ground-truth headings.

A wooden boat (brown hull, white sail along the centre line) seen by a fixed ¾
camera, rendered at 2× and downsampled so edges are soft like video-model
output, with per-pixel noise. The clip opens with a short wobble that settles
on side-on — what H3 does before it starts turning — then turns ~372°.

    python tests/fixture_spin.py --out tests/data/synth_spin [--ccw]

truth.json holds the heading of every frame (0 = bow right, clockwise seen
from above, y down — so 90 = bow toward the camera).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MAGENTA = (255, 0, 255)
HULL, DECK, SAIL = (150, 95, 45), (190, 130, 70), (240, 235, 220)


def hull2d(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex hull, monotone chain."""
    pts = sorted(set(pts))

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    lo, up = half(pts), half(reversed(pts))
    return lo[:-1] + up[:-1]


def render(theta: float, size: int, elev: float = 40.0) -> Image.Image:
    k = 2
    W = size * k
    im = Image.new("RGB", (W, W), MAGENTA)
    dr = ImageDraw.Draw(im)
    R, cx, cy = 0.33 * W, W / 2, 0.58 * W
    se, ce = math.sin(math.radians(elev)), math.cos(math.radians(elev))
    t = math.radians(theta)
    fx, fy = math.cos(t), math.sin(t)                  # forward (east, south)

    def proj(u: float, v: float, z: float) -> tuple[float, float]:
        X, Y = u * fx - v * fy, u * fy + v * fx        # local → world
        return cx + R * X, cy + R * (Y * se - z * ce)

    ring = [(math.cos(a), 0.3 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 48)]
    deck = [proj(u, v, 0.15) for u, v in ring]
    keel = [proj(0.85 * u, 0.6 * v, -0.12) for u, v in ring]
    dr.polygon(hull2d(deck + keel), fill=HULL)
    dr.polygon(deck, fill=DECK)
    base, top, boom = proj(0.1, 0, 0.15), proj(0.1, 0, 1.1), proj(-0.75, 0, 0.3)
    dr.polygon([base, top, boom], fill=SAIL)
    dr.line([base, top], fill=(60, 40, 30), width=k * 2)
    return im.resize((size, size), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--settle", type=int, default=14)
    ap.add_argument("--turn", type=int, default=140, help="frames for the turn")
    ap.add_argument("--degrees", type=float, default=372.0)
    ap.add_argument("--ccw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("f*.png"):
        p.unlink()
    rng = np.random.default_rng(args.seed)
    sign = -1.0 if args.ccw else 1.0
    angles = [-25.0 * sign * (1 - i / (args.settle - 1)) for i in range(args.settle)]
    angles += [sign * args.degrees * i / (args.turn - 1) for i in range(1, args.turn)]
    truth = []
    for i, a in enumerate(angles):
        im = np.asarray(render(a, args.size)).astype(np.float32)
        im += rng.normal(0, 4, im.shape)
        Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(out / f"f{i}.png")
        truth.append(round(a % 360, 2))
    (out / "truth.json").write_text(json.dumps({"settle": args.settle, "heading": truth}),
                                    encoding="utf-8")
    print(f"{len(angles)} frames -> {out}")


if __name__ == "__main__":
    main()

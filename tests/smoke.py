"""Smoke test: run the whole post-processing chain on synthetic frames.

No model weights needed. Generation (`src/generate_h3.py`) requires ComfyUI
plus ~40 GB of weights, so it is not covered here; everything after it is
pure CPU and is exercised end to end.

    python tests/smoke.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"


def run(args: list[str]) -> None:
    r = subprocess.run([sys.executable, *args], cwd=ROOT, text=True,
                       capture_output=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"failed: {' '.join(args)}")


def main() -> int:
    run(["tests/fixture.py", "--out", str(DATA / "synth_walk"),
         "--frames", "39", "--steps", "2"])
    src = DATA / "synth_walk"

    run(["src/pick_cycle.py", str(src), "-k", "9", "--osc", "2"])
    pick = src.parent / f"{src.name}_pick9osc2"

    run(["src/quant_sprite.py", str(pick), "--ref", "assets/side_bear.png",
         "--colors", "16", "--scale", "4", "--outline",
         "--outline-color", "0,0,0", "--alpha"])
    final = pick.parent / f"{pick.name}_q16_s4_ol_a"

    frames = []
    for p in sorted(final.glob("f*.png")):
        im = Image.open(p)
        frames.append((np.asarray(im.convert("RGB")), np.asarray(im)[..., 3]))
    if not frames:
        raise SystemExit("no output frames")

    h, w = frames[0][0].shape[:2]
    semi = sum(int(((a > 0) & (a < 255)).sum()) for _, a in frames)
    pal = {tuple(int(x) for x in c)
           for rgb, a in frames for c in np.unique(rgb[a > 127], axis=0)}
    lone = 0
    for rgb, a in frames:
        m = a > 127
        same = np.zeros(m.shape, int)
        pc, pm = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), "edge"), np.pad(m, 1)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == dx == 1:
                    continue
                same += ((pc[dy:dy + h, dx:dx + w] == rgb).all(2)
                         & pm[dy:dy + h, dx:dx + w] & m)
        lone += int((m & (same == 0)).sum())
    bad_ring = 0
    for rgb, a in frames:
        m = a > 127
        p = np.pad(m, 1)
        ring = m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
        bad_ring += int((rgb[ring].sum(1) != 0).sum())

    print(f"frames            {len(frames)}")
    print(f"size              {w}x{h}")
    print(f"palette           {len(pal)} colours (shared across all frames)")
    print(f"semi-transparent  {semi} px   (0 = hard alpha, no soft fringe)")
    print(f"isolated specks   {lone} px   (0 = no lone off-colour pixels)")
    print(f"outline breaks    {bad_ring} px   (0 = unbroken 1px outline)")
    print(f"\noutput -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

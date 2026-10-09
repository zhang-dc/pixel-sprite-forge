"""Smoke test for the rotation use case: synthetic turn → pick_rotation →
quant_sprite, checked against ground-truth headings. No model weights needed.

    python tests/smoke_spin.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
N = 32


def run(args: list[str]) -> str:
    r = subprocess.run([sys.executable, *args], cwd=ROOT, text=True,
                       capture_output=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"failed: {' '.join(args)}")
    return r.stdout


def check(ccw: bool) -> list[str]:
    name = "synth_spin_ccw" if ccw else "synth_spin"
    src = DATA / name
    run(["tests/fixture_spin.py", "--out", str(src)] + (["--ccw"] if ccw else []))
    run(["src/pick_rotation.py", str(src), "-n", str(N), "--hue", "15,50"])
    dst = src.parent / f"{name}_rot{N}"
    rep = json.loads((dst / "report.json").read_text(encoding="utf-8"))
    truth = json.loads((src / "truth.json").read_text(encoding="utf-8"))
    fails = []
    want = "counterclockwise" if ccw else "clockwise"
    if rep["direction"] != want:
        fails.append(f"{name}: direction {rep['direction']}, want {want}")
    if not rep["gate_ok"]:
        fails.append(f"{name}: gates {rep['gate']}")
    errs = []
    for p in rep["picks"]:
        d = (truth["heading"][p["frame"]] - p["k"] * 360 / N + 180) % 360 - 180
        errs.append(abs(d))
        if p["frame"] < truth["settle"] - 1:
            fails.append(f"{name}: k={p['k']} picked settling frame f{p['frame']}")
    if max(errs) > 8:
        fails.append(f"{name}: true heading error up to {max(errs):.1f}°")

    run(["src/quant_sprite.py", str(dst), "--ref", str(dst / "f00.png"),
         "--colors", "16", "--scale", str(rep["scale"]), "--outline", "--alpha",
         "--out", str(dst.parent / f"{dst.name}_q")])
    outs = sorted((dst.parent / f"{dst.name}_q").glob("f*.png"))
    alphas = [np.asarray(Image.open(p).convert("RGBA"))[..., 3] for p in outs]
    if len(outs) != N:
        fails.append(f"{name}: {len(outs)} sprites, want {N}")
    if {a.shape for a in alphas} != {(64, 64)}:
        fails.append(f"{name}: sprite sizes {({a.shape for a in alphas})}")
    if sum(int(((a > 0) & (a < 255)).sum()) for a in alphas):
        fails.append(f"{name}: semi-transparent pixels")
    print(f"{name:16s} {rep['direction']:16s} turned {rep['turned_deg']}°  "
          f"max step {rep['max_step']}°  true-heading error ≤ {max(errs):.1f}°  "
          f"scale {rep['scale']}")
    return fails


def main() -> int:
    fails = check(False) + check(True)
    for f in fails:
        print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

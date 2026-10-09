"""Pick N evenly spaced headings out of a turn-in-place clip (N-direction sprites).

    python src/pick_rotation.py data/trainset/h3/<run-dir> -n 32 [--hue 15,50]

Heading comes from the body's horizontal width, which is independent of camera
elevation: w(θ) = 2·√(a²cos²θ + b²sin²θ), widest side-on, narrowest head-on.
Width extremes anchor 0/90/180/270/360°, acos interpolates between them, and the
sign of the axis tilt in the first quarter gives the direction. Frames before
the first anchor are the model settling in place and are never picked.

Writes `<run-dir>_rot<N>/` (f00.. at one common scale, report.json, review.png)
and prints the quant_sprite command. Gates: ≥ 4 anchors, tilt ≥ 3°, step ≤ 12°,
every heading within half a step, head-on width ≤ 0.6 × side-on.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def wrap180(d: float) -> float:
    return (d + 180) % 360 - 180


def bg_of(a: np.ndarray) -> np.ndarray:
    edge = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3),
                           a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    return np.median(edge, 0)


def fg(a: np.ndarray, bg: np.ndarray, tol: int = 90) -> np.ndarray:
    return np.abs(a.astype(int) - bg).sum(-1) > tol


def body_mask(a: np.ndarray, hue: tuple[float, float] | None) -> np.ndarray:
    m = fg(a, bg_of(a))
    if hue is None:
        return m
    f = a.astype(np.float32) / 255.0
    mx, mn = f.max(-1), f.min(-1)
    s = (mx - mn) / np.maximum(mx, 1e-6)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    h = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    return m & (h > hue[0]) & (h < hue[1]) & (s > 0.35) & (mx > 0.25) & (mx < 0.95)


def width(m: np.ndarray) -> float:
    c = np.where(m.any(0))[0]
    return float(c[-1] - c[0] + 1) if len(c) else float("nan")


def axis_deg(m: np.ndarray) -> float | None:
    """Screen angle of the mask's main axis, 0..180, y down."""
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    w, v = np.linalg.eigh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()])))
    vx, vy = v[:, 1]
    return math.degrees(math.atan2(vy, vx)) % 180


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    med = np.array([np.median(pad[i:i + k]) for i in range(len(x))])
    pad = np.pad(med, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def headings(masks: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Heading per frame in degrees (nan = unknown) plus diagnostics."""
    w = np.array([width(m) for m in masks])
    ok = ~np.isnan(w)
    w = np.interp(np.arange(len(w)), np.where(ok)[0], w[ok])
    ws = smooth(w)
    L, B = float(np.percentile(ws, 95)), float(np.percentile(ws, 5))
    hi, lo = B + 0.75 * (L - B), B + 0.25 * (L - B)
    state = np.where(ws >= hi, 1, np.where(ws <= lo, -1, 0))
    runs: list[list[int]] = []                     # [state, first, last]
    for i, st in enumerate(state):
        if st == 0:
            continue
        if runs and runs[-1][0] == st:
            runs[-1][2] = i
        else:
            runs.append([int(st), i, i])
    while runs and runs[0][0] != 1:                # must start side-on
        runs.pop(0)
    anchors: list[int] = []
    for k, (st, a, b) in enumerate(runs[:5]):
        seg = ws[a:b + 1]
        if st == 1 and k == 0:
            # the clip often wobbles in place before turning: take the
            # *latest* frame that is near the peak
            cand = np.where(seg >= seg.max() - 0.03 * (L - B))[0]
            anchors.append(a + int(cand[-1]))
        else:
            anchors.append(a + int(seg.argmax() if st == 1 else seg.argmin()))
    info: dict = {"L": round(L, 1), "B": round(B, 1), "anchors": anchors,
                  "width": [round(float(x), 1) for x in ws]}
    cw = np.full(len(masks), np.nan)
    if len(anchors) < 2:
        info.update(direction="unknown", direction_confident=False,
                    turned=0.0, narrow=1.0)
        return cw, info
    # Frames before the first anchor stay nan. Marking them 0° makes the 0°
    # pick land on frame 0 — still settling, visibly off side-on.
    cw[anchors[0]] = 0.0

    def phi(i: int, Lq: float, Bq: float) -> float:
        c = math.sqrt(min(1.0, max(0.0, (ws[i] ** 2 - Bq ** 2)
                                   / max(1e-6, Lq ** 2 - Bq ** 2))))
        return math.degrees(math.acos(c))

    for q in range(len(anchors) - 1):
        a, b = anchors[q], anchors[q + 1]
        Lq, Bq = max(ws[a], ws[b]), min(ws[a], ws[b])
        from_wide = ws[a] >= ws[b]
        for i in range(a, b + 1):
            p = phi(i, Lq, Bq)
            cw[i] = 90 * q + (p if from_wide else 90 - p)
    last, q = anchors[-1], len(anchors) - 1
    from_wide = ws[last] >= (L + B) / 2
    for i in range(last + 1, len(masks)):
        p = phi(i, L, B)
        cw[i] = 90 * q + (p if from_wide else 90 - p)
    cw = np.fmax.accumulate(cw)
    # acos is infinitely steep near side-on; smooth 7 frames in time
    okc = ~np.isnan(cw)
    if okc.sum() > 7:
        pad = np.pad(cw[okc], (3, 3), mode="edge")
        cw[okc] = np.maximum.accumulate(
            np.convolve(pad, np.ones(7) / 7, mode="valid"))

    a0, a1 = anchors[0], anchors[1]
    tilts = []
    for i in range(a0 + (a1 - a0) // 4, a1 - (a1 - a0) // 4 + 1):
        ang = axis_deg(masks[i])
        if ang is not None:
            tilts.append(ang - 180 if ang > 90 else ang)
    tilt = float(np.median(tilts)) if tilts else 0.0
    info["tilt_q1_deg"] = round(tilt, 1)
    info["direction"] = "clockwise" if tilt >= 0 else "counterclockwise"
    info["direction_confident"] = abs(tilt) >= 3
    info["narrow"] = round(float(min(ws[a] for a in anchors[1::2])
                                 / max(ws[a] for a in anchors[0::2])), 2)
    info["turned"] = round(float(np.nanmax(cw)), 1)
    th = cw % 360 if info["direction"] == "clockwise" else (360 - cw) % 360
    return th, info


def common_scale(frames: list[np.ndarray], size: int) -> int:
    """One downsample factor for the whole ring: same object, same size in
    every direction. Longest side ≤ size·s and area ≤ 45 % of the canvas."""
    s = 1
    for a in frames:
        m = fg(a, bg_of(a))
        r, c = np.where(m.any(1))[0], np.where(m.any(0))[0]
        bh, bw = r[-1] - r[0] + 1, c[-1] - c[0] + 1
        t = max(1, math.ceil(max(bw, bh) / size))
        while bw * bh > 0.45 * (size * t) ** 2:
            t += 1
        s = max(s, t)
    return s


def recenter(a: np.ndarray, canvas: int) -> np.ndarray:
    bg = bg_of(a)
    m = fg(a, bg)
    r, c = np.where(m.any(1))[0], np.where(m.any(0))[0]
    crop = a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
    out = np.empty((canvas, canvas, 3), np.uint8)
    out[:] = bg.astype(np.uint8)
    y0, x0 = (canvas - crop.shape[0]) // 2, (canvas - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out


def review(path: Path, frames, th, picks, n: int) -> None:
    """Every 4th source frame with its heading (picked ones boxed), then the picks."""
    t, cols = 72, 24
    shown = list(range(0, len(frames), 4))
    rows = math.ceil(len(shown) / cols)
    prow = math.ceil(n / cols)
    im = Image.new("RGB", (cols * t, (rows + prow) * (t + 14) + 10), (24, 28, 36))
    dr = ImageDraw.Draw(im)
    for j, i in enumerate(shown):
        x, y = (j % cols) * t, (j // cols) * (t + 14)
        im.paste(Image.fromarray(frames[i]).resize((t, t), Image.LANCZOS), (x, y))
        if any(abs(i - p) < 2 for p in picks):
            dr.rectangle([x, y, x + t - 1, y + t - 1], outline=(255, 220, 60), width=2)
        dr.text((x + 2, y + t), f"f{i} " + ("-" if math.isnan(th[i]) else f"{th[i]:.0f}"),
                fill=(200, 200, 200))
    y0 = rows * (t + 14) + 10
    for k, i in enumerate(picks):
        x, y = (k % cols) * t, y0 + (k // cols) * (t + 14)
        im.paste(Image.fromarray(frames[i]).resize((t, t), Image.LANCZOS), (x, y))
        dr.text((x + 2, y + t), f"{k}: {k * 360 / n:.0f}", fill=(255, 255, 255))
    im.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="frame directory (f0.png, f1.png, ...)")
    ap.add_argument("-n", type=int, default=32, help="number of directions")
    ap.add_argument("--hue", default="",
                    help="lo,hi: measure width on this hue band only (degrees)")
    ap.add_argument("--size", type=int, default=64, help="final sprite size")
    ap.add_argument("--max-step", type=float, default=12.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    # Numeric order: a plain string sort puts f100..f109 between f10 and f11,
    # which scrambles every anchor and makes a real turn look like a flip.
    fs = sorted((p for p in src.glob("f*.png") if re.fullmatch(r"f\d+", p.stem)),
                key=lambda p: int(p.stem[1:]))
    if len(fs) < 20:
        raise SystemExit(f"{src}: need at least 20 frames, found {len(fs)}")
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in fs]
    hue = tuple(float(x) for x in args.hue.split(",")) if args.hue else None
    th, info = headings([body_mask(a, hue) for a in frames])

    n, step = args.n, 360 / args.n
    seq = [x for x in th if not math.isnan(x)]
    jump = max([abs(wrap180(b - a)) for a, b in zip(seq, seq[1:])] or [0.0])
    picks, errs = [], []
    for k in range(n):
        d = np.array([abs(wrap180(x - k * step)) if not math.isnan(x) else 999.0
                      for x in th])
        i = int(d.argmin())
        picks.append(i)
        errs.append(float(d[i]))
    gate = {
        "structure": len(info["anchors"]) >= 4,
        "direction": bool(info["direction_confident"]),
        "max_step": jump <= args.max_step,
        "coverage": max(errs) <= step / 2,
        "look": info["narrow"] <= 0.6,
    }
    gate = {k: bool(v) for k, v in gate.items()}

    s = common_scale([frames[i] for i in picks], args.size)
    dst = Path(args.out) if args.out else src.parent / f"{src.name}_rot{n}"
    dst.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(picks):
        Image.fromarray(recenter(frames[i], args.size * s)).save(dst / f"f{k:02d}.png")
    report = {
        "frames": len(frames), "turned_deg": info["turned"],
        "direction": info["direction"], "tilt_q1_deg": info.get("tilt_q1_deg"),
        "anchors": info["anchors"], "narrow": info["narrow"],
        "max_step": round(jump, 1), "max_pick_err": round(max(errs), 1),
        "scale": s, "gate": gate, "gate_ok": all(gate.values()),
        "picks": [{"k": k, "frame": i,
                   "heading": None if math.isnan(th[i]) else round(float(th[i]), 1),
                   "err": round(e, 1)} for k, (i, e) in enumerate(zip(picks, errs))],
        "heading": [None if math.isnan(x) else round(float(x), 1) for x in th],
    }
    (dst / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    review(dst / "review.png", frames, th, picks, n)

    flag = "PASS" if report["gate_ok"] else \
        "FAIL " + ",".join(k for k, v in gate.items() if not v)
    print(f"{flag}  turned {info['turned']}° {info['direction']} "
          f"(anchors {info['anchors']}), max step {jump:.1f}°, "
          f"max pick error {max(errs):.1f}°")
    print(f"-> {dst}  (review.png, report.json)")
    print(f"next: python src/quant_sprite.py {dst} --ref <reference> "
          f"--colors 24 --scale {s} --outline --alpha")


if __name__ == "__main__":
    main()

"""Build README previews from a finished sprite folder: animated GIF + sheet.

    python tools/make_preview.py tests/data/<final-dir> --out examples/bear_walk

Nearest-neighbour upscaling only — anything smoother would defeat the point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHECK_A, CHECK_B = (58, 58, 66), (44, 44, 50)


def checker(w: int, h: int, cell: int) -> Image.Image:
    a = np.zeros((h, w, 3), np.uint8)
    y, x = np.mgrid[0:h, 0:w]
    a[((y // cell + x // cell) % 2) == 0] = CHECK_A
    a[((y // cell + x // cell) % 2) == 1] = CHECK_B
    return Image.fromarray(a)


def on_checker(im: Image.Image, cell: int) -> Image.Image:
    bg = checker(im.width, im.height, cell)
    bg.paste(im, (0, 0), im if im.mode == "RGBA" else None)
    return bg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="examples/sprite")
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--checker", action="store_true",
                    help="铺棋盘格底而不是透明底")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    fs = sorted(src.glob("f*.png"))
    if not fs:
        raise SystemExit(f"no frames in {src}")

    ims = [Image.open(p).convert("RGBA") for p in fs]
    z = args.zoom
    big = [im.resize((im.width * z, im.height * z), Image.NEAREST)
           for im in ims]

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # 单帧 PNG，保留透明通道（用来做素材）
    fr = out.parent / out.name
    fr.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(ims):
        im.save(fr / f"f{i:02d}.png")

    cell = max(4, z * 4)
    if args.checker:
        frames = [on_checker(b, cell) for b in big]
        frames[0].save(f"{out}.gif", save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=False)
    else:
        # 透明 GIF：GIF 只支持 1 位透明，得留一个调色板槽当透明色。
        # 先转成 P 模式并把完全透明的像素指到那个槽上；
        # disposal=2 让每帧播完清屏，否则上一帧的残影会糊在下一帧后面。
        pf = []
        for b in big:
            p = b.convert("RGB").quantize(colors=255, method=Image.MAXCOVERAGE)
            idx = np.asarray(p)
            alpha = np.asarray(b)[..., 3]
            idx = np.where(alpha > 127, idx, 255).astype(np.uint8)
            q = Image.fromarray(idx, "P")
            q.putpalette(p.getpalette()[:255 * 3] + [0, 0, 0])
            q.info["transparency"] = 255
            pf.append(q)
        pf[0].save(f"{out}.gif", save_all=True, append_images=pf[1:],
                   duration=int(1000 / args.fps), loop=0,
                   transparency=255, disposal=2, optimize=False)

    # 帧序列表：一行铺开
    w, h = big[0].size
    if args.checker:
        sheet = checker(w * len(big), h, cell)
        for i, b in enumerate(big):
            sheet.paste(b, (i * w, 0), b)
    else:
        sheet = Image.new("RGBA", (w * len(big), h), (0, 0, 0, 0))
        for i, b in enumerate(big):
            sheet.paste(b, (i * w, 0), b)
    sheet.save(f"{out}_sheet.png")

    print(f"{len(ims)} frames {ims[0].width}x{ims[0].height} (x{z})")
    print(f"  {out}.gif")
    print(f"  {out}_sheet.png")
    print(f"  {fr}/  ({len(ims)} PNGs with alpha)")


if __name__ == "__main__":
    main()

"""量"边缘/描边有多锐"，并定位是在哪一步丢的。

## 三个指标

**轮廓过渡宽度**　沿轮廓法向，从背景走到前景要几个像素。
　　　　　　　　　手绘像素画 = 1.0（一步到位）；视频模型会糊成 2~4 像素。
　　　　　　　　　这是"生成过程中边缘丢失"最直接的数值形态。

**描边覆盖率**　　轮廓上有多少比例的像素是"明显暗于内侧"的描边像素。
　　　　　　　　　参考图这种手绘素材接近 100%；生成图会断断续续。

**描边纯度**　　　描边像素的颜色集中在几个色上。手绘描边通常就一个黑色；
　　　　　　　　　糊过之后会散成一片深浅不一的暗色。

## 为什么要分步量

同一套指标分别打在 参考图 / 生成原帧 / 抽帧后 / 量化后 上，
才能看出是**生成**丢的还是**后处理**丢的 —— 两者的修法完全不同。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import quantize as Q

ROOT = Path(__file__).resolve().parent.parent


def fg_mask(a: np.ndarray) -> np.ndarray:
    q = (a // 24 * 24).reshape(-1, 3)
    v, c = np.unique(q, axis=0, return_counts=True)
    return np.abs(a.astype(int) - v[c.argmax()]).sum(2) > 90


def bg_color(a: np.ndarray) -> np.ndarray:
    q = (a // 24 * 24).reshape(-1, 3)
    v, c = np.unique(q, axis=0, return_counts=True)
    return v[c.argmax()].astype(float)


def transition_width(a: np.ndarray, m: np.ndarray) -> float:
    """轮廓过渡宽度：沿 4 个方向扫描，数"既不像背景也不像前景"的像素。

    判据用到背景色的 OKLab 距离：完全是背景 -> 0，完全脱离背景 -> 大。
    取 15%~85% 两个分位之间跨了几个像素，就是过渡宽度。
    """
    bg = bg_color(a)
    d = np.sqrt(((Q.to_oklab(a.astype(np.uint8))
                  - Q.to_oklab(bg[None, None].astype(np.uint8))) ** 2).sum(-1))
    if not m.any():
        return float("nan")
    hi = np.percentile(d[m], 75)
    if hi <= 1e-6:
        return float("nan")
    lo_t, hi_t = 0.15 * hi, 0.85 * hi
    widths = []
    H, W = d.shape
    for axis in (0, 1):
        arr = d if axis == 0 else d.T
        n = arr.shape[0]
        for i in range(0, n, max(1, n // 64)):       # 抽 64 条扫描线就够
            row = arr[i]
            inside = row > hi_t
            if not inside.any():
                continue
            j = int(np.argmax(inside))               # 第一个"完全前景"位置
            k = j
            while k > 0 and row[k - 1] > lo_t:       # 往回退到"完全背景"
                k -= 1
            widths.append(j - k + 1)
    return float(np.mean(widths)) if widths else float("nan")


def outline_stats(a: np.ndarray, m: np.ndarray,
                  drop: float = 0.16) -> tuple[float, int]:
    """(描边覆盖率, 描边用了几个颜色)。"""
    if not m.any():
        return float("nan"), 0
    ol = Q.outline_mask(a, m, drop)
    # 轮廓 = 前景里紧挨背景的一圈
    pad = np.pad(m, 1)
    nb = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    border = m & ~nb
    if not border.any():
        return float("nan"), 0
    cov = float((ol & border).sum() / border.sum())
    ncol = len(np.unique(a[ol], axis=0)) if ol.any() else 0
    return cov, ncol


def scan(label: str, paths: list[Path], drop: float) -> None:
    if not paths:
        print(f"  {label:<26} —— 没有帧")
        return
    tw, cv, nc = [], [], []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"))
        m = fg_mask(a)
        tw.append(transition_width(a, m))
        c, n = outline_stats(a, m, drop)
        cv.append(c)
        nc.append(n)
    print(f"  {label:<26} 过渡宽度 {np.nanmean(tw):>5.2f} px   "
          f"描边覆盖 {np.nanmean(cv) * 100:>5.1f}%   "
          f"描边色数 {np.mean(nc):>6.1f}   ({len(paths)} 张)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="assets/side_bear.png")
    ap.add_argument("--drop", type=float, default=0.16)
    ap.add_argument("dirs", nargs="*", help="要对比的帧目录")
    args = ap.parse_args()

    print("越小越锐：过渡宽度 1.00 = 一步到位（手绘像素画的水平）")
    print("越大越好：描边覆盖率")
    print("越小越纯：描边色数\n")

    rp = ROOT / args.ref if not Path(args.ref).is_absolute() else Path(args.ref)
    if rp.is_file():
        scan("参考图（基准）", [rp], args.drop)

    for d in args.dirs:
        p = Path(d)
        if not p.is_absolute():
            p = ROOT / p
        scan(p.name[-26:], sorted(p.glob("f*.png")), args.drop)


if __name__ == "__main__":
    main()

"""程序生成一段"视频模型风格"的走路动画，当最小验证集。

## 为什么不用真实素材

这套流水线是从真实的视频模型产物上调出来的，但那些素材要么是商业游戏
的（不能再分发），要么是托管服务生成的。所以验证集全部程序生成 ——
仓库里不含任何第三方素材，克隆下来就能跑。

## 要模拟视频模型输出的哪些特征

流水线要解决的问题都来自这些特征，合成数据必须带上，否则测了个寂寞：

    品红底          生成时用纯色底方便抠像
    软边            模型输出的轮廓是渐变的，不是硬边（真实测得过渡 3~4 像素）
    边缘混色        角色色和品红底混出的紫调，会污染调色板
    颜色抖动        同一块区域逐帧有轻微色差，是"像素漂移"的来源
    亚像素晃动      角色整体在帧间轻微平移
    步态            展宽（两腿水平跨度）随步频振荡，抽帧器靠这个定位

## 步态定义

一个完整走路循环 = 两步（左脚一次 + 右脚一次），展宽序列上是两个波峰。
默认生成 39 帧 / 2 步，对应真实 H3 的 39 帧档。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

BG = (249, 5, 251)                      # 品红底
BODY = (150, 100, 60)                   # 棕色身体
BODY_DARK = (95, 62, 35)                # 暗部
ARMOUR = (140, 145, 155)                # 灰色护甲
OUTLINE = (12, 10, 14)                  # 描边


def _disc(h: int, w: int, cy: float, cx: float, ry: float, rx: float
          ) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0


def _bar(h: int, w: int, y0: float, x0: float, ang: float,
         length: float, width: float) -> np.ndarray:
    """一根有宽度的线段，用来当四肢。"""
    y, x = np.mgrid[0:h, 0:w]
    dy, dx = y - y0, x - x0
    ca, sa = math.cos(ang), math.sin(ang)
    t = dx * ca + dy * sa                       # 沿肢体方向
    n = -dx * sa + dy * ca                      # 垂直方向
    return (t >= 0) & (t <= length) & (np.abs(n) <= width / 2)


def render(phase: float, size: int, jitter: tuple[float, float]
           ) -> tuple[np.ndarray, np.ndarray]:
    """画一帧。phase 0~1 走完**一步**。返回 (RGB, 前景掩码)。"""
    S = size
    oy, ox = jitter
    cy, cx = S * 0.50 + oy, S * 0.50 + ox
    hip_y, hip_x = cy + S * 0.10, cx
    # 一步 = 两腿一开一合；相位差 pi 让两条腿反相
    swing = math.sin(phase * 2 * math.pi)
    leg_len, leg_w = S * 0.24, S * 0.055
    near = _bar(S, S, hip_y, hip_x, math.pi / 2 - swing * 0.55, leg_len, leg_w)
    far = _bar(S, S, hip_y, hip_x, math.pi / 2 + swing * 0.55, leg_len, leg_w)
    torso = _disc(S, S, cy - S * 0.02, cx, S * 0.15, S * 0.095)
    head = _disc(S, S, cy - S * 0.22, cx + S * 0.01, S * 0.085, S * 0.075)
    arm = _bar(S, S, cy - S * 0.10, cx, math.pi / 2 - swing * -0.45,
               S * 0.20, S * 0.045)

    img = np.zeros((S, S, 3), np.uint8)
    img[:] = BG
    for m, col in ((far, BODY_DARK), (torso, ARMOUR), (head, BODY),
                   (arm, BODY_DARK), (near, BODY)):
        img[m] = col
    fg = far | torso | head | arm | near
    return img, fg


def soften(img: np.ndarray, fg: np.ndarray, rng: np.random.Generator,
           blur: int, noise: float) -> np.ndarray:
    """把硬边糊成视频模型那样的软边，并加逐帧颜色抖动。"""
    out = img.astype(np.float64)
    # 盒式模糊，重复几次近似高斯 —— 只用 numpy，不引入 scipy
    for _ in range(blur):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
               + 2 * out) / 6.0
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data/synth_walk")
    ap.add_argument("--frames", type=int, default=39)
    ap.add_argument("--steps", type=int, default=2, help="走几步")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--wobble", type=float, default=0.8,
                    help="亚像素晃动幅度（像素）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(args.frames):
        phase = (i / args.frames) * args.steps
        j = (rng.normal(0, args.wobble), rng.normal(0, args.wobble))
        img, fg = render(phase % 1.0, args.size, j)
        img = soften(img, fg, rng, args.blur, args.noise)
        Image.fromarray(img).save(dst / f"f{i:02d}.png")
    print(f"{args.frames} 帧 {args.size}x{args.size} / {args.steps} 步 -> {dst}")


if __name__ == "__main__":
    main()

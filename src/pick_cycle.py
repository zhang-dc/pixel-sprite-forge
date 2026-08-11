"""从 H3 生成的长序列里抽出一个完整走路循环，降采样成精灵帧。

## 为什么要抽

H3 的帧数被锁在 17k+5 网格上，且实测：

    **步数** = (帧数 - 5) / 17        22->1步  39->2步  56->3步  124->7步

（因为 latent_t = ((n-5)//17)*5+2，展宽波动数 = (latent_t-2)/5 = (n-5)//17）

⚠️ 这个数是**步数**不是走路循环数。一个完整循环 = 两步（左脚 + 右脚），
   展宽序列上是**两个波峰**。所以：

       22 帧只有一步 —— 用户实测"一步多，另一只腿还没迈完就结束"，对上了
       39 帧两步    —— 能凑出一个完整循环，是最小可用帧数，也最干净
      124 帧七步    —— 每步只占 17 帧，步的定位精度反而下降，抽出来噪声大

所以拿不到"9 帧走完一圈"，只能生成 39 帧再抽。

## 为什么不能均匀抽

均匀抽会把腿分得最开、并得最拢这些**极值姿势**抽丢，出来的精灵看着像
在滑步。像素动画的可读性全靠这几个极值帧。

做法：用下半身前景的水平展宽当步态相位（腿分开=大，并拢=小），
先定位一个完整周期，再在周期内按**相位等距**取帧 —— 相位等距而不是
时间等距，保证极值一定被取到。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def fg_mask(a: np.ndarray) -> np.ndarray:
    """前景 = 与背景主色差得远的像素。"""
    q = (a // 24 * 24).reshape(-1, 3)
    v, c = np.unique(q, axis=0, return_counts=True)
    return np.abs(a.astype(int) - v[c.argmax()]).sum(2) > 90


def spread(p: Path) -> float:
    a = np.asarray(Image.open(p).convert("RGB"))
    m = fg_mask(a)
    rows = np.where(m.any(1))[0]
    if len(rows) < 8:
        return np.nan
    leg = m[rows[0] + int(len(rows) * 0.62):]
    cols = np.where(leg.any(0))[0]
    return (cols[-1] - cols[0]) / a.shape[1] if len(cols) > 1 else np.nan


def fill(v: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return np.nan_to_num(v)
    return np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])


def cycle_bounds(v: np.ndarray, n_steps: int,
                 n_osc: int = 2) -> tuple[int, int]:
    """定位 n_osc 个展宽波动的区间。

    ⚠️ 一个展宽波动（并拢->分开->并拢）= **一步**，不是一个完整走路循环。
       完整循环要两步（左脚一次 + 右脚一次），所以 n_osc 默认 2。
       实测 步数 = (帧数-5)/17，所以 39 帧（2 步）是能凑出一个
       完整循环的最小帧数；22 帧只有一步，抽出来是半个循环。
    """
    step = len(v) / max(1, n_steps)          # 一步占多少帧
    w = max(2, int(step * 0.3))
    cuts = [int(np.nanargmin(v[:max(2, int(step))]))]
    for _ in range(n_osc):
        guess = cuts[-1] + int(round(step))
        lo = max(cuts[-1] + w, guess - w)
        hi = min(len(v), guess + w + 1)
        if hi - lo < 2:
            cuts.append(min(len(v) - 1, cuts[-1] + int(round(step))))
        else:
            cuts.append(lo + int(np.nanargmin(v[lo:hi])))
    return cuts[0], cuts[-1]


def pick_by_phase(v: np.ndarray, i0: int, i1: int, k: int) -> list[int]:
    """在 [i0, i1) 内按相位等距取 k 帧。

    相位用展宽的累计变化量定义 —— 变化快的地方（腿在甩）多给帧，
    变化慢的地方（腿快并拢）少给帧。这样极值不会被跳过。
    """
    seg = v[i0:i1 + 1]
    if len(seg) <= k:
        return list(range(i0, i1 + 1))
    d = np.abs(np.diff(seg))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    if cum[-1] < 1e-9:
        return list(np.linspace(i0, i1, k).round().astype(int))
    targets = np.linspace(0, cum[-1], k, endpoint=False)
    idx = [i0 + int(np.searchsorted(cum, t)) for t in targets]
    out = []
    for i in idx:                       # 去重且保持递增
        i = min(i, i1)
        if not out or i > out[-1]:
            out.append(i)
    while len(out) < k and out[-1] < i1:
        out.append(out[-1] + 1)
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="H3 输出目录")
    ap.add_argument("-k", type=int, default=9, help="抽成几帧")
    ap.add_argument("--osc", type=int, default=2,
                    help="跨几个展宽波动。1 波动 = 1 步，"
                         "完整走路循环 = 2 步，所以默认 2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    fs = sorted(src.glob("f*.png"))
    if len(fs) < 8:
        raise SystemExit(f"帧不够：{src} 只有 {len(fs)} 张")

    # 实测：步数 = (帧数-5)/17
    n_steps = max(1, (len(fs) - 5) // 17)
    v = fill(np.array([spread(p) for p in fs]))
    i0, i1 = cycle_bounds(v, n_steps, args.osc)
    idx = pick_by_phase(v, i0, i1, args.k)

    dst = Path(args.out) if args.out else src.parent / f"{src.name}_pick{args.k}osc{args.osc}"
    dst.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(idx):
        shutil.copy(fs[i], dst / f"f{j:02d}.png")

    print(f"{src.name}: {len(fs)} 帧 / 推断 {n_steps} 步")
    print(f"  取 {args.osc} 步的区间 [{i0}, {i1}]（长度 {i1 - i0 + 1}）")
    print(f"  抽出帧号 {idx}")
    print(f"  展宽    {[f'{v[i] * 100:.0f}' for i in idx]}")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()

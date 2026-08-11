"""把 H3 抽出的帧量化成像素素材，并量化"颜色保持得怎么样"。

## 要解决的两件事

**一、帧间颜色一致。** 逐帧各减各的会让调色板逐帧漂移，播起来闪烁。
所以走 quantize.reduce()：所有帧**共用一套**调色板。

**二、颜色要保持成参考图那套。** 视频模型输出的色相跟参考图会有偏移。
两种做法都提供：

    自建调色板  从生成帧里跑 k-means 出调色板（默认）
    锁参考图    直接用参考图自己的调色板，生成帧只能吸附上去（--lock-ref）

后者更严格，代价是生成帧里参考图没有的颜色会被硬拉过去。

## 量什么

    独占色率   只在单帧出现过的颜色占比。真手绘像素动画 = 0.0%，
               这是"过渡帧糊掉"的数值形态 —— 用户最早抱怨的就是它
    色数       每帧实际用了几个颜色
    离参考图   量化后与参考图调色板的平均 OKLab 距离，越小越保住原色
    帧间调色板抖动  相邻帧用色集合的差异，衡量闪烁
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import quantize as Q

ROOT = Path(__file__).resolve().parent.parent


def bg_color(a: np.ndarray) -> np.ndarray:
    q = (a // 24 * 24).reshape(-1, 3)
    v, c = np.unique(q, axis=0, return_counts=True)
    return v[c.argmax()].astype(float)


def fg_mask(a: np.ndarray) -> np.ndarray:
    """前景 = 与背景主色差得远的像素。H3 出的是品红底。"""
    return np.abs(a.astype(int) - bg_color(a)).sum(2) > 90


def color_set(a: np.ndarray, m: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(x) for x in c) for c in np.unique(a[m], axis=0)}


def dekey(a: np.ndarray, lo: int = 90, hi: int = 260
          ) -> tuple[np.ndarray, np.ndarray]:
    """先去背景键，再量化。返回 (去污后的图, 前景掩码)。

    ## 为什么必须在量化之前

    品红底和角色色在边缘会混出一圈**中间色**（紫调、洋红调）。这圈像素
    落在前景掩码里，于是：
      - k-means 会拿出好几个槽位去表示这些紫调，真正的角色色反被挤掉
      - 描边、暗部被混成偏紫的深色，看起来"脏"
    量化之后再处理就晚了 —— 紫调已经固化成调色板里的正式颜色。

    ## 做法

      严格前景  离背景很远（> hi）的像素，一定不含背景成分
      宽松前景  离背景较远（> lo）的像素，含边缘那圈混色
      去污      宽松减严格的那一圈，用最近的**严格前景**像素的颜色填掉

    这样轮廓大小不变（还是宽松掩码），但边缘那圈的颜色换成了干净的角色色。
    """
    bg = bg_color(a)
    d = np.abs(a.astype(int) - bg).sum(2)
    strict, loose = d > hi, d > lo
    if not strict.any():
        return a.copy(), loose
    out = a.copy()
    src = strict.copy()
    fill = loose & ~strict
    # 从严格前景往外一圈圈长，最多 6 圈；每圈用 4 邻域已定色的均值
    for _ in range(6):
        if not fill.any():
            break
        p = np.pad(src, 1)
        nb = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
        edge = fill & nb
        if not edge.any():
            break
        pv = np.pad(np.where(src[..., None], out, 0).astype(np.int32),
                    ((1, 1), (1, 1), (0, 0)))
        pc = np.pad(src.astype(np.int32), 1)
        ssum = (pv[:-2, 1:-1] + pv[2:, 1:-1] + pv[1:-1, :-2] + pv[1:-1, 2:])
        scnt = (pc[:-2, 1:-1] + pc[2:, 1:-1] + pc[1:-1, :-2]
                + pc[1:-1, 2:])[..., None]
        out[edge] = (ssum[edge] / np.maximum(1, scnt[edge])).astype(np.uint8)
        src = src | edge
        fill = fill & ~edge
    return out, loose



def purge_key_colours(pal: np.ndarray, bg: np.ndarray,
                      min_cos: float = 0.95,
                      min_chroma: float = 0.15) -> np.ndarray:
    """剔除调色板里"跟背景同色相"的条目。

    去键处理的是**空间上**的混色环，但量化阶段仍可能把残余的键色
    单独分出一个槽位：实测品红底下会留下 (93,17,93) 和 (43,5,47) 两个
    紫调条目，共 220 像素、占前景 0.42%。这些点散在角色身上，
    像素画里非常刺眼。

    判据两条，缺一不可（实测数据，品红底 (249,5,251) 彩度 0.3170）：

        颜色              色相cos   彩度/背景   判定
        (93, 17, 93)      1.000     0.442      键色残留，剔
        (43,  5, 47)      0.997     0.274      键色残留，剔
        (2, 1, 2)         1.000     0.031      近黑色，色相虽同但几乎无彩度，留
        (162, 17, 12)     0.484     0.558      红腰带，色相差得远，留

    只看色相会误伤近黑色（它的色度向量方向不稳定）；只看彩度会漏掉
    暗一点的键色残留。第一版把彩度门槛设成 0.6，两个残留全漏了。
    """
    if len(pal) < 2:
        return pal
    lab = Q.to_oklab(pal)
    bl = Q.to_oklab(bg.reshape(1, 1, 3).astype(np.uint8)).reshape(3)
    bc = bl[1:]
    nb = np.linalg.norm(bc)
    if nb < 1e-6:
        return pal
    ch = lab[:, 1:]
    n = np.linalg.norm(ch, axis=1)
    cos = (ch @ bc) / np.maximum(1e-6, n * nb)
    # 同色相（夹角小）且本身有一定彩度的，判为键色残留
    keyish = (cos > min_cos) & (n > min_chroma * nb)
    if keyish.all():
        return pal
    return pal[~keyish]


def despeckle(m: np.ndarray, k: int = 2) -> np.ndarray:
    """去掉孤立像素：8 邻域里同类少于 k 个的翻面。

    生成图的轮廓外常挂着零星半透明残留，降采样后会变成孤立噪点，
    像素素材上特别显眼。
    """
    p = np.pad(m, 1)
    nb = sum(p[a:a + m.shape[0], b:b + m.shape[1]]
             for a in (0, 1, 2) for b in (0, 1, 2)) - m
    out = m.copy()
    out[m & (nb < k)] = False
    out[~m & (nb >= 7)] = True          # 内部空洞补上
    return out


def clean_specks(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """把孤立的单像素杂色并进邻域主色。

    降采样的块内多数表决在轮廓附近会留下零星异色点：实测一个颜色
    只在 1 帧出现、只占 1 个像素。像素画里这种孤点非常显眼，而且是
    帧间闪烁（独占色）的直接来源 —— 联合调色板保证了不引入新色，
    但保证不了每个色都在每帧出现。

    判据：8 邻域里与自己同色的邻居少于 2 个，就换成邻域里最多的那个色。
    """
    out = rgb.copy()
    h, w = m.shape
    pad_c = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    pad_m = np.pad(m, 1)
    same = np.zeros((h, w), np.int32)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            same += ((pad_c[dy:dy + h, dx:dx + w] == rgb).all(2)
                     & pad_m[dy:dy + h, dx:dx + w] & m)
    lone = m & (same < 2)
    for y, x in zip(*np.where(lone)):
        vals, cnt = {}, {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if (dy or dx) and 0 <= yy < h and 0 <= xx < w and m[yy, xx]:
                    t = tuple(int(v) for v in rgb[yy, xx])
                    cnt[t] = cnt.get(t, 0) + 1
                    vals[t] = rgb[yy, xx]
        if cnt:
            out[y, x] = vals[max(cnt, key=cnt.get)]
    return out


def ring_1px(m: np.ndarray) -> np.ndarray:
    """轮廓内圈：前景里 4 邻域挨着背景的那一圈，正好 1 像素宽。"""
    p = np.pad(m, 1)
    inner = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return m & ~inner


def add_outline(rgb: np.ndarray, m: np.ndarray,
                color: np.ndarray) -> np.ndarray:
    """把轮廓那一圈刷成描边色。

    ⚠️ 必须在**降采样之后**做，才能保证描边是 1 个精灵像素宽。
       降采样之前做，缩完描边会被块内表决吃掉或变成 1/s 像素宽。
    """
    out = rgb.copy()
    out[ring_1px(m)] = color
    return out


def report(frames: list[np.ndarray], masks: list[np.ndarray],
           ref_pal: np.ndarray | None, label: str) -> None:
    sets = [color_set(f, m) for f, m in zip(frames, masks)]
    allc: dict[tuple[int, int, int], int] = {}
    for s in sets:
        for c in s:
            allc[c] = allc.get(c, 0) + 1
    solo = sum(1 for c, n in allc.items() if n == 1)
    jitter = (np.mean([len(sets[i] ^ sets[i + 1]) for i in range(len(sets) - 1)])
              if len(sets) > 1 else 0.0)
    line = (f"  {label:<14} 总色数 {len(allc):>3}   "
            f"每帧色数 {np.mean([len(s) for s in sets]):>5.1f}   "
            f"独占色 {solo}/{len(allc)} = {solo / max(1, len(allc)) * 100:>5.1f}%   "
            f"帧间抖动 {jitter:>5.1f}")
    if ref_pal is not None and len(ref_pal):
        pool = np.concatenate([f[m] for f, m in zip(frames, masks) if m.any()])
        lab, rlab = Q.to_oklab(pool), Q.to_oklab(ref_pal)
        d = np.sqrt(((lab[:, None, :] - rlab[None]) ** 2).sum(-1)).min(1).mean()
        line += f"   离参考图 {d:.4f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="帧目录")
    ap.add_argument("--ref", default="assets/side_bear.png")
    ap.add_argument("--colors", type=int, default=16)
    ap.add_argument("--scale", type=int, default=0,
                    help="降采样倍数（0 = 不降）")
    ap.add_argument("--lock-ref", action="store_true",
                    help="锁定用参考图的调色板，不自建")
    ap.add_argument("--outline", action="store_true",
                    help="重建 1 像素描边（在降采样之后做）")
    ap.add_argument("--outline-color", default="",
                    help="描边色 R,G,B；留空则取调色板里最暗的色")
    ap.add_argument("--keep-key-colours", action="store_true",
                    help="保留跟背景同色相的调色板条目（默认剔除）")
    ap.add_argument("--no-clean", action="store_true",
                    help="不清降采样留下的孤立单像素杂色")
    ap.add_argument("--no-dekey", action="store_true",
                    help="不做去键（默认会先剥掉品红混色环再量化）")
    ap.add_argument("--alpha", action="store_true",
                    help="输出透明底 RGBA（精灵素材要的形态）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    fs = sorted(src.glob("f*.png"))
    if not fs:
        raise SystemExit(f"没有帧：{src}")
    raw = [np.asarray(Image.open(p).convert("RGB")) for p in fs]
    frames = [a.copy() for a in raw]
    masks = [fg_mask(f) for f in frames]
    if not args.no_dekey:
        # ⚠️ 必须在量化**之前**去键。品红底和角色色在边缘混出的一圈紫调
        #    会占掉 k-means 的槽位、把描边和暗部染脏，量化后就固化成
        #    调色板里的正式颜色，再想去掉已经晚了。这也是像素漂移的来源：
        #    那圈混色逐帧不同，量化时被吸到不同的槽位上。
        pairs = [dekey(f) for f in frames]
        frames = [p[0] for p in pairs]
        masks = [p[1] for p in pairs]

    rp = ROOT / args.ref if not Path(args.ref).is_absolute() else Path(args.ref)
    ref_pal = None
    if rp.is_file():
        ra = np.asarray(Image.open(rp).convert("RGB"))
        rm = fg_mask(ra)
        # 参考图自己就是像素图，它的调色板直接取唯一色即可；
        # 色太多（被压过或有渐变）时才 k-means 压到目标色数
        uniq = np.unique(ra[rm], axis=0)
        ref_pal = uniq if len(uniq) <= args.colors * 2 else \
            Q.palette(ra[rm], args.colors)

    print(f"{src.name}: {len(fs)} 帧 {frames[0].shape[1]}x{frames[0].shape[0]}"
          f"   参考图调色板 {len(ref_pal) if ref_pal is not None else 0} 色")
    report(frames, masks, ref_pal, "量化前")

    if args.lock_ref:
        if ref_pal is None:
            raise SystemExit("--lock-ref 需要参考图")
        pal = ref_pal
        out = [Q.apply(f, pal, m) for f, m in zip(frames, masks)]
    else:
        # 联合减色：所有帧共用一套调色板，避免逐帧漂移导致闪烁
        out = Q.reduce(frames, args.colors, masks)
        pool = np.concatenate([f[m] for f, m in zip(out, masks) if m.any()])
        pal = np.unique(pool, axis=0)
        if not args.keep_key_colours:
            # 去键清的是空间上的混色环，但量化仍可能给残余键色单独分槽位
            kept = purge_key_colours(pal, bg_color(raw[0]).astype(np.uint8))
            if len(kept) < len(pal):
                print(f"  剔除键色残留 {len(pal) - len(kept)} 个调色板条目")
                pal = kept
                out = [Q.apply(f, pal, m) for f, m in zip(out, masks)]
    report(out, masks, ref_pal, f"量化后{args.colors}色")

    if args.scale > 1:
        small, smasks = [], []
        for f, m in zip(out, masks):
            s, sm = Q.downsample(f, m, args.scale, pal)
            small.append(s)
            smasks.append(sm)
        out, masks = small, smasks
        # ⚠️ Q.downsample 对没有前景的块输出 0（纯黑），不保留原底色。
        #    不填回去的话，"取调色板里最暗的色"当描边 = 黑描边画在黑底上，
        #    肉眼和指标都看不见 —— 这个坑踩过一次。
        bg = bg_color(frames[0]).astype(np.uint8)
        for f, m in zip(out, masks):
            f[~m] = bg
        report(out, masks, ref_pal, f"降采样1/{args.scale}")

    if args.scale > 1 and not args.no_clean:
        # 降采样在轮廓附近会留下孤立异色点（实测有颜色只在 1 帧出现、
        # 只占 1 个像素），必须在描边之前清掉
        out = [clean_specks(f, m) for f, m in zip(out, masks)]
        report(out, masks, ref_pal, "清孤立杂色")

    if args.outline:
        # 先清掉孤立噪点，否则描边会沿着噪点长出毛刺
        masks = [despeckle(m) for m in masks]
        if args.outline_color:
            oc = np.array([int(x) for x in args.outline_color.split(",")],
                          np.uint8)
        else:
            oc = pal[int(np.argmin(Q.to_oklab(pal)[:, 0]))]
        out = [add_outline(f, m, oc) for f, m in zip(out, masks)]
        if not args.no_clean:
            # 描边会改变邻域构成，可能又孤立出新的单像素杂色 ——
            # 所以清理要在描边**之后**再跑一次，不能只跑降采样那一次
            out = [clean_specks(f, m) for f, m in zip(out, masks)]
        report(out, masks, ref_pal, f"描边 {tuple(int(x) for x in oc)}")

    dst = Path(args.out) if args.out else src.parent / (
        f"{src.name}_q{args.colors}"
        + ("lock" if args.lock_ref else "")
        + (f"_s{args.scale}" if args.scale > 1 else "")
        + ("_ol" if args.outline else "")
        + ("_a" if args.alpha else ""))
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for i, (f, m) in enumerate(zip(out, masks)):
        if args.alpha:
            # 硬 alpha：0 或 255，不留半透明 —— 半透明边就是"糊"的来源
            rgba = np.dstack([f, np.where(m, 255, 0).astype(np.uint8)])
            Image.fromarray(rgba, "RGBA").save(dst / f"f{i:02d}.png")
        else:
            Image.fromarray(f).save(dst / f"f{i:02d}.png")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()

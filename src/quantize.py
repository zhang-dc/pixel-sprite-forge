"""在感知色彩空间里做减色，而不是在 RGB 里。

## 为什么不能在 RGB 里量化

原来的做法是 PIL 中位切分出调色板，再按 **RGB 的 L1 距离**吸附像素。
两个毛病：

**距离不对应视觉。** 白 (246,245,243) 到洋红 (248,5,249) 的 L1 是 263，
到灰 (155,149,141) 是 320 —— 数值上洋红更近，因为 R/B 两通道几乎相同、
只差 G。于是熊的白眼睛被吸成了洋红。RGB 三通道等权，
而人眼对绿最敏感、对蓝最不敏感，等权本身就是错的。

**槽位按体积分。** 中位切分按颜色在 RGB 空间里占的**体积**切，
面积小但独特的颜色（眼白、点缀色、饰边）会被并进大块里。
实测 20 色时骑士胸前的红战袍会整块消失。

## 改成什么

  空间   OKLab —— 为"等距离 = 等视觉差"设计的空间，比 Lab 更均匀，
         转换只有几步矩阵乘和立方根，比色貌模型便宜得多
  分槽   在 OKLab 里跑 k-means（用中位切分的结果做初值），
         按**感知误差**分配槽位而不是按体积
  吸附   OKLab 里的欧氏最近邻

代价是比纯 PIL 慢一些（这些图尺寸下是毫秒级），换来的是不再出现
"白变洋红"这类漂移。
"""
from __future__ import annotations

import numpy as np

# sRGB -> 线性 -> LMS -> OKLab 的常数，取自 Björn Ottosson 的定义
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..,3) uint8 sRGB -> (..,3) float OKLab。"""
    s = rgb.astype(np.float64) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    lms = lin @ _M1.T
    return np.cbrt(np.clip(lms, 0, None)) @ _M2.T


def nearest(px_lab: np.ndarray, pal_lab: np.ndarray) -> np.ndarray:
    """每个像素在调色板里的最近邻下标（OKLab 欧氏）。分块算，别把内存吃光。"""
    out = np.empty(len(px_lab), np.int64)
    step = max(1, 2_000_000 // max(1, len(pal_lab)))
    for i in range(0, len(px_lab), step):
        d = px_lab[i:i + step, None, :] - pal_lab[None]
        out[i:i + step] = (d * d).sum(-1).argmin(1)
    return out


def palette(px: np.ndarray, colors: int, iters: int = 12,
            seed: int = 0) -> np.ndarray:
    """在 OKLab 里求调色板，返回 (n,3) uint8 sRGB。

    初值用**均匀分位采样**而不是随机点：随机初值在色数少时容易
    整簇塌到同一个颜色上，表现为"调色板里有重复色、实际可用色更少"。
    """
    if len(px) == 0:
        return np.zeros((0, 3), np.uint8)
    lab = to_oklab(px)
    k = int(min(colors, len(np.unique(px, axis=0))))
    if k <= 1:
        return np.unique(px, axis=0)[:1]

    order = np.argsort(lab[:, 0])            # 按明度排，取等分位当初值
    cen = lab[order[np.linspace(0, len(order) - 1, k).astype(int)]].copy()

    for _ in range(iters):
        idx = nearest(lab, cen)
        moved = 0.0
        for c in range(k):
            m = idx == c
            if not m.any():
                # 空簇：抢误差最大的那个像素，避免簇数悄悄变少
                d = ((lab - cen[idx]) ** 2).sum(-1)
                cen[c] = lab[int(np.argmax(d))]
                continue
            new = lab[m].mean(0)
            moved = max(moved, float(np.abs(new - cen[c]).max()))
            cen[c] = new
        if moved < 1e-4:
            break

    # 簇心转回 sRGB：取该簇里**离簇心最近的真实像素**，
    # 而不是簇心本身的逆变换 —— 逆变换会造出原图里不存在的颜色，
    # 像素画里那是新的杂色。
    idx = nearest(lab, cen)
    out = []
    for c in range(k):
        m = idx == c
        if not m.any():
            continue
        d = ((lab[m] - cen[c]) ** 2).sum(-1)
        out.append(px[m][int(np.argmin(d))])
    return np.array(out, np.uint8)


def apply(rgb: np.ndarray, pal: np.ndarray, mask: np.ndarray | None = None
          ) -> np.ndarray:
    """把图吸附到调色板上（OKLab 最近邻）。mask 外的像素原样保留。"""
    out = rgb.copy()
    m = np.ones(rgb.shape[:2], bool) if mask is None else mask
    if not m.any() or not len(pal):
        return out
    out[m] = pal[nearest(to_oklab(rgb[m]), to_oklab(pal))]
    return out


def outline_mask(rgb: np.ndarray, keep: np.ndarray,
                 drop: float = 0.16) -> np.ndarray:
    """描边像素：明显暗于**局部**邻域的那些。

    ⚠️ 不能用全局阈值。角色身上本来就有深色区域（黑毛、暗甲），
       全局判暗会把整块深色区当描边。描边的特征是"比周围暗一截"，
       是个局部对比关系。

    用 OKLab 的 L 通道：先取 3x3 邻域的最大 L 当作"周围有多亮"，
    比它低 drop 以上的算描边。drop 用 OKLab 的绝对量纲，
    与色相无关，深色和浅色区域用同一把尺子。
    """
    lab_l = to_oklab(rgb)[..., 0]
    m = lab_l.copy()
    m[~keep] = -1.0
    loc = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.roll(m, (dy, dx), (0, 1))
            if dy:
                (sh[0] if dy == 1 else sh[-1])[:] = -1.0
            if dx:
                (sh[:, 0] if dx == 1 else sh[:, -1])[:] = -1.0
            loc = np.maximum(loc, sh)
    return keep & (loc - lab_l > drop)


def _thin(m: np.ndarray) -> np.ndarray:
    """把 2 像素宽的描边收成 1 像素：去掉四邻域全是描边的内部点。

    降采样时"块内有描边就算描边"会让描边加粗一倍，
    加粗的描边在像素画里比断掉还难看。只去内部点，端点和拐角保留，
    所以不会把线打断。
    """
    inner = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(m, (dy, dx), (0, 1))
        if dy:
            (sh[0] if dy == 1 else sh[-1])[:] = False
        if dx:
            (sh[:, 0] if dx == 1 else sh[:, -1])[:] = False
        inner &= sh
    return m & ~inner


def downsample(rgb: np.ndarray, keep: np.ndarray, s: int, pal: np.ndarray,
               edge_t: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """**先量化后降采样**的块内多数表决。返回 (小图, 掩码)。

    ⚠️ 顺序很要紧。原来是先降采样再量化：块内取原始色的众数，
       而 s=2 时一块只有 4 个像素，实测**过半的块没有明显多数**
       （把握度中位 0.50，64~83% 的块低于 0.6）。平票时选谁由
       Counter 的插入顺序决定 —— 等于随机，这就是像素漂移的源头。

       先量化则块内只剩调色板里的少数几个色，出现明显多数的概率大得多。

    平票仍会有，按**更暗的色优先**破除。像素画里描边是深色的细线，
    平票时选亮色会把描边啃掉；选暗色顶多让描边粗一点，那是可接受的。
    """
    H, W = rgb.shape[0] // s, rgb.shape[1] // s
    idx = np.full(rgb.shape[:2], -1, np.int32)
    if keep.any() and len(pal):
        idx[keep] = nearest(to_oklab(rgb[keep]), to_oklab(pal))

    # 调色板按明度排序，序号小 = 更暗，平票时取序号小的
    order = np.argsort(to_oklab(pal)[:, 0])
    rank = np.empty(len(pal), np.int32)
    rank[order] = np.arange(len(pal))
    ranked = np.where(idx >= 0, rank[np.clip(idx, 0, None)], -1)

    # ⚠️ 描边要单独走一条通道。描边是**细而暗的线**，块内多数表决时
    #    它天然是少数派，一定会被吃掉 —— 表现为轮廓断断续续、
    #    内部色块糊到一起。所以先把描边识别出来，
    #    块内描边占比超过阈值就整块判为描边，其余块才走多数表决。
    edge = outline_mask(rgb, keep)
    ei = edge[:H * s, :W * s].reshape(H, s, W, s).mean((1, 3))

    # ⚠️ 内部块的表决必须**排除描边像素**。一个块只要沾到几个描边像素，
    #    那些暗像素就在参与投票，把内部色拉暗、拉串 ——
    #    表现为描边旁边一圈颜色发脏、色块之间互相污染。
    #    描边有自己的通道，不该再在内部通道里出现第二次。
    inner = keep & ~edge
    ranked_in = np.where(inner, ranked, -1)

    ri = ranked_in[:H * s, :W * s].reshape(H, s, W, s).transpose(0, 2, 1, 3)
    ri = ri.reshape(H, W, s * s)
    ki = keep[:H * s, :W * s].reshape(H, s, W, s).mean((1, 3))
    # 描边像素的调色板序号（用于给描边块取色）
    re = np.where(edge[:H * s, :W * s], ranked[:H * s, :W * s], -1)
    re = re.reshape(H, s, W, s).transpose(0, 2, 1, 3).reshape(H, W, s * s)
    # 内部像素被排干净时的兜底：整块都是描边，那就按描边处理
    ra = ranked[:H * s, :W * s].reshape(H, s, W, s).transpose(0, 2, 1, 3)
    ra = ra.reshape(H, W, s * s)

    msk = ki >= 0.5
    is_edge = _thin(msk & (ei >= edge_t))   # 收成 1 像素，别把描边加粗一倍

    out = np.zeros((H, W, 3), np.uint8)
    ys, xs = np.nonzero(msk)
    for y, x in zip(ys, xs):
        v = re[y, x] if is_edge[y, x] else ri[y, x]
        v = v[v >= 0]
        if not len(v):                      # 该通道无票，退回全部像素
            v = ra[y, x]
            v = v[v >= 0]
        if not len(v):
            msk[y, x] = False
            continue
        u, c = np.unique(v, return_counts=True)      # unique 已按序号升序
        out[y, x] = pal[order[int(u[int(np.argmax(c))])]]
    return out, msk


def reduce(frames: list[np.ndarray], colors: int,
           masks: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """多帧**联合**减色：共用一套调色板。

    联合而不是逐帧，是为了让帧间用色一致 —— 逐帧各减各的会让调色板
    逐帧漂移，播起来就是闪烁。
    """
    if masks is None:
        masks = [np.ones(f.shape[:2], bool) for f in frames]
    pool = np.concatenate([f[m] for f, m in zip(frames, masks)]) \
        if any(m.any() for m in masks) else np.zeros((0, 3), np.uint8)
    pal = palette(pool, colors)
    return [apply(f, pal, m) for f, m in zip(frames, masks)]

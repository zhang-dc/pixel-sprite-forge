"""MiniMax H3（Ref2VA）—— 参考图 + 提示词 -> 动画帧。

## 为什么试这个

Wan 这条链上姿态图是死的（适配器 0 个姿态权重，全静止/倒序/正常走路
输出逐像素相同），动作只能靠提示词。H3 是 2026-08-02 开的权重，
Ref2VA 变体原生吃「参考图 + 文字」，正好是我们要的形态。

## 帧数被锁死在 17k+5 的网格上

    def align_frame_count(n):
        while n % 17 != 5:
            n += 1

合法值只有 5, 22, 39, 56, 73, 90, 107, 124 ...
**请求 9 帧会被顶到 22 帧**，我们要的 9~13 帧拿不到。
节点自己的提示写着训练区间是 124~362 帧（5~15 秒），22 帧远低于训练区间。
所以这里默认跑 22 帧，再自己抽成 9~11 帧的精灵序列。

## 图的结构（照官方 ref2va 工作流搭）

    UnetLoaderGGUF / UNETLoader ─┐
    CLIPLoaderGGUF(minimax) ─────┼─> MiniMaxH3ReferenceToVideo ─> BasicGuider ─┐
    VAELoader(video) ────────────┤          ^ ref_image_0                      │
    VAELoader(audio) ────────────┘                                             v
    KSamplerSelect(res_multistep) + BasicScheduler(simple) ─> SamplerCustomAdvanced
                                                                     │
                                                        VAEDecode(video) -> SaveImage

⚠️ 官方图用的是 **BasicGuider**：没有负面提示词、CFG=1。别照 Wan 的习惯
   接 CFGGuider，那会要求 negative 而 H3 的 conditioning 里没有。

## 显存纪律

一次生成跑完显存不释放（WSL 不还给宿主），下一个配置前必须
`wsl --shutdown`。不清的话采样会从正常速度退化到几十秒一步，
表现出来就是"这个参数没效果" —— 今天已有多轮结论被这个污染过。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "trainset" / "h3"
BASE = "http://127.0.0.1:8189"

MODEL_GGUF = "minimax_h3_ref2va_pruned-Q4_K_M.gguf"
MODEL_FP8 = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
# ⚠️ leejet 那个 GGUF 文本编码器是给 stable-diffusion.cpp 的，
#    ComfyUI-GGUF 直接拒收：
#        ValueError: This gguf file is incompatible with llama.cpp!
#    所以走 safetensors，原生 CLIPLoader（type=minimax）读。
#    我推荐那个仓库时只看了体积和下载速度，没验格式 —— 这是疏忽。
CLIP_GGUF = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
CLIP_ST = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

# <Picture 1> 是 H3 约定的参考图引用写法，节点文档里写明了要在提示词里
# 用同样的标记去指代参考图。
PROMPT = (
    "The pixel art game character in <Picture 1> walking in place, "
    "side view, full body, seen from the side. "
    "Keep the exact same character design, colours and pixel art style "
    "as <Picture 1>. One complete walk cycle, ending in the same pose "
    "it started in. Flat solid magenta background, static camera, "
    "the character stays centred and does not move across the frame."
)

# 要的是**动作覆盖量**，不是帧数。同样的帧数里走更多步，动作密度才上去；
# 光把帧数拉长而步频不变，只是把同一段慢动作摊得更长，等于没变。
# 所以按步数下指令，帧数固定。
# 描边/锐利度的提示词档。
#
# ⚠️ H3 走 BasicGuider —— **没有负面提示词、CFG=1**，所以"不要抗锯齿"
#    这类否定只能写在正面里，不能像 Wan 那样丢进 negative。
#
# 背景：参考图自己就不是干净的像素画（215x311 有 12864 个不同颜色，
# 近 1/5 的像素颜色独一无二），边缘过渡宽度 2.02px；生成后放大到 3.97px。
# 现在的锐利全靠后处理造出来，这里试试能不能在生成阶段就保住。
EDGE_TXT = {
    "none": "",
    # 只要描边
    "ol": (" Every shape has a bold solid black outline one pixel thick, "
           "including around each arm and each leg."),
    # 描边 + 硬边填色 + 明确否定抗锯齿
    "hard": (" Every shape has a bold solid black outline. "
             "All colours are flat hard-edged fills with no gradients, "
             "no soft shading, no anti-aliasing and no blur. "
             "Every edge is a sharp crisp pixel step, high contrast."),
}


# 视角档。参考图必须跟着换 —— H3 靠参考图保持身份，用侧面参考图去要
# 正面动画，模型只能自己编一个正面，多半不像。
#
# 抽帧器（pick_cycle）用**下半身水平展宽**当步态相位：侧面走路时两腿
# 前后分开，展宽振荡明显；正面/背面走路两腿是左右交替、前后位移在画面上
# 被压缩，展宽振荡会弱很多。所以正背面的抽帧可能需要换相位判据，
# 这一点尚未验证。
VIEWS = {
    "side": ("side view, full body, seen from the side",
             "walking in place, the character stays centred and does not "
             "move across the frame"),
    "front": ("front view, facing the camera, full body",
              "walking towards the camera on the spot, the character stays "
              "centred and does not get closer or further away"),
    "back": ("back view, seen from behind, full body",
             "walking away from the camera on the spot, the character stays "
             "centred and does not get closer or further away"),
    "quarter": ("three-quarter view, turned 45 degrees away from the camera, "
                "full body",
                "walking forward on the spot, the character stays centred "
                "and does not move across the frame"),
}


def walk_prompt(steps: int, view: str = "side") -> str:
    angle, motion = VIEWS.get(view, VIEWS["side"])
    return (
        f"The pixel art game character in <Picture 1> {motion}, {angle}. "
        f"Keep the exact same character design, colours and pixel art "
        f"style as <Picture 1>. "
        f"The character takes {steps} full steps during this clip, "
        f"alternating left foot and right foot, lifting each knee high "
        f"and swinging the legs wide apart between steps. "
        f"Flat solid magenta background, static camera."
    )


# 原地转一整圈（下游 pick_rotation.py）。--ref 给 2×2 四视图拼图；
# 朝向顺序逐个写出来，转向才稳定是顺时针。
def spin_prompt(subject: str) -> str:
    order = ("first the front points to the right, then the front turns "
             "toward the viewer, then the front points to the left, then "
             "the front points away from the viewer, and finally the front "
             "points to the right again")
    return (
        f"<Picture 1> is a reference sheet showing one {subject} from four "
        f"sides: front right, front toward the viewer, front left, front "
        f"away from the viewer. The video shows only one single {subject}, "
        f"alone in the center, seen by a fixed high three-quarter camera. "
        f"It turns in place clockwise as seen from above through one full "
        f"circle: {order}. Steady even rotation speed. It looks exactly "
        f"like the one in <Picture 1> from every side and keeps the same "
        f"size. The camera does not move. Flat solid magenta background."
    )


def align(n: int) -> int:
    """H3 的帧数网格：17k+5。"""
    while n % 17 != 5:
        n += 1
    return n


def upload(p: Path, as_name: str = "") -> str:
    b = uuid.uuid4().hex[:8]
    fn = as_name or p.name
    head = (f'--{b}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{fn}"\r\nContent-Type: image/png\r\n\r\n')
    tail = (f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"'
            f'\r\n\r\ntrue\r\n--{b}--\r\n')
    req = urllib.request.Request(
        BASE + "/upload/image",
        data=head.encode() + p.read_bytes() + tail.encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["name"]


def graph(ref: str, prompt: str, w: int, h: int, n: int, seed: int,
          steps: int, gguf: bool, shift_v: float) -> dict:
    if gguf:
        loader = {"class_type": "UnetLoaderGGUF",
                  "inputs": {"unet_name": MODEL_GGUF}}
        clip = {"class_type": "CLIPLoaderGGUF",
                "inputs": {"clip_name": CLIP_GGUF, "type": "minimax"}}
    else:
        loader = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": MODEL_FP8,
                             "weight_dtype": "default"}}
        clip = {"class_type": "CLIPLoader",
                "inputs": {"clip_name": CLIP_ST, "type": "minimax"}}
    g: dict = {
        "1": loader,
        "2": clip,
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "5": {"class_type": "LoadImage", "inputs": {"image": ref}},
        # 参考图走 Autogrow。键名必须是**点号拼接的扁平键**：
        #     ref_images.ref_image_0
        # 依据 comfy_api/latest/_io.py:
        #     finalize_prefix() 用 "." join，parse_class_inputs() 把
        #     Autogrow 的模板名拼在其 id 之后，build_nested_inputs()
        #     再按 path.split(".") 还原成嵌套 dict 传给 execute()。
        # 试错记录（两次都不报错，但都是错的）：
        #     "ref_image_0": [...]              -> unexpected keyword argument
        #     "ref_images": {"ref_image_0":...} -> **静默丢弃**，退化成纯文生视频
        # 后者最阴险：不报错、能出图、动作还不错，但换成完全不同的参考图
        # 输出逐像素相同（差 0.00），参考图根本没参与。
        "6": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                         "audio_vae": ["4", 0], "prompt": prompt,
                         "width": w, "height": h, "length": n,
                         "ref_image_size": "match",
                         "ref_images.ref_image_0": ["5", 0]}},
        "7": {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["1", 0], "shift_video": shift_v,
                         "shift_audio": 3.0}},
        # ⚠️ BasicGuider：无负面提示词、CFG=1，官方图就是这么接的
        "8": {"class_type": "BasicGuider",
              "inputs": {"model": ["7", 0], "conditioning": ["6", 0]}},
        "9": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "BasicScheduler",
               "inputs": {"model": ["7", 0], "scheduler": "simple",
                          "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["11", 0], "guider": ["8", 0],
                          "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode",
               "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0], "filename_prefix": "h3"}},
    }
    return g


def run_one(g: dict, timeout: int = 2400) -> list[dict]:
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/prompt", data=json.dumps({"prompt": g}).encode(),
        headers={"Content-Type": "application/json"}), timeout=60).read())
    pid = r["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/history/{pid}",
                                        timeout=30) as h:
                hist = json.loads(h.read())
        except urllib.error.URLError:
            time.sleep(3)
            continue
        if pid in hist:
            imgs = [i for v in hist[pid].get("outputs", {}).values()
                    for i in v.get("images", [])]
            if imgs:
                return imgs
            raise RuntimeError(json.dumps(hist[pid].get("status", {}),
                                          ensure_ascii=False)[:900])
        time.sleep(3)
    raise TimeoutError(f"超时 {timeout}s")


def fetch(info: dict) -> bytes:
    q = urllib.parse.urlencode({"filename": info["filename"],
                                "subfolder": info.get("subfolder", ""),
                                "type": info.get("type", "output")})
    with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=180) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="assets/side_bear.png")
    ap.add_argument("--frames", type=int, default=22,
                    help="会被顶到 17k+5 网格上：5, 22, 39, 56 ...")
    ap.add_argument("--width", type=int, default=512, help="须为 32 的倍数")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=float, default=12.0)
    # 默认走 safetensors（fp8 主模型 + nvfp4 文本编码器），不碰 GGUF ——
    # 手上那两个 GGUF 是 sd.cpp 格式，ComfyUI 读不了
    ap.add_argument("--gguf", action="store_true",
                    help="改用 GGUF 权重（目前手上这份不兼容，会报错）")
    ap.add_argument("--prompt", default="")
    # 动作覆盖量的旋钮：同样帧数里走几步
    ap.add_argument("--walk", type=int, default=0,
                    help="要求角色在这段里走几步（0 = 用默认提示词）")
    ap.add_argument("--view", default="side", choices=sorted(VIEWS),
                    help="视角。参考图要跟着换成对应朝向那张")
    # 整圈要 158 帧：开头约 20~36 帧是原地晃，124 帧实测只转到 ~300°
    ap.add_argument("--spin", default="",
                    help="原地转一圈，填主体描述，如 'small wooden sailing "
                         "ship'；--ref 给四视图拼图。建议 --frames 158")
    ap.add_argument("--edge", default="none", choices=sorted(EDGE_TXT),
                    help="描边/锐利度档，见 EDGE_TXT")
    ap.add_argument("--tag", default="h3")
    args = ap.parse_args()

    for v, name in ((args.width, "width"), (args.height, "height")):
        if v % 32:
            sys.exit(f"{name}={v} 不是 32 的倍数")

    try:
        with urllib.request.urlopen(BASE + "/system_stats", timeout=10):
            pass
    except Exception:                                       # noqa: BLE001
        sys.exit(f"ComfyUI 没在跑（{BASE}）")

    rp = ROOT / args.ref
    if not rp.is_file():
        sys.exit(f"参考图不存在：{rp}")
    ref_name = upload(rp)

    n = align(max(5, args.frames))
    if n != args.frames:
        print(f"⚠️ 帧数 {args.frames} 不在 17k+5 网格上，顶到 {n}")

    prompt = args.prompt or (spin_prompt(args.spin) if args.spin else
                             walk_prompt(args.walk, args.view)
                             if args.walk else PROMPT)
    prompt += EDGE_TXT[args.edge]
    # ⚠️ 走几步必须进目录名，否则不同配置写进同一个目录互相覆盖
    tag = (f"{args.tag}_{'gguf' if args.gguf else 'fp8'}"
           f"_{args.width}x{args.height}_n{n}_s{args.steps}"
           f"_sh{args.shift:g}_seed{args.seed}"
           + (f"_w{args.walk}" if args.walk else "")
           + (f"_e{args.edge}" if args.edge != "none" else "")
           + (f"_{args.view}" if args.view != "side" else "")
           + ("_spin" if args.spin else ""))
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)

    print(f"参考图 {rp.name} · {n} 帧 · {args.width}x{args.height} · "
          f"{args.steps} 步 · shift {args.shift}")
    t0 = time.time()
    print(f"  提示词: {prompt}")
    g = graph(ref_name, prompt, args.width, args.height, n,
              args.seed, args.steps, args.gguf, args.shift)
    try:
        imgs = run_one(g)
    except Exception as e:                                  # noqa: BLE001
        print(f"  失败 {type(e).__name__}: {str(e)[:800]}")
        return
    for i, info in enumerate(imgs):
        (d / f"f{i:02d}.png").write_bytes(fetch(info))
    print(f"  {len(imgs)} 帧 / {time.time() - t0:.0f}s -> {d}")


if __name__ == "__main__":
    main()

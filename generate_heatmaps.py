"""生成异常检测热力图：把 patch 级异常图叠加到原图上。

训练我方方法（轻量 Adapter），然后对测试集正常/异常样本各取几张，
输出 224×224 的异常热力图（jet 叠加）。
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.model import TextSideAnomalyModel, load_trained_model
from text_side_anomaly.prompts import BRAIN_MRI_PROMPT_SETS, ThreeLevelPrompts

PNEUMONIA_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest x-ray"],
        "2": ["a normal chest x-ray with clear lungs"],
        "3": ["a normal chest x-ray with sharp lung markings"],
    },
    abnormal={
        "1": ["a chest x-ray with pneumonia"],
        "2": ["a chest x-ray with pulmonary opacities"],
        "3": ["a chest x-ray with hazy infiltrates"],
    },
)


def train_model(device, epochs=6, batch_size=64):
    cfg = Config(device=str(device), epochs=epochs, batch_size=batch_size, lr=1e-3)
    model = TextSideAnomalyModel(cfg).to(device)
    anchors = PNEUMONIA_PROMPTS
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = TotalLoss(margin=cfg.margin, w_text=cfg.w_text, w_global=cfg.w_global, w_local=0.0)

    ds = SliceAnomalyDataset("data_pneumonia/train", mask_root=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            enc = model.encode_anchors(anchors)
            out = model(images, enc)
            loss = crit(enc, out, labels, None)
            opt.zero_grad()
            loss["total"].backward()
            opt.step()
    return model, anchors


@torch.no_grad()
def anomaly_map_for(model, anchors, image_tensor, device):
    model.eval()
    enc = model.encode_anchors(anchors)
    out = model(image_tensor.unsqueeze(0).to(device), enc)
    m = out["anomaly_map"][0].cpu().numpy()          # (14,14)
    score = out["cls_probs"][0].item()
    return m, score


# 局部异常图 (pa - pn) 的固定尺度：|amap| 达到 AMAP_SCALE 即视为满热，<=0 全冷。
# 不要用逐图 min-max，否则正常图也会被拉出一块最红区域。
AMAP_SCALE = 0.3


def overlay_heatmap(gray_pil: Image.Image, amap: np.ndarray, score: float, out_path: str):
    """把 (14,14) 异常图用全局固定尺度映射到 [0,1]，jet 叠加到灰度图上。

    amap 是 (pa - pn) 余弦相似度差，>0 表示 patch 更靠近异常锚点。
    按图像级分数 score 做全局门控：正常图(score 低)整体保持冷色。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    h, w = gray_pil.size[1], gray_pil.size[0]

    # 固定尺度归一化（非逐图 min-max）
    local = np.clip(amap / AMAP_SCALE, 0.0, 1.0)
    up = Image.fromarray((local * 255).astype(np.uint8)).convert("L")
    up = up.resize((w, h), Image.BILINEAR)
    norm = np.asarray(up, dtype=np.float32) / 255.0
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)   # RGB 热力图

    # 全局门控：score<0.5 时热力逐渐熄灭，正常图保持冷色
    gate = float(np.clip((score - 0.5) * 2.0, 0.0, 1.0))
    alpha = 0.45 * gate

    gray = np.asarray(gray_pil.convert("RGB"), dtype=np.float32)
    blend = (1.0 - alpha) * gray + alpha * heat
    Image.fromarray(blend.astype(np.uint8)).save(out_path)


def load_model(path, device):
    """按 checkpoint 记录的文本/视觉适配器重建，避免出图时丢掉视觉分支。"""
    model = load_trained_model(path, device)
    prompt_name = (model.checkpoint_meta or {}).get("prompt_set")
    if prompt_name in BRAIN_MRI_PROMPT_SETS:
        anchors = BRAIN_MRI_PROMPT_SETS[prompt_name]
    elif prompt_name == "pneumonia":
        anchors = PNEUMONIA_PROMPTS
    else:
        from thymoma_local import PROMPT_SETS
        if prompt_name not in PROMPT_SETS:
            raise RuntimeError(f"checkpoint 提示词版本缺失或未知：{prompt_name!r}，无法选择出图锚点")
        anchors = PROMPT_SETS[prompt_name]
    print(f"[heatmap] 使用 checkpoint 提示词 {prompt_name}")
    if list(anchors.levels) != list(model.cfg.levels):
        raise RuntimeError(
            f"提示词层 {list(anchors.levels)} 与 checkpoint 的 {list(model.cfg.levels)} 不一致"
        )
    print(f"[heatmap] 已加载 {path} stage={model.training_stage}")
    return model, anchors


def heatmap_route(args) -> str:
    """--layout four_panel 走四列图。已导出的预测只重新排版。"""
    if getattr(args, "predictions", None):
        return "replay"
    if getattr(args, "layout", None) == "four_panel":
        return "checkpoint_four_panel"
    return "single_overlay"


def _preview_files(data_root: str):
    npz_files = sorted(
        os.path.join(data_root, name)
        for name in os.listdir(data_root)
        if name.lower().endswith(".npz")
    )
    if npz_files:
        return [("slice", path) for path in npz_files[:6]]
    rows = []
    for cls in ("normal", "abnormal"):
        folder = os.path.join(data_root, cls)
        if not os.path.isdir(folder):
            continue
        names = sorted(os.listdir(folder))[:3]
        rows.extend((cls, os.path.join(folder, name)) for name in names)
    return rows


def _load_preview(path: str, image_size: int):
    if path.lower().endswith(".npz"):
        payload = np.load(path)
        gray = payload["image"].astype(np.float32)
        gt = payload["mask224"].astype(np.float32) if "mask224" in payload.files else None
        return gray, gt, os.path.basename(path)
    gray_image = Image.open(path).convert("L").resize((image_size, image_size))
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    return gray, None, os.path.basename(path)


def save_checkpoint_four_panel(model, anchors, data_root, device, out_dir, args):
    """checkpoint 直接出图也走后处理四列，不看图像分数把门控关掉。"""
    from text_side_anomaly.postprocess import OperatingThresholds, postprocess_one, split_manual_thresholds
    from text_side_anomaly.visualize import render_four_panel, save_contact_sheet
    import thymoma_local as thymoma

    files = _preview_files(data_root)
    if not files:
        raise RuntimeError(f"{data_root} 里没有可出图的样本")
    meta = dict(getattr(model, "checkpoint_meta", None) or {})
    meta["topk_ratio"] = float(getattr(args, "topk_ratio", 0.05))
    meta["mask_mode"] = getattr(args, "mask_mode", "fixed")
    meta["score_mode"] = getattr(args, "score_mode", None)
    meta["global_weight"] = getattr(args, "global_weight", None)
    config = thymoma.postprocess_config_for_checkpoint(meta)
    spaces = split_manual_thresholds(
        getattr(args, "image_threshold", None),
        getattr(args, "pixel_threshold", None),
        score_space=getattr(args, "score_space", None),
        post_image_threshold=getattr(args, "post_image_threshold", None),
        post_pixel_threshold=getattr(args, "post_pixel_threshold", None),
        raw_image_threshold=getattr(args, "raw_image_threshold", None),
        raw_pixel_threshold=getattr(args, "raw_pixel_threshold", None),
    )
    post = spaces["post_v1_probability_score"]
    operating = OperatingThresholds(
        image_threshold=post["image_threshold"],
        pixel_threshold=post["pixel_threshold"],
        seed_threshold=getattr(args, "seed_threshold", None),
        grow_floor=getattr(args, "grow_floor", None),
        score_space="post_v1_probability_score",
        image_threshold_source="preset" if post["image_threshold"] is not None else "none",
        pixel_threshold_source="preset" if post["pixel_threshold"] is not None else "none",
    )
    temperature = float(getattr(model.cfg, "temperature", 0.07))
    image_size = int(model.cfg.image_size)
    os.makedirs(out_dir, exist_ok=True)
    panels = []
    for _cls, path in files:
        gray, gt, name = _load_preview(path, image_size)
        tensor = _clip_tensor(gray)
        amap, score = anomaly_map_for(model, anchors, tensor, device)
        prediction = postprocess_one(amap, score, gray, temperature, config, operating)
        info = prediction.get("mask_info") or {}
        panel = render_four_panel(
            gray, prediction["score_map_high"], prediction.get("pred_mask"), gt, gt is not None,
            mask_status=info.get("mask_status", "unavailable"),
            sample_title=name,
        )
        Image.fromarray(panel).save(os.path.join(out_dir, f"{_cls}_{name}_four.png"))
        panels.append(panel)
    if not panels:
        raise RuntimeError("没有预测记录，不能把出图标成完成")
    save_contact_sheet(
        panels, os.path.join(out_dir, "overview.png"),
        "Original | Heatmap | Heatmap + GT | Refined mask + GT",
        "局部异常分数 0–1",
    )
    print(f"[heatmap] 四列图已写入 {out_dir}，未使用单幅叠加")


def _clip_tensor(gray: np.ndarray) -> torch.Tensor:
    image = np.stack([gray, gray, gray], axis=0).astype(np.float32)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    array = (image - mean[:, None, None]) / std[:, None, None]
    return torch.from_numpy(array).float()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="加载已训练 checkpoint。会按其中的架构装上视觉层内适配器")
    parser.add_argument("--data-root", type=str, default=None,
                        help="含 normal/abnormal 子目录的二维切片目录；非肺炎 checkpoint 必须指定")
    parser.add_argument("--out-dir", type=str, default="heatmaps", help="热力图输出目录")
    parser.add_argument("--predictions", type=str, default=None,
                        help="已导出的 predictions.jsonl。给出后只重新排版，不再加载模型")
    parser.add_argument("--layout", type=str, default=None, choices=[None, "four_panel"])
    parser.add_argument("--image-threshold", type=float, default=None)
    parser.add_argument("--pixel-threshold", type=float, default=None)
    parser.add_argument("--score-space", default=None, choices=["raw_margin", "post_v1_probability_score"])
    parser.add_argument("--post-image-threshold", type=float, default=None)
    parser.add_argument("--post-pixel-threshold", type=float, default=None)
    parser.add_argument("--raw-image-threshold", type=float, default=None)
    parser.add_argument("--raw-pixel-threshold", type=float, default=None)
    parser.add_argument("--seed-threshold", type=float, default=None)
    parser.add_argument("--grow-floor", type=float, default=None)
    parser.add_argument("--mask-mode", default="fixed", choices=["fixed", "adaptive_seeded"])
    parser.add_argument("--topk-ratio", type=float, default=0.05)
    parser.add_argument("--score-mode", default=None, choices=["local_topk", "fusion", "cls_baseline"])
    parser.add_argument("--global-weight", type=float, default=None)
    args = parser.parse_args()
    if heatmap_route(args) == "replay":
        import json
        from text_side_anomaly.visualize import render_four_panel, save_contact_sheet

        folder = os.path.dirname(os.path.abspath(args.predictions))
        panels = []
        with open(args.predictions, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                arrays = np.load(os.path.join(folder, "arrays", item["array"]))
                pred = None if "pred_mask" not in arrays else arrays["pred_mask"] > 0
                gt = None if "gt_mask" not in arrays else arrays["gt_mask"] > 0
                panels.append(render_four_panel(
                    arrays["gray01"], arrays["score_map_high"], pred, gt,
                    bool(item.get("gt_available")),
                    mask_status=item.get("mask_status") or "unavailable",
                    sample_title=str(item.get("sample_id")),
                ))
        os.makedirs(args.out_dir, exist_ok=True)
        save_contact_sheet(panels, os.path.join(args.out_dir, "overview.png"),
                           "replay", "局部异常分数 0–1")
        print(f"[heatmap] 已按导出结果排版到 {args.out_dir}，未加载 CLIP")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if heatmap_route(args) == "checkpoint_four_panel" and not args.ckpt:
        parser.error("四列图请提供 --ckpt，出图时不会临时训练")
    if args.ckpt:
        model, anchors = load_model(args.ckpt, device)
        if args.data_root is None and model.checkpoint_meta.get("prompt_set") != "pneumonia":
            parser.error("非肺炎 checkpoint 出图请指定 --data-root，避免误读取 data_pneumonia/test")
    else:
        print("训练中（约 2 分钟）...")
        model, anchors = train_model(device)

    data_root = args.data_root or "data_pneumonia/test"
    if heatmap_route(args) == "checkpoint_four_panel":
        save_checkpoint_four_panel(model, anchors, data_root, device, args.out_dir, args)
        return
    npz_names = [name for name in os.listdir(data_root) if name.lower().endswith(".npz")]
    if npz_names:
        parser.error("胸腺瘤 npz 不能走单幅叠加，请加 --layout four_panel")
    image_size = model.cfg.image_size
    os.makedirs(args.out_dir, exist_ok=True)
    for cls, label in [("normal", 0), ("abnormal", 1)]:
        d = os.path.join(data_root, cls)
        files = sorted(os.listdir(d))[:3]
        for fn in files:
            path = os.path.join(d, fn)
            gray = Image.open(path).convert("L").resize((image_size, image_size))
            img = np.asarray(gray, dtype=np.float32) / 255.0
            img = np.stack([img] * 3, axis=0)
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            x = (img - mean[:, None, None]) / std[:, None, None]
            amap, score = anomaly_map_for(model, anchors, torch.from_numpy(x).float(), device)
            out = os.path.join(args.out_dir, f"{cls}_{fn}_score{score:.2f}.png")
            overlay_heatmap(gray, amap, score, out)
            print(f"  {cls} {fn} 异常分数={score:.3f} "
                  f"amap[min/mean/max]={amap.min():+.3f}/{amap.mean():+.3f}/{amap.max():+.3f} -> {out}")

    print(f"热力图已生成到 {args.out_dir}/ 目录")


if __name__ == "__main__":
    main()

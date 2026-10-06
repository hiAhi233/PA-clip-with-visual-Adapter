"""脑 MRI 训练、续训和独立评估。

K 是正常与异常合计的支持样本数。二维时 K 是切片文件数，三维时 K 是体积文件数。
``--init-ckpt`` 只加载权重并继承支持集，不是精确断点恢复。
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import uuid
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from .config import Config
from .dataset import (
    SliceAnomalyDataset,
    VolumeAnomalyDataset,
    VolumeFixedSliceDataset,
    choose_fixed_zs,
    load_manifest_splits,
    scan_split,
    validate_modalities,
)
from .fewshot import (
    SAMPLING_POLICY,
    SPLIT_SEED,
    experiment_identity,
    file_sha256,
    make_support_manifest,
    prompt_snapshot,
    restore_support_indices,
    sample_support_indices,
    saved_z_indices,
    stable_digest,
    support_digest,
    validate_masks,
    validate_splits,
    validate_support_held_out,
    validate_support_manifest,
)
from .losses import TotalLoss
from .inference import apply_postprocess, collect_raw_predictions, export_predictions
from .metrics import choose_image_threshold, choose_pixel_threshold, image_metrics, mask_overlap_metrics, pixel_metrics
from .postprocess import (
    OperatingThresholds,
    PostprocessConfig,
    adaptive_candidates,
    build_postprocess_config,
    choose_adaptive_operating_point,
    score_adaptive_preview,
    should_search_adaptive_thresholds,
    split_manual_thresholds,
    supervision_metadata,
)
from .model import _is_v2_checkpoint, _torch_load
from .prompts import BRAIN_MRI_PROMPT_SETS, DEFAULT_BRAIN_MRI_PROMPT_SET


def _csv_ints(text: str):
    return [int(part) for part in text.split(",") if part.strip()]


def _csv_strs(text: str):
    return [part.strip() for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="脑 MRI 小样本异常检测训练与评估")
    parser.add_argument("--data_root", type=str, required=True, help="训练集根目录，内含 normal/ 与 abnormal/")
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--data_format", type=str, default="slice", choices=["slice", "volume"])
    parser.add_argument("--bottleneck", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--lambda_t", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="runs/brain_mri")
    parser.add_argument("--prompt-set", type=str, default=None, choices=sorted(BRAIN_MRI_PROMPT_SETS),
                        help="新训练默认 brain_mri_sentence。brain_mri 是旧版。显式切换表示更换锚点的新实验")
    parser.add_argument("--organ", type=str, default="brain")
    parser.add_argument("--inlayer", action="store_true")
    parser.add_argument("--inlayer-layers", type=str, default="8,9,10,11")
    parser.add_argument("--inlayer-positions", type=str, default="attn,ffn")
    parser.add_argument("--inlayer-bottleneck", type=int, default=64)
    parser.add_argument("--visual-inlayer", action="store_true")
    parser.add_argument("--visual-inlayer-layers", type=str, default="5,8,11")
    parser.add_argument("--visual-inlayer-positions", type=str, default="attn,ffn")
    parser.add_argument("--visual-inlayer-bottleneck", type=int, default=64)
    parser.add_argument("--visual-inlayer-lambda", type=float, default=0.1)
    parser.add_argument("--train-stage", type=str, default=None, choices=["text", "visual", "joint", "tvs"])
    parser.add_argument("--init-ckpt", type=str, default=None, help="加载权重并继承支持集；不是精确断点恢复")
    parser.add_argument("--n-support", type=int, default=None, help="正常与异常合计的支持样本数")
    parser.add_argument("--k", type=int, default=None, help="与 --n-support 相同")
    parser.add_argument("--seed", type=int, default=None, help="新实验默认 0；续训默认沿用已保存的种子")
    parser.add_argument("--steps", type=int, default=None, help="本次运行的优化步数；不补满最后一个 epoch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-data-root", type=str, default=None)
    parser.add_argument("--val-mask-root", type=str, default=None)
    parser.add_argument("--eval-data-root", type=str, default=None)
    parser.add_argument("--eval-mask-root", type=str, default=None)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--case-id-regex", type=str, default=None)
    parser.add_argument("--support-manifest", type=str, default=None)
    parser.add_argument("--resample-support", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--pixel-threshold", type=float, default=None)
    parser.add_argument("--image-threshold", type=float, default=None)
    parser.add_argument("--score-space", type=str, default=None,
                        choices=["raw_margin", "post_v1_probability_score"],
                        help="image/pixel 阈值属于原始相似度差，还是后处理的 0～1 分数")
    parser.add_argument("--post-image-threshold", type=float, default=None,
                        help="后处理 0～1 图像分数的手动阈值，不作用于原始 anomaly_map")
    parser.add_argument("--post-pixel-threshold", type=float, default=None,
                        help="后处理 0～1 分数图的手动像素阈值")
    parser.add_argument("--raw-image-threshold", type=float, default=None,
                        help="原始相似度差的图像阈值，不作用于 0～1 后处理分数")
    parser.add_argument("--raw-pixel-threshold", type=float, default=None,
                        help="原始 anomaly_map 的像素阈值")
    parser.add_argument("--oracle-metrics", action="store_true")
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--volume-support-slices", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--score-mode", type=str, default=None, choices=["local_topk", "fusion", "cls_baseline"])
    parser.add_argument("--topk-ratio", type=float, default=0.05)
    parser.add_argument("--global-weight", type=float, default=None)
    parser.add_argument("--mask-mode", type=str, default="fixed", choices=["fixed", "adaptive_seeded"])
    parser.add_argument("--seed-threshold", type=float, default=None)
    parser.add_argument("--grow-floor", type=float, default=None)
    parser.add_argument("--rows-per-page", type=int, default=8)
    parser.add_argument("--panel-size", type=int, default=256)
    return parser


def parse_and_validate_args(argv=None):
    args = build_parser().parse_args(argv)
    _validate_flags(args)
    return args


def make_args(data_root: str, **overrides):
    """给测试和批量入口填好解析后的默认值。"""
    args = build_parser().parse_args(["--data_root", data_root])
    for key, value in overrides.items():
        setattr(args, key, value)
    _validate_flags(args)
    return args


def _validate_flags(args) -> None:
    if args.n_support is not None and args.k is not None and int(args.n_support) != int(args.k):
        raise ValueError("--n-support 与 --k 给出的数值不一致")
    if args.support_manifest and args.resample_support:
        raise ValueError("--support-manifest 与 --resample-support 不能同时使用")
    if args.eval_only and args.init_ckpt:
        raise ValueError("--eval-only 与 --init-ckpt 不能同时使用")
    if args.eval_only and not args.ckpt:
        raise ValueError("--eval-only 需要 --ckpt")
    if args.eval_only and args.train_stage == "tvs":
        raise ValueError("评估只加载已有权重，TVS 是训练日程")
    if args.train_stage == "tvs" and not (args.inlayer and args.visual_inlayer):
        raise ValueError("TVS 需要同时打开 --inlayer 和 --visual-inlayer")
    if args.steps is not None and int(args.steps) < 1:
        raise ValueError("训练步数必须为正")
    if args.train_stage == "tvs" and args.steps is not None and int(args.steps) < 2 and not args.eval_only:
        raise ValueError("从头运行 TVS 时总步数至少为 2")
    if int(args.batch_size) < 1:
        raise ValueError("batch size 必须为正")
    if int(args.volume_support_slices) < 1:
        raise ValueError("每个体积的支持切片数必须为正")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负")


def _requested_k(args) -> Optional[int]:
    if args.n_support is not None:
        return int(args.n_support)
    if args.k is not None:
        return int(args.k)
    return None


def peek_checkpoint(path: Optional[str]) -> dict:
    """只读取元数据，不构建模型。"""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 checkpoint：{path}")
    blob = _torch_load(path, map_location="cpu")
    if not _is_v2_checkpoint(blob):
        return {}
    blob.pop("model_state_dict", None)
    blob.pop("optimizer_state_dict", None)
    return blob


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _read_run_status(run_dir: str):
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(path):
        return None, None
    payload = _load_json(path)
    return payload.get("status"), payload.get("experiment_id")


def _resolve_run_dir(save_dir: str, stem: str, experiment_id: str, fresh: bool):
    os.makedirs(save_dir, exist_ok=True)
    primary = os.path.join(save_dir, stem)
    status, saved_id = _read_run_status(primary)
    if not fresh and saved_id == experiment_id and status in ("trained", "evaluated", "visualized"):
        return primary, status
    if not os.path.exists(primary):
        return primary, None
    return primary + "_" + uuid.uuid4().hex[:8], None


def _group_name(args) -> str:
    if args.train_stage == "tvs":
        return "tvs"
    if args.train_stage == "joint":
        return "tv_joint"
    if args.train_stage == "visual":
        return "visual"
    if args.inlayer:
        return "text_inlayer"
    return "text"


def _resolve_stage(args) -> None:
    if args.train_stage is None:
        if args.visual_inlayer and args.inlayer:
            args.train_stage = "joint"
        elif args.visual_inlayer:
            args.train_stage = "visual"
        else:
            args.train_stage = "text"


def _adapter_specs(args, cfg: Config):
    inlayer = None
    if args.inlayer:
        inlayer = {
            "organs": [args.organ],
            "organ": args.organ,
            "layers": _csv_ints(args.inlayer_layers),
            "positions": tuple(_csv_strs(args.inlayer_positions)),
            "bottleneck": args.inlayer_bottleneck,
        }
    visual = None
    if args.visual_inlayer:
        visual = {
            "organs": [args.organ],
            "organ": args.organ,
            "layers": list(cfg.visual_inlayer_layers),
            "positions": tuple(cfg.visual_inlayer_positions),
            "bottleneck": cfg.visual_inlayer_bottleneck,
            "lambda_t": cfg.visual_inlayer_lambda,
        }
    return inlayer, visual


def _scan_splits(args):
    if args.split_manifest:
        groups = load_manifest_splits(args.split_manifest)
        return groups["train"], groups["val"], groups["test"], True
    regex = args.case_id_regex
    train = scan_split(args.data_root, args.mask_root or None, args.data_format, regex)
    val = scan_split(args.val_data_root, args.val_mask_root or None, args.data_format, regex) if args.val_data_root else []
    test = scan_split(args.eval_data_root, args.eval_mask_root or None, args.data_format, regex) if args.eval_data_root else []
    return train, val, test, bool(regex)


def _dataset_from_records(args, records, strategy: str):
    if args.data_format == "volume":
        return VolumeAnomalyDataset(
            args.data_root,
            mask_root=args.mask_root,
            image_size=224,
            grid=14,
            slice_strategy=strategy,
            records=records,
        )
    return SliceAnomalyDataset(args.data_root, mask_root=args.mask_root, image_size=224, grid=14, records=records)


def _bind_volume_slices(samples, indices, count: int, seed: int) -> str:
    import nibabel as nib

    pending = [samples[index] for index in indices if not samples[index].get("z_indices")]
    if not pending:
        return "volume_fixed_slices"
    rng = np.random.RandomState(int(seed))
    for sample in pending:
        volume = np.asarray(nib.load(sample["path"]).dataobj)
        depth = int(volume.shape[2])
        mask = None
        if sample.get("mask_path"):
            mask_volume = np.asarray(nib.load(sample["mask_path"]).dataobj)
            if mask_volume.shape[:3] != volume.shape[:3]:
                raise ValueError(
                    f"{sample['sample_id']} 的掩码形状 {mask_volume.shape} 与图像形状 {volume.shape} 不一致"
                )
            mask = mask_volume > 0
        if volume.ndim == 4 and sample.get("modality") in (None, ""):
            sample["modality"] = 0
        zs, policy = choose_fixed_zs(depth, mask, count, rng, int(sample["label"]))
        sample["z_indices"] = zs
        sample["z_policy"] = policy
    return "volume_fixed_slices"


def _middle_pairs(samples):
    import nibabel as nib

    pairs = []
    for index, sample in enumerate(samples):
        depth = int(nib.load(sample["path"]).shape[2])
        pairs.append((index, depth // 2))
    return pairs


def _support_dataset(args, dataset, samples, indices):
    if args.data_format != "volume":
        return Subset(dataset, list(indices)), len(indices)
    pairs = []
    for index in indices:
        zs = samples[index].get("z_indices") or []
        if len(zs) < 1:
            raise ValueError(f"体积 {samples[index]['sample_id']} 没有固定的支持切片")
        for z in zs:
            pairs.append((index, int(z)))
    return VolumeFixedSliceDataset(dataset, pairs), len(pairs)


def _data_version(samples) -> str:
    rows = []
    for sample in samples:
        path = sample.get("path")
        mask_path = sample.get("mask_path")
        rows.append({
            "sample_id": sample["sample_id"],
            "label": int(sample["label"]),
            "relative_path": sample.get("relative_path"),
            "sha256": file_sha256(path) if path and os.path.isfile(path) else "",
            "mask_relative_path": sample.get("mask_relative_path"),
            "mask_sha256": file_sha256(mask_path) if mask_path and os.path.isfile(mask_path) else "",
            "modality": sample.get("modality"),
        })
    return stable_digest(rows)


def _worker_init(worker_id: int, base_seed: int) -> None:
    random.seed(base_seed + worker_id)
    np.random.seed((base_seed + worker_id) % (2 ** 32))


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int, drop_last: bool = False):
    generator = None
    worker_init = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init = functools.partial(_worker_init, base_seed=int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init,
    )


def _make_criterion(cfg: Config, stage: str, local_weight: float) -> TotalLoss:
    visual = stage == "visual"
    return TotalLoss(
        margin=cfg.margin,
        w_text=0.0 if visual else cfg.w_text,
        w_global=cfg.w_global,
        w_local=local_weight,
        w_div=0.0,
    )


def _seed_model(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_model_factory(cfg: Config, inlayer, visual):
    from .model import TextSideAnomalyModel

    return TextSideAnomalyModel(cfg, inlayer=inlayer, visual_inlayer=visual)


def _optimizer_param_ids(optimizer):
    return sorted(id(param) for group in optimizer.param_groups for param in group["params"])


def _run_steps(model, loader, optimizer, criterion, anchors, device, steps: int, cached_enc, local: bool, seen: list, total_steps: int, offset: int):
    if steps < 1:
        return cached_enc, 0
    iterator = iter(loader)
    completed = 0
    model.train()
    while completed < steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device) if local else None
        enc = cached_enc if cached_enc is not None else model.encode_anchors(anchors)
        outputs = model(images, enc)
        losses = criterion(enc, outputs, labels, masks)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        completed += 1
        seen.extend(list(batch["sample_id"]))
        absolute = offset + completed
        if absolute == 1 or absolute == total_steps or absolute % 20 == 0:
            print(f"[train] step={absolute}/{total_steps} total={float(losses['total'].detach().cpu()):.4f}")
    return cached_enc, completed


def _predict(model, loader, anchors, device):
    model.eval()
    scores, labels, maps, masks, sample_ids = [], [], [], [], []
    with torch.no_grad():
        encoded = model.encode_anchors(anchors)
        for batch in loader:
            outputs = model(batch["image"].to(device), encoded)
            scores.append(outputs["cls_probs"].detach().cpu())
            labels.append(batch["label"])
            maps.append(outputs["anomaly_map"].detach().cpu())
            masks.append(batch["mask"])
            sample_ids.extend(list(batch["sample_id"]))
    if not scores:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty.astype(np.int64), np.zeros((0, 1, 1)), np.zeros((0, 1, 1)), []
    return (
        torch.cat(scores).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(maps).numpy(),
        torch.cat(masks).numpy(),
        sample_ids,
    )


def _threshold_spaces(args) -> dict:
    return split_manual_thresholds(
        getattr(args, "image_threshold", None),
        getattr(args, "pixel_threshold", None),
        score_space=getattr(args, "score_space", None),
        post_image_threshold=getattr(args, "post_image_threshold", None),
        post_pixel_threshold=getattr(args, "post_pixel_threshold", None),
        raw_image_threshold=getattr(args, "raw_image_threshold", None),
        raw_pixel_threshold=getattr(args, "raw_pixel_threshold", None),
    )


def render_identity(args, evaluation_id: str) -> str:
    """只描述排版。改面板尺寸或每页行数时重新出图，不重新训练或评估。"""
    return stable_digest({
        "evaluation_id": evaluation_id,
        "layout": "four_panel",
        "panel_size": int(getattr(args, "panel_size", 256)),
        "rows_per_page": int(getattr(args, "rows_per_page", 8)),
    })


def evaluation_identity(args) -> str:
    """后处理参数单独标识。变了就重新评估，训练实验标识保持不变。"""
    spaces = _threshold_spaces(args)
    payload = {
        "adaptive_objective": "lesion_minus_normal_fp_then_dice_v1",
        "global_weight": getattr(args, "global_weight", None),
        "grow_floor": getattr(args, "grow_floor", None),
        "mask_mode": getattr(args, "mask_mode", "fixed"),
        "post_image_threshold": spaces["post_v1_probability_score"]["image_threshold"],
        "post_pixel_threshold": spaces["post_v1_probability_score"]["pixel_threshold"],
        "raw_image_threshold": spaces["raw_margin"]["image_threshold"],
        "raw_pixel_threshold": spaces["raw_margin"]["pixel_threshold"],
        "score_mode": getattr(args, "score_mode", None),
        "score_space": getattr(args, "score_space", None),
        "seed_threshold": getattr(args, "seed_threshold", None),
        "topk_ratio": float(getattr(args, "topk_ratio", 0.05)),
    }
    return stable_digest(payload)


def _select_thresholds(args, val_loader, model, anchors, device, protocol):
    reasons = []
    raw = _threshold_spaces(args)["raw_margin"]
    image_threshold = raw["image_threshold"]
    pixel_threshold = raw["pixel_threshold"]
    image_source = protocol["image_threshold_source"]
    pixel_source = protocol["pixel_threshold_source"]
    if image_threshold is not None:
        image_source = "preset"
    if pixel_threshold is not None:
        pixel_source = "preset"
    if val_loader is not None and (image_threshold is None or pixel_threshold is None):
        scores, labels, maps, masks, _ = _predict(model, val_loader, anchors, device)
        if image_threshold is None and image_source == "val":
            try:
                image_threshold = choose_image_threshold(scores, labels)
                image_source = "val"
            except ValueError as exc:
                image_threshold = None
                image_source = "unavailable"
                reasons.append(str(exc))
        if pixel_threshold is None and pixel_source == "val":
            try:
                pixel_threshold = choose_pixel_threshold(maps, masks)
                pixel_source = "val"
            except ValueError as exc:
                pixel_threshold = None
                pixel_source = "unavailable"
                reasons.append(str(exc))
    return image_threshold, pixel_threshold, image_source, pixel_source, reasons


def _score_split(model, loader, anchors, device, image_threshold, pixel_threshold, include_oracle: bool, has_masks: bool):
    scores, labels, maps, masks, sample_ids = _predict(model, loader, anchors, device)
    image = image_metrics(scores, labels, threshold=image_threshold, include_oracle=include_oracle)
    pixel = None
    if has_masks:
        pixel = pixel_metrics(maps, masks, threshold=pixel_threshold, include_oracle=include_oracle)
    return image, pixel, sample_ids


def _safe_name(sample_id: str) -> str:
    return str(sample_id).replace("\\", "_").replace("/", "_").replace(":", "_")


def _gray_from_clip(image) -> np.ndarray:
    """还原 _gray2clip 的灰度图，作为引导滤波的原图。"""
    gray = image[0].detach().float().cpu().numpy()
    gray = gray * 0.26862954 + 0.48145466
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def _dataset_records(loader):
    dataset = loader.dataset
    seen = set()
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        records = getattr(dataset, "records", None)
        if records:
            return {record["sample_id"]: record for record in records}
        dataset = getattr(dataset, "dataset", None) or getattr(dataset, "base", None)
    return {}


def _display_gt(mask_path, height: int, width: int, z=None):
    """用原始掩码描轮廓。14×14 的训练网格放大后会变成方块，不能拿来画绿线。"""
    if not mask_path or not os.path.isfile(mask_path):
        return None
    lower = mask_path.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        import nibabel as nib

        volume = np.asarray(nib.load(mask_path).dataobj)
        if volume.ndim < 3:
            return None
        index = volume.shape[2] // 2 if z is None else int(z)
        if index < 0 or index >= volume.shape[2]:
            return None
        plane = volume[:, :, index]
        image = Image.fromarray((plane > 0).astype(np.uint8) * 255)
    else:
        image = Image.open(mask_path).convert("L")
    image = image.resize((width, height), Image.BILINEAR)
    array = np.where(np.asarray(image) > 127, 255, 0).astype(np.uint8)
    if not array.any():
        return None
    return array


def _model_temperature(model) -> float:
    cfg = getattr(model, "cfg", None)
    return 0.07 if cfg is None else float(cfg.temperature)


def _checkpoint_supervision(model, cfg, trained_here: bool) -> dict:
    """评估时读模型上的 checkpoint 记录。只有这次确实按 cfg 训练过，才用这次的损失权重补上。"""
    meta = dict(getattr(model, "checkpoint_meta", None) or {})
    if meta.get("global_supervision_enabled") is None and trained_here:
        meta["global_supervision_enabled"] = float(cfg.w_global) > 0
        meta["w_global"] = float(cfg.w_global)
        model.checkpoint_meta = meta
    return supervision_metadata(meta)


def _build_post_config(args, model, cfg, trained_here: bool) -> PostprocessConfig:
    metadata = _checkpoint_supervision(model, cfg, trained_here)
    return build_postprocess_config(
        getattr(args, "score_mode", None),
        metadata,
        topk_ratio=float(getattr(args, "topk_ratio", 0.05)),
        mask_mode=getattr(args, "mask_mode", "fixed"),
        global_weight=getattr(args, "global_weight", None),
    )


def _search_adaptive_thresholds(val_raw, config, temperature):
    preview = apply_postprocess(val_raw, config, OperatingThresholds(), temperature)
    scored = []
    for seed, floor in adaptive_candidates(preview):
        trial = apply_postprocess(
            val_raw, config, OperatingThresholds(seed_threshold=seed, grow_floor=floor), temperature,
        )
        overlap = mask_overlap_metrics(
            [np.zeros_like(item["gray01"], dtype=bool) if item["pred_mask"] is None else item["pred_mask"] for item in trial],
            [np.zeros_like(item["gray01"], dtype=bool) if item["gt_mask_high"] is None else item["gt_mask_high"] for item in trial],
            [bool(item.get("gt_available")) for item in trial],
        )
        quality = score_adaptive_preview(trial)
        scored.append({
            "seed": seed,
            "floor": floor,
            "dice": overlap.get("dice_micro"),
            **quality,
        })
    return choose_adaptive_operating_point(scored)


def _formal_postprocess(model, eval_loader, val_loader, anchors, device, args, cfg, trained_here: bool) -> dict:
    temperature = _model_temperature(model)
    config = _build_post_config(args, model, cfg, trained_here)
    spaces = _threshold_spaces(args)
    post = spaces["post_v1_probability_score"]
    val_raw = collect_raw_predictions(model, val_loader, anchors, device) if val_loader is not None else []
    pixel_threshold = post["pixel_threshold"]
    pixel_source = "preset" if pixel_threshold is not None else "none"
    seed_threshold = getattr(args, "seed_threshold", None)
    grow_floor = getattr(args, "grow_floor", None)
    threshold_source = "preset" if seed_threshold is not None or grow_floor is not None else "none"
    selection = None
    calibration_reason = None
    if config.mask_mode == "fixed" and pixel_threshold is None and val_raw:
        preview = apply_postprocess(val_raw, config, OperatingThresholds(), temperature)
        usable = [item for item in preview if item.get("gt_available")]
        if usable and len(np.unique(np.concatenate([item["gt_mask_high"].reshape(-1) for item in usable]) > 0)) >= 2:
            pixel_threshold = choose_pixel_threshold(
                np.stack([item["score_map_refined"] for item in usable]),
                np.stack([item["gt_mask_high"].astype(np.float32) for item in usable]),
            )
            pixel_source = "val"
    elif config.mask_mode == "adaptive_seeded" and should_search_adaptive_thresholds(seed_threshold, grow_floor) and val_raw:
        selection = _search_adaptive_thresholds(val_raw, config, temperature)
        if selection and selection.get("ok"):
            seed_threshold, grow_floor = selection["seed"], selection["floor"]
            threshold_source = "val"
        else:
            seed_threshold, grow_floor = None, None
            threshold_source = "unavailable"
            calibration_reason = (selection or {}).get("reason") or "无法完成校准"
    elif config.mask_mode == "adaptive_seeded" and should_search_adaptive_thresholds(seed_threshold, grow_floor) and not val_raw:
        threshold_source = "unavailable"
        calibration_reason = "验证数据不足，没有验证集，无法完成校准"
    image_threshold = post["image_threshold"]
    image_source = "preset" if image_threshold is not None else "none"
    if image_threshold is None and val_raw:
        preview = apply_postprocess(val_raw, config, OperatingThresholds(
            pixel_threshold=pixel_threshold, seed_threshold=seed_threshold, grow_floor=grow_floor,
        ), temperature)
        labels = np.array([item["label"] for item in preview])
        if len(np.unique(labels)) >= 2:
            image_threshold = choose_image_threshold(
                np.array([item["score_fused"] for item in preview]), labels,
            )
            image_source = "val"
    operating = OperatingThresholds(
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        seed_threshold=seed_threshold,
        grow_floor=grow_floor,
        score_space="post_v1_probability_score",
        image_threshold_source=image_source,
        pixel_threshold_source=pixel_source if config.mask_mode == "fixed" else threshold_source,
    )
    processed = apply_postprocess(
        collect_raw_predictions(model, eval_loader, anchors, device), config, operating, temperature,
    )
    fused_scores = np.array([item["score_fused"] for item in processed]) if processed else np.zeros((0,))
    fused_labels = np.array([item["label"] for item in processed]) if processed else np.zeros((0,), dtype=np.int64)
    image = image_metrics(fused_scores, fused_labels, threshold=image_threshold) if processed else {}
    overlap = mask_overlap_metrics(
        [np.zeros_like(item["gray01"], dtype=bool) if item["pred_mask"] is None else item["pred_mask"] for item in processed],
        [np.zeros_like(item["gray01"], dtype=bool) if item["gt_mask_high"] is None else item["gt_mask_high"] for item in processed],
        [bool(item["gt_available"] and item["pred_mask"] is not None) for item in processed],
    ) if processed else {}
    thresholds = operating.to_dict()
    if calibration_reason:
        thresholds["calibration"] = "unavailable"
        thresholds["calibration_reason"] = calibration_reason
    elif threshold_source == "val":
        thresholds["calibration"] = "ok"
    if selection is not None and selection.get("ok"):
        thresholds["selection"] = {
            "normal_false_positive_rate": selection.get("normal_false_positive_rate"),
            "lesion_detection_rate": selection.get("lesion_detection_rate"),
            "dice_micro": selection.get("dice"),
        }
    if calibration_reason:
        print(f"[post] {calibration_reason}")
    return {
        "config": config.to_dict(),
        "thresholds": thresholds,
        "raw_thresholds": spaces["raw_margin"],
        "records": processed,
        "image": image,
        "pixel_highres": overlap,
        "metric_protocol": "post_v1_val_threshold",
        "score_space": operating.score_space,
    }


def _load_exported_records(run_dir: str, evaluation_id: str):
    """从已经导出的预测恢复出图所需字段，不再跑模型。"""
    folder = os.path.join(run_dir, "evaluations", evaluation_id)
    manifest = os.path.join(folder, "predictions.jsonl")
    if not os.path.isfile(manifest):
        return []
    records = []
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            with np.load(os.path.join(folder, "arrays", item["array"])) as arrays:
                pred = None if "pred_mask" not in arrays.files else np.array(arrays["pred_mask"]) > 0
                gt = None if "gt_mask" not in arrays.files else np.array(arrays["gt_mask"]) > 0
                records.append({
                    "sample_id": item["sample_id"],
                    "file_id": item.get("file_id"),
                    "slice_z": item.get("slice_z"),
                    "label": item.get("label"),
                    "pred_label": item.get("pred_label"),
                    "raw_map": np.array(arrays["raw_map"]),
                    "score_map_high": np.array(arrays["score_map_high"]),
                    "gray01": np.array(arrays["gray01"]),
                    "pred_mask": pred,
                    "gt_mask_high": gt,
                    "gt_available": bool(item.get("gt_available")),
                    "mask_info": {"mask_status": item.get("mask_status") or "unavailable"},
                })
    return records


def _save_heatmaps(records, out_dir: str, panel_size: int = 256, rows_per_page: int = 8) -> None:
    """四列图只读取已经算好的预测。没有预测就不能写成出图完成。"""
    from .visualize import render_four_panel, save_contact_sheet

    if not records:
        raise RuntimeError("没有预测记录，不能把出图标成完成")
    os.makedirs(out_dir, exist_ok=True)
    figure_dir = os.path.join(out_dir, "figures")
    os.makedirs(os.path.join(figure_dir, "rows"), exist_ok=True)
    panels = []
    for item in records:
        stem = item.get("file_id") or _safe_name(item["sample_id"])
        np.save(os.path.join(out_dir, stem + ".npy"), item["raw_map"])
        info = item.get("mask_info") or {}
        gt = item.get("gt_mask_high")
        panel = render_four_panel(
            item["gray01"], item["score_map_high"], item.get("pred_mask"), gt,
            bool(item.get("gt_available")), panel_size=panel_size,
            mask_status=info.get("mask_status", "unavailable"),
            sample_title=f"{item['sample_id']}  y={item.get('label')}  z={item.get('slice_z')}",
        )
        Image.fromarray(panel).save(os.path.join(figure_dir, "rows", stem + ".png"))
        Image.fromarray(panel).save(os.path.join(out_dir, stem + "_display.png"))
        panels.append(panel)
    atomic_write_json(os.path.join(out_dir, "display_note.json"), {
        "kind": "display_only",
        "score": "score_map_high",
        "note": "四列图使用后处理得到的 0～1 局部分数，色阶固定为 0 到 1，不逐图拉伸。"
                "指标使用 npy 里的原始 anomaly_map 以及导出的 pred_mask，不用热图颜色。",
    })
    if panels:
        _clear_previous_overview_pages(figure_dir)
        save_contact_sheet(
            panels, os.path.join(figure_dir, "overview.png"),
            "Original | Heatmap | Heatmap + GT | Refined mask + GT",
            "局部异常分数 0–1；绿色为 GT 轮廓，红色为预测掩码",
            rows_per_page=rows_per_page,
        )


def _clear_previous_overview_pages(figure_dir: str) -> None:
    """只删除上一次的分页总览，不动单行图和原始分数。"""
    if not os.path.isdir(figure_dir):
        return
    for name in os.listdir(figure_dir):
        if name == "overview.png" or (name.startswith("overview_") and name.endswith(".png")):
            os.remove(os.path.join(figure_dir, name))


def _checkpoint_payload(model, path, optimizer, **extra) -> None:
    temporary = path + ".tmp"
    model.save_checkpoint(temporary, optimizer=optimizer, **extra)
    os.replace(temporary, path)


def _build_model(args, cfg, inlayer, visual, device, model_factory, checkpoint, stage: str):
    model = model_factory(cfg, inlayer, visual).to(device)
    if checkpoint:
        model.load_checkpoint(
            checkpoint,
            map_location=device,
            allow_missing_visual=True,
            resume_stage=stage,
            organ=args.organ,
        )
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(stage, organ=args.organ)
    return model


def _run_identifier(experiment_id: str, run_dir: str, saved: Optional[str] = None) -> str:
    if saved:
        return saved
    return stable_digest({"experiment_id": experiment_id, "run_dir": os.path.basename(run_dir)})


def _evaluation_complete(run_dir: str) -> bool:
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        return False
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("image"), dict) and bool(payload.get("evaluation_split"))


def _write_eval_state(run_config, run_dir, evaluation_id, evaluation_status, visualization_status, status) -> None:
    run_config["status"] = status
    run_config["evaluation_id"] = evaluation_id
    run_config["evaluation_status"] = evaluation_status
    run_config["visualization_status"] = visualization_status
    atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)


def _result_from_disk(run_dir: str, reused: bool) -> dict:
    config = _load_json(os.path.join(run_dir, "run_config.json"))
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = _load_json(metrics_path) if os.path.isfile(metrics_path) else {}
    thresholds_path = os.path.join(run_dir, "thresholds.json")
    thresholds = _load_json(thresholds_path) if os.path.isfile(thresholds_path) else {}
    print(f"[reuse] {run_dir} status={config.get('status')}")
    return {
        "experiment_id": config.get("experiment_id"),
        "evaluation_id": config.get("evaluation_id"),
        "run_dir": run_dir,
        "status": config.get("status"),
        "evaluation_status": config.get("evaluation_status"),
        "visualization_status": config.get("visualization_status"),
        "reused": reused,
        "metrics": metrics,
        "thresholds": thresholds,
        "support_digest": config.get("support_digest"),
        "support_ids": [item["sample_id"] for item in config.get("support_set", {}).get("samples", [])],
        "evaluation_split": metrics.get("evaluation_split") or config.get("evaluation_protocol", {}).get("evaluation_split"),
        "seen_sample_ids": [],
        "eval_sample_ids": [],
        "completed_steps": config.get("steps_this_run"),
        "stage_optimizer_ids": [],
        "train_pool": config.get("train_pool"),
        "checkpoint": config.get("checkpoint_path") or os.path.join(run_dir, "checkpoint_final.pt"),
        "normal_count": (config.get("support_config") or {}).get("normal_count"),
        "abnormal_count": (config.get("support_config") or {}).get("abnormal_count"),
        "prompt_set": config.get("prompt_set"),
        "run_id": _run_identifier(config.get("experiment_id") or "", run_dir, config.get("run_id")),
    }


def _saved_checkpoint(run_dir: str) -> str:
    """复用训练或评估记录前，确认权重还在，且内容和记录的摘要一致。"""
    config = _load_json(os.path.join(run_dir, "run_config.json"))
    path = config.get("checkpoint_path") or os.path.join(run_dir, "checkpoint_final.pt")
    status = config.get("status")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"记录状态为 {status}，但权重不存在：{path}。不能继续评估；重新训练请使用 --fresh"
        )
    expected = config.get("checkpoint_sha256")
    if expected and file_sha256(path) != expected:
        raise RuntimeError(
            f"权重摘要与记录不一致：{path}。不能继续评估；重新训练请使用 --fresh"
        )
    return path


def _adapter_identity_from_spec(spec: dict):
    text = None
    if spec.get("text_inlayer_enabled"):
        text = {
            "layers": list(spec.get("text_inlayer_layers") or []),
            "positions": list(spec.get("text_inlayer_positions") or []),
            "bottleneck": spec.get("text_inlayer_bottleneck"),
        }
    visual = None
    if spec.get("visual_inlayer_enabled"):
        visual = {
            "layers": list(spec.get("visual_inlayer_layers") or []),
            "positions": list(spec.get("visual_inlayer_positions") or []),
            "bottleneck": spec.get("visual_inlayer_bottleneck"),
            "lambda_t": spec.get("visual_inlayer_lambda"),
        }
    return text, visual


def _build_eval_model(args, meta, device, stage: str, arch_factory=None):
    """只评估时按 checkpoint 里的架构重建，不看本次命令行的适配器开关。"""
    spec = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    if not spec.get("model_name"):
        raise RuntimeError("评估 checkpoint 没有架构配置，无法按保存的结构重建模型")
    if arch_factory is None:
        from .model import TextSideAnomalyModel
        arch_factory = TextSideAnomalyModel.from_arch_spec
    model = arch_factory(spec, str(device))
    if hasattr(model, "to"):
        model = model.to(device)
    model.load_checkpoint(
        args.ckpt,
        map_location=device,
        allow_missing_visual=False,
        resume_stage=stage,
        organ=args.organ,
    )
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(stage, organ=args.organ)
    return model


def run_experiment(args, model_factory=None, arch_factory=None) -> dict:
    _validate_flags(args)
    _resolve_stage(args)
    explicit_factory = model_factory is not None
    if model_factory is None:
        model_factory = default_model_factory

    weight_path = args.ckpt if args.eval_only else args.init_ckpt
    meta = peek_checkpoint(weight_path)
    saved_prompt = meta.get("prompt_set")
    if weight_path and not saved_prompt and args.prompt_set is None:
        raise RuntimeError(
            "checkpoint 未记录提示词版本，请通过 --prompt-set 明确选择 "
            "brain_mri（旧版）或 brain_mri_sentence（新版）"
        )
    if args.eval_only and args.prompt_set and saved_prompt and args.prompt_set != saved_prompt:
        raise RuntimeError("评估必须使用 checkpoint 记录的提示词；更换提示词请作为新实验训练")
    prompt_name = args.prompt_set or saved_prompt or DEFAULT_BRAIN_MRI_PROMPT_SET
    if prompt_name not in BRAIN_MRI_PROMPT_SETS:
        raise RuntimeError(
            f"脑 MRI 不能使用提示词 {prompt_name!r}，可选 {sorted(BRAIN_MRI_PROMPT_SETS)}"
        )
    if saved_prompt and prompt_name != saved_prompt:
        print(f"[prompt] 显式切换提示词 {saved_prompt} -> {prompt_name}，本次属于更换锚点的新实验")
    anchors = BRAIN_MRI_PROMPT_SETS[prompt_name]
    snapshot = prompt_snapshot(anchors)
    prompt_digest = stable_digest(snapshot)

    user_k = _requested_k(args)
    saved_support = meta.get("support_set") if isinstance(meta.get("support_set"), dict) else None
    external_manifest = _load_json(args.support_manifest) if args.support_manifest else None
    if external_manifest and saved_support:
        validate_support_manifest(external_manifest)
        validate_support_manifest(saved_support)
        if support_digest(external_manifest) != support_digest(saved_support):
            raise ValueError("给定的支持集清单与 checkpoint 中的支持集不一致，不会自动替换")
    restore_manifest = None
    if external_manifest is not None:
        restore_manifest = external_manifest
    elif saved_support is not None and not args.resample_support:
        restore_manifest = saved_support

    if restore_manifest is not None and not args.resample_support:
        saved_k = restore_manifest.get("requested_k")
        saved_seed = restore_manifest.get("seed", meta.get("seed"))
        if user_k is not None and user_k != saved_k:
            raise ValueError(
                f"K 与已保存支持集冲突：保存 {saved_k}，当前 {user_k}。重新抽样请加 --resample-support"
            )
        if args.seed is not None and saved_seed is not None and int(args.seed) != int(saved_seed):
            raise ValueError(
                f"seed 与已保存支持集冲突：保存 {saved_seed}，当前 {args.seed}。重新抽样请加 --resample-support"
            )
        seed = int(args.seed if args.seed is not None else (0 if saved_seed is None else saved_seed))
        draw_mode = "restore"
        k_for_sample = saved_k
    elif args.resample_support:
        draw_mode = "resample"
        inherited_k = None if not saved_support else saved_support.get("requested_k")
        inherited_seed = None if not saved_support else saved_support.get("seed", meta.get("seed"))
        k_for_sample = user_k if user_k is not None else inherited_k
        if args.seed is not None:
            seed = int(args.seed)
        elif inherited_seed is not None:
            seed = int(inherited_seed)
        else:
            seed = 0
        print("[support] 按 --resample-support 重新抽取支持集，本次使用新的实验标识")
    else:
        draw_mode = "migrate" if args.init_ckpt else "new"
        if args.init_ckpt and not saved_support:
            print("[init] 初始化迁移：checkpoint 没有支持集，按本次目录和 K 建立新记录，这不是精确恢复")
        seed = 0 if args.seed is None else int(args.seed)
        k_for_sample = user_k

    train_samples, val_samples, test_samples, require_cases = _scan_splits(args)
    split_info = validate_splits(train_samples, val_samples, test_samples, require_cases=require_cases)
    validate_modalities(list(train_samples) + list(val_samples) + list(test_samples))
    verified = bool(split_info["case_disjoint_verified"])
    if args.eval_only and restore_manifest is not None:
        held_out = validate_support_held_out(
            restore_manifest.get("samples") or [], val_samples, test_samples
        )
        verified = bool(verified and held_out["case_disjoint_verified"])
    if not train_samples and not args.eval_only:
        raise ValueError("训练集为空")
    has_test = bool(test_samples)
    eval_saved_support = bool(args.eval_only and has_test and restore_manifest is not None)

    if draw_mode == "restore" and not eval_saved_support:
        if args.data_format == "volume":
            for item in restore_manifest.get("samples", []):
                saved_z_indices(item)
        indices = restore_support_indices(train_samples, restore_manifest, args.data_root)
    elif args.eval_only:
        if restore_manifest is not None:
            validate_support_manifest(restore_manifest)
        if not has_test and not train_samples:
            raise ValueError("评估需要独立测试集；没有测试集时才需要原来的训练数据")
        indices = []
        if eval_saved_support:
            print("[eval] 使用 checkpoint 里的支持集记录，不重新加载训练文件")
    else:
        indices, _ = sample_support_indices(train_samples, k_for_sample, seed, SAMPLING_POLICY)

    volume_policy = None
    if args.data_format == "volume" and indices:
        if draw_mode == "restore":
            volume_policy = restore_manifest.get("volume_sampling_policy") or "volume_fixed_slices"
        else:
            volume_policy = _bind_volume_slices(train_samples, indices, int(args.volume_support_slices), seed)

    local_supervision = bool(args.mask_root) or any(sample.get("mask_path") for sample in train_samples)
    if indices and local_supervision:
        validate_masks(train_samples, indices, True)
    if args.val_mask_root or any(sample.get("mask_relative_path") for sample in val_samples):
        validate_masks(val_samples, range(len(val_samples)), True)
    if args.eval_mask_root or any(sample.get("mask_relative_path") for sample in test_samples):
        validate_masks(test_samples, range(len(test_samples)), True)

    selected = [train_samples[index] for index in indices]
    case_ids = [sample.get("case_id") for sample in selected]
    unique_cases = len(set(case_ids)) if case_ids and all(case_ids) else None
    unit = "volume" if args.data_format == "volume" else "slice"
    normal_count = sum(int(sample["label"]) == 0 for sample in selected)
    abnormal_count = sum(int(sample["label"]) == 1 for sample in selected)
    split_id = stable_digest({
        "train": _data_version(train_samples),
        "val": _data_version(val_samples),
        "test": _data_version(test_samples),
        "split_seed": SPLIT_SEED,
    })
    sampling_config = {
        "requested_k": None if not selected else (restore_manifest.get("requested_k") if draw_mode == "restore" else k_for_sample),
        "actual_support": len(selected),
        "seed": seed,
        "support_unit": unit,
        "sampling_policy": SAMPLING_POLICY,
        "unique_cases": unique_cases,
        "split_id": split_id,
        "case_disjoint_verified": verified,
        "volume_sampling_policy": volume_policy,
    }
    manifest = make_support_manifest(train_samples, indices, args.data_root, sampling_config) if selected else {
        "schema_version": 2,
        "task": "brain_mri",
        "samples": [],
        "support_digest": stable_digest([]),
        "requested_k": k_for_sample,
        "actual_support": 0,
        "class_counts": {"normal": 0, "abnormal": 0},
        "unique_cases": None,
        "case_disjoint_verified": verified,
        "sampling_policy": SAMPLING_POLICY,
        "seed": seed,
        "support_unit": unit,
    }
    if eval_saved_support:
        manifest = dict(restore_manifest)
        manifest["support_digest"] = support_digest(restore_manifest)
        manifest["case_disjoint_verified"] = verified
        counts = manifest.get("class_counts") or {}
        normal_count = int(counts.get("normal", 0))
        abnormal_count = int(counts.get("abnormal", 0))
        unique_cases = manifest.get("unique_cases")
        sampling_config["requested_k"] = manifest.get("requested_k")
        sampling_config["actual_support"] = manifest.get("actual_support")
        sampling_config["unique_cases"] = unique_cases
        sampling_config["volume_sampling_policy"] = manifest.get("volume_sampling_policy")
    if args.data_format == "volume" and not eval_saved_support:
        manifest["training_slices"] = sum(len(sample.get("z_indices") or []) for sample in selected)
    cases_text = "unknown" if unique_cases is None else str(unique_cases)
    support_shown = manifest.get("actual_support") if eval_saved_support else len(selected)
    print(
        f"[support] train_pool={len(train_samples)} support={support_shown} "
        f"normal={normal_count} abnormal={abnormal_count} unit={unit} cases={cases_text} "
        f"policy={SAMPLING_POLICY}"
    )
    print(f"[prompt] set={prompt_name} levels={list(anchors.levels)}")

    model_stage = "text" if args.train_stage == "tvs" else args.train_stage
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        data_root=args.data_root,
        mask_root=args.mask_root or None,
        data_format=args.data_format,
        bottleneck=args.bottleneck,
        lambda_t=args.lambda_t,
        margin=args.margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        train_stage=model_stage,
        freeze_output_text_adapter=bool(args.inlayer or args.visual_inlayer),
        visual_inlayer_enabled=args.visual_inlayer,
        visual_inlayer_layers=_csv_ints(args.visual_inlayer_layers),
        visual_inlayer_positions=_csv_strs(args.visual_inlayer_positions),
        visual_inlayer_bottleneck=args.visual_inlayer_bottleneck,
        visual_inlayer_lambda=args.visual_inlayer_lambda,
        levels=list(anchors.levels),
        device=str(device),
    )
    local_weight = cfg.w_local if local_supervision else 0.0
    inlayer, visual = _adapter_specs(args, cfg)
    training_items = len(selected) if args.data_format != "volume" else sum(len(sample.get("z_indices") or []) for sample in selected)
    if not args.eval_only and training_items < 1:
        raise ValueError("支持集为空，无法训练")
    effective_batch = min(int(args.batch_size), training_items) if training_items else int(args.batch_size)
    batches_per_pass = max(1, math.ceil(training_items / effective_batch)) if training_items else 1
    target_steps = int(args.steps) if args.steps is not None else int(args.epochs) * batches_per_pass
    if not args.eval_only and target_steps < 1:
        raise ValueError("训练步数必须为正")
    if args.train_stage == "tvs" and not args.eval_only and target_steps < 2:
        raise ValueError("从头运行 TVS 时总步数至少为 2")
    if args.train_stage == "tvs":
        step_plan = {"text": target_steps // 2, "visual": target_steps - target_steps // 2}
    else:
        step_plan = {model_stage: target_steps}

    has_test = bool(test_samples)
    has_val = bool(val_samples)
    spaces = _threshold_spaces(args)
    raw_space = spaces["raw_margin"]
    post_space = spaces["post_v1_probability_score"]
    protocol = {
        "evaluation_split": "test" if has_test else "support",
        "image_threshold_source": "preset" if raw_space["image_threshold"] is not None else ("val" if has_val else "none"),
        "pixel_threshold_source": "preset" if raw_space["pixel_threshold"] is not None else (
            "val" if has_val and (args.val_mask_root or args.split_manifest) else "none"
        ),
        "image_threshold": raw_space["image_threshold"],
        "pixel_threshold": raw_space["pixel_threshold"],
        "post_image_threshold": post_space["image_threshold"],
        "post_pixel_threshold": post_space["pixel_threshold"],
        "score_space": getattr(args, "score_space", None),
        "oracle_metrics": bool(args.oracle_metrics),
        "volume_eval": "middle_slice_diagnostic" if args.data_format == "volume" else None,
    }
    identity_protocol = {
        "evaluation_split": protocol["evaluation_split"],
        "oracle_metrics": protocol["oracle_metrics"],
        "volume_eval": protocol["volume_eval"],
    }
    init_sha = file_sha256(args.init_ckpt) if args.init_ckpt else ""
    eval_sha = file_sha256(args.ckpt) if args.eval_only else ""
    source_step = meta.get("step") or 0
    saved_spec = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    use_saved_arch = bool(args.eval_only and saved_spec.get("model_name"))
    identity = {
        "task": "brain_mri",
        "data_format": args.data_format,
        "support_unit": unit,
        "requested_k": sampling_config["requested_k"],
        "support_digest": manifest.get("support_digest"),
        "sampling_policy": SAMPLING_POLICY,
        "draw_mode": draw_mode,
        "split_id": split_id,
        "split_seed": SPLIT_SEED,
        "support_seed": seed,
        "model_seed": seed,
        "loader_seed": seed,
        "prompt_set": prompt_name,
        "prompt_digest": prompt_digest,
        "model_name": cfg.model_name,
        "bottleneck": cfg.bottleneck,
        "lambda_t": cfg.lambda_t,
        "margin": cfg.margin,
        "ms_layers": list(cfg.ms_layers),
        "inlayer": _adapter_identity_from_spec(saved_spec)[0] if use_saved_arch else (
            None if inlayer is None else {
                "layers": list(inlayer["layers"]),
                "positions": list(inlayer["positions"]),
                "bottleneck": inlayer["bottleneck"],
            }
        ),
        "visual_inlayer": _adapter_identity_from_spec(saved_spec)[1] if use_saved_arch else (
            None if visual is None else {
                "layers": list(visual["layers"]),
                "positions": list(visual["positions"]),
                "bottleneck": visual["bottleneck"],
                "lambda_t": visual["lambda_t"],
            }
        ),
        "train_stage": args.train_stage,
        "step_plan": step_plan,
        "steps": None if args.eval_only else target_steps,
        "batch_size": int(args.batch_size),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "loss_weights": {
            "w_text": cfg.w_text,
            "w_global": cfg.w_global,
            "w_local": local_weight,
            "w_div": 0.0,
            "visual_w_text": 0.0,
            "visual_w_div": 0.0,
        },
        "init_checkpoint_sha256": init_sha,
        "eval_checkpoint_sha256": eval_sha,
        "volume_sampling_policy": volume_policy,
        "volume_support_slices": int(args.volume_support_slices) if args.data_format == "volume" else None,
        "evaluation_protocol": identity_protocol,
    }
    experiment_id = experiment_identity(identity)
    evaluation_id = evaluation_identity(args)
    render_id = render_identity(args, evaluation_id)
    k_label = "full" if sampling_config["requested_k"] is None else str(sampling_config["requested_k"])
    stem = f"{_group_name(args)}_k{k_label}_s{seed}_{experiment_id[:16]}"
    run_dir, existing_status = _resolve_run_dir(args.save_dir, stem, experiment_id, bool(args.fresh))
    run_id = _run_identifier(experiment_id, run_dir)
    ckpt_path = os.path.join(run_dir, "checkpoint_final.pt")
    verified_ckpt = None
    if existing_status in ("trained", "evaluated", "visualized"):
        verified_ckpt = _saved_checkpoint(run_dir)
        saved_config = _load_json(os.path.join(run_dir, "run_config.json"))
        run_id = _run_identifier(experiment_id, run_dir, saved_config.get("run_id"))
        metrics_ready = _evaluation_complete(run_dir)
        same_evaluation = saved_config.get("evaluation_id") == evaluation_id
        evaluation_ok = saved_config.get("evaluation_status") == "ok" and same_evaluation and metrics_ready
        same_render = saved_config.get("render_id") == render_id
        visualization_ok = saved_config.get("visualization_status") == "ok" and same_evaluation and same_render
        if not evaluation_ok and existing_status in ("evaluated", "visualized"):
            if not same_evaluation:
                print("[eval] 后处理参数已变，加载已有权重重新评估")
            elif saved_config.get("evaluation_status") == "failed":
                print("[eval] 上次后处理失败，加载已有权重重新评估")
            else:
                print("[eval] 指标文件缺失，加载已有权重重新评估")
        if existing_status in ("evaluated", "visualized") and evaluation_ok and (
            args.no_heatmaps or visualization_ok
        ):
            result = _result_from_disk(run_dir, reused=True)
            result["checkpoint"] = verified_ckpt
            result["run_id"] = run_id
            result["evaluation_id"] = evaluation_id
            result["render_id"] = saved_config.get("render_id")
            return result
        if evaluation_ok and not args.no_heatmaps and not visualization_ok:
            exported = _load_exported_records(run_dir, evaluation_id)
            if exported:
                print("[render] 评估结果未变，只按新的排版重新出图")
                _save_heatmaps(
                    exported,
                    os.path.join(run_dir, "heatmaps"),
                    panel_size=int(getattr(args, "panel_size", 256)),
                    rows_per_page=int(getattr(args, "rows_per_page", 8)),
                )
                saved_config["status"] = "visualized"
                saved_config["visualization_status"] = "ok"
                saved_config["render_id"] = render_id
                atomic_write_json(os.path.join(run_dir, "run_config.json"), saved_config)
                result = _result_from_disk(run_dir, reused=False)
                result["checkpoint"] = verified_ckpt
                result["run_id"] = run_id
                result["evaluation_id"] = evaluation_id
                result["render_id"] = render_id
                result["completed_steps"] = 0
                result["seen_sample_ids"] = []
                return result

    os.makedirs(run_dir, exist_ok=True)
    support_config = {
        "requested_k": sampling_config["requested_k"],
        "actual_support": len(selected),
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "unique_cases": unique_cases,
        "support_unit": unit,
        "sampling_policy": SAMPLING_POLICY,
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "support_seed": seed,
        "model_seed": seed,
        "loader_seed": seed,
        "case_disjoint_verified": verified,
        "volume_sampling_policy": volume_policy,
        "draw_mode": draw_mode,
    }
    run_config = {
        "status": "prepared" if existing_status != "trained" else "trained",
        "experiment_id": experiment_id,
        "evaluation_id": evaluation_id,
        "evaluation_status": "not_run",
        "visualization_status": "not_run",
        "run_id": run_id,
        "task": "brain_mri",
        "prompt_set": prompt_name,
        "prompt_digest": prompt_digest,
        "prompt_snapshot": snapshot,
        "support_config": support_config,
        "support_set": manifest,
        "support_digest": manifest.get("support_digest"),
        "split_id": split_id,
        "case_disjoint_verified": verified,
        "data_format": args.data_format,
        "evaluation_protocol": protocol,
        "train_pool": len(train_samples),
        "steps_this_run": None if args.eval_only else target_steps,
        "source_step": source_step,
        "requested_batch_size": int(args.batch_size),
        "effective_batch_size": effective_batch,
        "init_checkpoint_sha256": init_sha,
        "eval_only": bool(args.eval_only),
        "checkpoint_path": args.ckpt if args.eval_only else ckpt_path,
        "checkpoint_sha256": eval_sha if args.eval_only else "",
        "step_plan": step_plan,
        "seeds": {
            "split_seed": SPLIT_SEED,
            "support_seed": seed,
            "model_seed": seed,
            "loader_seed": seed,
        },
    }
    resume_saved = existing_status in ("trained", "evaluated", "visualized")
    if resume_saved:
        run_config["checkpoint_path"] = verified_ckpt
        run_config["checkpoint_sha256"] = _load_json(os.path.join(run_dir, "run_config.json")).get("checkpoint_sha256") or file_sha256(verified_ckpt)
        run_config["status"] = existing_status
    else:
        atomic_write_json(os.path.join(run_dir, "support_manifest.json"), manifest)
        atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)

    _seed_model(seed)
    load_from = verified_ckpt if resume_saved else weight_path
    infer_stage = "visual" if args.train_stage == "tvs" else model_stage
    if args.eval_only and meta.get("training_stage") in ("text", "visual", "joint"):
        stage_for_load = meta["training_stage"]
    elif resume_saved:
        stage_for_load = infer_stage
    else:
        stage_for_load = model_stage
    if args.eval_only and (use_saved_arch or not explicit_factory):
        model = _build_eval_model(args, meta, device, stage_for_load, arch_factory)
    else:
        model = _build_model(args, cfg, inlayer, visual, device, model_factory, load_from, stage_for_load)
    print(f"[train] device={device} stage={args.train_stage} support_items={training_items} batch={effective_batch}")

    seen = []
    stage_optimizer_ids = []
    completed = 0
    optimizer = None
    if not args.eval_only and not resume_saved:
        train_ds = _dataset_from_records(args, train_samples, "lesion")
        support_ds, _ = _support_dataset(args, train_ds, train_samples, indices)
        train_loader = _loader(support_ds, effective_batch, True, seed, args.num_workers, drop_last=False)
        if args.train_stage == "tvs":
            plan = [("text", step_plan["text"]), ("visual", step_plan["visual"])]
        else:
            plan = [(model_stage, target_steps)]
        cached_enc = None
        for stage_name, stage_steps in plan:
            if getattr(model, "training_stage", model_stage) != stage_name and hasattr(model, "set_train_stage"):
                model.set_train_stage(stage_name, organ=args.organ)
            elif hasattr(model, "set_train_stage"):
                model.set_train_stage(stage_name, organ=args.organ)
            optimizer = model.build_optimizer()
            stage_optimizer_ids.append(_optimizer_param_ids(optimizer))
            criterion = _make_criterion(cfg, stage_name, local_weight)
            if stage_name == "visual":
                model.eval()
                with torch.no_grad():
                    cached_enc = model.encode_anchors(anchors)
                print("[anchor] 视觉阶段文本权重固定，锚点只编码一次")
            else:
                cached_enc = None
            _, finished = _run_steps(
                model, train_loader, optimizer, criterion, anchors, device,
                stage_steps, cached_enc, local_supervision, seen, target_steps, completed,
            )
            completed += finished
        print(f"[train] step={completed}/{target_steps}")
        extra = dict(
            prompt_set=prompt_name,
            step=completed,
            seed=seed,
            support_set=manifest,
            split_id=split_id,
            prompt_digest=prompt_digest,
            prompt_snapshot=snapshot,
            support_config=support_config,
            support_digest=manifest.get("support_digest"),
            experiment_id=experiment_id,
            evaluation_protocol=protocol,
            global_supervision_enabled=float(cfg.w_global) > 0,
            w_global=float(cfg.w_global),
            task="brain_mri",
            source_step=source_step,
            steps_this_run=target_steps,
            init_checkpoint_sha256=init_sha,
            case_disjoint_verified=verified,
            requested_batch_size=int(args.batch_size),
            effective_batch_size=effective_batch,
        )
        _checkpoint_payload(model, ckpt_path, optimizer, **extra)
        run_config["status"] = "trained"
        run_config["completed_steps"] = completed
        run_config["checkpoint_path"] = ckpt_path
        run_config["checkpoint_sha256"] = file_sha256(ckpt_path)
        atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)
        meta_now = dict(getattr(model, "checkpoint_meta", None) or {})
        meta_now["global_supervision_enabled"] = float(cfg.w_global) > 0
        meta_now["w_global"] = float(cfg.w_global)
        model.checkpoint_meta = meta_now
        print(f"[ckpt] {ckpt_path}")
    elif resume_saved:
        print(f"[ckpt] 已有训练结果，继续评估 {verified_ckpt}")

    eval_samples = test_samples if has_test else selected
    eval_name = "test" if has_test else "support"
    if args.data_format == "volume":
        eval_base = _dataset_from_records(args, eval_samples, "middle")
        eval_ds = VolumeFixedSliceDataset(eval_base, _middle_pairs(eval_samples))
    else:
        eval_ds = _dataset_from_records(args, eval_samples, "middle")
    eval_loader = _loader(eval_ds, max(1, min(int(args.batch_size), max(1, len(eval_ds)))), False, seed, args.num_workers)
    val_loader = None
    if has_val:
        if args.data_format == "volume":
            val_base = _dataset_from_records(args, val_samples, "middle")
            val_ds = VolumeFixedSliceDataset(val_base, _middle_pairs(val_samples))
        else:
            val_ds = _dataset_from_records(args, val_samples, "middle")
        val_loader = _loader(val_ds, max(1, min(int(args.batch_size), max(1, len(val_ds)))), False, seed, args.num_workers)

    image_threshold, pixel_threshold, image_source, pixel_source, reasons = _select_thresholds(
        args, val_loader, model, anchors, device, protocol
    )
    has_eval_masks = any(sample.get("mask_path") for sample in eval_samples)
    image_metrics_out, pixel_metrics_out, eval_ids = _score_split(
        model, eval_loader, anchors, device, image_threshold, pixel_threshold,
        bool(args.oracle_metrics), has_eval_masks,
    )
    if image_metrics_out.get("reason"):
        reasons.append(image_metrics_out["reason"])
        print(f"[eval] {image_metrics_out['reason']}")
    if pixel_metrics_out and pixel_metrics_out.get("reason"):
        reasons.append(pixel_metrics_out["reason"])
        print(f"[eval] {pixel_metrics_out['reason']}")
    metrics = {
        "evaluation_split": eval_name,
        "image": image_metrics_out,
        "pixel": pixel_metrics_out,
        "reasons": reasons,
        "test_size": len(test_samples) if has_test else None,
        "support_size": len(selected),
    }
    formal = None
    evaluation_status = "failed"
    visualization_status = "not_run"
    trained_here = not args.eval_only and not resume_saved
    try:
        formal = _formal_postprocess(
            model, eval_loader, val_loader, anchors, device, args, cfg, trained_here,
        )
        metrics["postprocess"] = {
            key: formal[key] for key in (
                "config", "thresholds", "raw_thresholds", "image", "pixel_highres", "metric_protocol", "score_space",
            )
        }
        export_predictions(formal["records"], os.path.join(run_dir, "evaluations", evaluation_id))
        evaluation_status = "ok"
    except ValueError as exc:
        reasons.append(f"后处理未完成：{exc}")
        print(f"[post] {exc}")
        if "全局监督" in str(exc):
            _write_eval_state(
                run_config, run_dir, evaluation_id, "failed", "not_run",
                "trained" if (args.eval_only or os.path.isfile(verified_ckpt or ckpt_path)) else run_config.get("status"),
            )
            raise
    except Exception as exc:
        reasons.append(f"后处理未完成：{exc}")
        print(f"[post] {exc}")
    post_thresholds = {} if formal is None else formal["thresholds"]
    thresholds = {
        "raw_margin": {
            "image_threshold": image_threshold,
            "image_threshold_source": image_source,
            "pixel_threshold": pixel_threshold,
            "pixel_threshold_source": pixel_source if has_eval_masks or pixel_threshold is not None else "no_mask",
            "score_space": "raw_margin",
        },
        "post_v1_probability_score": post_thresholds,
        "image_threshold": image_threshold,
        "image_threshold_source": image_source,
        "pixel_threshold": pixel_threshold,
        "pixel_threshold_source": pixel_source if has_eval_masks or pixel_threshold is not None else "no_mask",
        "reasons": reasons,
    }
    atomic_write_json(os.path.join(run_dir, "metrics.json"), metrics)
    atomic_write_json(os.path.join(run_dir, "thresholds.json"), thresholds)
    weights_ready = bool(args.eval_only or os.path.isfile(verified_ckpt or ckpt_path))
    status = "trained" if evaluation_status != "ok" and weights_ready else ("evaluated" if evaluation_status == "ok" else run_config.get("status"))
    if not args.no_heatmaps and evaluation_status == "ok":
        try:
            _save_heatmaps(
                formal["records"],
                os.path.join(run_dir, "heatmaps"),
                panel_size=int(getattr(args, "panel_size", 256)),
                rows_per_page=int(getattr(args, "rows_per_page", 8)),
            )
            visualization_status = "ok"
            status = "visualized"
            run_config["render_id"] = render_id
        except Exception as exc:
            visualization_status = "failed"
            reasons.append(f"出图未完成：{exc}")
            print(f"[heatmap] 出图失败，训练权重已保留：{exc}")
    elif args.no_heatmaps and evaluation_status == "ok":
        visualization_status = "not_run"
    run_config["status"] = status
    run_config["evaluation_id"] = evaluation_id
    run_config["evaluation_status"] = evaluation_status
    run_config["visualization_status"] = visualization_status
    run_config["evaluation_error"] = None if evaluation_status == "ok" else reasons[-1] if reasons else "后处理未完成"
    atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)
    print(f"[eval] evaluation_split={eval_name} size={len(eval_ds)} image={_json_safe(image_metrics_out)}")
    if eval_name != "test":
        print("[eval] 未提供独立测试集，以上是 support diagnostic，不能当作测试集结果")

    reported_checkpoint = args.ckpt if args.eval_only else (verified_ckpt or ckpt_path)
    print(f"[done] status={status} evaluation={evaluation_status} visualization={visualization_status} ckpt={reported_checkpoint}")
    return {
        "experiment_id": experiment_id,
        "evaluation_id": evaluation_id,
        "render_id": render_id,
        "run_id": run_id,
        "run_dir": run_dir,
        "status": status,
        "evaluation_status": evaluation_status,
        "visualization_status": visualization_status,
        "reused": False,
        "metrics": metrics,
        "thresholds": thresholds,
        "support_digest": manifest.get("support_digest"),
        "support_ids": [sample["sample_id"] for sample in manifest.get("samples", [])],
        "evaluation_split": eval_name,
        "seen_sample_ids": seen,
        "eval_sample_ids": eval_ids,
        "completed_steps": completed if not args.eval_only else 0,
        "stage_optimizer_ids": stage_optimizer_ids,
        "train_pool": len(train_samples),
        "checkpoint": reported_checkpoint,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
    }


def main(argv=None):
    args = parse_and_validate_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()

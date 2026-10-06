"""一次前向后统一做后处理。评估标签只跟随结果，不进入 postprocess_one。"""

from __future__ import annotations

import os

import numpy as np
import torch

from .fewshot import stable_digest
from .postprocess import OperatingThresholds, PostprocessConfig, postprocess_one


def sample_file_id(sample_id: str, slice_z, modality) -> str:
    return stable_digest({
        "sample_id": sample_id,
        "slice_z": None if slice_z in (None, -1) else int(slice_z),
        "modality": None if modality in (None, -1) else int(modality),
    })[:16]


def _gray_from_batch(batch, index: int) -> np.ndarray:
    if "gray01" in batch:
        return batch["gray01"][index].detach().cpu().numpy().astype(np.float32)
    image = batch["image"][index, 0].detach().float().cpu().numpy()
    return np.clip(image * 0.26862954 + 0.48145466, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def collect_raw_predictions(model, loader, anchors, device):
    model.eval()
    rows = []
    encoded = model.encode_anchors(anchors)
    for batch in loader:
        outputs = model(batch["image"].to(device), encoded)
        raw_maps = outputs["anomaly_map"].detach().cpu().numpy()
        scores = outputs["cls_probs"].detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        zs = batch["slice_z"].detach().cpu().numpy() if "slice_z" in batch else None
        modalities = batch["modality_index"].detach().cpu().numpy() if "modality_index" in batch else None
        available = batch["gt_available"].detach().cpu().numpy() if "gt_available" in batch else None
        gt = batch["gt_mask_high"].detach().cpu().numpy() if "gt_mask_high" in batch else None
        for index, sample_id in enumerate(list(batch["sample_id"])):
            slice_z = -1 if zs is None else int(zs[index])
            modality = -1 if modalities is None else int(modalities[index])
            rows.append({
                "sample_id": sample_id,
                "case_id": "" if "case_id" not in batch else batch["case_id"][index],
                "slice_z": slice_z,
                "modality": modality,
                "label": int(labels[index]),
                "raw_map": np.asarray(raw_maps[index], dtype=np.float32).copy(),
                "score_global": float(scores[index]),
                "gray01": _gray_from_batch(batch, index),
                "gt_mask_high": None if gt is None else np.asarray(gt[index]) > 0.5,
                "gt_available": False if available is None else bool(available[index]),
                "file_id": sample_file_id(sample_id, slice_z, modality),
            })
    return rows


def apply_postprocess(raw_records, config: PostprocessConfig, thresholds: OperatingThresholds, temperature: float):
    processed = []
    for record in raw_records:
        prediction = postprocess_one(
            record["raw_map"], record["score_global"], record["gray01"],
            temperature, config, thresholds,
        )
        merged = dict(record)
        merged.update(prediction)
        processed.append(merged)
    return processed


def export_predictions(records, output_dir: str) -> str:
    array_dir = os.path.join(output_dir, "arrays")
    os.makedirs(array_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "predictions.jsonl")
    lines = []
    for record in records:
        array_path = os.path.join(array_dir, record["file_id"] + ".npz")
        payload = {
            "raw_map": record["raw_map"],
            "score_map_high": record["score_map_high"],
            "score_map_refined": record["score_map_refined"],
            "gray01": record["gray01"],
        }
        if record.get("pred_mask") is not None:
            payload["pred_mask"] = np.asarray(record["pred_mask"], dtype=np.uint8)
        if record.get("gt_mask_high") is not None:
            payload["gt_mask"] = np.asarray(record["gt_mask_high"], dtype=np.uint8)
        np.savez(array_path, **payload)
        info = record.get("mask_info") or {}
        lines.append({
            "sample_id": record["sample_id"],
            "case_id": record.get("case_id") or "",
            "slice_z": record.get("slice_z"),
            "modality": record.get("modality"),
            "file_id": record["file_id"],
            "score_global": record["score_global"],
            "score_local": record["score_local"],
            "score_fused": record["score_fused"],
            "score_mode": record["score_mode"],
            "label": None if record.get("label") is None else int(record["label"]),
            "pred_label": record["pred_label"],
            "gt_available": record.get("gt_available"),
            "mask_status": info.get("mask_status"),
            "array": os.path.basename(array_path),
        })
    with open(manifest_path, "w", encoding="utf-8") as handle:
        import json
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return manifest_path

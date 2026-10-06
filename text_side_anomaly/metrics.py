"""异常检测通用指标。

图像级（分类）：AUROC、AP（平均精度）、F1、ACC。
像素级（定位）：Dice、IoU、像素 AUROC。
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


_UNSET = object()


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size and not np.isfinite(values).all():
        raise ValueError(f"{name} 含有非有限分数，无法计算指标")
    return values


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def _best_image_operating_point(scores: np.ndarray, labels: np.ndarray):
    """按分数从低到高移动阈值，和「每个唯一分数都做一次 >= 比较」的结果一致。

    相同分数整组处理。F1 并列时保留更低的阈值，以匹配原来的升序扫描。
    """
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order].astype(np.int64)
    total_pos = int((ordered_labels == 1).sum())
    total_neg = int(ordered_labels.size - total_pos)
    tp = total_pos
    fp = total_neg
    best_f1, best_thr = 0.0, 0.5
    start = 0
    while start < ordered_scores.size:
        thr = float(ordered_scores[start])
        fn = total_pos - tp
        f1 = _binary_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
        stop = start + 1
        while stop < ordered_scores.size and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        removed = ordered_labels[start:stop]
        tp -= int((removed == 1).sum())
        fp -= int((removed == 0).sum())
        start = stop
    return best_thr, best_f1


def choose_image_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """只应在验证集上调用，返回使 F1 最大的分数阈值。"""
    scores = _finite(scores, "图像分数")
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("验证集不同时包含正常和异常，无法选择图像阈值")
    threshold, _ = _best_image_operating_point(scores, labels)
    return float(threshold)


def image_metrics(scores: np.ndarray, labels: np.ndarray, threshold=_UNSET, include_oracle: bool = False) -> Dict[str, float]:
    """图像级异常分数 → 指标。

    不传 threshold 时保持旧行为：在当前这份数据上搜索 F1 最优阈值，结果写在 f1/acc。
    传入 threshold=None 表示没有操作点，只返回 AUROC/AP。传入具体阈值时按该阈值计算，
    不再把这份数据上的最优值写进 f1/acc。include_oracle 才额外给出 *_oracle。
    """
    scores = _finite(scores, "图像分数")
    labels = np.asarray(labels)
    legacy = threshold is _UNSET
    if len(np.unique(labels)) < 2:
        result = {"auroc": float("nan"), "ap": float("nan"), "f1": float("nan"), "acc": float("nan")}
        if not legacy:
            result["reason"] = "评估集不同时包含正常和异常，无法计算 AUROC/F1"
        if include_oracle:
            result["f1_oracle"] = float("nan")
            result["acc_oracle"] = float("nan")
        return result

    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    oracle_thr, oracle_f1 = _best_image_operating_point(scores, labels)
    oracle_acc = float(accuracy_score(labels, scores >= oracle_thr))
    if legacy:
        return {"auroc": auroc, "ap": ap, "f1": oracle_f1, "acc": oracle_acc}
    if threshold is None:
        result = {"auroc": auroc, "ap": ap, "f1": float("nan"), "acc": float("nan")}
    else:
        pred = scores >= float(threshold)
        tp = int(((labels == 1) & pred).sum())
        fp = int(((labels == 0) & pred).sum())
        fn = int(((labels == 1) & ~pred).sum())
        result = {
            "auroc": auroc,
            "ap": ap,
            "f1": _binary_f1(tp, fp, fn),
            "acc": float(accuracy_score(labels, pred)),
            "threshold": float(threshold),
        }
    if include_oracle:
        result["f1_oracle"] = oracle_f1
        result["acc_oracle"] = oracle_acc
        result["threshold_oracle"] = float(oracle_thr)
    return result


def _dice_iou(tp: int, pred_count: int, gt_count: int):
    dice = float(2 * tp / (pred_count + gt_count + 1e-6))
    iou = float(tp / (pred_count + gt_count - tp + 1e-6))
    return dice, iou


def _best_pixel_operating_point(flat_m: np.ndarray, flat_g: np.ndarray):
    order = np.argsort(flat_m, kind="mergesort")
    ordered_scores = flat_m[order]
    ordered_pos = flat_g[order] > 0
    gt_count = int(ordered_pos.sum())
    tp = gt_count
    pred_count = int(ordered_pos.size)
    best_dice, best_iou, best_thr = 0.0, 0.0, 0.5
    start = 0
    while start < ordered_scores.size:
        thr = float(ordered_scores[start])
        dice, iou = _dice_iou(tp, pred_count, gt_count)
        if dice > best_dice:
            best_dice, best_iou, best_thr = dice, iou, thr
        stop = start + 1
        while stop < ordered_scores.size and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        removed = ordered_pos[start:stop]
        tp -= int(removed.sum())
        pred_count -= int(removed.size)
        start = stop
    return best_thr, best_dice, best_iou


def choose_pixel_threshold(maps: np.ndarray, masks: np.ndarray) -> float:
    """只应在验证集上调用。anomaly_map 是相似度差，不是概率。"""
    flat_m = _finite(np.asarray(maps).reshape(-1), "异常图")
    flat_g = np.asarray(masks).reshape(-1)
    if len(np.unique(flat_g > 0)) < 2:
        raise ValueError("验证集掩码不同时包含病灶和背景，无法选择像素阈值")
    threshold, _, _ = _best_pixel_operating_point(flat_m, flat_g)
    return float(threshold)


def mask_overlap_metrics(predictions, ground_truth, available) -> Dict[str, float]:
    """对最终 bool 掩码汇总 Dice/IoU。没有 GT 的样本不进入分割指标。"""
    tp = fp = fn = 0
    per_image = []
    normal_fp = []
    for pred, gt, ok in zip(predictions, ground_truth, available):
        if not ok:
            continue
        pred = np.asarray(pred, dtype=bool)
        gt = np.asarray(gt, dtype=bool)
        inter = int((pred & gt).sum())
        pred_count = int(pred.sum())
        gt_count = int(gt.sum())
        tp += inter
        fp += pred_count - inter
        fn += gt_count - inter
        if pred_count == 0 and gt_count == 0:
            per_image.append(1.0)
        elif pred_count == 0 or gt_count == 0:
            per_image.append(0.0)
        else:
            per_image.append(float(2 * inter / (pred_count + gt_count)))
        if gt_count == 0:
            normal_fp.append(pred_count / max(pred.size, 1))
    if not per_image:
        return {
            "dice_micro": float("nan"), "iou_micro": float("nan"),
            "dice_macro": float("nan"), "normal_false_positive_area_ratio": float("nan"),
            "reason": "没有可用的高分辨率标注，不能计算分割指标",
        }
    if tp + fp + fn == 0:
        dice_micro, iou_micro = 1.0, 1.0
    else:
        dice_micro = float(2 * tp / (2 * tp + fp + fn))
        iou_micro = float(tp / (tp + fp + fn))
    return {
        "dice_micro": dice_micro,
        "iou_micro": iou_micro,
        "dice_macro": float(np.mean(per_image)),
        "normal_false_positive_area_ratio": float(np.mean(normal_fp)) if normal_fp else float("nan"),
    }


def pixel_metrics(maps: np.ndarray, masks: np.ndarray, threshold=_UNSET, include_oracle: bool = False) -> Dict[str, float]:
    """像素级异常图 → 定位指标。

    不传 threshold 时保持旧行为：在当前掩码上搜索 Dice 最优阈值。
    脑部正式结果应传入验证集阈值；threshold=None 时不把这份数据上的最优 Dice 当作正式结果。
    """
    flat_m = _finite(np.asarray(maps, dtype=np.float64).reshape(-1), "异常图")
    flat_g = np.asarray(masks).reshape(-1)
    legacy = threshold is _UNSET
    if len(np.unique(flat_g > 0)) < 2:
        result = {"dice": float("nan"), "iou": float("nan"), "pixel_auroc": float("nan")}
        if not legacy:
            result["reason"] = "评估掩码不同时包含病灶和背景，无法计算像素 AUROC/Dice"
        if include_oracle:
            result["dice_oracle"] = float("nan")
            result["iou_oracle"] = float("nan")
        return result

    pixel_auroc = float(roc_auc_score(flat_g > 0, flat_m))
    oracle_thr, oracle_dice, oracle_iou = _best_pixel_operating_point(flat_m, flat_g)
    if legacy:
        return {"dice": oracle_dice, "iou": oracle_iou, "pixel_auroc": pixel_auroc}
    if threshold is None:
        result = {"dice": float("nan"), "iou": float("nan"), "pixel_auroc": pixel_auroc}
    else:
        pred = flat_m >= float(threshold)
        positive = flat_g > 0
        tp = int((pred & positive).sum())
        dice, iou = _dice_iou(tp, int(pred.sum()), int(positive.sum()))
        result = {"dice": dice, "iou": iou, "pixel_auroc": pixel_auroc, "threshold": float(threshold)}
    if include_oracle:
        result["dice_oracle"] = oracle_dice
        result["iou_oracle"] = oracle_iou
        result["threshold_oracle"] = float(oracle_thr)
    return result

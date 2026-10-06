"""把模型的原始异常图变成连续分数和二值掩码。

这里不做读文件和上色。真实标签与 GT 不能作为输入；换掉它们，分数和掩码必须保持不变。
显示用的逐图拉伸也不能送进这些公式。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class PostprocessConfig:
    version: str = "post_v1"
    score_mode: str = "local_topk"
    topk_ratio: float = 0.05
    global_weight: float = 0.0
    refine_method: str = "guided"
    guided_radius: int = 4
    guided_eps: float = 1e-3
    roi_mode: str = "none"
    min_component_ratio: float = 0.0
    max_hole_ratio: float = 0.0
    mask_mode: str = "fixed"
    adaptive_method: str = "otsu"
    histogram_bins: int = 256
    connectivity: int = 8

    def validate(self) -> None:
        if self.version != "post_v1":
            raise ValueError(f"未知后处理版本 {self.version}")
        if self.score_mode not in ("local_topk", "fusion", "cls_baseline"):
            raise ValueError(f"未知 score_mode {self.score_mode}")
        if not 0 < float(self.topk_ratio) <= 1:
            raise ValueError("topk_ratio 必须在 (0, 1] 内，它不是少样本的 K")
        if not 0 <= float(self.global_weight) <= 1:
            raise ValueError("global_weight 必须在 [0, 1] 内")
        if self.refine_method not in ("guided", "none"):
            raise ValueError(f"未知 refine_method {self.refine_method}")
        if int(self.guided_radius) < 1:
            raise ValueError("guided_radius 必须为正")
        if float(self.guided_eps) <= 0:
            raise ValueError("guided_eps 必须为正")
        if self.roi_mode not in ("none",):
            raise ValueError("第一版 roi_mode 只接受 none；前景 ROI 需要单独验证")
        if float(self.min_component_ratio) < 0 or float(self.max_hole_ratio) < 0:
            raise ValueError("形态学比例不能为负")
        if self.mask_mode not in ("fixed", "adaptive_seeded"):
            raise ValueError(f"未知 mask_mode {self.mask_mode}")
        if self.adaptive_method != "otsu":
            raise ValueError("第一版动态边界只实现 otsu")
        if int(self.histogram_bins) != 256 or int(self.connectivity) != 8:
            raise ValueError("直方图必须是 256 bin，连通性必须是 8 邻域")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OperatingThresholds:
    image_threshold: Optional[float] = None
    pixel_threshold: Optional[float] = None
    seed_threshold: Optional[float] = None
    grow_floor: Optional[float] = None
    score_space: str = "post_v1_probability_score"
    image_threshold_source: str = "none"
    pixel_threshold_source: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


def supervision_metadata(metadata: Optional[dict] = None) -> dict:
    """全局监督只认 checkpoint 里写下的字段。新建 Config 的默认 w_global 不能算数。"""
    metadata = metadata or {}
    if "global_supervision_enabled" in metadata and metadata["global_supervision_enabled"] is not None:
        enabled = bool(metadata["global_supervision_enabled"])
        weight = metadata.get("w_global")
        if weight is None and isinstance(metadata.get("loss"), dict):
            weight = metadata["loss"].get("w_global")
        return {
            "global_supervision_enabled": enabled,
            "w_global": None if weight is None else float(weight),
        }
    loss = metadata.get("loss") if isinstance(metadata.get("loss"), dict) else None
    if loss and loss.get("w_global") is not None:
        weight = float(loss["w_global"])
        return {"global_supervision_enabled": weight > 0, "w_global": weight}
    if metadata.get("w_global") is not None:
        weight = float(metadata["w_global"])
        return {"global_supervision_enabled": weight > 0, "w_global": weight}
    return {"global_supervision_enabled": False, "w_global": 0.0}


def resolve_score_mode(requested: Optional[str], metadata: Optional[dict] = None) -> str:
    """没有确认全局监督时，不能把 CLS 分数当成已训练好的分类器。"""
    enabled = bool(supervision_metadata(metadata)["global_supervision_enabled"])
    mode = requested or "local_topk"
    if mode not in ("local_topk", "fusion", "cls_baseline"):
        raise ValueError(f"未知 score_mode {mode}")
    if mode in ("fusion", "cls_baseline") and not enabled:
        raise ValueError(
            f"当前 checkpoint 没有可用的全局监督，不能使用 {mode}。请改用 local_topk。"
        )
    return mode


def build_postprocess_config(requested_mode, metadata=None, *, topk_ratio: float = 0.05,
                             mask_mode: str = "fixed", global_weight: Optional[float] = None) -> PostprocessConfig:
    """融合权重只在 checkpoint 确认有全局监督、并且模式不是 local_topk 时才启用。"""
    mode = resolve_score_mode(requested_mode, metadata)
    if mode == "local_topk":
        weight = 0.0
    elif global_weight is None:
        weight = 0.5
    else:
        weight = float(global_weight)
    config = PostprocessConfig(score_mode=mode, topk_ratio=float(topk_ratio), global_weight=weight, mask_mode=mask_mode)
    config.validate()
    return config


def _unit_interval(value, name: str):
    if value is None:
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name}={number} 不在 0～1。"
            "这是后处理分数的阈值；原始相似度差请改用 score_space=raw_margin，不要套到 0～1 图上。"
        )
    return number


def split_manual_thresholds(image_threshold=None, pixel_threshold=None, *, score_space=None,
                            post_image_threshold=None, post_pixel_threshold=None,
                            raw_image_threshold=None, raw_pixel_threshold=None) -> dict:
    """旧的原始分数阈值和后处理 0～1 阈值分开存放，不能共用同一个数。"""
    if score_space not in (None, "raw_margin", "post_v1_probability_score"):
        raise ValueError(f"未知 score_space {score_space}")
    raw_image, raw_pixel = raw_image_threshold, raw_pixel_threshold
    post_image, post_pixel = post_image_threshold, post_pixel_threshold
    if score_space == "raw_margin":
        if raw_image is None:
            raw_image = image_threshold
        if raw_pixel is None:
            raw_pixel = pixel_threshold
    else:
        # 未声明或明确是 0～1 时，命令行 image/pixel 进入后处理。
        # 原始分数只能走 raw_*，避免 0.8/0.6 同时套到两种图上。
        if post_image is None:
            post_image = image_threshold
        if post_pixel is None:
            post_pixel = pixel_threshold
    return {
        "raw_margin": {
            "image_threshold": None if raw_image is None else float(raw_image),
            "pixel_threshold": None if raw_pixel is None else float(raw_pixel),
            "score_space": "raw_margin",
        },
        "post_v1_probability_score": {
            "image_threshold": _unit_interval(post_image, "image_threshold"),
            "pixel_threshold": _unit_interval(post_pixel, "pixel_threshold"),
            "score_space": "post_v1_probability_score",
        },
    }


def should_search_adaptive_thresholds(seed_threshold, grow_floor) -> bool:
    """两个都没给才自动搜索。只写了边界下限时不能把下限改掉。"""
    return seed_threshold is None and grow_floor is None


def _quantile_values(values, quantiles) -> list:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, q)) for q in quantiles]


def adaptive_candidates(records) -> list:
    """种子和边界下限跟验证图的分数走，不使用固定的 0.3/0.5/0.7。"""
    stats = []
    for item in records:
        score = np.asarray(item["score_map_refined"], dtype=np.float64)
        if score.size == 0 or not np.isfinite(score).all():
            continue
        maximum = float(score.max())
        gt = item.get("gt_mask_high")
        gt_any = gt is not None and bool(np.asarray(gt).astype(bool).any())
        label = int(item.get("label", -1))
        stats.append({
            "max": maximum,
            "normal": label == 0 or (bool(item.get("gt_available")) and not gt_any),
            "lesion": label == 1 or gt_any,
        })
    if not stats:
        return []
    seeds = set(_quantile_values([row["max"] for row in stats if row["normal"]], (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)))
    lesion_max = [row["max"] for row in stats if row["lesion"]]
    seeds.update(_quantile_values(lesion_max, (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)))
    positive = [row["max"] for row in stats if row["max"] > 0]
    if positive:
        start = max(min(positive) * 0.5, 1e-6)
        stop = min(max(max(positive), start * 1.01), 1.0)
        if stop > start:
            seeds.update(float(value) for value in np.geomspace(start, stop, num=8))
        else:
            seeds.add(float(start))
    weakest = [value for value in lesion_max if value > 0]
    if weakest:
        low = min(weakest)
        seeds.add(float(low))
        seeds.add(float(max(low * 0.5, 1e-6)))
    normal_peaks = [row["max"] for row in stats if row["normal"]]
    if normal_peaks:
        peak = float(max(normal_peaks))
        # 必须有一个严格高于正常图最高分的种子，否则恒定分数会被整张涂成掩码。
        if peak < 1.0:
            reject_all = min(1.0, peak + max(1e-3, abs(peak) * 0.01))
            if reject_all <= peak:
                reject_all = float(np.nextafter(np.float64(peak), np.float64(2.0)))
            if peak < reject_all <= 1.0:
                seeds.add(float(reject_all))
    seed_list = sorted({round(min(max(float(value), 0.0), 1.0), 6) for value in seeds if value > 0 and np.isfinite(value)})
    if len(seed_list) > 12:
        index = np.linspace(0, len(seed_list) - 1, 12).round().astype(int)
        seed_list = [seed_list[int(i)] for i in sorted(set(index.tolist()))]
    normal_max = [row["max"] for row in stats if row["normal"]]
    pairs = []
    for seed in seed_list:
        floors = {0.0, round(seed * 0.25, 6), round(seed * 0.5, 6), round(min(seed * 0.75, seed), 6)}
        for value in normal_max:
            if 0.0 <= value <= seed:
                floors.add(round(float(value), 6))
        for floor in sorted(floors):
            if 0.0 <= floor <= seed <= 1.0:
                pairs.append((float(seed), float(floor)))
    return pairs


def score_adaptive_preview(records) -> dict:
    """同时统计正常图是否出现任何预测，以及病灶图是否与 GT 重叠。"""
    normal_flags = []
    lesion_hits = []
    for item in records:
        pred = item.get("pred_mask")
        pred_any = pred is not None and bool(np.asarray(pred).astype(bool).any())
        gt = item.get("gt_mask_high")
        gt_arr = None if gt is None else np.asarray(gt).astype(bool)
        gt_any = gt_arr is not None and bool(gt_arr.any())
        label = int(item.get("label", -1))
        is_normal = label == 0 or (bool(item.get("gt_available")) and not gt_any)
        is_lesion = label == 1 or gt_any
        if is_normal and not gt_any:
            normal_flags.append(float(pred_any))
        if is_lesion and gt_arr is not None:
            overlap = pred is not None and bool(np.logical_and(np.asarray(pred).astype(bool), gt_arr).any())
            lesion_hits.append(float(overlap))
    return {
        "normal_false_positive_rate": None if not normal_flags else float(np.mean(normal_flags)),
        "lesion_detection_rate": None if not lesion_hits else float(np.mean(lesion_hits)),
    }


def choose_adaptive_operating_point(rows):
    """先看病灶检出减去正常图误报，再用 Dice 打破平局。100% 误报不能入选。"""
    usable = [row for row in rows if row.get("seed") is not None and row.get("floor") is not None]
    if not usable:
        return {"ok": False, "reason": "没有阈值候选，无法完成校准"}
    measured = [row for row in usable if row.get("normal_false_positive_rate") is not None]
    if not measured:
        return {"ok": False, "reason": "验证数据不足，没有正常图可以估计误报，无法完成校准"}
    eligible = [row for row in measured if float(row["normal_false_positive_rate"]) < 1.0]
    if not eligible:
        return {"ok": False, "reason": "所有候选的正常图误报率都是 100%，无法完成校准"}

    def sort_key(row):
        lesion = row.get("lesion_detection_rate")
        false_positive = float(row["normal_false_positive_rate"])
        dice = row.get("dice")
        dice = -1.0 if dice is None or dice != dice else float(dice)
        primary = -false_positive if lesion is None else float(lesion) - false_positive
        return (primary, dice, -float(row["seed"]), -float(row["floor"]))

    chosen = dict(max(eligible, key=sort_key))
    chosen["ok"] = True
    return chosen


def _finite_map(array, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} 必须是二维图，收到形状 {values.shape}")
    if values.size and not np.isfinite(values).all():
        raise ValueError(f"{name} 含有非有限值")
    return values


def _box_filter(values: np.ndarray, radius: int) -> np.ndarray:
    height, width = values.shape
    cumulative = np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.arange(height), np.arange(width)
    y0, y1 = np.clip(ys - radius, 0, height - 1), np.clip(ys + radius + 1, 0, height)
    x0, x1 = np.clip(xs - radius, 0, width - 1), np.clip(xs + radius + 1, 0, width)
    total = (
        cumulative[np.ix_(y1, x1)] - cumulative[np.ix_(y0, x1)]
        - cumulative[np.ix_(y1, x0)] + cumulative[np.ix_(y0, x0)]
    )
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return (total / area).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """以 guide 的边缘为准，对 src 做保边平滑。数值函数，不上色。"""
    guide = _finite_map(guide, "guide")
    src = _finite_map(src, "src")
    if guide.shape != src.shape:
        raise ValueError(f"guide {guide.shape} 与 src {src.shape} 尺寸不一致")
    mean_i, mean_p = _box_filter(guide, radius), _box_filter(src, radius)
    corr_ip = _box_filter(guide * src, radius)
    corr_ii = _box_filter(guide * guide, radius)
    a = (corr_ip - mean_i * mean_p) / (corr_ii - mean_i * mean_i + eps)
    b = mean_p - a * mean_i
    return _box_filter(a, radius) * guide + _box_filter(b, radius)


def _bilinear_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    array = np.asarray(image, dtype=np.float32)
    resized = Image.fromarray(array, mode="F").resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def sigmoid_score(raw_map: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or float(temperature) <= 0:
        raise ValueError("temperature 必须是正数")
    raw = _finite_map(raw_map, "raw_map")
    scaled = np.clip(raw / float(temperature), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def topk_pool(score_map: np.ndarray, ratio: float, roi=None):
    """在高分辨率连续图上取最高的一部分像素。k 不来自 GT，也不是少样本 K。"""
    score = _finite_map(score_map, "score_map")
    if not 0 < float(ratio) <= 1:
        raise ValueError("topk_ratio 必须在 (0, 1] 内")
    valid = np.ones(score.shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
    if valid.shape != score.shape:
        raise ValueError("ROI 与分数图尺寸不一致")
    values = score[valid]
    fallback = False
    if values.size == 0:
        values = score.reshape(-1)
        fallback = True
    if values.size == 0:
        raise ValueError("分数图为空，无法做 Top-k")
    count = min(int(values.size), max(1, int(math.ceil(float(ratio) * values.size))))
    chosen = np.partition(values, values.size - count)[-count:]
    return float(np.mean(chosen)), fallback


def refine_score_map(score_map_high: np.ndarray, gray01: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    high = _finite_map(score_map_high, "score_map_high")
    gray = _finite_map(gray01, "gray01")
    if high.shape != gray.shape:
        raise ValueError("高分辨率分数图必须与灰度图同尺寸")
    if config.refine_method == "none":
        return np.clip(high, 0.0, 1.0)
    refined = guided_filter(gray, high, radius=int(config.guided_radius), eps=float(config.guided_eps))
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def label_components(mask: np.ndarray) -> np.ndarray:
    """8 邻域连通分量，返回从 1 开始的标签，0 为背景。"""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    parent = [0]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for y in range(height):
        for x in range(width):
            if not binary[y, x]:
                continue
            adjacent = []
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if yy < 0 or xx < 0 or xx >= width:
                    continue
                if labels[yy, xx]:
                    adjacent.append(find(int(labels[yy, xx])))
            if not adjacent:
                current += 1
                parent.append(current)
                labels[y, x] = current
            else:
                root = min(adjacent)
                labels[y, x] = root
                for item in adjacent:
                    parent[find(item)] = root
    for y in range(height):
        for x in range(width):
            if labels[y, x]:
                labels[y, x] = find(int(labels[y, x]))
    return labels


def _remove_small_components(mask: np.ndarray, min_ratio: float) -> np.ndarray:
    if min_ratio <= 0:
        return mask
    labels = label_components(mask)
    minimum = float(min_ratio) * mask.size
    keep = np.zeros(mask.shape, dtype=bool)
    for label in np.unique(labels):
        if label == 0:
            continue
        component = labels == label
        if int(component.sum()) >= minimum:
            keep |= component
    return keep


def _fill_small_holes(mask: np.ndarray, max_ratio: float) -> np.ndarray:
    if max_ratio <= 0:
        return mask
    holes = label_components(~mask)
    maximum = float(max_ratio) * mask.size
    filled = mask.copy()
    height, width = mask.shape
    for label in np.unique(holes):
        if label == 0:
            continue
        component = holes == label
        touches_border = bool(component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any())
        if touches_border:
            continue
        if int(component.sum()) <= maximum:
            filled |= component
    return filled


def otsu_threshold(values: np.ndarray, bins: int = 256) -> tuple:
    """在固定 [0, 1]、256 bin 上计算 Otsu。返回 (阈值, 状态)。"""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, "constant_map"
    if float(flat.min()) == float(flat.max()):
        return float(flat.min()), "constant_map"
    histogram, edges = np.histogram(np.clip(flat, 0.0, 1.0), bins=int(bins), range=(0.0, 1.0))
    total = float(histogram.sum())
    if total <= 0:
        return 0.0, "constant_map"
    probability = histogram.astype(np.float64) / total
    omega = np.cumsum(probability)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mu = np.cumsum(probability * centers)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    variance = np.zeros_like(omega)
    valid = denominator > 0
    variance[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    index = int(np.argmax(variance))
    return float(edges[index + 1]), "ok"


def make_prediction_mask(score_map_refined, pixel_threshold, config: PostprocessConfig, roi=None):
    score = _finite_map(score_map_refined, "score_map_refined")
    if pixel_threshold is None:
        return None, {"mask_status": "unavailable", "reason": "pixel threshold missing"}
    region = np.ones(score.shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
    mask = region & (score >= float(pixel_threshold))
    mask = _remove_small_components(mask, float(config.min_component_ratio))
    mask = _fill_small_holes(mask, float(config.max_hole_ratio))
    mask &= region
    return mask, {"mask_status": "predicted", "effective_threshold": float(pixel_threshold)}


def make_seeded_adaptive_mask(score_map_refined, seed_threshold, grow_floor, roi=None,
                              adaptive_method: str = "otsu", bins: int = 256):
    """种子约束的动态边界。不接收 GT、真实标签或 CLS 判定。"""
    score = _finite_map(score_map_refined, "score_map_refined")
    if seed_threshold is None or grow_floor is None:
        return None, {"mask_status": "unavailable", "adaptive_status": "not_run", "reason": "seed threshold missing"}
    seed_threshold = float(seed_threshold)
    grow_floor = float(grow_floor)
    if not 0 <= grow_floor <= seed_threshold <= 1:
        raise ValueError("需要 0 <= grow_floor <= seed_threshold <= 1")
    region = np.ones(score.shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
    if region.shape != score.shape:
        raise ValueError("ROI 与分数图尺寸不一致")
    if not region.any():
        region = np.ones(score.shape, dtype=bool)
        roi_fallback = True
    else:
        roi_fallback = False
    seeds = region & (score >= seed_threshold)
    info = {
        "roi_fallback": roi_fallback,
        "max_refined_score": float(score[region].max()) if region.any() else float(score.max()),
        "seed_threshold": seed_threshold,
        "grow_floor": grow_floor,
        "seed_pixel_count": int(seeds.sum()),
        "dynamic_threshold": None,
        "effective_grow_threshold": None,
        "candidate_component_count": 0,
        "kept_component_count": 0,
    }
    if not seeds.any():
        info.update({"mask_status": "no_seed", "adaptive_status": "not_run"})
        return np.zeros(score.shape, dtype=bool), info
    values = score[region]
    if adaptive_method != "otsu":
        raise ValueError(f"未知动态阈值算法 {adaptive_method}")
    dynamic, status = otsu_threshold(values, bins=bins)
    if status == "constant_map":
        grow = grow_floor
    else:
        grow = float(np.clip(dynamic, grow_floor, seed_threshold))
    candidates = region & (score >= grow)
    labels = label_components(candidates)
    seed_labels = set(np.unique(labels[seeds])) - {0}
    kept = np.isin(labels, list(seed_labels)) if seed_labels else np.zeros(score.shape, dtype=bool)
    info.update({
        "mask_status": "predicted",
        "adaptive_status": status,
        "dynamic_threshold": None if status == "constant_map" else dynamic,
        "effective_grow_threshold": grow,
        "candidate_component_count": int(len(set(np.unique(labels)) - {0})),
        "kept_component_count": int(len(seed_labels)),
    })
    return kept, info


def fuse_image_scores(score_global: float, score_local: float, config: PostprocessConfig) -> float:
    if config.score_mode == "local_topk":
        return float(score_local)
    if config.score_mode == "cls_baseline":
        return float(score_global)
    weight = float(config.global_weight)
    return float(weight * float(score_global) + (1.0 - weight) * float(score_local))


def postprocess_one(raw_map, score_global, gray01, temperature, config: PostprocessConfig,
                    thresholds: OperatingThresholds, roi=None) -> dict:
    """从一张原始异常图得到连续分数和掩码。不读取 GT。"""
    config.validate()
    raw = _finite_map(raw_map, "raw_map").copy()
    original = raw.copy()
    gray = _finite_map(gray01, "gray01")
    low = sigmoid_score(raw, temperature)
    high = _bilinear_resize(low, gray.shape[0], gray.shape[1])
    refined = refine_score_map(high, gray, config)
    local, roi_fallback = topk_pool(high, config.topk_ratio, roi)
    fused = fuse_image_scores(score_global, local, config)
    image_threshold = thresholds.image_threshold
    pred_label = None if image_threshold is None else bool(fused >= float(image_threshold))
    if config.mask_mode == "adaptive_seeded":
        pred_mask, mask_info = make_seeded_adaptive_mask(
            refined, thresholds.seed_threshold, thresholds.grow_floor, roi,
            config.adaptive_method, config.histogram_bins,
        )
    else:
        pred_mask, mask_info = make_prediction_mask(refined, thresholds.pixel_threshold, config, roi)
    if not np.array_equal(raw, original):
        raise RuntimeError("后处理修改了原始异常图")
    return {
        "raw_map": original,
        "score_map_high": high.astype(np.float32),
        "score_map_refined": refined.astype(np.float32),
        "pred_mask": pred_mask,
        "score_global": float(score_global),
        "score_local": float(local),
        "score_fused": float(fused),
        "score_mode": config.score_mode,
        "pred_label": pred_label,
        "roi_fallback": bool(roi_fallback or mask_info.get("roi_fallback")),
        "mask_info": mask_info,
        "postprocess_id": config.version,
    }

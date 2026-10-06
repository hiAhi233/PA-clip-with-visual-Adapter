"""脑 MRI 支持集抽样、清单和划分检查。

这里不加载模型。K 是正常与异常合计的支持样本数，不是每类各 K 个。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

SAMPLING_POLICY = "total_k_balanced_case_round_robin_v1"
SPLIT_SEED = 0


def stable_digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Optional[str]) -> str:
    if not path:
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_quotas(n_normal: int, n_abnormal: int, k: int):
    """合计 K 个，两类都够且 K 为奇数时异常多一个。某一类不够就用另一类补足。"""
    if k <= 0:
        raise ValueError(f"K 必须为正，收到 {k}")
    total = int(n_normal) + int(n_abnormal)
    if k > total:
        raise ValueError(f"K={k} 超过训练集样本数 {total}，不能把不足的集合当成 K-shot")
    if n_abnormal <= 0:
        return k, 0
    if n_normal <= 0:
        return 0, k
    target_abnormal = k // 2 + (k % 2)
    target_normal = k - target_abnormal
    if n_abnormal < target_abnormal:
        abnormal = n_abnormal
        normal = k - abnormal
    elif n_normal < target_normal:
        normal = n_normal
        abnormal = k - normal
    else:
        normal, abnormal = target_normal, target_abnormal
    if normal > n_normal or abnormal > n_abnormal or normal + abnormal != k:
        raise RuntimeError(f"配额计算错误：normal={normal}, abnormal={abnormal}, K={k}")
    return int(normal), int(abnormal)


def _validate_samples(samples: Sequence[dict]) -> None:
    if not samples:
        raise ValueError("训练集为空")
    seen = set()
    for sample in samples:
        label = int(sample["label"])
        if label not in (0, 1):
            raise ValueError(f"标签只允许 0/1，样本 {sample.get('sample_id')} 为 {label}")
        sample_id = sample["sample_id"]
        if sample_id in seen:
            raise ValueError(f"样本标识重复：{sample_id}")
        seen.add(sample_id)


def _round_robin(indices: Sequence[int], samples: Sequence[dict], quota: int, rng: np.random.RandomState):
    if quota == 0:
        return [], False
    grouped: Dict[str, List[int]] = {}
    unknown = False
    for index in indices:
        case_id = samples[index].get("case_id")
        if not case_id:
            unknown = True
            case_id = f"sample:{samples[index]['sample_id']}"
        grouped.setdefault(str(case_id), []).append(index)
    cases = sorted(grouped)
    rng.shuffle(cases)
    for case_id in cases:
        rng.shuffle(grouped[case_id])
    picked: List[int] = []
    round_index = 0
    while len(picked) < quota:
        progressed = False
        for case_id in cases:
            if round_index < len(grouped[case_id]):
                picked.append(grouped[case_id][round_index])
                progressed = True
                if len(picked) >= quota:
                    break
        if not progressed:
            break
        round_index += 1
    if len(picked) != quota:
        raise RuntimeError(f"轮转抽样得到 {len(picked)} 个，期望 {quota}")
    return picked, unknown


def sample_support_indices(samples: Sequence[dict], k: Optional[int], seed: int, policy: str = SAMPLING_POLICY):
    """只从给定的 train 样本里选索引。返回 (索引, 计数信息)。"""
    if policy != SAMPLING_POLICY:
        raise ValueError(f"未知抽样规则 {policy}")
    _validate_samples(samples)
    ordered = sorted(range(len(samples)), key=lambda index: samples[index]["sample_id"])
    if k is None:
        indices = ordered
    else:
        if int(k) <= 0:
            raise ValueError(f"K 必须为正，收到 {k}")
        normal = [index for index in ordered if int(samples[index]["label"]) == 0]
        abnormal = [index for index in ordered if int(samples[index]["label"]) == 1]
        normal_quota, abnormal_quota = class_quotas(len(normal), len(abnormal), int(k))
        rng = np.random.RandomState(int(seed))
        normal_picked, unknown_normal = _round_robin(normal, samples, normal_quota, rng)
        abnormal_picked, unknown_abnormal = _round_robin(abnormal, samples, abnormal_quota, rng)
        indices = normal_picked + abnormal_picked
        if len(indices) != int(k) or len(set(indices)) != len(indices):
            raise RuntimeError("支持集索引数量或唯一性不符合 K")
        if any(index < 0 or index >= len(samples) for index in indices):
            raise RuntimeError("支持集索引越出训练集")
    selected = [samples[index] for index in indices]
    case_ids = [sample.get("case_id") for sample in selected]
    if case_ids and all(case_ids):
        unique_cases = len(set(case_ids))
    else:
        unique_cases = None
    info = {
        "requested_k": None if k is None else int(k),
        "actual_support": len(indices),
        "normal_count": sum(int(sample["label"]) == 0 for sample in selected),
        "abnormal_count": sum(int(sample["label"]) == 1 for sample in selected),
        "unique_cases": unique_cases,
        "sampling_policy": policy,
        "cases_unknown": unique_cases is None,
    }
    return indices, info


def make_support_manifest(samples: Sequence[dict], indices: Sequence[int], data_root: str, sampling_config: dict) -> dict:
    chosen = []
    for index in indices:
        sample = samples[index]
        chosen.append({
            "sample_id": sample["sample_id"],
            "case_id": sample.get("case_id"),
            "relative_path": sample["relative_path"],
            "label": int(sample["label"]),
            "mask_relative_path": sample.get("mask_relative_path"),
            "z_indices": None if sample.get("z_indices") is None else [int(z) for z in sample["z_indices"]],
            "z_policy": sample.get("z_policy"),
            "modality": sample.get("modality"),
        })
    manifest = {
        "schema_version": 2,
        "task": "brain_mri",
        "data_root_name": os.path.basename(os.path.abspath(data_root)) if data_root else "",
        "support_unit": sampling_config["support_unit"],
        "requested_k": sampling_config.get("requested_k"),
        "actual_support": len(chosen),
        "seed": int(sampling_config["seed"]),
        "sampling_policy": sampling_config.get("sampling_policy", SAMPLING_POLICY),
        "class_counts": {
            "normal": sum(item["label"] == 0 for item in chosen),
            "abnormal": sum(item["label"] == 1 for item in chosen),
        },
        "unique_cases": sampling_config.get("unique_cases"),
        "split_id": sampling_config.get("split_id"),
        "case_disjoint_verified": bool(sampling_config.get("case_disjoint_verified", False)),
        "volume_sampling_policy": sampling_config.get("volume_sampling_policy"),
        "samples": chosen,
    }
    manifest["support_digest"] = stable_digest(_support_rows(chosen))
    return manifest


def _support_rows_v1(items):
    """schema 1 的摘要不含通道。旧 checkpoint 仍按这个算法核对。"""
    return [
        [
            item["sample_id"], int(item["label"]), item.get("relative_path"),
            item.get("z_indices"), item.get("z_policy"),
        ]
        for item in items
    ]


def _support_rows_v2(items):
    return [
        [
            item["sample_id"], int(item["label"]), item.get("relative_path"),
            item.get("z_indices"), item.get("z_policy"), item.get("modality"),
        ]
        for item in items
    ]


def _support_rows(items, schema_version: int = 2):
    if int(schema_version) >= 2:
        return _support_rows_v2(items)
    return _support_rows_v1(items)


def support_digest(manifest: dict) -> str:
    """按清单版本重算摘要，不采用清单里可能过期的字段。"""
    samples = manifest.get("samples") or []
    version = int(manifest.get("schema_version") or 1)
    return stable_digest(_support_rows(samples, version))


def _check_support_counts(manifest: dict, samples) -> None:
    if manifest.get("actual_support") is not None and int(manifest["actual_support"]) != len(samples):
        raise ValueError(
            f"支持集 actual_support={manifest['actual_support']}，但样本只有 {len(samples)} 条"
        )
    if manifest.get("requested_k") is not None and int(manifest["requested_k"]) != len(samples):
        raise ValueError(
            f"支持集 requested_k={manifest['requested_k']}，但样本只有 {len(samples)} 条"
        )
    counts = manifest.get("class_counts")
    if isinstance(counts, dict) and ("normal" in counts or "abnormal" in counts):
        normal = sum(int(item["label"]) == 0 for item in samples)
        abnormal = sum(int(item["label"]) == 1 for item in samples)
        if int(counts.get("normal", normal)) != normal or int(counts.get("abnormal", abnormal)) != abnormal:
            raise ValueError("支持集类别数量与样本列表不一致")


def validate_support_manifest(manifest: dict) -> str:
    """核对摘要和条数。schema 1 用旧算法验证，通过后迁移到 schema 2。"""
    samples = list(manifest.get("samples") or [])
    stored = manifest.get("support_digest")
    version = int(manifest.get("schema_version") or 1)
    digest_v1 = stable_digest(_support_rows_v1(samples))
    digest_v2 = stable_digest(_support_rows_v2(samples))
    if version >= 2:
        if stored and stored != digest_v2:
            raise ValueError("支持集摘要与样本列表不一致，拒绝使用这份清单")
        _check_support_counts(manifest, samples)
        manifest["support_digest"] = digest_v2
        return digest_v2
    if stored and stored != digest_v1 and stored != digest_v2:
        raise ValueError("支持集摘要与样本列表不一致，拒绝使用这份清单")
    _check_support_counts(manifest, samples)
    manifest["schema_version"] = 2
    manifest["support_digest"] = digest_v2
    return digest_v2


def _modality_key(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _sample_aliases(sample: dict):
    aliases = set()
    for key in ("sample_id", "relative_path"):
        value = sample.get(key)
        if value:
            aliases.add(str(value).replace("\\", "/"))
    return aliases


def validate_support_held_out(support_samples, val_samples=None, test_samples=None) -> dict:
    """评估集不能包含 checkpoint 支持集里的样本或患者。患者信息不全时不能记为已验证。"""
    support = list(support_samples or [])
    groups = {"val": list(val_samples or []), "test": list(test_samples or [])}
    support_aliases = set()
    for sample in support:
        support_aliases.update(_sample_aliases(sample))
    overlapped = []
    for name, samples in groups.items():
        shared = set()
        for sample in samples:
            shared.update(_sample_aliases(sample) & support_aliases)
        if shared:
            overlapped.append(f"{name}: {sorted(shared)[:5]}")
    if overlapped:
        raise ValueError("评估集包含训练支持样本：" + "; ".join(overlapped))

    support_cases = {sample.get("case_id") for sample in support if sample.get("case_id")}
    case_overlaps = []
    for name, samples in groups.items():
        shared = support_cases & {sample.get("case_id") for sample in samples if sample.get("case_id")}
        if shared:
            case_overlaps.append(f"{name}: {sorted(shared)[:5]}")
    if case_overlaps:
        raise ValueError("评估集包含训练支持患者：" + "; ".join(case_overlaps))

    eval_samples = groups["val"] + groups["test"]
    complete = bool(support) and bool(eval_samples)
    complete = complete and all(sample.get("case_id") for sample in support)
    complete = complete and all(sample.get("case_id") for sample in eval_samples)
    return {"case_disjoint_verified": complete}


def restore_support_indices(samples: Sequence[dict], saved_manifest: dict, data_root: str = ""):
    """按清单中的样本身份恢复。样本缺失、标签变化或重复都报错，不另抽替补。"""
    del data_root  # 身份使用相对路径和 sample_id，搬迁数据根目录不应改变匹配。
    validate_support_manifest(saved_manifest)
    _validate_samples(samples)
    by_id = {}
    for index, sample in enumerate(samples):
        by_id[sample["sample_id"]] = index
    indices = []
    seen = set()
    for item in saved_manifest.get("samples", []):
        sample_id = item["sample_id"]
        if sample_id in seen:
            raise ValueError(f"支持集清单中的样本重复：{sample_id}")
        seen.add(sample_id)
        if sample_id not in by_id:
            raise ValueError(f"支持样本不存在，不能用其他样本代替：{sample_id}")
        current = samples[by_id[sample_id]]
        if int(current["label"]) != int(item["label"]):
            raise ValueError(
                f"支持样本 {sample_id} 的标签从 {item['label']} 变为 {current['label']}"
            )
        saved_path = item.get("relative_path")
        if saved_path and current.get("relative_path") and saved_path != current["relative_path"]:
            raise ValueError(f"支持样本 {sample_id} 的相对路径已变化：{saved_path} -> {current['relative_path']}")
        if item.get("z_indices") is not None:
            current["z_indices"] = [int(z) for z in item["z_indices"]]
        saved_modality = _modality_key(item.get("modality"))
        current_modality = _modality_key(current.get("modality"))
        if saved_modality is not None and current_modality is not None and saved_modality != current_modality:
            raise ValueError(
                f"支持样本 {sample_id} 的通道从 {item.get('modality')} 变为 {current.get('modality')}"
            )
        if saved_modality is not None and current_modality is None:
            current["modality"] = item.get("modality")
        indices.append(by_id[sample_id])
    if not indices:
        raise ValueError("支持集清单是空的")
    return indices


def saved_z_indices(item: dict) -> List[int]:
    """恢复时只读取清单里的 z。没有记录就报错，避免悄悄重选切片。"""
    if not item.get("z_indices"):
        raise ValueError(f"三维支持样本 {item.get('sample_id')} 没有保存 z 索引，拒绝重新选片")
    return [int(z) for z in item["z_indices"]]


def normalized_sample_path(sample: dict) -> str:
    """把样本实际文件收成可比较的路径。没有路径时返回空字符串。"""
    path = sample.get("path")
    if not path:
        return ""
    return os.path.normcase(os.path.realpath(path))


def validate_splits(train_samples, val_samples=None, test_samples=None, require_cases: bool = False) -> dict:
    """检查样本标识、实际文件和患者是否跨 train/val/test。返回是否完成患者隔离验证。"""
    groups = {
        "train": list(train_samples or []),
        "val": list(val_samples or []),
        "test": list(test_samples or []),
    }
    id_sets = {}
    path_sets = {}
    for name, samples in groups.items():
        ids = [sample["sample_id"] for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} 内部样本标识重复")
        id_sets[name] = set(ids)
        paths = [normalized_sample_path(sample) for sample in samples]
        paths = [path for path in paths if path]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{name} 内部文件路径重复")
        path_sets[name] = set(paths)
    overlaps = []
    path_overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = id_sets[left] & id_sets[right]
        if shared:
            overlaps.append(f"{left}/{right}: {sorted(shared)[:5]}")
        shared_paths = path_sets[left] & path_sets[right]
        if shared_paths:
            path_overlaps.append(f"{left}/{right}: {sorted(shared_paths)[:3]}")
    if overlaps:
        raise ValueError("同一样本进入了多个划分：" + "; ".join(overlaps))
    if path_overlaps:
        raise ValueError("同一文件进入了多个划分：" + "; ".join(path_overlaps))

    verified = False
    if require_cases:
        case_sets = {}
        for name, samples in groups.items():
            missing = [sample["sample_id"] for sample in samples if not sample.get("case_id")]
            if missing:
                raise ValueError(f"{name} 有样本缺少患者编号，例如 {missing[:5]}")
            case_sets[name] = {sample["case_id"] for sample in samples}
        shared_cases = []
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = case_sets[left] & case_sets[right]
            if shared:
                shared_cases.append(f"{left}/{right}: {sorted(shared)[:5]}")
        if shared_cases:
            raise ValueError("同一患者进入了多个划分：" + "; ".join(shared_cases))
        verified = True
    return {"case_disjoint_verified": verified}


def validate_masks(samples: Sequence[dict], indices: Sequence[int], local_supervision: bool) -> None:
    """给了掩码目录时，选中的异常样本必须真有掩码。全零只能表示正常组织，不能表示标注缺失。"""
    if not local_supervision:
        return
    missing = []
    for index in indices:
        sample = samples[index]
        if int(sample["label"]) == 1 and not sample.get("mask_found"):
            missing.append(sample.get("relative_path") or sample["sample_id"])
    if missing:
        preview = "\n".join(missing[:20])
        extra = "" if len(missing) <= 20 else f"\n... 另有 {len(missing) - 20} 个"
        raise ValueError(f"以下异常样本缺少掩码，已停止训练：\n{preview}{extra}")


def split_digest(train_samples, val_samples=None, test_samples=None) -> str:
    payload = {
        name: sorted(sample["sample_id"] for sample in (samples or []))
        for name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples))
    }
    return stable_digest(payload)


def prompt_snapshot(prompts) -> dict:
    return {
        "levels": list(prompts.levels),
        "normal": {level: list(prompts.normal[level]) for level in prompts.levels},
        "abnormal": {level: list(prompts.abnormal[level]) for level in prompts.levels},
    }


def experiment_identity(config: dict) -> str:
    return stable_digest(config)

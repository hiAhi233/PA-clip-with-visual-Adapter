"""脑 MRI 的多 K、多种子调度。

单组训练仍走 text_side_anomaly.train.run_experiment。
一次运行只用一个提示词版本，默认 brain_mri_sentence。
"""

from __future__ import annotations

import argparse
import json
import os

from text_side_anomaly.dataset import load_manifest_splits, scan_split
from text_side_anomaly.fewshot import validate_splits
from text_side_anomaly.prompts import DEFAULT_BRAIN_MRI_PROMPT_SET
from text_side_anomaly.train import make_args, run_experiment


GROUPS = {
    "T": {"inlayer": True, "visual_inlayer": False, "train_stage": "text"},
    "V": {"inlayer": False, "visual_inlayer": True, "train_stage": "visual"},
    "TVJ": {"inlayer": True, "visual_inlayer": True, "train_stage": "joint"},
    "TVS": {"inlayer": True, "visual_inlayer": True, "train_stage": "tvs"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="脑 MRI 小样本批量调度")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mask-root", default=None)
    parser.add_argument("--val-data-root", default=None)
    parser.add_argument("--val-mask-root", default=None)
    parser.add_argument("--eval-data-root", default=None)
    parser.add_argument("--eval-mask-root", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--case-id-regex", default=None)
    parser.add_argument("--data-format", default="slice", choices=["slice", "volume"])
    parser.add_argument("--ks", required=True, help="合计支持样本数，例如 5,10,20")
    parser.add_argument("--seeds", required=True, help="例如 0,1,2")
    parser.add_argument("--groups", default="TVJ", help="T,V,TVJ,TVS 的子集")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-set", default=None)
    parser.add_argument("--results", default=os.path.join("runs", "brain_mri", "results.jsonl"))
    parser.add_argument("--out-root", default=os.path.join("runs", "brain_mri"))
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-mode", default=None, choices=["local_topk", "fusion", "cls_baseline"])
    parser.add_argument("--topk-ratio", type=float, default=0.05)
    parser.add_argument("--global-weight", type=float, default=None)
    parser.add_argument("--mask-mode", default="fixed", choices=["fixed", "adaptive_seeded"])
    parser.add_argument("--seed-threshold", type=float, default=None)
    parser.add_argument("--grow-floor", type=float, default=None)
    parser.add_argument("--image-threshold", type=float, default=None)
    parser.add_argument("--pixel-threshold", type=float, default=None)
    parser.add_argument("--score-space", default=None, choices=["raw_margin", "post_v1_probability_score"])
    parser.add_argument("--post-image-threshold", type=float, default=None)
    parser.add_argument("--post-pixel-threshold", type=float, default=None)
    parser.add_argument("--raw-image-threshold", type=float, default=None)
    parser.add_argument("--raw-pixel-threshold", type=float, default=None)
    return parser


def _csv_ints(text: str):
    return [int(part) for part in str(text).split(",") if part.strip()]


def _csv_names(text: str):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def iter_jobs(args):
    """展开组别、K 和种子。不展开提示词。"""
    prompt = args.prompt_set or DEFAULT_BRAIN_MRI_PROMPT_SET
    if "," in str(prompt):
        raise ValueError("批量运行只使用一个提示词版本，不会把新旧词表各跑一遍")
    groups = _csv_names(args.groups)
    unknown = [name for name in groups if name not in GROUPS]
    if unknown:
        raise ValueError(f"未知训练组 {unknown}，可选 {sorted(GROUPS)}")
    for group in groups:
        spec = GROUPS[group]
        for k in _csv_ints(args.ks):
            for seed in _csv_ints(args.seeds):
                job = make_args(
                    args.data_root,
                    mask_root=args.mask_root,
                    val_data_root=args.val_data_root,
                    val_mask_root=args.val_mask_root,
                    eval_data_root=args.eval_data_root,
                    eval_mask_root=args.eval_mask_root,
                    split_manifest=args.split_manifest,
                    case_id_regex=args.case_id_regex,
                    data_format=args.data_format,
                    n_support=k,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    prompt_set=prompt,
                    save_dir=args.out_root,
                    init_ckpt=args.init_ckpt,
                    no_heatmaps=args.no_heatmaps,
                    fresh=args.fresh,
                    device=args.device,
                    score_mode=args.score_mode,
                    topk_ratio=args.topk_ratio,
                    global_weight=args.global_weight,
                    mask_mode=args.mask_mode,
                    seed_threshold=args.seed_threshold,
                    grow_floor=args.grow_floor,
                    image_threshold=args.image_threshold,
                    pixel_threshold=args.pixel_threshold,
                    score_space=args.score_space,
                    post_image_threshold=args.post_image_threshold,
                    post_pixel_threshold=args.post_pixel_threshold,
                    raw_image_threshold=args.raw_image_threshold,
                    raw_pixel_threshold=args.raw_pixel_threshold,
                    inlayer=spec["inlayer"],
                    visual_inlayer=spec["visual_inlayer"],
                    train_stage=spec["train_stage"],
                )
                yield group, job


def _load_records(path: str):
    run_ids = set()
    config_ids = set()
    evaluation_ids = set()
    if not os.path.isfile(path):
        return run_ids, config_ids, evaluation_ids
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("run_id"):
                run_ids.add(item["run_id"])
            if item.get("experiment_id"):
                config_ids.add(item["experiment_id"])
            if item.get("evaluation_id"):
                evaluation_ids.add(item["evaluation_id"])
    return run_ids, config_ids, evaluation_ids


def should_append(record: dict, run_ids, config_ids, reused: bool, evaluation_ids=None) -> bool:
    """同一训练可以有多次后处理评估。评估标识变了要另记一行，不能当成已完成复用。"""
    evaluation_ids = set() if evaluation_ids is None else evaluation_ids
    eval_id = record.get("evaluation_id")
    if eval_id and eval_id not in evaluation_ids:
        return True
    if record.get("run_id") in run_ids:
        return False
    if reused and record.get("experiment_id") in config_ids:
        return False
    return True


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _record(group: str, job, result: dict) -> dict:
    image = (result.get("metrics") or {}).get("image") or {}
    pixel = (result.get("metrics") or {}).get("pixel") or {}
    return {
        "group": group,
        "k": job.n_support,
        "seed": job.seed,
        "prompt_set": job.prompt_set,
        "train_stage": job.train_stage,
        "steps": job.steps,
        "normal_count": result.get("normal_count"),
        "abnormal_count": result.get("abnormal_count"),
        "support_digest": result.get("support_digest"),
        "evaluation_split": result.get("evaluation_split"),
        "auroc": image.get("auroc"),
        "ap": image.get("ap"),
        "f1": image.get("f1"),
        "f1_oracle": image.get("f1_oracle"),
        "dice": pixel.get("dice"),
        "iou": pixel.get("iou"),
        "dice_oracle": pixel.get("dice_oracle"),
        "status": result.get("status"),
        "checkpoint": result.get("checkpoint"),
        "support_manifest": os.path.join(result["run_dir"], "support_manifest.json"),
        "experiment_id": result.get("experiment_id"),
        "evaluation_id": result.get("evaluation_id"),
        "evaluation_status": result.get("evaluation_status"),
        "visualization_status": result.get("visualization_status"),
        "run_id": result.get("run_id"),
        "score_mode": ((result.get("metrics") or {}).get("postprocess") or {}).get("config", {}).get("score_mode"),
        "topk_ratio": getattr(job, "topk_ratio", None),
        "mask_mode": getattr(job, "mask_mode", None),
    }


def _summarize(records) -> None:
    grouped = {}
    for record in records:
        key = (
            record["group"], record["k"], record["steps"], record["prompt_set"],
            record["train_stage"], record.get("evaluation_split"),
        )
        grouped.setdefault(key, []).append(record)
    print("[summary] 按完整协议分组，不同组别或步数不会混成一个均值")
    for key, rows in grouped.items():
        group, k, steps, prompt, stage, split = key
        scores = [row["auroc"] for row in rows if isinstance(row.get("auroc"), (int, float))]
        mean = sum(scores) / len(scores) if scores else None
        mean_text = "null" if mean is None else f"{mean:.4f}"
        print(
            f"  {group} K={k} steps={steps} prompt={prompt} stage={stage} "
            f"split={split} seeds={len(rows)} mean_auroc={mean_text}"
        )


def _check_shared_split(args) -> None:
    if args.split_manifest:
        groups = load_manifest_splits(args.split_manifest)
        train, val, test = groups["train"], groups["val"], groups["test"]
        require_cases = True
    else:
        train = scan_split(args.data_root, args.mask_root, args.data_format, args.case_id_regex)
        val = scan_split(args.val_data_root, args.val_mask_root, args.data_format, args.case_id_regex) if args.val_data_root else []
        test = scan_split(args.eval_data_root, args.eval_mask_root, args.data_format, args.case_id_regex) if args.eval_data_root else []
        require_cases = bool(args.case_id_regex)
    validate_splits(train, val, test, require_cases=require_cases)
    print(
        f"[split] 所有 K 与 seed 共用同一划分 train={len(train)} val={len(val)} test={len(test)} "
        f"case_checked={require_cases}"
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    _check_shared_split(args)
    run_ids, config_ids, evaluation_ids = _load_records(args.results)
    records = []
    for group, job in iter_jobs(args):
        print(f"[job] group={group} K={job.n_support} seed={job.seed} stage={job.train_stage}")
        result = run_experiment(job)
        record = _record(group, job, result)
        records.append(record)
        if should_append(record, run_ids, config_ids, bool(result.get("reused")), evaluation_ids):
            _append_jsonl(args.results, record)
            if record.get("run_id"):
                run_ids.add(record["run_id"])
            if record.get("experiment_id"):
                config_ids.add(record["experiment_id"])
            if record.get("evaluation_id"):
                evaluation_ids.add(record["evaluation_id"])
    _summarize(records)
    return records


if __name__ == "__main__":
    main()

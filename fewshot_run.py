"""小样本 + 层内适配器：全自动对照实验。

一条命令跑完，**全程无需人工介入**：

    python fewshot_run.py
    python fewshot_run.py --ks 5,10,20 --seeds 0,1,2 --steps 2700
    python fewshot_run.py --groups A,C            # 只跑部分组

跑四组对照（每组多种子，报 mean / 区间）：

    A  输出端 adapter + 全监督  —— 复现现有基线 Dice 0.6295，确认脚本没跑偏
    B  层内 adapter  + 全监督   —— 隔离「插在哪」这一个变量
    C  层内 adapter  + 小样本   —— K = 5 / 10 / 20 的退化曲线
    D  输出端 adapter + 小样本  —— 隔离小样本下「插在哪」是否更关键

设计要点：
  - **按优化步数对齐**（默认 2700 步 ≈ 原 15 epoch × 179 步），而不是按 epoch。
    小样本时每"epoch"只有 1 个 batch，按 epoch 对齐就没法比了。
  - 支持集只从 **train 划分**抽，按病例抽；与 test 病例断言不相交。
  - 层内 adapter 会改变文本塔，所以**每步都要重算锚点**（`encode_anchors`）——
    既是为了数值正确，也是因为复用上一步的锚点会 backward 到已释放的计算图。
  - 每步 `model.lock_backbone_eval()`：`model.train()` 会把 BERT 的 dropout(0.1) 打开，
    而评估时是关的；层内 adapter 在塔内，这个不一致会直接污染结果。
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.metrics import pixel_metrics
from text_side_anomaly.model import TextSideAnomalyModel, _is_v2_checkpoint, _torch_load
from text_side_anomaly.prompts import resolve_prompt_set
from text_side_anomaly.thymoma_dataset import (
    ThymomaSliceDataset,
    case_ids,
    make_slice_splits,
    sample_support,
)

import thymoma_local as TL

NPZ_DIR = "thymoma_slices"
ORGAN = "thymoma"
PROMPT_SET = "sentence"          # 改动 7 那版整句提示词，与基线 ckpt 同源


def _c(x) -> str:
    """Windows 控制台中文乱码兜底：把不可编码字符换成 '?'。"""
    try:
        return str(x).encode(sys.stdout.encoding or "utf-8", "replace").decode(
            sys.stdout.encoding or "utf-8")
    except Exception:
        return str(x)


# ---------------------------------------------------------------------- #
# 断点续跑
# ---------------------------------------------------------------------- #
# 本机是 RTX 4060 Laptop，跑满时会把适配器拉爆直接硬断电（Kernel-Power 41），
# 已经崩过两次。所以每跑完一组就立刻落盘 —— 不能等 12 组全跑完再统一写，
# 那样一崩就是 0 产出。
RESULTS_JSONL = "_fewshot_results.jsonl"


SPLIT_ID = "thymoma_slices:seed0"
TEXT_LAYERS_DEFAULT = [8, 9, 10, 11]
TEXT_POSITIONS_DEFAULT = ["attn", "ffn"]
VISUAL_LAYERS_DEFAULT = [5, 8, 11]
VISUAL_POSITIONS_DEFAULT = ["attn", "ffn"]
LEGACY_LOSS = {
    "margin": 0.3, "margin_lo": -0.3, "margin_d": 0.7,
    "w_text": 1.0, "w_global": 0.0, "w_local": 1.0, "w_div": 0.0, "w_level": 0.0,
}
VISUAL_MODES = {"visual", "tv_seq", "tv_joint"}


def loss_for_stage(stage: str, base: dict) -> dict:
    """视觉阶段关掉只依赖文本锚点的项，避免把它们算进视觉适配器的损失。"""
    out = dict(base)
    if stage == "visual":
        out["w_text"] = 0.0
        out["w_div"] = 0.0
    return out


def _loss_id(mode: str, step_plan: str, base: dict) -> str:
    if mode == "tv_seq":
        payload = {
            "text": loss_for_stage("text", base),
            "visual": loss_for_stage("visual", base),
            "plan": step_plan,
        }
    elif mode == "visual":
        payload = loss_for_stage("visual", base)
    else:
        payload = loss_for_stage("text", base)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


IDENTITY_KEYS = [
        "identity_version", "steps", "batch_size", "prompt_set", "prompt_digest",
        "optimizer", "model_config", "metric_protocol",
        "mode", "k", "seed", "train_stage", "text_layers", "text_positions", "text_bottleneck",
        "visual_layers", "visual_positions", "visual_bottleneck", "visual_lambda",
        "init_ckpt_id", "loss_id", "split_id", "step_plan",
]


def experiment_identity(fields: dict) -> str:
    return json.dumps({key: fields[key] for key in IDENTITY_KEYS}, sort_keys=True, ensure_ascii=False)


def artifact_stem(fields: dict) -> str:
    digest = hashlib.sha256(experiment_identity(fields).encode("utf-8")).hexdigest()[:16]
    k = fields["k"] if fields["k"] is not None else "full"
    return f"{fields['mode']}_k{k}_s{fields['seed']}_{digest}"


def normalize_identity(record: dict) -> Optional[dict]:
    """缺少预算等关键字段的历史记录仅保留作档案，不能据此跳过新实验。"""
    if not isinstance(record, dict) or record.get("identity_version") != 2:
        return None
    if any(key not in record for key in IDENTITY_KEYS):
        return None
    return {key: record[key] for key in IDENTITY_KEYS}


def ckpt_identity(path: Optional[str]) -> str:
    if not path:
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_done(path: str) -> Dict[str, dict]:
    """读已完成实验。键是完整配置，不再只看 mode/k/seed。"""
    done: Dict[str, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # 崩在写一半的那行：丢掉，重跑这一组
                continue
            fields = normalize_identity(record)
            if fields is None:
                continue
            key = experiment_identity(fields)
            if record.get("experiment_id") != key:
                continue
            if not all(name in record for name in ("dice", "auroc", "iou", "n_trainable")):
                continue
            checkpoint = record.get("checkpoint_path")
            if not checkpoint or not os.path.isfile(checkpoint):
                continue
            if not record.get("checkpoint_sha256") or ckpt_identity(checkpoint) != record["checkpoint_sha256"]:
                continue
            done[key] = record
    return done


def append_result(path: str, r: dict) -> None:
    """追加一行并 fsync —— 断电掉的是这一行还是下一行，不能是已完成的这行。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # 上次断电可能留下没有换行的半条 JSON，先隔开，确保新记录仍可被读取。
    needs_newline = False
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------- #
# 建模
# ---------------------------------------------------------------------- #
def build_model(device, levels, mode: str, seed: int, hparams: dict):
    """output / inlayer 保持原对照。visual、tv_seq、tv_joint 才装视觉层内适配器。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    text_on = mode in {"inlayer", "tv_seq", "tv_joint"}
    visual_on = mode in VISUAL_MODES
    stage = {"visual": "visual", "tv_joint": "joint"}.get(mode, "text")
    cfg = Config(
        device=str(device), levels=levels, train_stage=stage,
        freeze_output_text_adapter=text_on or visual_on,
        visual_inlayer_enabled=visual_on,
        visual_inlayer_layers=list(hparams["visual_layers"]),
        visual_inlayer_positions=list(hparams["visual_positions"]),
        visual_inlayer_bottleneck=hparams["visual_bottleneck"],
        visual_inlayer_lambda=hparams["visual_lambda"],
    )
    inlayer = None
    if text_on:
        inlayer = {
            "organs": [ORGAN], "organ": ORGAN,
            "bottleneck": hparams["text_bottleneck"],
            "layers": list(hparams["text_layers"]),
            "positions": tuple(hparams["text_positions"]),
        }
    visual = None
    if visual_on:
        visual = {
            "organs": [ORGAN], "organ": ORGAN,
            "bottleneck": hparams["visual_bottleneck"],
            "lambda_t": hparams["visual_lambda"],
            "layers": list(hparams["visual_layers"]),
            "positions": tuple(hparams["visual_positions"]),
        }
    model = TextSideAnomalyModel(cfg, inlayer=inlayer, visual_inlayer=visual).to(device)
    return cfg, model


# ---------------------------------------------------------------------- #
# 训练 / 评估
# ---------------------------------------------------------------------- #
def train(model, prompts, support_files, device, steps: int, batch_size: int,
          lr: float = 1e-4, seed: int = 0, tag: str = "", stage: Optional[str] = None,
          loss_base: Optional[dict] = None):
    if steps < 1 or batch_size < 1:
        raise ValueError("steps 和 batch_size 必须是正整数")
    if not support_files:
        raise ValueError("训练支持集为空")
    torch.manual_seed(seed)
    if stage is not None:
        model.set_train_stage(stage, organ=ORGAN)
    ds = ThymomaSliceDataset(NPZ_DIR, files=support_files)
    bs = max(1, min(batch_size, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                        drop_last=False)
    opt = model.build_optimizer(lr=lr, weight_decay=1e-5)
    spec = loss_for_stage(model.training_stage, loss_base or LEGACY_LOSS)
    crit = TotalLoss(
        margin=spec["margin"], w_text=spec["w_text"], w_global=spec["w_global"],
        w_local=spec["w_local"], margin_lo=spec["margin_lo"], margin_d=spec["margin_d"],
        w_div=spec["w_div"], w_level=spec["w_level"],
    )

    cached_enc = None
    if model.training_stage == "visual":
        model.eval()
        with torch.no_grad():
            cached_enc = model.encode_anchors(prompts)

    t0 = time.time()
    it = iter(loader)
    run = 0.0
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        model.train()
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        # 文本或联合阶段每步重算锚点。视觉阶段文本权重不变，使用上面缓存的锚点。
        enc = cached_enc if cached_enc is not None else model.encode_anchors(prompts)
        out = model(images, enc)
        loss = crit(enc, out, labels, masks)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        run += loss["total"].item()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    [{tag}] step {step+1}/{steps} loss={run/(step+1):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return run / steps


@torch.no_grad()
def evaluate(model, prompts, files, device, batch_size: int = 16) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(ThymomaSliceDataset(NPZ_DIR, files=files),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    enc = model.encode_anchors(prompts)
    maps, masks = [], []
    for batch in loader:
        out = model(batch["image"].to(device), enc)
        maps.append(out["anomaly_map"].detach().cpu())
        masks.append(batch["mask"].detach() if torch.is_tensor(batch["mask"]) else batch["mask"])
    return pixel_metrics(torch.cat(maps).numpy(), torch.cat(masks).numpy())


# ---------------------------------------------------------------------- #
# 单组
# ---------------------------------------------------------------------- #
def _csv_ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _csv_strs(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def step_plan_for(mode: str, init_ckpt: Optional[str]) -> str:
    init_ckpt = init_ckpt if mode in VISUAL_MODES else None
    if mode == "tv_seq" and init_ckpt:
        return "visual_from_ckpt"
    if mode == "tv_seq":
        return "half_half"
    if init_ckpt:
        return "from_ckpt"
    return "single"


def identity_fields(mode: str, k: Optional[int], seed: int, hparams: dict,
                    step_plan: str, init_ckpt_id: str, steps: int, batch_size: int) -> dict:
    text_on = mode in {"inlayer", "tv_seq", "tv_joint"}
    visual_on = mode in VISUAL_MODES
    stage = {"visual": "visual", "tv_joint": "joint", "tv_seq": "sequential"}.get(mode, "text")
    return {
        "identity_version": 2,
        "steps": steps,
        "batch_size": batch_size,
        "prompt_set": hparams["prompt_set"],
        "prompt_digest": hparams["prompt_digest"],
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5},
        "model_config": hparams["model_config"],
        "metric_protocol": "test_oracle_dice_v1",
        "mode": mode,
        "k": k,
        "seed": seed,
        "train_stage": stage,
        "text_layers": list(hparams["text_layers"]) if text_on else [],
        "text_positions": list(hparams["text_positions"]) if text_on else [],
        "text_bottleneck": hparams["text_bottleneck"] if text_on else None,
        "visual_layers": list(hparams["visual_layers"]) if visual_on else [],
        "visual_positions": list(hparams["visual_positions"]) if visual_on else [],
        "visual_bottleneck": hparams["visual_bottleneck"] if visual_on else None,
        "visual_lambda": hparams["visual_lambda"] if visual_on else None,
        "init_ckpt_id": init_ckpt_id if visual_on else "",
        "loss_id": _loss_id(mode, step_plan, hparams["loss"]),
        "split_id": hparams["split_id"],
        "step_plan": step_plan,
    }


def run_one(mode: str, k: Optional[int], seed: int, device, splits, prompts,
            steps: int, batch_size: int, ckpt_dir: str, heat_dir: Optional[str],
            hparams: dict, init_ckpt: Optional[str]):
    init_ckpt = init_ckpt if mode in VISUAL_MODES else None
    if steps < 1 or batch_size < 1 or (mode == "tv_seq" and not init_ckpt and steps < 2):
        raise ValueError("步数和 batch size 必须为正；从头训练 TVS 至少需要 2 步")
    fields = identity_fields(mode, k, seed, hparams, step_plan_for(mode, init_ckpt),
                             ckpt_identity(init_ckpt), steps, batch_size)
    safe = artifact_stem(fields)
    ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"{safe}.pt"))
    # 同配置 --fresh 或上次中断重跑也保留之前已经生成的产物。
    if os.path.exists(ckpt_path) or (heat_dir and os.path.exists(heat_dir)):
        revision = uuid.uuid4().hex[:8]
        ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"{safe}_{revision}.pt"))
        if heat_dir:
            heat_dir = f"{heat_dir}_{revision}"
    names = {
        "output": "输出端", "inlayer": "文本层内", "visual": "视觉层内",
        "tv_seq": "先文本后视觉", "tv_joint": "双侧联合",
    }
    cnt = "全监督" if k is None else f"K={k}"
    tag = f"{names.get(mode, mode)}/{cnt}/s{seed}"
    print(f"\n{'='*72}\n[{tag}] 开始（支持集 {'全部' if k is None else k} 张）\n{'='*72}", flush=True)

    support = sample_support(splits["train"], k, seed)
    sup_cases = case_ids(support)
    test_cases = case_ids(splits["test"])
    leak = sup_cases & test_cases
    assert not leak, f"支持集与 test 病例重叠，会泄漏：{leak}"

    _, model = build_model(device, prompts.levels, mode, seed, hparams)
    if init_ckpt and mode in VISUAL_MODES:
        resume = {"tv_seq": "visual", "tv_joint": "joint"}.get(mode, "visual")
        blob = model.load_checkpoint(
            init_ckpt, map_location=device, allow_missing_visual=True,
            resume_stage=resume, organ=ORGAN,
        )
        resolve_prompt_set(hparams["prompt_set"], blob.get("prompt_set") if _is_v2_checkpoint(blob) else None,
                           TL.PROMPT_SETS, PROMPT_SET, loading=True)
        del blob
        print(f"  从 {init_ckpt} 继续，阶段={model.training_stage}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  支持集 {len(support)} 张（{len(sup_cases)} 个病例），可训练参数 {n_train}")

    if mode == "tv_seq" and not init_ckpt:
        first = steps // 2
        second = steps - first
        print(f"  两阶段步数 {first}+{second}，合计 {steps}")
        text_loss = train(model, prompts, support, device, first, batch_size, seed=seed,
                     tag=tag + "/text", stage="text", loss_base=hparams["loss"])
        visual_loss = train(model, prompts, support, device, second, batch_size, seed=seed,
                     tag=tag + "/visual", stage="visual", loss_base=hparams["loss"])
        loss = (text_loss * first + visual_loss * second) / steps
    else:
        loss = train(model, prompts, support, device, steps, batch_size, seed=seed,
                     tag=tag, stage=model.training_stage, loss_base=hparams["loss"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m = evaluate(model, prompts, splits["test"], device, batch_size)
    print(f"  → Dice(test 最优阈值)={m['dice']:.4f}  AUROC={m['pixel_auroc']:.4f}  IoU={m['iou']:.4f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_checkpoint(
        ckpt_path, prompt_set=hparams["prompt_set"], step=steps, seed=seed,
        support_set=list(support), split_id=hparams["split_id"],
        experiment_id=experiment_identity(fields), experiment_config=fields,
        global_supervision_enabled=False, w_global=0.0,
    )
    formal = TL.calibrate_thymoma_predictions(model, prompts, splits["val"], splits["test"], device)
    if formal["metrics"].get("calibration") == "ok":
        print(
            f"  → 后处理 Dice={formal['metrics']['dice_micro']:.4f}  "
            f"阈值={formal['metrics']['pixel_threshold']:.4f}（验证集冻结）"
        )
    else:
        print(f"  → 后处理未校准：{formal['metrics'].get('reason')}")
    if heat_dir:
        TL.prepare_heatmap_dir(heat_dir, clean=False)
        with open(os.path.join(heat_dir, "formal_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(formal["metrics"], handle, ensure_ascii=False, indent=2)
        TL.save_formal_figures(
            model, prompts, splits["test"], device, heat_dir, n=6,
            thresholds=formal["thresholds"],
        )

    result = dict(fields)
    result.update({
        "n_support": len(support), "n_trainable": n_train, "loss": loss,
        "dice": m["dice"], "auroc": m["pixel_auroc"], "iou": m["iou"],
        "postprocess_dice": formal["metrics"].get("dice_micro"),
        "postprocess_calibration": formal["metrics"].get("calibration"),
        "experiment_id": experiment_identity(result),
        "checkpoint_path": ckpt_path, "checkpoint_sha256": ckpt_identity(ckpt_path),
        "heatmap_dir": os.path.abspath(heat_dir) if heat_dir else None,
    })
    return result


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2700,
                    help="每组每种的优化步数（默认 2700 ≈ 原 15 epoch × 179 步）")
    ap.add_argument("--ks", type=str, default="5",
                    help="小样本的 K 列表（默认只跑 K=5：一次只用支持集里 5 张切片）")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prompt-set", choices=sorted(TL.PROMPT_SETS), default=None,
                    help="新实验默认 sentence；初始化已有权重时继承其版本，显式指定必须与其一致")
    ap.add_argument("--groups", type=str, default="A,B,C,D",
                    help="A-D 是原文本对照。V=只训练视觉，TVS=先文本后视觉，TVJ=双侧一起训练")
    ap.add_argument("--ckpt-dir", type=str, default="fewshot_ckpt")
    ap.add_argument("--results", type=str, default=None,
                    help="断点记录。只跑 V/TVS/TVJ 时默认写到独立文件，避免和旧文本实验混用")
    ap.add_argument("--text-inlayer-layers", type=str, default="8,9,10,11")
    ap.add_argument("--text-inlayer-positions", type=str, default="attn,ffn")
    ap.add_argument("--text-bottleneck", type=int, default=64)
    ap.add_argument("--visual-inlayer-layers", type=str, default="5,8,11")
    ap.add_argument("--visual-inlayer-positions", type=str, default="attn,ffn")
    ap.add_argument("--visual-inlayer-bottleneck", type=int, default=64)
    ap.add_argument("--visual-inlayer-lambda", type=float, default=0.1)
    ap.add_argument("--init-ckpt", type=str, default=None,
                    help="只对 V/TVS/TVJ 生效。允许缺少视觉适配器参数；TVS 会跳过文本阶段，"
                         "把全部 --steps 用在视觉阶段")
    ap.add_argument("--no-heatmaps", action="store_true")
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="每组之间等待秒数：让供电与温度回落，削掉连续满载的瞬态尖峰。"
                         "只影响总耗时，**不改变任何训练结果**。0 = 关闭")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略 _fewshot_results.jsonl 从头重跑（默认自动跳过已完成的组）")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    groups = {g.strip().upper() for g in args.groups.split(",")}
    visual_groups = {"V", "TVS", "TVJ"}
    if not groups or not groups <= {"A", "B", "C", "D", *visual_groups}:
        ap.error("--groups 只能包含 A,B,C,D,V,TVS,TVJ")
    if args.steps < 1 or args.batch_size < 1 or not seeds or not ks or any(k < 1 for k in ks):
        ap.error("--steps、--batch-size、--ks 必须为正，种子列表不能为空")
    if "TVS" in groups and not args.init_ckpt and args.steps < 2:
        ap.error("TVS 的文本/视觉两个阶段至少各需要 1 步，--steps 必须 >= 2")
    if len(set(seeds)) != len(seeds) or len(set(ks)) != len(ks):
        ap.error("--seeds 和 --ks 不能包含重复项")
    effective_init = args.init_ckpt if groups & visual_groups else None
    if args.init_ckpt and not effective_init:
        print("[setup] 当前未选择视觉组，--init-ckpt 不参与这些实验")
    init_prompt = args.prompt_set or PROMPT_SET
    if effective_init:
        blob = _torch_load(effective_init, map_location="cpu")
        try:
            init_prompt = resolve_prompt_set(
                args.prompt_set, blob.get("prompt_set") if _is_v2_checkpoint(blob) else None,
                TL.PROMPT_SETS, PROMPT_SET, loading=True,
            )
        except ValueError as exc:
            ap.error(str(exc))
        del blob
    if groups and groups <= visual_groups:
        if args.ckpt_dir == "fewshot_ckpt":
            args.ckpt_dir = "fewshot_ckpt_visual"
        if args.results is None:
            args.results = "_fewshot_visual_results.jsonl"
    elif args.results is None:
        args.results = RESULTS_JSONL
    hparams = {
        "text_layers": _csv_ints(args.text_inlayer_layers),
        "text_positions": _csv_strs(args.text_inlayer_positions),
        "text_bottleneck": args.text_bottleneck,
        "visual_layers": _csv_ints(args.visual_inlayer_layers),
        "visual_positions": _csv_strs(args.visual_inlayer_positions),
        "visual_bottleneck": args.visual_inlayer_bottleneck,
        "visual_lambda": args.visual_inlayer_lambda,
        "loss": dict(LEGACY_LOSS),
    }
    defaults = Config()
    hparams["model_config"] = {
        key: getattr(defaults, key) for key in (
            "model_name", "image_size", "text_hidden", "visual_hidden", "max_text_len",
            "bottleneck", "lambda_t", "lambda_t_learnable", "adapter_dropout",
            "fusion_learnable", "ms_layers", "temperature",
        )
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = make_slice_splits(NPZ_DIR, seed=0)
    split_manifest = {name: sorted(os.path.abspath(f) for f in files) for name, files in splits.items()}
    hparams["split_id"] = SPLIT_ID + ":" + hashlib.sha256(
        json.dumps(split_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    print(f"[setup] device={device} 新实验提示词={args.prompt_set or PROMPT_SET} "
          f"初始化权重提示词={init_prompt} 训练切片={len(splits['train'])} "
          f"test={len(splits['test'])}")

    # 组定义：(标签, mode, K 列表)
    plan = []
    if "A" in groups:
        plan.append(("A 输出端 + 全监督", "output", [None]))
    if "B" in groups:
        plan.append(("B 层内 + 全监督", "inlayer", [None]))
    if "C" in groups:
        plan.append(("C 层内 + 小样本", "inlayer", ks))
    if "D" in groups:
        plan.append(("D 输出端 + 小样本", "output", ks))
    if "V" in groups:
        plan.append(("V 视觉层内 + 小样本", "visual", ks))
    if "TVS" in groups:
        plan.append(("TVS 先文本后视觉", "tv_seq", ks))
    if "TVJ" in groups:
        plan.append(("TVJ 双侧联合", "tv_joint", ks))

    # 先摊平成任务表：要判断「是不是最后一组」，最后一组跑完不必再冷却
    jobs = [(label, mode, k, seed)
            for label, mode, klist in plan
            for k in klist
            for seed in seeds]

    done = {} if args.fresh else load_done(args.results)
    if done:
        print(f"[resume] 已有 {len(done)} 组完成，自动跳过（加 --fresh 可强制重跑）")
    print(f"[setup] checkpoint={args.ckpt_dir} 记录={args.results}")

    init_id = ckpt_identity(effective_init)
    results = []
    for i, (label, mode, k, seed) in enumerate(jobs):
        job_init = effective_init if mode in VISUAL_MODES else None
        job_params = dict(hparams)
        job_params["prompt_set"] = init_prompt if job_init else (args.prompt_set or PROMPT_SET)
        prompts = TL.PROMPT_SETS[job_params["prompt_set"]]
        job_params["prompt_digest"] = hashlib.sha256(json.dumps(
            {"normal": prompts.normal, "abnormal": prompts.abnormal},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        fields = identity_fields(mode, k, seed, job_params, step_plan_for(mode, job_init),
                                 init_id, args.steps, args.batch_size)
        key = experiment_identity(fields)
        if key in done:
            r = done[key]
            print(f"[跳过] {label} seed={seed} 已完成 Dice={r['dice']:.4f}", flush=True)
            results.append(r)
            continue

        heat = None
        if not args.no_heatmaps and seed == seeds[0]:
            heat = f"heatmaps_fewshot_{artifact_stem(fields)}"
        r = run_one(mode, k, seed, device, splits, prompts,
                    args.steps, args.batch_size, args.ckpt_dir, heat, job_params, job_init)
        append_result(args.results, r)          # ← 先落盘，再冷却
        results.append(r)

        if args.cooldown > 0 and i < len(jobs) - 1:
            print(f"[cooldown] 等待 {args.cooldown:.0f}s 让供电/温度回落…", flush=True)
            time.sleep(args.cooldown)

    # ---------- 汇总 ----------
    lines = ["", "=" * 84,
             f"小样本 + 层内适配器 对照汇总（提示词={sorted({r['prompt_set'] for r in results})}，步数={args.steps}，"
             f"种子={seeds}）",
             "=" * 84,
             "Dice / IoU 沿用历史口径：在 test 掩码上选最优阈值（oracle），不是固定阈值泛化指标。",
             f"{'配置':<26}{'支持集':>7}{'可训练参数':>12}{'Dice':>20}{'AUROC':>12}",
             "-" * 84]
    print("\n".join(lines))
    for label, mode, klist in plan:
        for k in klist:
            rs = [r for r in results if r["mode"] == mode and r["k"] == k]
            if not rs:
                continue
            d = np.array([r["dice"] for r in rs])
            a = np.array([r["auroc"] for r in rs])
            cnt = "全部" if k is None else str(k)
            row = (f"{label:<26}{cnt:>7}{rs[0]['n_trainable']:>12}"
                   f"{d.mean():>10.4f} [{d.min():.4f},{d.max():.4f}]"
                   f"{a.mean():>12.4f}")
            print(row)
            lines.append(row)
    ref = [r for r in results if r["mode"] == "output" and r["k"] is None]
    if ref:
        d = np.array([r["dice"] for r in ref])
        tail = (f"\n基线参考：输出端 + 全监督 Dice = {d.mean():.4f} "
                f"[{d.min():.4f}, {d.max():.4f}]（docx 改动 7/8 记录 0.6275）")
        print(tail)
        lines.append(tail)

    summary_id = hashlib.sha256("\n".join(r["experiment_id"] for r in results).encode("utf-8")).hexdigest()[:16]
    summary_path = os.path.splitext(args.results)[0] + f"_summary_{summary_id}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[done] 汇总已写入 {summary_path}")


if __name__ == "__main__":
    main()

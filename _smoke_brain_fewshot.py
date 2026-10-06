"""脑 MRI 少步 CUDA 检查：真实 BiomedCLIP，2 步，不跑正式训练。"""

import os
import tempfile
from pathlib import Path

import torch
from PIL import Image

from text_side_anomaly.config import Config
from text_side_anomaly.dataset import SliceAnomalyDataset
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.prompts import BRAIN_MRI_SENTENCE_PROMPTS
from text_side_anomaly.train import make_args, run_experiment


def _png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=value).save(path)


def _up_abs(module) -> float:
    total = 0.0
    for name, param in module.named_parameters():
        if ".up." in name:
            total += float(param.detach().abs().sum())
    return total


def _clip_has_grad(model) -> bool:
    return any(
        param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        for param in model.clip.parameters()
    )


def _batch(loader):
    return next(iter(loader))


def _step(model, batch, anchors, criterion, optimizer, device, cached):
    model.train()
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    masks = batch["mask"].to(device)
    encoded = cached if cached is not None else model.encode_anchors(anchors)
    loss = criterion(encoded, model(images, encoded), labels, masks)["total"]
    optimizer.zero_grad()
    loss.backward()
    if _clip_has_grad(model):
        raise RuntimeError("冻结主干出现了梯度")
    optimizer.step()
    return encoded


def _build(device):
    cfg = Config(
        data_format="slice",
        train_stage="text",
        freeze_output_text_adapter=True,
        visual_inlayer_enabled=True,
        levels=list(BRAIN_MRI_SENTENCE_PROMPTS.levels),
        device=str(device),
    )
    inlayer = {
        "organs": ["brain"], "organ": "brain", "layers": [8, 9, 10, 11],
        "positions": ("attn", "ffn"), "bottleneck": 64,
    }
    visual = {
        "organs": ["brain"], "organ": "brain", "layers": [5, 8, 11],
        "positions": ("attn", "ffn"), "bottleneck": 64, "lambda_t": 0.1,
    }
    return TextSideAnomalyModel(cfg, inlayer=inlayer, visual_inlayer=visual).to(device), cfg


def _check_stages(model, cfg, loader, device):
    anchors = BRAIN_MRI_SENTENCE_PROMPTS
    batch = _batch(loader)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(initial)
    model.set_train_stage("text", organ="brain")
    if model.visual_inlayer_bank.active is not None:
        raise RuntimeError("文本阶段视觉适配器没有旁路")
    text_up = _up_abs(model.inlayer_bank)
    _step(model, batch, anchors, TotalLoss(margin=cfg.margin, w_div=0.0), model.build_optimizer(), device, None)
    if _up_abs(model.inlayer_bank) <= text_up:
        raise RuntimeError("文本层内适配器没有更新")
    if _up_abs(model.visual_inlayer_bank) != 0:
        raise RuntimeError("文本阶段视觉适配器被更新了")

    model.load_state_dict(initial)
    model.set_train_stage("visual", organ="brain")
    model.eval()
    with torch.no_grad():
        cached = model.encode_anchors(anchors)
        frozen_text = [param.detach().clone() for param in model.inlayer_bank.parameters()]
    visual_up = _up_abs(model.visual_inlayer_bank)
    _step(
        model, batch, anchors,
        TotalLoss(margin=cfg.margin, w_text=0.0, w_div=0.0),
        model.build_optimizer(), device, cached,
    )
    with torch.no_grad():
        again = model.encode_anchors(anchors)
    for level in cached:
        for kind in ("normal", "abnormal"):
            if not torch.allclose(cached[level][kind], again[level][kind]):
                raise RuntimeError("视觉阶段文本锚点发生了变化")
    for before, after in zip(frozen_text, model.inlayer_bank.parameters()):
        if not torch.equal(before, after.detach()):
            raise RuntimeError("视觉阶段文本参数发生了变化")
    if _up_abs(model.visual_inlayer_bank) <= visual_up:
        raise RuntimeError("视觉层内适配器没有更新")

    model.load_state_dict(initial)
    model.set_train_stage("joint", organ="brain")
    text_up = _up_abs(model.inlayer_bank)
    visual_up = _up_abs(model.visual_inlayer_bank)
    _step(model, batch, anchors, TotalLoss(margin=cfg.margin, w_div=0.0), model.build_optimizer(), device, None)
    if _up_abs(model.inlayer_bank) <= text_up or _up_abs(model.visual_inlayer_bank) <= visual_up:
        raise RuntimeError("联合阶段没有同时更新两侧适配器")
    print("[smoke] 三个阶段的参数更新和冻结检查通过")
    return batch


def _check_roundtrip(model, cfg, batch, device, directory):
    path = os.path.join(directory, "roundtrip.pt")
    model.save_checkpoint(
        path, prompt_set="brain_mri_sentence", task="brain_mri",
        support_set={"actual_support": 4}, seed=0, step=1,
    )
    model.eval()
    with torch.no_grad():
        encoded = model.encode_anchors(BRAIN_MRI_SENTENCE_PROMPTS)
        first = model(batch["image"].to(device), encoded)["cls_probs"]
    restored = TextSideAnomalyModel.from_arch_spec(model.architecture_spec(), device=str(device))
    restored.load_checkpoint(path, map_location=device, resume_stage="joint", organ="brain")
    restored.eval()
    with torch.no_grad():
        encoded = restored.encode_anchors(BRAIN_MRI_SENTENCE_PROMPTS)
        second = restored(batch["image"].to(device), encoded)["cls_probs"]
    if not torch.allclose(first, second, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"重新加载后输出不一致，最大差 {(first - second).abs().max().item()}")
    print("[smoke] 保存再加载后输出一致")


def _check_experiment(data_root, mask_root, directory):
    result = run_experiment(make_args(
        data_root, mask_root=mask_root, n_support=4, seed=0, steps=2, batch_size=2,
        inlayer=True, visual_inlayer=True, train_stage="joint",
        prompt_set="brain_mri_sentence", save_dir=directory, no_heatmaps=True, device="cuda",
    ))
    blob = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    if blob.get("prompt_set") != "brain_mri_sentence":
        raise RuntimeError("checkpoint 提示词与本次运行不一致")
    if blob.get("task") != "brain_mri" or blob.get("training_stage") != "joint":
        raise RuntimeError("checkpoint 任务或阶段元数据不正确")
    support = blob.get("support_set") or {}
    if support.get("actual_support") != 4 or support.get("class_counts") != {"normal": 2, "abnormal": 2}:
        raise RuntimeError(f"支持集计数不正确：{support.get('class_counts')}")
    if blob.get("steps_this_run") != 2 or result["completed_steps"] != 2:
        raise RuntimeError("本次步数没有按 2 记录")
    if set(result["seen_sample_ids"]) != set(result["support_ids"]):
        raise RuntimeError("训练看到的样本和支持集不一致")
    print("[smoke] K、提示词、阶段和步数与 checkpoint 一致")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 才能做这次短检查")
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, value in (("n0.png", 30), ("n1.png", 40)):
            _png(root / "train" / "normal" / name, value)
        for name, value in (("a0.png", 200), ("a1.png", 210)):
            _png(root / "train" / "abnormal" / name, value)
            _png(root / "masks" / "abnormal" / name, 255)
        dataset = SliceAnomalyDataset(str(root / "train"), mask_root=str(root / "masks"), image_size=224, grid=14)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        model, cfg = _build(device)
        batch = _check_stages(model, cfg, loader, device)
        _check_roundtrip(model, cfg, batch, device, tmp)
        _check_experiment(str(root / "train"), str(root / "masks"), str(root / "runs"))
    print("[smoke] 通过")


if __name__ == "__main__":
    main()

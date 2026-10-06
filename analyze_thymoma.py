"""胸腺瘤定位精度补充分析：预测热区是否落在 GT 肿瘤上。

加载已训练 checkpoint，对 test 集逐切片统计：
    - top-1 命中率（最热 cell 是否落在肿瘤内）
    - top-k 命中率（k = 该切片肿瘤 cell 数）
    - 病灶 cell 与背景 cell 的 amap 均值差
"""

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from text_side_anomaly.config import Config
from text_side_anomaly.model import (
    TextSideAnomalyModel,
    _is_v2_checkpoint,
    _torch_load,
)
from text_side_anomaly.thymoma_dataset import ThymomaSliceDataset, make_slice_splits
from text_side_anomaly.prompts import resolve_prompt_set
from thymoma_local import PROMPT_SETS


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="thymoma_local.pt")
    parser.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default=None)
    args = parser.parse_args()
    ckpt = args.ckpt
    blob = _torch_load(ckpt, map_location="cpu")
    prompt_name = resolve_prompt_set(
        args.prompt_set, blob.get("prompt_set") if _is_v2_checkpoint(blob) else None,
        PROMPT_SETS, "sentence", loading=True,
    )
    anchors = PROMPT_SETS[prompt_name]
    if _is_v2_checkpoint(blob):
        model = TextSideAnomalyModel.from_arch_spec(blob["config"], device=str(device))
        model.load_checkpoint_blob(blob)
        if list(anchors.levels) != list(model.cfg.levels):
            raise RuntimeError(
                f"提示词 {prompt_name} 的层 {list(anchors.levels)} 与 checkpoint 的 {list(model.cfg.levels)} 不一致"
            )
        print(f"已按 checkpoint 架构加载 stage={model.training_stage} prompt={prompt_name}")
    else:
        cfg = Config(device=str(device), levels=anchors.levels)
        model = TextSideAnomalyModel(cfg).to(device)
        model.load_checkpoint_blob(blob)
    del blob
    model.eval()
    enc = model.encode_anchors(anchors)

    splits = make_slice_splits("thymoma_slices", seed=0)
    test_ds = ThymomaSliceDataset("thymoma_slices", files=splits["test"])
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    n = 0
    top1_hit = 0.0
    topk_hit = 0.0
    pos_means, neg_means = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()          # (B,14,14)
        with torch.no_grad():
            amaps = model(images, enc)["anomaly_map"].cpu().numpy()  # (B,14,14)
        for i in range(amaps.shape[0]):
            am = amaps[i].reshape(-1)
            gt = masks[i].reshape(-1)
            if gt.sum() == 0:
                continue
            top1_hit += float(gt[int(np.argmax(am))] > 0)
            k = int(gt.sum())
            topk = np.argsort(am)[::-1][:k]
            topk_hit += float((gt[topk] > 0).mean())
            pos_means.append(am[gt > 0].mean())
            neg_means.append(am[gt == 0].mean())
            n += 1

    print(f"测试切片数（含病灶）: {n}")
    if n == 0:
        raise ValueError("测试集没有含病灶的切片，无法计算命中率")
    print(f"top-1 命中率（最热 cell 落在肿瘤内）: {top1_hit / n:.3f}")
    print(f"top-k 命中率（k=肿瘤 cell 数）: {topk_hit / n:.3f}")
    print(f"病灶 cell amap 均值: {np.mean(pos_means):+.4f}  vs  背景 cell amap 均值: {np.mean(neg_means):+.4f}")


if __name__ == "__main__":
    main()

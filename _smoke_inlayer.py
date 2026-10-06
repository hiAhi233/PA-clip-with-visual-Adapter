"""层内适配器 smoke：验证 inlayer_adapter.py 那条从未执行过的路径。

只验证不变量，不看指标：
  1. 挂载点数 = 层数 × 位置数，且前向后 hits 全部非零（证明猴补丁真的被调用）
  2. step 0 严格恒等：|r|=0，塔内适配器开/关的文本嵌入逐位相同
  3. 输出端 adapter 置零冻结后，可训参数只剩 inlayer_bank.*
  4. 真跑 3 步：层内参数确实在动、text_adapter 一动不动，且 loss 能反传
  5. attention 包装保留了二元组返回（跑通即证明；BertLayer.forward 里 out, _ = ...）
"""

import sys

import numpy as np
import torch

from text_side_anomaly.config import Config
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.model import TextSideAnomalyModel
from text_side_anomaly.thymoma_dataset import (
    ThymomaSliceDataset,
    case_ids,
    make_slice_splits,
    sample_support,
)

import thymoma_local as TL

ORGAN = "thymoma"
NPZ = "thymoma_slices"
LAYERS = [8, 9, 10, 11]
POSITIONS = ("attn", "ffn")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    prompts = TL.PROMPT_SETS["sentence"]
    cfg = Config(device=str(device), levels=prompts.levels)
    inlayer = {"organs": [ORGAN], "organ": ORGAN, "bottleneck": 64,
               "layers": LAYERS, "positions": POSITIONS}
    torch.manual_seed(0)
    model = TextSideAnomalyModel(cfg, inlayer=inlayer).to(device)

    # 「不要加在输出后面」：输出端压成恒等并冻住（与 thymoma_local.py --inlayer 同一套）
    with torch.no_grad():
        model.text_adapter.lambda_t.data.zero_()
    for p in model.text_adapter.parameters():
        p.requires_grad = False

    ok = True

    # ---- 1. 挂载点 / 恒等起点 ---- #
    n_mount = len(LAYERS) * len(POSITIONS)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    # 允许的非层内可训参数只有融合权重（3 层 → 3 个）；主干与输出端 adapter 都必须冻住
    not_inlayer = [n for n in trainable
                   if not n.startswith("inlayer_bank") and n != "fusion_weights"]
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_level = len(prompts.levels)
    print(f"[1] 可训参数={n_param}（{len(trainable)} 项），其中非 inlayer_bank 的："
          f"{[n for n in trainable if not n.startswith('inlayer_bank')]}")
    if not_inlayer:
        print("    ✗ 输出端 adapter 或主干还有可训参数（融合权重除外）")
        ok = False
    n_lambda = sum(1 for n in trainable if n.endswith("lambda_t"))
    per_slot = 768 * 64 + 64 + 64 * 768 + 768 + 1
    print(f"    层内 λ_t 个数={n_lambda}（应={n_mount}）；每槽 {per_slot}，"
          f"共 {n_mount} 槽 + 融合 {n_level} = {n_mount * per_slot + n_level}")
    if n_param != n_mount * per_slot + n_level:
        print("    ✗ 参数量与预期不符")
        ok = False

    # ---- 2. step 0 恒等 ---- #
    ids = model.tokenizer(["a normal chest CT"]).to(device)
    with torch.no_grad():
        t1 = model.encode_text_tokens(ids)
        # 包装闭包持有 bank，把 model.inlayer_bank 换成 None 关不掉适配器。
        model.inlayer_bank.set_active(None)
        t0 = model.encode_text_tokens(ids)
        model.inlayer_bank.set_active(ORGAN)
        for p in model.text_adapter.parameters():
            p.requires_grad = False
    d = (t1 - t0).abs().max().item()
    rs = [ad.up.weight.abs().max().item() for ad in model.inlayer_bank.adapters[ORGAN]]
    print(f"[2] 开/关层内适配器的嵌入最大差={d:.3e}；各槽 |W_up|max 最大={max(rs):.3e}")
    if d > 1e-6 or max(rs) > 0:
        print("    ✗ step 0 不是严格恒等")
        ok = False

    # ---- 3. 真跑 3 步（用 5 张支持集，与正式小样本同配置）---- #
    splits = make_slice_splits(NPZ, seed=0)
    support = sample_support(splits["train"], 5, seed=0)
    leak = case_ids(support) & case_ids(splits["test"])
    print(f"[3] 支持集 {len(support)} 张（{len(case_ids(support))} 个病例）"
          f" 与 test 重叠={leak or '无'}")
    if leak:
        ok = False

    ds = ThymomaSliceDataset(NPZ, files=support)
    batch = torch.stack([ds[i]["image"] for i in range(len(ds))]).to(device)
    masks = torch.stack([ds[i]["mask"] for i in range(len(ds))]).to(device)
    labels = torch.ones(len(ds), dtype=torch.long, device=device)

    before = {k: v.detach().clone() for k, v in model.inlayer_bank.state_dict().items()}
    txt_before = model.text_adapter.lambda_t.detach().clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    crit = TotalLoss(margin=0.3, w_text=1.0, w_global=0.0, w_local=1.0,
                     margin_lo=None, margin_d=0.7, w_div=0.0, w_level=0.0)
    for step in range(3):
        model.train()
        model.lock_backbone_eval()
        enc = model.encode_anchors(prompts)
        out = model(batch, enc)
        loss = crit(enc, out, labels, masks)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        print(f"    step {step + 1} loss={loss['total'].item():.4f}")

    hits = model.inlayer_hits
    print(f"[3] hits={hits}")
    if len(hits) != n_mount or any(v == 0 for v in hits.values()):
        print(f"    ✗ 期望 {n_mount} 个挂载点全部被调用")
        ok = False

    moved = sum(1 for k, v in model.inlayer_bank.state_dict().items()
                if not torch.equal(before[k], v))
    txt_moved = not torch.equal(txt_before, model.text_adapter.lambda_t.detach())
    print(f"[4] 有变化的层内张量 {moved}/{len(before)}；输出端 λ_t 变了={txt_moved}")
    if moved == 0:
        print("    ✗ 层内适配器没拿到梯度")
        ok = False
    if txt_moved:
        print("    ✗ 输出端 adapter 不该动")
        ok = False

    print("\n===== smoke " + ("通过 ✅" if ok else "失败 ❌") + " =====")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""视觉层内适配器的结构检查。不跑正式训练，只确认安装、梯度和 checkpoint。

用法（在 paclip 目录下）：
    python tests/test_visual_inlayer_adapter.py
"""

import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_side_anomaly.config import Config
from text_side_anomaly.inlayer_adapter import OrganAdapterBank
from text_side_anomaly.losses import TotalLoss
from text_side_anomaly.model import TextSideAnomalyModel, load_trained_model
from text_side_anomaly.visual_inlayer_adapter import (
    install_visual_inlayer_adapters,
    uninstall_visual_inlayer_adapters,
)

ORGAN = "thymoma"
OTHER = "brain"
LAYERS = [5, 8, 11]
POSITIONS = ("attn", "ffn")
TEXT_LAYERS = [8, 9, 10, 11]
PER_SLOT = 768 * 64 + 64 + 64 * 768 + 768 + 1


class Report:
    def __init__(self):
        self.ok = True

    def check(self, name, passed, detail):
        mark = "通过" if passed else "失败"
        print(f"[{mark}] {name} {detail}")
        self.ok = self.ok and passed


def anchors_for(model, device):
    anchors = {}
    for level in model.cfg.levels:
        anchors[level] = {
            "normal": F.normalize(torch.randn(model.projection_dim, device=device), dim=0),
            "abnormal": F.normalize(torch.randn(model.projection_dim, device=device), dim=0),
        }
    return anchors


def logits_of(model, image, anchors):
    model.eval()
    with torch.no_grad():
        return model(image, anchors)["patch_logits"].detach().clone()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visual-smoke] device={device}")
    report = Report()
    torch.manual_seed(0)

    cfg = Config(
        device=str(device),
        levels=["1", "2", "3"],
        ms_layers=[5, 8, 11],
        train_stage="visual",
        freeze_output_text_adapter=True,
        visual_inlayer_layers=list(LAYERS),
        visual_inlayer_positions=list(POSITIONS),
        visual_inlayer_bottleneck=64,
        visual_inlayer_lambda=0.1,
    )
    inlayer = {
        "organs": [ORGAN, OTHER], "organ": ORGAN,
        "layers": TEXT_LAYERS, "positions": POSITIONS, "bottleneck": 64, "lambda_t": 0.1,
    }
    visual = {
        "organs": [ORGAN, OTHER], "organ": ORGAN,
        "layers": LAYERS, "positions": POSITIONS, "bottleneck": 64, "lambda_t": 0.1,
    }
    model = TextSideAnomalyModel(cfg, inlayer=inlayer, visual_inlayer=visual).to(device)
    model.set_train_stage("visual", organ=ORGAN)
    image = torch.randn(1, 3, 224, 224, device=device)
    trunk = model.clip.visual.trunk
    expected = {f"L{layer}.{pos}" for layer in LAYERS for pos in POSITIONS}

    # 1. 两条视觉路径都打到 3 层 × 2 个位置
    for path in ("forward_features", "forward_intermediates"):
        model.visual_inlayer_hits.clear()
        with torch.no_grad():
            if path == "forward_features":
                trunk.forward_features(image)
            else:
                trunk.forward_intermediates(image, indices=list(LAYERS), norm=True, output_fmt="NLC")
        hits = dict(model.visual_inlayer_hits)
        report.check(
            f"安装覆盖/{path}",
            set(hits) == expected and all(value > 0 for value in hits.values()) and len(hits) == 6,
            f"hits={hits}",
        )

    # 10 的前半：维度。无多层读取时也要能反传。
    model.eval()
    with torch.no_grad():
        cls, patch, hw = model.encode_image(image)
    report.check(
        "数据维度",
        tuple(cls.shape) == (1, 512) and tuple(patch.shape) == (1, 196, 512) and hw == (14, 14),
        f"cls={tuple(cls.shape)} patch={tuple(patch.shape)} grid={hw}",
    )
    saved_layers = list(model.cfg.ms_layers)
    model.cfg.ms_layers = []
    single_cls, single_patch, single_hw = model.encode_image(image)
    # CLS 已归一化，平方均值是常数；选取一个坐标才能实际检验梯度传播。
    single_loss = single_cls[:, 0].mean()
    single_loss.backward()
    report.check(
        "无多层读取仍可训练",
        tuple(single_cls.shape) == (1, 512)
        and tuple(single_patch.shape) == (1, 196, 512)
        and single_hw == (14, 14)
        and model.visual_inlayer_bank.adapters[ORGAN][0].up.weight.grad is not None,
        f"grid={single_hw}",
    )
    model.zero_grad()
    model.cfg.ms_layers = saved_layers

    # 2/3. 零初始化与真实旁路
    model.eval()
    anchors = anchors_for(model, device)
    with torch.no_grad():
        on = model.encode_image(image)
        model.visual_inlayer_bank.set_active(None)
        off = model.encode_image(image)
        model.set_train_stage("visual", organ=ORGAN)
    gap = max((a - b).abs().max().item() for a, b in zip(on[:2], off[:2]))
    up_max = max(adapter.up.weight.abs().max().item() for adapter in model.visual_inlayer_bank.adapters[ORGAN])
    report.check("初始化恒等", gap == 0 and up_max == 0, f"最大差={gap:.3e} |W_up|={up_max:.3e}")

    # 6. 早层改动只影响该层及之后
    adapter = model.visual_inlayer_bank.get(5, "attn")
    with torch.no_grad():
        before4, before5, before8 = _layers(model, image, (4, 5, 8))
        adapter.up.weight.normal_(std=0.05)
        model.visual_inlayer_bank.set_active(None)
        base4, base5, base8 = _layers(model, image, (4, 5, 8))
        model.set_train_stage("visual", organ=ORGAN)
        after4, after5, after8 = _layers(model, image, (4, 5, 8))
        adapter.up.weight.zero_()
    report.check(
        "层内传播",
        torch.equal(after4, before4)
        and torch.equal(base4, before4)
        and not torch.equal(after5, base5)
        and not torch.equal(after8, base8),
        f"层4差={(after4 - base4).abs().max().item():.3e} "
        f"层5差={(after5 - base5).abs().max().item():.3e} "
        f"层8差={(after8 - base8).abs().max().item():.3e}",
    )

    # 4/5. 梯度：第一步只有 up；up 更新后 down 才有梯度
    mask = torch.zeros(1, 14, 14, device=device)
    mask[:, :4, :4] = 1
    labels = torch.ones(1, dtype=torch.long, device=device)
    crit = TotalLoss(margin=0.3, w_text=0.0, w_global=0.0, w_local=1.0, w_div=0.0, w_level=0.0)
    opt = model.build_optimizer(lr=1e-4, weight_decay=0.0)
    backbone = trunk.blocks[0].norm1.weight.detach().clone()
    text_up = model.inlayer_bank.adapters[ORGAN][0].up.weight.detach().clone()
    visual_before = {key: value.detach().clone() for key, value in model.visual_inlayer_bank.state_dict().items()}

    model.train()
    opt.zero_grad()
    loss = crit(anchors, model(image, anchors), labels, mask)
    loss["total"].backward()
    up_grad = adapter.up.weight.grad
    down_grad = adapter.down.weight.grad
    report.check(
        "首步 up 有梯度",
        up_grad is not None and torch.isfinite(up_grad).all() and up_grad.abs().sum().item() > 0,
        f"up={None if up_grad is None else up_grad.abs().sum().item():.3e}",
    )
    report.check(
        "首步 down 梯度为零",
        down_grad is not None and torch.isfinite(down_grad).all() and down_grad.abs().sum().item() == 0,
        f"down={None if down_grad is None else down_grad.abs().sum().item():.3e}",
    )
    report.check(
        "主干没有参数梯度",
        trunk.blocks[0].norm1.weight.grad is None and not trunk.blocks[0].norm1.weight.requires_grad,
        "",
    )
    opt.step()
    model.train()
    opt.zero_grad()
    loss = crit(anchors, model(image, anchors), labels, mask)
    loss["total"].backward()
    down_grad = adapter.down.weight.grad
    report.check(
        "up 更新后 down 有梯度",
        down_grad is not None and torch.isfinite(down_grad).all() and down_grad.abs().sum().item() > 0,
        f"down={None if down_grad is None else down_grad.abs().sum().item():.3e}",
    )
    opt.step()
    model.train()
    opt.zero_grad()
    loss = crit(anchors, model(image, anchors), labels, mask)
    loss["total"].backward()
    opt.step()

    moved = sum(1 for key, value in model.visual_inlayer_bank.state_dict().items() if not torch.equal(visual_before[key], value))
    text_same = torch.equal(text_up, model.inlayer_bank.adapters[ORGAN][0].up.weight)
    backbone_same = torch.equal(backbone, trunk.blocks[0].norm1.weight)
    report.check("更新后视觉变化且文本/主干不变", moved > 0 and text_same and backbone_same,
                 f"变化张量={moved} 文本不变={text_same} 主干不变={backbone_same}")

    # 7. 训练后的旁路，以及“把属性设成 None”关不掉闭包
    model.eval()
    with torch.no_grad():
        adapted = model.encode_image(image)[1]
        held = model.visual_inlayer_bank
        model.visual_inlayer_bank = None
        still = model.encode_image(image)[1]
        model.visual_inlayer_bank = held
        model.visual_inlayer_bank.set_active(None)
        bypassed = model.encode_image(image)[1]
        model.set_train_stage("visual", organ=ORGAN)
        restored = model.encode_image(image)[1]
    report.check(
        "旁路恢复",
        torch.equal(still, adapted)
        and not torch.equal(bypassed, adapted)
        and torch.equal(restored, adapted)
        and torch.equal(bypassed, off[1]),
        f"属性置空差={(still - adapted).abs().max().item():.3e} "
        f"真旁路差={(bypassed - adapted).abs().max().item():.3e}",
    )

    # 8. 三个阶段、换器官、eval/train 往返
    stage_counts = {}
    stage_ok = True
    for stage in ("text", "visual", "joint"):
        model.set_train_stage(stage, organ=ORGAN)
        stage_opt = model.build_optimizer(lr=1e-4, weight_decay=0.0)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stage_counts[stage] = count
        try:
            model.assert_optimizer_matches_stage(stage_opt)
        except AssertionError as exc:
            stage_ok = False
            print(f"    {stage} {exc}")
    report.check(
        "阶段参数集合",
        stage_ok and stage_counts == {"text": 8 * PER_SLOT + 3, "visual": 6 * PER_SLOT, "joint": 14 * PER_SLOT + 3},
        str(stage_counts),
    )
    model.set_train_stage("visual", organ=OTHER)
    other_opt = model.build_optimizer(lr=1e-4, weight_decay=0.0)
    other_names = model.trainable_parameter_names()
    organ_ok = all(f".{OTHER}." in name for name in other_names) and not any(f".{ORGAN}." in name for name in other_names)
    grad_snapshot = {name: param.requires_grad for name, param in model.named_parameters()}
    active_snapshot = (
        model.inlayer_bank.active,
        model.visual_inlayer_bank.active,
        model.training_stage,
    )
    model.eval()
    model.train()
    grad_same = all(param.requires_grad == grad_snapshot[name] for name, param in model.named_parameters())
    active_same = active_snapshot == (model.inlayer_bank.active, model.visual_inlayer_bank.active, model.training_stage)
    try:
        model.assert_optimizer_matches_stage(other_opt)
        roundtrip_ok = True
    except AssertionError:
        roundtrip_ok = False
    model.set_train_stage("text", organ=ORGAN)
    switched = False
    try:
        model.assert_optimizer_matches_stage(other_opt)
    except AssertionError:
        switched = True
    report.check(
        "器官切换与 eval/train",
        organ_ok and grad_same and active_same and roundtrip_ok and switched,
        f"器官正确={organ_ok} 往返={grad_same and active_same} 旧优化器失效={switched}",
    )
    model.set_train_stage("visual", organ=ORGAN)

    # 9. checkpoint 往返、错层、缺参数、重复安装
    model.eval()
    reference = logits_of(model, image, anchors)
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        model.save_checkpoint(
            path, optimizer=opt, prompt_set="sentence", step=3, seed=0,
            support_set=["synthetic"], split_id="synthetic",
        )
        rebuilt = load_trained_model(path, device)
        rebuilt.eval()
        again = logits_of(rebuilt, image, anchors)
        gap = (reference - again).abs().max().item()
        report.check("checkpoint 往返", gap == 0, f"logits 最大差={gap:.3e}")

        bad = dict(rebuilt.architecture_spec())
        bad["visual_inlayer_layers"] = [11]
        bad_arch = False
        try:
            rebuilt._check_architecture(bad, allow_missing_visual=False)
        except RuntimeError:
            bad_arch = True
        report.check("错误层配置报错", bad_arch, "")

        missing = model.state_dict()
        dropped = next(key for key in missing if key.startswith("visual_inlayer_bank."))
        broken = {key: value for key, value in missing.items() if key != dropped}
        missing_raises = False
        try:
            model._load_weights(broken, allow_missing_visual=False)
        except RuntimeError:
            missing_raises = True
        report.check("缺少视觉参数报错", missing_raises, dropped)

        rejected_partial = False
        try:
            model._load_weights(broken, allow_missing_visual=True)
        except RuntimeError:
            rejected_partial = True
        report.check("迁移开关不能放行残缺视觉参数", rejected_partial, "")

        text_only = {key: value for key, value in missing.items() if not key.startswith("visual_inlayer_bank.")}
        rejected_trained_target = False
        try:
            model._load_weights(text_only, allow_missing_visual=True)
        except RuntimeError:
            rejected_trained_target = True
        report.check("迁移旧文本权重不能残留已训练视觉参数", rejected_trained_target, "")
        with torch.no_grad():
            for slot in model.visual_inlayer_bank.modules():
                if hasattr(slot, "up"):
                    slot.up.weight.zero_()
                    slot.up.bias.zero_()
        model._load_weights(text_only, allow_missing_visual=True)
        report.check("完整缺失的旧文本权重可迁移到零残差视觉分支", True, "")

        rejected_declared = False
        damaged = {"format_version": 2, "config": model.architecture_spec(),
                   "model_state_dict": text_only}
        try:
            model.load_checkpoint_blob(damaged, allow_missing_visual=True)
        except RuntimeError:
            rejected_declared = True
        report.check("声明含视觉分支的 checkpoint 缺权重时拒绝加载", rejected_declared, "")
        model.load_checkpoint(path, map_location=device)
    finally:
        if os.path.exists(path):
            os.remove(path)

    duplicate = False
    try:
        install_visual_inlayer_adapters(model.clip, model.visual_inlayer_bank)
    except RuntimeError:
        duplicate = True
    report.check("重复安装报错", duplicate, "")

    bad_bank = OrganAdapterBank([ORGAN], layers=[99], positions=POSITIONS, d_model=768, bottleneck=64)
    out_of_range = False
    try:
        install_visual_inlayer_adapters(model.clip, bad_bank)
    except ValueError:
        out_of_range = True
    report.check("越界层号报错", out_of_range, "")

    uninstall_visual_inlayer_adapters(model.visual_inlayer_installation)
    model.visual_inlayer_hits.clear()
    with torch.no_grad():
        trunk.forward_features(image)
    report.check("卸载后不再命中", model.visual_inlayer_hits == {}, str(model.visual_inlayer_hits))
    installation = install_visual_inlayer_adapters(model.clip, model.visual_inlayer_bank)
    model.visual_inlayer_installation = installation
    model.visual_inlayer_hits = installation.hits
    model.set_train_stage("visual", organ=ORGAN)
    with torch.no_grad():
        trunk.forward_features(image)
    report.check("重新安装后恢复命中", set(model.visual_inlayer_hits) == expected, str(dict(model.visual_inlayer_hits)))

    # 联合阶段的 logits 形状
    model.set_train_stage("joint", organ=ORGAN)
    out = model(image, anchors)
    report.check(
        "logits 形状",
        tuple(out["patch_logits"].shape) == (1, 2, 14, 14) and tuple(out["cls_logits"].shape) == (1, 2),
        f"patch={tuple(out['patch_logits'].shape)} cls={tuple(out['cls_logits'].shape)}",
    )

    print("\n===== visual smoke " + ("通过" if report.ok else "失败") + " =====")
    return 0 if report.ok else 1


def _layers(model, image, indices):
    with torch.no_grad():
        _, inter = model.clip.visual.trunk.forward_intermediates(
            image, indices=list(indices), norm=False, output_fmt="NLC"
        )
    return [tensor.detach().clone() for tensor in inter]


if __name__ == "__main__":
    raise SystemExit(main())

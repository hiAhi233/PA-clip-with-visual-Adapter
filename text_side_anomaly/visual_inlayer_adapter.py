"""视觉塔层内瓶颈适配器的安装与卸载。

文本侧的包装挂在 HF BertLayer 上，返回类型和插入点都不同，不能复用。
这里只处理 timm `VisionTransformer.Block`：适配器接在 attention 残差相加之后，
以及 FFN 残差相加之后，输出继续进入后续 block。

支持的 block 前向（timm 1.0.28+，本机核验为 1.0.30）是：

    x = x + drop_path1(ls1(attn(norm1(x), attn_mask=..., is_causal=...)))
    x = A_attn(x)
    x = x + drop_path2(ls2(mlp(norm2(x))))
    y = A_ffn(x)

`A` 复用 `InLayerBottleneckAdapter`（参数名仍是 `lambda_t`，报告里称为视觉 λ_v）。
bank 只注册在主模型的 `visual_inlayer_bank` 上，forward 闭包引用它，不挂到 block 里。
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Sequence

import torch.nn as nn

from .inlayer_adapter import OrganAdapterBank

_FLAG = "_paclip_visual_inlayer_token"
_REQUIRED_ATTRS = ("norm1", "attn", "ls1", "drop_path1", "norm2", "mlp", "ls2", "drop_path2")
_SOURCE_SNIPPETS = (
    "self.drop_path1(self.ls1(self.attn(self.norm1(x)",
    "self.drop_path2(self.ls2(self.mlp(self.norm2(x))))",
)
_ALLOWED_POSITIONS = ("attn", "ffn")


class VisualInlayerInstallation:
    """一次安装的原始 forward 记录，供卸载。"""

    def __init__(self) -> None:
        self.records: List[dict] = []
        self.hits: Dict[str, int] = {}


def _visual_trunk(clip):
    visual = getattr(clip, "visual", None)
    trunk = getattr(visual, "trunk", None)
    if trunk is None or not hasattr(trunk, "blocks"):
        raise AttributeError(
            "找不到视觉塔 block 列表（期望 clip.visual.trunk.blocks）："
            "该基座的视觉塔可能不是 timm VisionTransformer。"
        )
    return trunk


def _assert_supported_block(block: nn.Module, index: int) -> None:
    """安装前确认 block 仍是未改过的 timm pre-norm Block。"""
    try:
        from timm.models.vision_transformer import Block
    except ImportError as exc:  # pragma: no cover
        raise ImportError("视觉层内适配器需要 timm") from exc

    if not isinstance(block, Block):
        raise TypeError(
            f"视觉 block[{index}] 类型是 {type(block).__name__}，不是 timm VisionTransformer.Block"
        )
    missing = [name for name in _REQUIRED_ATTRS if not hasattr(block, name)]
    if missing:
        raise TypeError(f"视觉 block[{index}] 缺少 {missing}，无法按残差后插入适配器")

    params = inspect.signature(type(block).forward).parameters
    for name in ("x", "attn_mask", "is_causal"):
        if name not in params:
            raise TypeError(
                f"视觉 block[{index}].forward 没有参数 {name}，与已支持的 timm Block 接口不一致"
            )
    source = inspect.getsource(type(block).forward)
    if any(snippet not in source for snippet in _SOURCE_SNIPPETS):
        raise TypeError(
            f"视觉 block[{index}] 的 forward 与已支持的 pre-norm 残差公式不一致，拒绝安装"
        )


def visual_encoder_blocks(clip) -> nn.ModuleList:
    """返回 `clip.visual.trunk.blocks`，并校验每一层的类型与前向接口。"""
    trunk = _visual_trunk(clip)
    blocks = trunk.blocks
    if len(blocks) < 1:
        raise RuntimeError("视觉塔 blocks 为空")
    for index, block in enumerate(blocks):
        _assert_supported_block(block, index)
    return blocks


def visual_hidden_dim(clip) -> int:
    """从视觉主干读取隐藏维度，并要求每个 block 的 LayerNorm 与之一致。"""
    trunk = _visual_trunk(clip)
    if not hasattr(trunk, "embed_dim"):
        raise AttributeError("视觉主干没有 embed_dim，无法确定适配器宽度")
    dim = int(trunk.embed_dim)
    blocks = visual_encoder_blocks(clip)
    for index, block in enumerate(blocks):
        width = int(block.norm1.normalized_shape[0])
        if width != dim:
            raise ValueError(
                f"视觉 block[{index}] 宽度 {width} 与 trunk.embed_dim {dim} 不一致"
            )
    return dim


def _validate_bank(bank: OrganAdapterBank, n_blocks: int, hidden_dim: int) -> List[int]:
    organs = bank.organ_names()
    if not organs or any(not str(name) for name in organs):
        raise ValueError(f"视觉适配器器官名无效：{organs}")
    if len(set(organs)) != len(organs):
        raise ValueError(f"视觉适配器器官名重复：{organs}")

    layers = [int(x) for x in bank.layers]
    if not layers:
        raise ValueError("视觉适配器至少要插入一个 block")
    if len(set(layers)) != len(layers):
        raise ValueError(f"视觉插入层重复：{layers}")
    out_of_range = [idx for idx in layers if idx < 0 or idx >= n_blocks]
    if out_of_range:
        raise ValueError(f"视觉插入层 {out_of_range} 超出 block 范围 [0, {n_blocks - 1}]")

    positions = tuple(bank.positions)
    if not positions or any(pos not in _ALLOWED_POSITIONS for pos in positions):
        raise ValueError(f"视觉插入位置只能是 attn/ffn，收到 {positions}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"视觉插入位置重复：{positions}")

    adapter = bank.adapters[organs[0]][0]
    if int(adapter.d_model) != int(hidden_dim):
        raise ValueError(
            f"视觉适配器宽度 {adapter.d_model} 与视觉主干隐藏维度 {hidden_dim} 不一致"
        )
    return layers


def make_visual_forward(block, bank: OrganAdapterBank, layer_idx: int, hits: Dict[str, int]):
    """绑定当前 block 与层号，避免循环变量晚绑定。"""
    original_forward = block.forward

    def apply_adapter(hidden, position: str):
        if position not in bank.positions:
            return hidden
        adapter = bank.get(layer_idx, position)
        if adapter is None:
            return hidden
        key = f"L{layer_idx}.{position}"
        hits[key] = hits.get(key, 0) + 1
        return adapter(hidden)

    def forward(x, attn_mask=None, is_causal: bool = False):
        # active 为 None 时走原始 block，这是真正的旁路；把模型属性置空关不掉闭包。
        if bank.active is None:
            return original_forward(x, attn_mask=attn_mask, is_causal=is_causal)

        x = x + block.drop_path1(
            block.ls1(block.attn(block.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
        )
        x = apply_adapter(x, "attn")
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return apply_adapter(x, "ffn")

    return forward


def install_visual_inlayer_adapters(clip, bank: OrganAdapterBank) -> VisualInlayerInstallation:
    """校验通过后，把 bank 挂到指定视觉 block 的 forward 上。

    必须在冻结 `clip` 原始参数之后调用。返回的 hits 与安装记录共用同一批对象，
    前向会就地更新 hits。任一校验失败时不会改动 block。
    """
    if not isinstance(bank, OrganAdapterBank):
        raise TypeError("视觉层内安装需要 OrganAdapterBank")
    hidden_dim = visual_hidden_dim(clip)
    blocks = visual_encoder_blocks(clip)
    layers = _validate_bank(bank, len(blocks), hidden_dim)

    for layer_idx in layers:
        block = blocks[layer_idx]
        _assert_supported_block(block, layer_idx)
        if getattr(block, _FLAG, None) is not None:
            raise RuntimeError(f"视觉 block[{layer_idx}] 已经安装过层内适配器，拒绝重复叠加")
    if getattr(_visual_trunk(clip), "grad_checkpointing", False):
        raise RuntimeError("视觉主干开了梯度检查点。第一版只保证 eager 前向，请先关掉再安装适配器")

    installation = VisualInlayerInstallation()
    token = object()
    try:
        for layer_idx in layers:
            block = blocks[layer_idx]
            installation.records.append({
                "block": block,
                "forward": block.forward,
                "token": token,
                "layer_idx": layer_idx,
            })
            block.forward = make_visual_forward(block, bank, layer_idx, installation.hits)
            setattr(block, _FLAG, token)
    except Exception:
        uninstall_visual_inlayer_adapters(installation)
        raise
    return installation


def uninstall_visual_inlayer_adapters(installation: Optional[VisualInlayerInstallation]) -> None:
    """把安装过的 block.forward 恢复成原来的 bound method。"""
    if installation is None:
        return
    for record in installation.records:
        block = record["block"]
        block.forward = record["forward"]
        if getattr(block, _FLAG, None) is record["token"]:
            delattr(block, _FLAG)
    installation.records.clear()


def visual_adapter_param_report(bank: OrganAdapterBank) -> str:
    parts = []
    for name, mods in bank.adapters.items():
        count = sum(p.numel() for p in mods.parameters())
        mark = " ← active" if name == bank.active else ""
        parts.append(f"{name}:{count}{mark}")
    return (
        f"[visual-inlayer] 器官={bank.organ_names()} 槽位={bank.n_slot} "
        f"λ 参数名=lambda_t（视觉 λ_v） | " + "  ".join(parts)
    )

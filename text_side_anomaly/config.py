"""超参数配置（brain MRI 异常检测）。

文档要求：
- bottleneck 维度从 {64, 128, 256} 做消融；
- 残差比例 λ_t 建议初始 0.05~0.10；
- 只训练 W_down、W_up（文本编码器主干冻结）；
- 文本侧损失 margin m 为允许的最大相似度。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ---- 主干模型 ----
    # open_clip 的 hf-hub: 前缀（配合 HF_ENDPOINT=https://hf-mirror.com 使用镜像）
    model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    image_size: int = 224
    text_hidden: int = 512           # BiomedCLIP 投影维度（Adapter 作用于此）
    visual_hidden: int = 768         # ViT 隐层维度
    max_text_len: int = 256

    # ---- 残差文本 Adapter ----
    bottleneck: int = 128            # 消融 {64, 128, 256}
    lambda_t: float = 0.05           # 初始 0.05~0.10
    lambda_t_learnable: bool = True
    adapter_dropout: float = 0.0

    # ---- 三层提示词 ----
    levels: List[str] = field(default_factory=lambda: ["1", "2", "3"])
    fusion_learnable: bool = True    # 三层融合权重是否可学习

    # ---- 多尺度 patch 特征 ----
    # ViT block 索引（0 起，共 12 层）。取中间层与末层融合，缓解末层语义化过重、
    # 跨 patch 区分度低（实测跨 patch 标准差仅 0.025）。空列表 = 关闭，退回原版单层行为。
    ms_layers: List[int] = field(default_factory=lambda: [5, 8, 11])

    # ---- 对齐与损失 ----
    margin: float = 0.3             # 文本侧损失：允许的最大相似度 m
    temperature: float = 0.07       # 对齐 logits 温度
    w_text: float = 1.0
    w_global: float = 1.0
    w_local: float = 1.0

    # ---- 训练 ----
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda"

    # ---- 视觉侧层内适配器（默认关闭，旧实验仍走原来的输出端/文本层内路径）----
    visual_inlayer_enabled: bool = False
    visual_inlayer_layers: List[int] = field(default_factory=lambda: [5, 8, 11])
    visual_inlayer_positions: List[str] = field(default_factory=lambda: ["attn", "ffn"])
    visual_inlayer_bottleneck: int = 64
    visual_inlayer_lambda: float = 0.1
    # text：训练文本侧，视觉层内适配器旁路；visual：冻结文本、训练视觉；joint：双侧一起训练
    train_stage: str = "text"
    # 由入口显式打开。打开后输出端 ResidualTextAdapter 的 λ 置零并冻结，加载 checkpoint 后会再执行一次
    freeze_output_text_adapter: bool = False

    # ---- 数据 ----
    data_root: str = "data"              # 含 normal/ 与 abnormal/ 子目录
    mask_root: Optional[str] = None      # 病灶掩码目录（训练局部对齐用）
    data_format: str = "volume"          # "volume"(3D .nii.gz) 或 "slice"(2D png)
    modality: Optional[int] = None       # 4D 体积时选用的通道索引；None 为单模态
    slice_strategy: str = "lesion"       # 3D 切片策略：lesion / middle / random
    normalize: bool = True               # 是否做百分位强度归一化

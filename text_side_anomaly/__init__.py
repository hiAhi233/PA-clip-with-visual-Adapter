"""文本侧改进方法：基于 BiomedCLIP 的 brain MRI 异常检测。

将一对 normal/abnormal 提示扩展为三层多属性文本锚点，在冻结的 BiomedCLIP
文本编码器上加入残差 Adapter，配合全局（CLS→图像级判断）+ 局部（patch→病灶
定位）对齐与 margin 文本侧损失。
"""

from .config import Config
from .prompts import ThreeLevelPrompts, DEFAULT_BRAIN_MRI_PROMPTS, build_text_anchors
from .text_adapter import ResidualTextAdapter
from .inlayer_adapter import (
    InLayerBottleneckAdapter,
    OrganAdapterBank,
    install_inlayer_adapters,
)
from .visual_inlayer_adapter import (
    install_visual_inlayer_adapters,
    uninstall_visual_inlayer_adapters,
    visual_encoder_blocks,
)
from .model import TextSideAnomalyModel, load_trained_model
from .losses import (
    text_separation_loss,
    global_alignment_loss,
    local_alignment_loss,
    TotalLoss,
)
from .dataset import SliceAnomalyDataset, VolumeAnomalyDataset
from .metrics import image_metrics, pixel_metrics

__all__ = [
    "Config",
    "ThreeLevelPrompts",
    "DEFAULT_BRAIN_MRI_PROMPTS",
    "build_text_anchors",
    "ResidualTextAdapter",
    "InLayerBottleneckAdapter",
    "OrganAdapterBank",
    "install_inlayer_adapters",
    "install_visual_inlayer_adapters",
    "uninstall_visual_inlayer_adapters",
    "visual_encoder_blocks",
    "TextSideAnomalyModel",
    "load_trained_model",
    "text_separation_loss",
    "global_alignment_loss",
    "local_alignment_loss",
    "TotalLoss",
    "SliceAnomalyDataset",
    "VolumeAnomalyDataset",
    "image_metrics",
    "pixel_metrics",
]

"""主模型：冻结 BiomedCLIP(open_clip) + 残差文本 Adapter + 全局/局部对齐。

BiomedCLIP 用 open_clip 加载（transformers 的 CLIPModel 无法识别该权重）。

对应文档：
    1) 三层 normal/abnormal 提示 → 多属性文本锚点；
    2) 冻结文本编码器上加残差 Adapter（W_down/W_up + λ_t，作用于 512 维投影空间）；
    3) 全局对齐（CLS → 图像级判断）+ 局部对齐（patch → 病灶定位）；
    4) 文本侧 margin 损失（见 losses.py）。

`inlayer` 是文本塔内部适配器，`visual_inlayer` 是视觉塔内部适配器。两者默认都不启用。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .inlayer_adapter import (
    DEFAULT_LAYERS,
    DEFAULT_POSITIONS,
    OrganAdapterBank,
    adapter_param_report,
    install_inlayer_adapters,
)
from .prompts import ThreeLevelPrompts
from .text_adapter import ResidualTextAdapter
from .visual_inlayer_adapter import (
    VisualInlayerInstallation,
    install_visual_inlayer_adapters,
    uninstall_visual_inlayer_adapters,
    visual_adapter_param_report,
    visual_hidden_dim,
)

CHECKPOINT_FORMAT = 2
_TRAIN_STAGES = ("text", "visual", "joint")


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_v2_checkpoint(blob) -> bool:
    return isinstance(blob, dict) and blob.get("format_version") == CHECKPOINT_FORMAT and "model_state_dict" in blob


def _is_state_dict(blob) -> bool:
    return isinstance(blob, dict) and bool(blob) and all(torch.is_tensor(v) for v in blob.values())


class TextSideAnomalyModel(nn.Module):
    """冻结 BiomedCLIP，训练文本/视觉层内适配器或输出端文本 Adapter。

    `inlayer` 给定时，在文本塔内部插入按器官分的瓶颈适配器。
    `visual_inlayer` 给定时，在视觉 block 的 attention / FFN 残差之后插入另一套适配器。
    两者都是 None 时，与原先只训练输出端 Adapter 的行为一致。
    """

    def __init__(self, cfg: Config, inlayer: Optional[dict] = None, visual_inlayer: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg

        # ---- 冻结的 BiomedCLIP ----
        import open_clip

        loaded = open_clip.create_model_from_pretrained(cfg.model_name)
        self.clip = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # ---- 文本层内适配器（可选）---- #
        # 必须在冻结循环之后安装，否则新模块会被一起冻住。
        self.inlayer_bank: Optional[OrganAdapterBank] = None
        self.inlayer_hits: Dict[str, int] = {}
        self._text_inlayer_init: Optional[dict] = None
        if inlayer:
            bank = OrganAdapterBank(
                organs=inlayer["organs"],
                layers=inlayer.get("layers", DEFAULT_LAYERS),
                positions=inlayer.get("positions", DEFAULT_POSITIONS),
                d_model=inlayer.get("d_model", 768),      # 塔内 hidden，不是 512
                bottleneck=inlayer.get("bottleneck", 64),
                lambda_t=inlayer.get("lambda_t", 0.1),
            )
            bank.set_active(inlayer.get("organ") or bank.organ_names()[0])
            self.inlayer_hits = install_inlayer_adapters(self.clip, bank)
            self.inlayer_bank = bank
            self._text_inlayer_init = {
                "organs": list(bank.organ_names()),
                "layers": list(bank.layers),
                "positions": list(bank.positions),
                "bottleneck": int(bank.adapters[bank.organ_names()[0]][0].bottleneck),
                "lambda_t": float(inlayer.get("lambda_t", 0.1)),
                "d_model": int(bank.adapters[bank.organ_names()[0]][0].d_model),
            }
            print(adapter_param_report(bank))
            n_mount = len(bank.layers) * len(bank.positions)
            print(f"[inlayer] 挂载 {n_mount} 个点：层{bank.layers} × 位置{list(bank.positions)}"
                  f"（hits 计数器在前向后填充）")

        # ---- 视觉层内适配器（可选，与文本 bank 不共享权重）---- #
        self.visual_inlayer_bank: Optional[OrganAdapterBank] = None
        self.visual_inlayer_hits: Dict[str, int] = {}
        self.visual_inlayer_installation: Optional[VisualInlayerInstallation] = None
        self._visual_inlayer_init: Optional[dict] = None
        if visual_inlayer:
            measured = visual_hidden_dim(self.clip)
            if int(cfg.visual_hidden) != measured:
                raise ValueError(
                    f"cfg.visual_hidden={cfg.visual_hidden} 与视觉主干实际宽度 {measured} 不一致"
                )
            requested = visual_inlayer.get("d_model")
            if requested is not None and int(requested) != measured:
                raise ValueError(
                    f"visual_inlayer d_model={requested} 与视觉主干实际宽度 {measured} 不一致"
                )
            organs = [str(name) for name in visual_inlayer["organs"]]
            if not organs or any(not name for name in organs) or len(set(organs)) != len(organs):
                raise ValueError(f"视觉适配器器官名无效或重复：{organs}")
            vbank = OrganAdapterBank(
                organs=organs,
                layers=visual_inlayer.get("layers", cfg.visual_inlayer_layers),
                positions=visual_inlayer.get("positions", cfg.visual_inlayer_positions),
                d_model=measured,
                bottleneck=visual_inlayer.get("bottleneck", cfg.visual_inlayer_bottleneck),
                lambda_t=visual_inlayer.get("lambda_t", cfg.visual_inlayer_lambda),
            )
            vbank.set_active(visual_inlayer.get("organ") or vbank.organ_names()[0])
            installation = install_visual_inlayer_adapters(self.clip, vbank)
            self.visual_inlayer_bank = vbank
            self.visual_inlayer_hits = installation.hits
            self.visual_inlayer_installation = installation
            self._visual_inlayer_init = {
                "organs": list(vbank.organ_names()),
                "layers": list(vbank.layers),
                "positions": list(vbank.positions),
                "bottleneck": int(vbank.adapters[vbank.organ_names()[0]][0].bottleneck),
                "lambda_t": float(visual_inlayer.get("lambda_t", cfg.visual_inlayer_lambda)),
                "d_model": measured,
            }
            print(visual_adapter_param_report(vbank))
            n_mount = len(vbank.layers) * len(vbank.positions)
            print(f"[visual-inlayer] 挂载 {n_mount} 个点：层{vbank.layers} × 位置{list(vbank.positions)}")

        if self.inlayer_bank is not None and self.visual_inlayer_bank is not None:
            if self.inlayer_bank.organ_names() != self.visual_inlayer_bank.organ_names():
                raise ValueError(
                    "文本和视觉适配器库的器官列表必须一致："
                    f"{self.inlayer_bank.organ_names()} vs {self.visual_inlayer_bank.organ_names()}"
                )

        text_organs = self.inlayer_bank.organ_names() if self.inlayer_bank is not None else None
        visual_organs = self.visual_inlayer_bank.organ_names() if self.visual_inlayer_bank is not None else None
        organs = text_organs or visual_organs or []
        self._organ = None
        if inlayer and inlayer.get("organ"):
            self._organ = str(inlayer["organ"])
        elif visual_inlayer and visual_inlayer.get("organ"):
            self._organ = str(visual_inlayer["organ"])
        elif organs:
            self._organ = organs[0]

        # open_clip tokenizer：list[str] -> (B, L) 长整型张量
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

        # BiomedCLIP 投影维度为 512
        self.projection_dim = getattr(self.clip, "embed_dim", cfg.text_hidden)

        # ---- 残差文本 Adapter（只训练 W_down/W_up + λ_t，作用于 512 维投影空间）----
        self.text_adapter = ResidualTextAdapter(
            d_model=self.projection_dim,
            bottleneck=cfg.bottleneck,
            lambda_t=cfg.lambda_t,
            lambda_t_learnable=cfg.lambda_t_learnable,
            dropout=cfg.adapter_dropout,
        )

        # ---- 三层融合权重（可学习，softmax 归一）----
        n_levels = len(cfg.levels)
        if cfg.fusion_learnable:
            self.fusion_weights = nn.Parameter(torch.zeros(n_levels))
        else:
            self.register_buffer("fusion_weights", torch.zeros(n_levels))

        self.checkpoint_meta: dict = {}
        self.freeze_output_text_adapter = bool(getattr(cfg, "freeze_output_text_adapter", False))
        self.training_stage = "text"
        self._arch_spec = self._make_arch_spec()
        self.set_train_stage(getattr(cfg, "train_stage", "text"), organ=self._organ)

    # ------------------------------------------------------------------ #
    # 阶段 / 器官
    # ------------------------------------------------------------------ #
    @property
    def active_organ(self) -> Optional[str]:
        if self.inlayer_bank is not None and self.inlayer_bank.active is not None:
            return self.inlayer_bank.active
        if self.visual_inlayer_bank is not None and self.visual_inlayer_bank.active is not None:
            return self.visual_inlayer_bank.active
        return self._organ

    def set_active_organ(self, organ: Optional[str]) -> None:
        """切换器官后按当前阶段重新冻结。视觉阶段不会把文本适配器关掉。"""
        if organ is not None:
            self._organ = organ
        self.set_train_stage(self.training_stage, organ=self._organ)

    def set_train_stage(self, stage: str, organ: Optional[str] = None) -> None:
        """配置哪一侧参与前向、哪一侧可训练。不创建优化器。"""
        if stage not in _TRAIN_STAGES:
            raise ValueError(f"未知训练阶段 {stage!r}，只能是 {list(_TRAIN_STAGES)}")
        if stage in {"visual", "joint"} and self.visual_inlayer_bank is None:
            raise RuntimeError(f"{stage} 阶段需要已经安装的视觉层内适配器")
        if stage == "joint" and self.inlayer_bank is None:
            raise RuntimeError("joint 阶段需要已经安装的文本层内适配器")

        organ = self._organ if organ is None else organ
        if stage == "visual" and self.inlayer_bank is not None and not organ:
            raise RuntimeError("视觉阶段必须保留文本层内适配器的激活器官，不能把它关掉")
        if stage in {"visual", "joint"} and self.visual_inlayer_bank is not None and not organ:
            raise RuntimeError("视觉层内适配器需要一个激活器官")

        self.training_stage = stage
        self.cfg.train_stage = stage
        if organ is not None:
            self._organ = organ

        for p in self.clip.parameters():
            p.requires_grad = False

        if self.inlayer_bank is not None:
            self.inlayer_bank.set_active(organ)
            if stage == "visual":
                for p in self.inlayer_bank.parameters():
                    p.requires_grad = False

        if self.visual_inlayer_bank is not None:
            if stage == "text":
                self.visual_inlayer_bank.set_active(None)
            else:
                self.visual_inlayer_bank.set_active(organ)

        self._apply_output_adapter_policy()
        self._apply_fusion_policy()

    def _apply_output_adapter_policy(self) -> None:
        if self.freeze_output_text_adapter:
            with torch.no_grad():
                self.text_adapter.lambda_t.zero_()
            for p in self.text_adapter.parameters():
                p.requires_grad = False
            return
        train_it = self.training_stage in {"text", "joint"}
        for p in self.text_adapter.parameters():
            p.requires_grad = train_it

    def _apply_fusion_policy(self) -> None:
        if not isinstance(self.fusion_weights, nn.Parameter):
            return
        self.fusion_weights.requires_grad = bool(
            self.cfg.fusion_learnable and self.training_stage in {"text", "joint"}
        )

    def train(self, mode: bool = True):
        """主干保持 eval。适配器的 train/eval 跟阶段走，不改 requires_grad。"""
        super().train(mode)
        self._apply_adapter_modes(mode)
        return self

    def lock_backbone_eval(self) -> None:
        """兼容旧训练循环：主干锁 eval，适配器模式跟当前阶段。"""
        self._apply_adapter_modes(self.training)

    def _apply_adapter_modes(self, mode: bool) -> None:
        self.clip.eval()
        if self.inlayer_bank is not None:
            self.inlayer_bank.train(bool(mode) and self.training_stage in {"text", "joint"})
        if self.visual_inlayer_bank is not None:
            self.visual_inlayer_bank.train(bool(mode) and self.training_stage in {"visual", "joint"})

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def trainable_report(self) -> str:
        buckets: Dict[str, List[int]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            top = name.split(".")[0]
            count, total = buckets.get(top, [0, 0])
            buckets[top] = [count + 1, total + param.numel()]
        parts = [f"{key}:{value[1]}({value[0]}张量)" for key, value in buckets.items()]
        total = sum(value[1] for value in buckets.values())
        return f"[stage={self.training_stage}] 可训练 {total} = " + (", ".join(parts) if parts else "无")

    def assert_stage_parameters(self) -> None:
        """当前 requires_grad 集合必须和阶段约定一致。"""
        trainable = set(self.trainable_parameter_names())
        if any(name.startswith("clip.") for name in trainable):
            raise AssertionError("主干参数不可训练")

        def _slot_names(prefix: str) -> List[str]:
            return [name for name, _ in self.named_parameters() if name.startswith(prefix)]

        stage = self.training_stage
        if stage == "text":
            if any(name.startswith("visual_inlayer_bank.") for name in trainable):
                raise AssertionError("文本阶段不能训练视觉层内适配器")
            if self.visual_inlayer_bank is not None and self.visual_inlayer_bank.active is not None:
                raise AssertionError("文本阶段视觉层内适配器必须旁路")
            if self.inlayer_bank is not None:
                organ = self.inlayer_bank.active
                if organ is None:
                    raise AssertionError("文本阶段文本层内适配器必须参与前向")
                names = _slot_names(f"inlayer_bank.adapters.{organ}.")
                if not names or any(name not in trainable for name in names):
                    raise AssertionError("文本阶段没有包含全部激活文本槽位")
                self._assert_inactive_organs_frozen("inlayer_bank", organ, trainable)
        elif stage == "visual":
            bank = self.visual_inlayer_bank
            if bank is None or bank.active is None:
                raise AssertionError("视觉阶段视觉层内适配器必须激活")
            names = _slot_names(f"visual_inlayer_bank.adapters.{bank.active}.")
            if not names or any(name not in trainable for name in names):
                raise AssertionError("视觉阶段没有包含全部激活视觉槽位")
            banned = [
                name for name in trainable
                if name.startswith("inlayer_bank.")
                or name.startswith("text_adapter.")
                or name == "fusion_weights"
            ]
            if banned:
                raise AssertionError(f"视觉阶段包含应冻结的参数：{banned[:6]}")
            self._assert_inactive_organs_frozen("visual_inlayer_bank", bank.active, trainable)
            if self.inlayer_bank is not None and self.inlayer_bank.active is None:
                raise AssertionError("视觉阶段不能关闭已经启用的文本层内适配器")
        elif stage == "joint":
            if self.inlayer_bank is None or self.inlayer_bank.active is None:
                raise AssertionError("联合阶段文本层内适配器必须激活")
            if self.visual_inlayer_bank is None or self.visual_inlayer_bank.active is None:
                raise AssertionError("联合阶段视觉层内适配器必须激活")
            for bank_name, bank in (
                ("inlayer_bank", self.inlayer_bank),
                ("visual_inlayer_bank", self.visual_inlayer_bank),
            ):
                names = _slot_names(f"{bank_name}.adapters.{bank.active}.")
                if not names or any(name not in trainable for name in names):
                    raise AssertionError(f"联合阶段没有包含全部激活 {bank_name} 槽位")
                self._assert_inactive_organs_frozen(bank_name, bank.active, trainable)

        if self.freeze_output_text_adapter and any(name.startswith("text_adapter.") for name in trainable):
            raise AssertionError("输出端文本适配器应按模式配置冻结")
        if (
            not self.freeze_output_text_adapter
            and stage in {"text", "joint"}
            and not any(name.startswith("text_adapter.") for name in trainable)
        ):
            raise AssertionError("当前阶段应训练输出端文本适配器")
        fusion_on = "fusion_weights" in trainable
        fusion_expected = isinstance(self.fusion_weights, nn.Parameter) and stage in {"text", "joint"} and self.cfg.fusion_learnable
        if fusion_on != fusion_expected:
            raise AssertionError(f"融合权重可训练性应为 {fusion_expected}，实际 {fusion_on}")

    def _assert_inactive_organs_frozen(self, bank_name: str, active: str, trainable: set) -> None:
        bank = getattr(self, bank_name)
        for organ in bank.organ_names():
            if organ == active:
                continue
            leaked = [name for name in trainable if name.startswith(f"{bank_name}.adapters.{organ}.")]
            if leaked:
                raise AssertionError(f"未激活器官 {organ} 的参数仍可训练")

    def assert_optimizer_matches_stage(self, optimizer) -> None:
        self.assert_stage_parameters()
        opt_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        model_ids = {id(param) for param in self.parameters() if param.requires_grad}
        if opt_ids != model_ids:
            raise AssertionError(
                f"优化器参数数 {len(opt_ids)} 与当前可训练参数数 {len(model_ids)} 不一致，需要按当前阶段重建优化器"
            )

    def build_optimizer(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):
        """在安装、选器官、阶段冻结都完成之后调用。"""
        params = [param for param in self.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(f"{self.training_stage} 阶段没有可训练参数")
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr if lr is None else lr,
            weight_decay=self.cfg.weight_decay if weight_decay is None else weight_decay,
        )
        self.assert_optimizer_matches_stage(optimizer)
        print(self.trainable_report())
        return optimizer

    # ------------------------------------------------------------------ #
    # checkpoint
    # ------------------------------------------------------------------ #
    def _make_arch_spec(self) -> dict:
        text = self._text_inlayer_init or {}
        visual = self._visual_inlayer_init or {}
        organs: Sequence[str] = []
        if self.inlayer_bank is not None:
            organs = self.inlayer_bank.organ_names()
        elif self.visual_inlayer_bank is not None:
            organs = self.visual_inlayer_bank.organ_names()
        return {
            "model_name": self.cfg.model_name,
            "image_size": int(self.cfg.image_size),
            "levels": list(self.cfg.levels),
            "ms_layers": list(self.cfg.ms_layers or []),
            "visual_hidden": int(self.cfg.visual_hidden),
            "text_hidden": int(self.projection_dim),
            "bottleneck": int(self.cfg.bottleneck),
            "lambda_t": float(self.cfg.lambda_t),
            "lambda_t_learnable": bool(self.cfg.lambda_t_learnable),
            "fusion_learnable": bool(self.cfg.fusion_learnable),
            "temperature": float(self.cfg.temperature),
            "margin": float(self.cfg.margin),
            "freeze_output_text_adapter": bool(self.freeze_output_text_adapter),
            "organs": list(organs),
            "text_inlayer_enabled": self.inlayer_bank is not None,
            "text_inlayer_layers": list(text.get("layers", [])),
            "text_inlayer_positions": list(text.get("positions", [])),
            "text_inlayer_bottleneck": text.get("bottleneck"),
            "text_inlayer_lambda": text.get("lambda_t"),
            "text_inlayer_d_model": text.get("d_model"),
            "visual_inlayer_enabled": self.visual_inlayer_bank is not None,
            "visual_inlayer_layers": list(visual.get("layers", [])),
            "visual_inlayer_positions": list(visual.get("positions", [])),
            "visual_inlayer_bottleneck": visual.get("bottleneck"),
            "visual_inlayer_lambda": visual.get("lambda_t"),
            "visual_inlayer_d_model": visual.get("d_model"),
        }

    def architecture_spec(self) -> dict:
        spec = dict(self._arch_spec)
        spec["train_stage"] = self.training_stage
        spec["active_organ"] = self.active_organ
        return spec

    def _check_architecture(self, spec: dict, allow_missing_visual: bool) -> None:
        problems = []

        def same(key, actual, expected):
            if list(actual) != list(expected) if isinstance(expected, (list, tuple)) else actual != expected:
                problems.append(f"{key}: checkpoint={expected!r} 模型={actual!r}")

        same("model_name", self._arch_spec["model_name"], spec.get("model_name"))
        same("levels", self._arch_spec["levels"], list(spec.get("levels", [])))
        same("ms_layers", self._arch_spec["ms_layers"], list(spec.get("ms_layers", [])))
        same(
            "freeze_output_text_adapter",
            self._arch_spec["freeze_output_text_adapter"],
            bool(spec.get("freeze_output_text_adapter")),
        )
        same("text_inlayer_enabled", self._arch_spec["text_inlayer_enabled"], bool(spec.get("text_inlayer_enabled")))
        if spec.get("text_inlayer_enabled"):
            same("text_inlayer_layers", self._arch_spec["text_inlayer_layers"], list(spec.get("text_inlayer_layers", [])))
            same("text_inlayer_positions", self._arch_spec["text_inlayer_positions"], list(spec.get("text_inlayer_positions", [])))
            same("text_inlayer_bottleneck", self._arch_spec["text_inlayer_bottleneck"], spec.get("text_inlayer_bottleneck"))
            same("organs", self._arch_spec["organs"], list(spec.get("organs", [])))
        visual_ckpt = bool(spec.get("visual_inlayer_enabled"))
        visual_model = bool(self._arch_spec["visual_inlayer_enabled"])
        if visual_ckpt != visual_model:
            if not (allow_missing_visual and visual_model and not visual_ckpt):
                problems.append(
                    f"visual_inlayer_enabled: checkpoint={visual_ckpt} 模型={visual_model}"
                )
        elif visual_ckpt:
            same("visual_inlayer_layers", self._arch_spec["visual_inlayer_layers"], list(spec.get("visual_inlayer_layers", [])))
            same("visual_inlayer_positions", self._arch_spec["visual_inlayer_positions"], list(spec.get("visual_inlayer_positions", [])))
            same("visual_inlayer_bottleneck", self._arch_spec["visual_inlayer_bottleneck"], spec.get("visual_inlayer_bottleneck"))
            same("organs", self._arch_spec["organs"], list(spec.get("organs", [])))
        if problems:
            raise RuntimeError("checkpoint 架构与当前模型不一致：\n  " + "\n  ".join(problems))

    def _load_weights(self, state: dict, allow_missing_visual: bool) -> None:
        model_keys = set(self.state_dict().keys())
        src_keys = set(state.keys())
        unexpected = sorted(src_keys - model_keys)
        missing = sorted(model_keys - src_keys)
        visual_missing = [key for key in missing if key.startswith("visual_inlayer_bank.")]
        other_missing = [key for key in missing if not key.startswith("visual_inlayer_bank.")]
        if unexpected:
            raise RuntimeError(f"checkpoint 含当前模型没有的参数：{unexpected[:8]}")
        if other_missing:
            raise RuntimeError(f"checkpoint 缺少参数：{other_missing[:8]}")
        if visual_missing and not allow_missing_visual:
            raise RuntimeError(f"checkpoint 缺少视觉层内适配器参数：{visual_missing[:8]}")
        if visual_missing:
            if any(key.startswith("visual_inlayer_bank.") for key in src_keys):
                raise RuntimeError("checkpoint 的视觉适配器参数不完整；只允许从完全没有视觉分支的旧权重初始化")
            # 迁移旧文本权重时不能偷偷保留目标模型先前训练过的视觉参数。
            for name, value in self.visual_inlayer_bank.state_dict().items():
                if not torch.isfinite(value).all() or (".up." in name and torch.count_nonzero(value)):
                    raise RuntimeError("新增视觉分支必须处于零残差状态；请新建模型后加载旧文本权重")
            print(
                f"[load] 缺少 {len(visual_missing)} 个 visual_inlayer_bank 参数，"
                "这些槽位保持零残差初始化"
            )
        self.load_state_dict(state, strict=False)
        if not missing and not unexpected:
            print("[load] 权重与模型一致")

    def load_checkpoint(
        self,
        path,
        map_location="cpu",
        allow_missing_visual: bool = False,
        resume_stage: Optional[str] = None,
        organ: Optional[str] = None,
    ) -> dict:
        blob = _torch_load(path, map_location=map_location)
        self.load_checkpoint_blob(
            blob,
            allow_missing_visual=allow_missing_visual,
            resume_stage=resume_stage,
            organ=organ,
        )
        return blob if isinstance(blob, dict) else {}

    def load_checkpoint_blob(
        self,
        blob,
        allow_missing_visual: bool = False,
        resume_stage: Optional[str] = None,
        organ: Optional[str] = None,
    ) -> None:
        if _is_v2_checkpoint(blob):
            spec = blob["config"]
            self._check_architecture(spec, allow_missing_visual=allow_missing_visual)
            visual_declared = bool(spec.get("visual_inlayer_enabled"))
            if not visual_declared and any(
                key.startswith("visual_inlayer_bank.") for key in blob["model_state_dict"]
            ):
                raise RuntimeError("checkpoint 的视觉分支元数据与权重不一致")
            self._load_weights(
                blob["model_state_dict"],
                allow_missing_visual=allow_missing_visual and not visual_declared,
            )
            self.checkpoint_meta = {
                key: blob.get(key)
                for key in (
                    "prompt_set", "prompt_digest", "prompt_snapshot", "training_stage",
                    "active_organ", "step", "seed", "split_id", "support_set",
                    "support_config", "support_digest", "experiment_id",
                    "evaluation_protocol", "task", "source_step", "steps_this_run",
                    "init_checkpoint_sha256", "case_disjoint_verified",
                    "global_supervision_enabled", "w_global", "loss",
                )
            }
            stage = resume_stage or blob.get("training_stage") or spec.get("train_stage") or self.training_stage
            chosen = organ or blob.get("active_organ") or spec.get("active_organ") or self._organ
        elif _is_state_dict(blob):
            self._load_weights(blob, allow_missing_visual=allow_missing_visual)
            stage = resume_stage or self.training_stage
            chosen = organ or self._organ
        else:
            raise RuntimeError("无法识别的 checkpoint：既不是格式 2，也不是纯 state_dict")
        self.set_train_stage(stage, organ=chosen)

    def save_checkpoint(self, path, optimizer=None, **extra) -> None:
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "model_state_dict": self.state_dict(),
            "config": self.architecture_spec(),
            "training_stage": self.training_stage,
            "active_organ": self.active_organ,
            "prompt_set": extra.pop("prompt_set", None),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "step": extra.pop("step", None),
            "seed": extra.pop("seed", None),
            "support_set": extra.pop("support_set", None),
            "split_id": extra.pop("split_id", None),
        }
        payload.update(extra)
        torch.save(payload, path)

    def load_compat(self, path, map_location="cpu", verbose: bool = True) -> None:
        """宽松加载。缺文本/视觉层内键会提示；加载后重新套用当前阶段的冻结规则。"""
        blob = _torch_load(path, map_location=map_location)
        sd = blob["model_state_dict"] if _is_v2_checkpoint(blob) else blob
        if not _is_state_dict(sd):
            raise RuntimeError(f"无法从 {path} 读出 state_dict")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if verbose:
            groups = {
                "inlayer_bank": [key for key in missing if key.startswith("inlayer_bank")],
                "visual_inlayer_bank": [key for key in missing if key.startswith("visual_inlayer_bank")],
            }
            others = [key for key in missing if not any(key.startswith(prefix) for prefix in groups)]
            for prefix, keys in groups.items():
                if keys:
                    print(f"[load] 缺 {len(keys)} 个 {prefix} 键（旧 checkpoint 会从该分支的初始值开始）")
            if others:
                print(f"[load] ⚠ 缺 {len(others)} 个其他键：{others[:4]}{' …' if len(others) > 4 else ''}")
            if unexpected:
                print(f"[load] ⚠ 多 {len(unexpected)} 个键：{unexpected[:4]}{' …' if len(unexpected) > 4 else ''}")
            if not missing and not unexpected:
                print("[load] 严格一致")
        stage = self.training_stage
        organ = self._organ
        if _is_v2_checkpoint(blob):
            stage = blob.get("training_stage") or stage
            organ = blob.get("active_organ") or organ
        self.set_train_stage(stage, organ=organ)

    @classmethod
    def from_arch_spec(cls, spec: dict, device: str = "cpu") -> "TextSideAnomalyModel":
        cfg = Config(
            model_name=spec["model_name"],
            image_size=int(spec["image_size"]),
            text_hidden=int(spec["text_hidden"]),
            visual_hidden=int(spec["visual_hidden"]),
            levels=list(spec["levels"]),
            ms_layers=list(spec.get("ms_layers") or []),
            bottleneck=int(spec["bottleneck"]),
            lambda_t=float(spec["lambda_t"]),
            lambda_t_learnable=bool(spec["lambda_t_learnable"]),
            fusion_learnable=bool(spec["fusion_learnable"]),
            temperature=float(spec["temperature"]),
            margin=float(spec["margin"]),
            train_stage=spec.get("train_stage") or "text",
            freeze_output_text_adapter=bool(spec["freeze_output_text_adapter"]),
            visual_inlayer_enabled=bool(spec["visual_inlayer_enabled"]),
            visual_inlayer_layers=list(spec.get("visual_inlayer_layers") or []),
            visual_inlayer_positions=list(spec.get("visual_inlayer_positions") or []),
            visual_inlayer_bottleneck=int(spec.get("visual_inlayer_bottleneck") or 64),
            visual_inlayer_lambda=float(spec.get("visual_inlayer_lambda") or 0.1),
            device=str(device),
        )
        organ = spec.get("active_organ") or (spec.get("organs") or [None])[0]
        inlayer = None
        if spec.get("text_inlayer_enabled"):
            inlayer = {
                "organs": list(spec["organs"]),
                "organ": organ,
                "layers": list(spec["text_inlayer_layers"]),
                "positions": tuple(spec["text_inlayer_positions"]),
                "bottleneck": spec["text_inlayer_bottleneck"],
                "lambda_t": spec["text_inlayer_lambda"],
                "d_model": spec.get("text_inlayer_d_model") or 768,
            }
        visual = None
        if spec.get("visual_inlayer_enabled"):
            visual = {
                "organs": list(spec["organs"]),
                "organ": organ,
                "layers": list(spec["visual_inlayer_layers"]),
                "positions": tuple(spec["visual_inlayer_positions"]),
                "bottleneck": spec["visual_inlayer_bottleneck"],
                "lambda_t": spec["visual_inlayer_lambda"],
                "d_model": spec.get("visual_inlayer_d_model"),
            }
        return cls(cfg, inlayer=inlayer, visual_inlayer=visual).to(torch.device(device))

    # ------------------------------------------------------------------ #
    # 文本侧
    # ------------------------------------------------------------------ #
    def encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, L) → 适配后归一化文本嵌入 (B, 512)。"""
        t = self.clip.encode_text(tokens)          # (B, 512) 投影后，未归一化
        t = self.text_adapter(t)                   # 残差适配
        return F.normalize(t, dim=-1)

    def _encode_list(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.cfg.device)  # (N, L)
        t = self.encode_text_tokens(tokens)
        return t.mean(dim=0) if t.size(0) > 1 else t.squeeze(0)

    def encode_anchors(self, prompts: ThreeLevelPrompts) -> Dict[str, Dict[str, torch.Tensor]]:
        """三层提示词 → 每层 normal/abnormal 文本锚点（归一化）。"""
        anchors: Dict[str, Dict[str, torch.Tensor]] = {}
        for lvl in prompts.levels:
            anchors[lvl] = {
                "normal": F.normalize(self._encode_list(prompts.normal[lvl]), dim=0),
                "abnormal": F.normalize(self._encode_list(prompts.abnormal[lvl]), dim=0),
            }
        return anchors

    # ------------------------------------------------------------------ #
    # 图像侧
    # ------------------------------------------------------------------ #
    def _patch_hw(self, n_tokens: int) -> Tuple[int, int]:
        trunk = self.clip.visual.trunk
        grid = getattr(getattr(trunk, "patch_embed", None), "grid_size", None)
        if grid is not None and len(grid) == 2 and int(grid[0]) * int(grid[1]) == n_tokens:
            return int(grid[0]), int(grid[1])
        side = int(round(n_tokens ** 0.5))
        if side * side != n_tokens:
            raise RuntimeError(
                f"patch 数 {n_tokens} 不能排成网格，且 trunk.patch_embed.grid_size 不可用"
            )
        return side, side

    def encode_image(self, pixel_values) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """只做一次视觉前向，得到 CLS 与 patch 特征。

        层内适配器已经包在 block.forward 上，forward_intermediates / forward_features
        都会经过它们。这里不再在投影之后额外加一层视觉适配。
        """
        trunk = self.clip.visual.trunk
        head = self.clip.visual.head
        ms_layers = getattr(self.cfg, "ms_layers", None)

        if ms_layers:
            final, inter = trunk.forward_intermediates(
                pixel_values, indices=list(ms_layers), norm=True, output_fmt="NLC"
            )
            f_cls = F.normalize(head(final[:, 0]), dim=-1)              # (B, 512)
            proj = torch.stack([head(t) for t in inter], dim=0)         # (L, B, N, 512)
            f_patch = F.normalize(proj.mean(dim=0), dim=-1)             # (B, N, 512)
        else:
            feats = trunk.forward_features(pixel_values)                # (B, 1+N, 768)
            proj = F.normalize(head(feats), dim=-1)                     # (B, 1+N, 512)
            f_cls = proj[:, 0]                                          # (B, 512)
            f_patch = proj[:, 1:]                                       # (B, N, 512)

        h, w = self._patch_hw(int(f_patch.size(1)))
        return f_cls, f_patch, (h, w)

    # ------------------------------------------------------------------ #
    # 前向：三层独立匹配后融合
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values, anchors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        f_cls, f_patch, (h, w) = self.encode_image(pixel_values)
        B = f_cls.size(0)
        levels = list(anchors.keys())
        L = len(levels)

        fw = F.softmax(self.fusion_weights, dim=0)          # (L,)

        cls_logits_list = []
        patch_logits_list = []
        anomaly_maps_list = []
        for lvl in levels:
            t_n = anchors[lvl]["normal"]
            t_a = anchors[lvl]["abnormal"]

            s_n = f_cls @ t_n
            s_a = f_cls @ t_a
            cls_logits_list.append(torch.stack([s_n, s_a], dim=1) / self.cfg.temperature)

            pn = f_patch @ t_n
            pa = f_patch @ t_a
            patch_logits_list.append(torch.stack([pn, pa], dim=1) / self.cfg.temperature)
            anomaly_maps_list.append((pa - pn).reshape(B, h, w))

        cls_logits = sum(fw[i] * cls_logits_list[i] for i in range(L))
        patch_logits = sum(
            fw[i] * patch_logits_list[i].reshape(B, 2, h, w) for i in range(L)
        )
        anomaly_map = sum(fw[i] * anomaly_maps_list[i] for i in range(L))

        return {
            "cls_logits": cls_logits,
            "cls_probs": cls_logits.softmax(dim=1)[:, 1],
            "patch_logits": patch_logits,
            # 逐层 patch logits (B, L, 2, h, w)：给「每层各自也要能定位」那份损失用。
            "patch_logits_per_level": torch.stack(
                [patch_logits_list[i].reshape(B, 2, h, w) for i in range(L)], dim=1
            ),
            "anomaly_map": anomaly_map,
            "anomaly_maps": torch.stack(anomaly_maps_list, dim=1),
            "cls_logits_per_level": torch.stack(cls_logits_list, dim=1),
            "fusion_weights": fw.detach(),
        }


def load_trained_model(
    path,
    device,
    allow_missing_visual: bool = False,
    resume_stage: Optional[str] = None,
    fallback: Optional[TextSideAnomalyModel] = None,
) -> TextSideAnomalyModel:
    """按 checkpoint 里的架构重建模型并严格加载。旧的纯 state_dict 需要调用方先建好模型。"""
    blob = _torch_load(path, map_location="cpu")
    if _is_v2_checkpoint(blob):
        stage = resume_stage or blob.get("training_stage") or blob["config"].get("train_stage")
        spec = dict(blob["config"])
        if stage:
            spec["train_stage"] = stage
        if blob.get("active_organ"):
            spec["active_organ"] = blob["active_organ"]
        model = TextSideAnomalyModel.from_arch_spec(spec, device=str(device))
        model.load_checkpoint_blob(
            blob,
            allow_missing_visual=allow_missing_visual,
            resume_stage=stage,
            organ=spec.get("active_organ"),
        )
        return model
    if fallback is None:
        raise RuntimeError("旧格式 checkpoint 没有架构配置。请用与训练时相同的参数先构建模型，再加载。")
    fallback.load_checkpoint_blob(
        blob,
        allow_missing_visual=allow_missing_visual,
        resume_stage=resume_stage,
    )
    return fallback

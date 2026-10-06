# 文本侧改进方法：基于 BiomedCLIP 的 brain MRI 异常检测

按文档《文本侧改进方法》实现：在冻结的 **BiomedCLIP** 之上，把一对
`normal/abnormal` 提示扩展为**三层多属性文本锚点**，加**残差 Adapter**，
配合**全局 + 局部对齐**完成图像级异常判断与病灶定位。

## 方法（对应文档四部分）

### 1. 三层提示词 → 多属性文本锚点

默认 `brain_mri_sentence` 按胸腺瘤 `sentence` 版的方式组织句式：

| 层级 | 正常 | 异常 |
|---|---|---|
| L1 整体状态 | a normal brain MRI | an abnormal brain MRI |
| L2 结构与病灶 | brain MRI showing preserved brain anatomy and symmetric ventricles | brain MRI showing a focal lesion in the brain parenchyma |
| L3 信号与边界 | preserved gray-white matter contrast with well-defined tissue boundaries on brain MRI | heterogeneous lesion signal with irregular indistinct margins on brain MRI |

L2 使用 `brain MRI showing ...`，L3 将属性放在句首，避免三层都重复
`a normal/abnormal brain MRI with ...`。未指定成像序列，不预设异常为高信号或低信号。
这种句式调整的定位收益仍需真实数据实验验证。

每层 `normal`/`abnormal` 各一组锚点 `t_n^l`、`t_a^l`，三层**各自独立**与图像
特征匹配，输出各自判别结果，最后**融合**（可学习权重）。见 `prompts.py`。

### 2. 残差文本 Adapter

冻结文本编码器，只训练 `W_down`、`W_up` 与残差比例 `λ_t`：

```
r = W_up( ReLU( W_down( LayerNorm(h) ) ) )
t = h + λ_t · r        # λ_t 初始 0.05~0.10；bottleneck ∈ {64,128,256}
```

见 `text_adapter.py`（`up` 零初始化，初始残差为 0，训练更稳）。

### 3. 全局 + 局部对齐

- **全局**：CLS → 三层 normal/abnormal 锚点，融合后 softmax → 图像级异常概率。
- **局部**：patch → 三层 normal/abnormal 锚点，融合后逐像素 softmax → 异常图；
  **病灶内 patch 对齐异常锚点、病灶外 patch 对齐正常锚点**。见 `model.py`。

### 4. 文本侧损失（margin m）

```
L_text = Σ_l  max(0, cos(t_a^l, t_n^l) − m)      # m 为允许的最大相似度
```

另有 `L_global`（CLS 交叉熵）、`L_local`（逐像素交叉熵）。见 `losses.py`。

## 目录结构

```
text_side_anomaly/
├── config.py          # 超参数（bottleneck/λ_t/margin/三层/温度）
├── prompts.py         # 三层 normal/abnormal 提示锚点
├── text_adapter.py    # 残差 Adapter（W_down/W_up + λ_t）
├── model.py           # 冻结 BiomedCLIP + 全局/局部对齐 + 三层融合
├── losses.py          # 文本分离 / 全局 / 局部 损失
├── metrics.py         # 异常检测通用指标（AUROC/AP/F1/Dice/IoU/pixelAUROC）
├── dataset.py         # 2D 切片 / 3D 体积(.nii.gz) 加载 + 掩码
├── train.py           # 训练 / 推理 / 评估
└── README.md
demo_synthetic.py      # 离线自测脚本（合成数据 + 模拟骨干）
```

## 依赖

```bash
pip install torch transformers pillow numpy scikit-learn
```

需要联网（或已缓存）下载 `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`。

## 数据格式

正式实验按患者把切片分到 train / val / test，再从 train 里抽支持集。K 是正常和异常合计的样本数，不是每一类各 K 个。二维数据的单位是切片文件，三维数据的单位是体积文件。

```
brain_mri/
  train/normal/          train/abnormal/
  val/normal/            val/abnormal/
  test/normal/           test/abnormal/
brain_mri_masks/
  train/abnormal/        val/abnormal/        test/abnormal/
```

掩码与异常图像同名，像素值大于 0 视为病灶。给了掩码目录却缺少异常样本的掩码时，训练会在加载模型前停止。没有患者清单时，可以用恰好包含一个捕获组的 `--case-id-regex`，例如 `^(patient[0-9]+)_`。两者都没有时，只能检查文件是否重复，结果会标明 `case_disjoint_verified=false`。

## 训练 + 评估

```powershell
python -m text_side_anomaly.train `
  --data_root brain_mri/train --mask_root brain_mri_masks/train `
  --val-data-root brain_mri/val --val-mask-root brain_mri_masks/val `
  --eval-data-root brain_mri/test --eval-mask-root brain_mri_masks/test `
  --data_format slice --case-id-regex '^(patient[0-9]+)_' `
  --n-support 5 --seed 0 --steps 2700 --batch_size 16 `
  --prompt-set brain_mri_sentence `
  --inlayer --visual-inlayer --train-stage joint `
  --save_dir runs/brain_mri
```

`--n-support 5` 与 `--k 5` 相同，表示合计 5 个支持样本。两类都足够且 K 为奇数时，异常比正常多一个，所以上例是正常 2、异常 3。`--steps` 是这次运行的优化步数，不会为了补满 epoch 而多更新。

验证集只用来选择图像阈值和像素阈值。测试集使用这组阈值计算一次；AUROC / AP 不依赖阈值。没有验证集、也没有手动阈值时，不报告 F1 / Dice。`--oracle-metrics` 会额外给出当前评估集上的最优阈值参考，字段名带 `_oracle`。未提供测试集时，日志和 `metrics.json` 写的是 `evaluation_split=support`，这只是支持集诊断。

同一配置的完成记录会直接复用。`--fresh` 另建目录，不覆盖已有 checkpoint。`--init-ckpt` 加载权重并默认沿用其中的支持集，它不是精确断点恢复；要重抽支持集必须显式加 `--resample-support`。

多个 K 和种子用批量入口，默认只用 `brain_mri_sentence`：

```powershell
python brain_fewshot_run.py `
  --data-root brain_mri/train `
  --val-data-root brain_mri/val `
  --eval-data-root brain_mri/test `
  --ks 5,10,20 --seeds 0,1,2 --groups TVJ `
  --steps 2700 --batch-size 16 `
  --out-root runs/brain_mri `
  --results runs/brain_mri/results.jsonl
```

组别 T、V、TVJ、TVS 分别是文本层内、视觉层内、双侧联合和先文本后视觉。TVS 的总步数至少为 2，两段共用同一份支持集。

三维训练使用 `volume_fixed_slices`：K 是体积数，每个体积另取 `--volume-support-slices` 张固定轴向切片，默认 1 张。异常体积只在病灶切片中抽取，正常体积使用全部轴向切片。评估默认只看中间切片，并记为 `middle_slice_diagnostic`，不会用测试掩码决定切哪一张。

产物在 `runs/brain_mri/<组别>_k<K>_s<seed>_<配置摘要>/`。`metrics.json` 里的 `image` / `pixel` 仍是原来的 CLS 分数和 14×14 图；`postprocess` 才是 224 工作分辨率上的 Top-k 分数、引导细化图和最终掩码指标。这两套数不能写成同一列来比较。

四列结果图在 `heatmaps/figures/`：原图、固定 0～1 色阶的热图、热图加 GT 轮廓、预测掩码加 GT 轮廓。色阶和预测阈值是两件事。没有阈值时第四列会写明掩码不可用，不会画成一张全黑的“正常”。胸腺瘤旧权重默认只用局部 Top-k，不把未经图像级监督的 CLS 分数融进判定。

新训练默认使用 `brain_mri_sentence`，checkpoint 会保存这个版本名。
`--prompt-set brain_mri` 可复现旧提示词。用 `--init-ckpt` 继续训练时，默认沿用
checkpoint 记录的版本；没有版本记录的旧权重必须显式指定 `--prompt-set`。
显式选择与 checkpoint 不同的版本表示更换文本锚点的新实验。

## 推理

```python
import torch
from text_side_anomaly.model import load_trained_model
from text_side_anomaly.prompts import BRAIN_MRI_PROMPT_SETS

model = load_trained_model(checkpoint_path, device)
model.eval()
prompts = BRAIN_MRI_PROMPT_SETS[model.checkpoint_meta["prompt_set"]]
with torch.no_grad():
    enc = model.encode_anchors(prompts)
    out = model(image_tensor.unsqueeze(0).to(device), enc)
# out["cls_probs"] 图像级异常概率；out["anomaly_map"] (14,14) 异常热图
```

已有二维脑 MRI 切片可用对应 checkpoint 出图，输入目录包含 `normal/abnormal`：

```bash
python generate_heatmaps.py --ckpt <checkpoint.pt> \
    --data-root <brain_mri_test_slices> --out-dir heatmaps_brain_mri
```

出图按 checkpoint 选择新旧词表；脑 MRI 权重不会回退使用肺炎提示词或默认肺炎数据目录。

## 离线自测（本仓库可直接跑）

`demo_synthetic.py` 用合成脑组织数据与模拟冻结骨干演示流程，指标不能代表真实数据效果。

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tests/test_visual_inlayer_adapter.py
```

第一项无需下载模型；第二项使用真实 BiomedCLIP（需要缓存或下载权重），检查视觉适配器
的安装、梯度、阶段冻结规则与 checkpoint 往返一致性。

## 胸腺瘤实验记录与续训

胸腺瘤的新训练默认仍为 `sentence`，其他词表保留用于指定的对照实验和历史权重复现。
`thymoma_local.py` 与 `fewshot_run.py` 加载初始化权重时默认继承其 `prompt_set`；
显式指定不一致版本会报错。纯 `state_dict` 没有这项元数据，必须显式声明训练时的词表。
`analyze_thymoma.py --ckpt <checkpoint.pt>` 也按保存的版本分析。

`fewshot_run.py` 的实验标识包含实际步数、batch size、提示词名称与内容摘要、
两侧适配器配置、损失、初始化权重摘要和数据划分清单。checkpoint 与热图目录带配置摘要；
同配置重新训练会另存已有产物。结果记录包含实际 checkpoint 路径和 SHA256，
权重丢失或被替换后不会复用原指标。
旧 JSONL 原样保留，但缺少完整预算字段的记录不再自动当成新实验的已完成结果。

单次训练的 `--steps` 精确控制优化步数，不补满最后一个 epoch。
从头运行 TVS 时文本和视觉阶段各至少 1 步，因此总步数必须至少为 2。
从旧文本 checkpoint 增加视觉分支时，只允许整套视觉权重缺失且目标分支为零残差；
已声明含视觉分支但缺参数的 checkpoint 会报错。

现有 Dice/IoU 沿用历史实验的 test 最优阈值（oracle）口径；图像 F1/ACC 也在当前
评估集选阈值。它们不是预先固定阈值的泛化指标。AUROC/AP 不需要选择阈值。

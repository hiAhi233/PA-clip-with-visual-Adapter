# 脑 MRI 小样本与训练流程完善修改方案

> 编写日期：2026-10-04。目标代码目录：`D:\图神经网络\小样本原型学习\PA-clip新设计2\paclip`。
>
> 本文是待实施的修改方案。按用户最新要求，脑部小样本功能不再继续代写，由用户按本文实现。临时新增但尚未接入训练入口的 `text_side_anomaly/fewshot.py` 已撤掉。
>
> 用户已确认：**K 表示正常与异常合计的支持样本数，不是每类各 K 个。**

## 1. 先明确当前完成到哪里

目前不能说脑部的小样本流程已经完善。胸腺瘤已经能限制训练支持集，脑 MRI 还不能。

| 项目 | 当前代码状态 | 本次待实施内容 |
|---|---|---|
| 胸腺瘤单次训练 | `thymoma_local.py --n-support K` | 保留已有入口 |
| 胸腺瘤批量对照 | `fewshot_run.py --ks 5,10,20 --seeds 0,1,2` | 保留已有入口 |
| 脑 MRI 单次小样本训练 | `text_side_anomaly/train.py` 尚无 K 参数，直接使用完整训练目录 | 增加 K、支持集固定、步数预算和记录 |
| 脑 MRI 多 K、多种子对照 | 尚无对应批量入口 | 新增专用调度脚本 |
| 脑 MRI 独立测试 | 当前 `train.py` 在训练结束后评估 `train_loader` | 分离 train / val / test |
| 脑 MRI 新提示词 | 已新增 `brain_mri_sentence`，作为脑部新训练默认值 | 接入小样本流程时继续使用 |
| 历史脑 MRI 提示词 | `brain_mri` 保留原来的六句话 | 仅用于历史权重复现或显式对照 |
| 文本、视觉层内适配器 | 当前均有实现和命令行开关 | 复用，不能为了增加 K 再另写一套模型 |

### 1.1 此前已经实际修改的部分

以下修复在用户要求“只出方案”之前已经落入代码，应在此基础上继续，不需要重复重写：

1. `fewshot_run.py` 的实验标识加入训练步数、batch size、提示词版本及内容摘要、相关模型配置等。
2. checkpoint、热图目录采用包含配置摘要的名称；同配置重新训练时保留已有产物。结果记录包含权重路径和 SHA256。
3. 缺少关键配置的历史 JSONL 记录保留，但不能再据此自动跳过新实验；权重丢失或内容被替换也不复用旧指标。
4. 胸腺瘤初始化和评估会继承 checkpoint 的提示词；显式给出不一致版本会报错。先确定提示词层级，再建立模型。
5. 脑 MRI checkpoint 出图会选择对应脑部提示词，非肺炎权重必须明确指定数据目录。
6. 权重加载只允许为完整缺少视觉分支的旧文本权重增加零残差视觉分支；不再允许残缺的已训练视觉参数混入初始化参数。
7. `thymoma_local.py --steps` 精确计步，不再补满最后一个 epoch；从头执行 TVS 至少需要两步。
8. 分析脚本加强 checkpoint 架构与提示词检查，减少按文件名猜结构、未知版本回退的情况。

此前验证：`tests/test_prompt_and_resume.py` 的 14 个回归测试通过；真实 BiomedCLIP 的 CUDA 视觉适配器冒烟测试通过，覆盖梯度、阶段冻结、权重加载和重载输出一致性。脑部新旧提示词编码也已检查。

这些结果**不等于脑 MRI 小样本功能已经实现，也不等于真实数据精度已经验证**。

### 1.2 本文的代码位置说明

下文行号以编写时文件为准，实施时优先按函数名定位：

| 文件 | 主要定位点 | 现有职责 |
|---|---|---|
| `text_side_anomaly/train.py` | `build_dataset`，约 38 行 | 建立完整脑部数据集 |
| 同上 | `predict_batch` / `evaluate`，约 48 / 69 行 | 收集预测和计算指标 |
| 同上 | `main`，约 76 行 | 建模、训练、保存、训练集评估 |
| 同上 | 参数定义，约 211 行以后 | 当前没有 K、固定步数和测试集入口 |
| `text_side_anomaly/dataset.py` | `SliceAnomalyDataset`，约 63 行 | 二维切片和掩码 |
| 同上 | `VolumeAnomalyDataset`，约 115 行 | 三维体积及切片选择 |
| `text_side_anomaly/prompts.py` | `BRAIN_MRI_PROMPT_SETS`，约 74 行 | 脑部新旧词表 |
| `text_side_anomaly/model.py` | `set_train_stage` / `build_optimizer` / checkpoint 方法 | 适配器训练阶段与权重保存加载 |
| `text_side_anomaly/metrics.py` | `image_metrics` / `pixel_metrics` | 当前使用评估集最优阈值计算部分指标 |
| `text_side_anomaly/thymoma_dataset.py` | `sample_support`，约 220 行 | 胸腺瘤按病例轮转抽切片的已有参考 |

## 2. 需要补齐的实际缺口

### 2.1 只有小样本参数还不够

当前脑部入口大致是：

```python
train_ds = build_dataset(cfg)
train_loader = DataLoader(train_ds, ...)
for epoch in range(cfg.epochs):
    for batch in train_loader:
        ...
evaluate(model, train_loader, ...)
```

需要同时解决：

- 哪 K 个样本进入训练，而不是只改变日志中的 K。
- 是否只从 train 中抽取；同一患者是否跨 train / val / test。
- 小 K 与全量是否有相同的优化步数预算。
- 续训是否沿用完全相同的支持集。
- 评估是否真正使用独立数据。
- 结果文件能否辨认 K、seed、训练阶段、提示词及完整配置。

### 2.2 胸腺瘤 K 的准确含义

现有胸腺瘤的 K 也是**切片数**，不是严格 K 位患者。`sample_support` 先打乱病例，再按病例轮转选切片，以尽量覆盖不同病例。

因此，脑部需要与它对齐的是“支持样本合计 K 个”的口径。不能把二维脑 MRI 的 K 张切片写成 K 位患者，也不能把三维的 K 个体积与 K 张切片直接作为相同监督量比较。

### 2.3 三维分支有一个已发现、尚未修改的错误

`VolumeAnomalyDataset.__getitem__` 中 mask 为 `(H, W, D)`，但 `_pick_slice` 当前用：

```python
sums = mask_slice.reshape(depth, -1).sum(axis=1)
```

reshape 不会把最后一维的深度轴移到第一维。这会把内存按错误方式分组，得到的索引不一定对应真实病灶所在的 z 切片。

待修改为按空间轴求和，并检查维度：

```python
if mask_slice.ndim != 3 or mask_slice.shape[2] != depth:
    raise ValueError("mask 应为 (H, W, D)，且深度与图像一致")
sums = mask_slice.sum(axis=(0, 1))
lesion_z = np.flatnonzero(sums > 0)
```

这项应独立提交和测试，不能混在支持集抽样中掩盖。

### 2.4 缺失掩码不能静默当作正常组织

当前数据集在找不到某个异常样本掩码时，会返回全零 mask。若已经启用局部损失，这会把“标注缺失”误当成“没有病灶”。

新流程要求：只要给了训练掩码目录，就检查**选中的异常支持样本**是否都有匹配掩码；缺失时在训练前报错并列出文件。正常样本的零掩码仍然合法。

如果完全没有掩码，明确关闭局部损失，此时可以进行图像级训练，但不能将其描述为有监督定位训练。

## 3. 统一 K、类别和随机性的定义

### 3.1 K 的规则

新增单次入口参数 `--n-support K`，可增加 `--k` 作为同一参数的别名。

| 场景 | 行为 |
|---|---|
| 不传 K，且不是恢复一个带支持集的实验 | 使用完整 train 集 |
| `1 <= K <= train 样本数` | 恰好选择 K 个支持样本 |
| `K == train 样本数` | 全量训练，但记录请求 K 和实际数量 |
| `K <= 0` | 报错 |
| `K > train 样本数` | 报错，不能静默变成全量后仍宣称 K-shot |
| train 为空 | 建模前报错 |

样本单位必须进入日志、manifest、checkpoint、结果 JSONL：

- `data_format=slice`：`support_unit="slice"`，K 是二维切片文件数。
- `data_format=volume`：`support_unit="volume"`，K 是三维体积文件数；另记参与训练的切片数。

### 3.2 正常与异常的分配

采用“合计 K 个、尽量均衡”的默认规则，写入 `sampling_policy`，不能由不同脚本各自解释。

建议具体规则：

1. 两类都存在且 K 为偶数时，正常和异常目标数量各为 `K / 2`。
2. 两类都存在且 K 为奇数时，异常比正常多一个。例如 K=5 是正常 2、异常 3。
3. 某一类不足目标数量时，使用它的全部可用样本，再从另一类补足；总数仍必须为 K。
4. 只有一种类别时，在该类别中选择 K 个，并记录真实类别分布；不能伪造另一类。
5. K=1 且有异常可选时默认选一个异常样本，记录 `normal=0, abnormal=1`。这是单类别支持集，不能称作“每类 1-shot”。

示例：

| 可用正常 / 异常 | K | 应选正常 / 异常 |
|---|---:|---|
| 100 / 100 | 5 | 2 / 3 |
| 100 / 100 | 10 | 5 / 5 |
| 1 / 100 | 5 | 1 / 4 |
| 100 / 1 | 5 | 4 / 1 |
| 0 / 100 | 5 | 0 / 5 |
| 2 / 2 | 5 | 报错：总量不足 |

### 3.3 尽量避免 K 张全来自同一个病例

如果提供了患者编号，在每个类别内按患者分组，先为各患者选择一张，再进入下一轮。每位患者内部的候选切片也要按固定随机种子打乱。

不要直接照搬胸腺瘤“文件名前三位就是病例号”的规则。脑 MRI 必须使用自身的数据命名约定或显式 manifest。

如果同一患者在同一 train 集中同时有正常切片和异常切片，可以分别满足类别配额，但要记录唯一患者总数，不能把它计作两个患者。

抽样输出至少包括：

```text
requested_k=5
actual_support=5
support_unit=slice
normal_count=2
abnormal_count=3
unique_cases=<实际患者数或 unknown>
sampling_policy=total_k_balanced_case_round_robin_v1
```

### 3.4 随机种子

至少新增 `--seed`，统一设置 Python、NumPy、PyTorch 和 DataLoader generator。DataLoader 多进程时设置 `worker_init_fn`，同步初始化每个 worker 的 Python 和 NumPy RNG。

数据划分必须独立于训练随机种子：

- `split_seed=0`：固定 train / val / test 患者划分。
- `support_seed=seed`：控制 train 内选择哪 K 个样本。
- `model_seed=seed`：控制适配器初始化。
- `loader_seed=seed`：控制训练样本顺序。

更细的三个种子可以后续开放成参数，但第一次实现就应在运行配置中分别记录，避免误把改变 seed 理解为只改变模型初始化。

相同数据、抽样规则和 seed 应复现同一支持集。不同 seed 不保证必然不同，特别是 K 等于总样本数时，测试中不要写这种错误断言。

## 4. 数据划分：先固定患者划分，再抽 K

### 4.1 推荐的数据组织

```text
brain_mri/
  train/
    normal/
    abnormal/
  val/
    normal/
    abnormal/
  test/
    normal/
    abnormal/

brain_mri_masks/
  train/abnormal/
  val/abnormal/
  test/abnormal/
```

train / val / test 要按患者划分，不能将同一体积导出的不同切片随机分散到不同集合。

现有 `--data_root` 继续表示 train 根目录。新增独立的验证和测试根目录，不把 test 文件与 train 拼在一起再抽 K。

### 4.2 患者标识的两种实现方式

**推荐：JSONL 清单。** 每行记录一张切片或一个体积，路径相对于统一的数据根目录：

```json
{"sample_id":"patient001_slice042","case_id":"patient001","split":"train","image":"train/abnormal/patient001_slice042.png","mask":"masks/train/abnormal/patient001_slice042.png","label":1,"modality":"T2"}
```

`sample_id` 唯一，`case_id` 必须来自数据源，不从任意文件名前缀推测。保存清单版本和内容摘要。

**兼容方式：预划分目录 + `--case-id-regex`。** 正则必须恰好包含一个捕获组。例如文件名为 `patient001_slice042.png` 时：

```text
--case-id-regex '^(patient[0-9]+)_'
```

如果用户既没有患者 manifest，也没有可验证的命名规则，只能检查文件是否相交，不能声称已验证患者隔离。正式对照结果应要求提供患者信息；调试模式可以明确标记 `case_disjoint_verified=false`。

### 4.3 训练前校验

在加载 BiomedCLIP 之前完成以下校验，避免运行很久才发现输入错误：

1. 训练集非空，K 合法，样本标识不重复。
2. train / val / test 的文件标识相互不重叠。
3. 有患者信息时，三个集合的患者集合两两不相交。
4. 支持集是 train 的子集，恰好 K 个，无重复。
5. 给了 mask_root 时，选中的异常支持样本具备 mask；需要像素指标的验证和测试样本也必须有完整异常标注。
6. 标签只允许 0 / 1，缺失类别要记录；AUROC 等需要双类别的指标缺少类别时返回 `null` 或 NaN，并说明原因。
7. 不同模态或 MRI 序列不能在数据预处理中无记录地混用。

严格患者隔离不仅检查最后选中的 K 个样本，还应检查完整的 train / val / test 清单，保证所有 seed 共用合法划分。

## 5. 文件级修改清单

| 文件 | 操作 | 内容 |
|---|---|---|
| `text_side_anomaly/fewshot.py` | 新增 | 抽样、清单、恢复支持集、划分检查；不加载模型 |
| `text_side_anomaly/train.py` | 修改 | 参数、训练与评估数据分离、K 接入、固定步数、结果保存 |
| `text_side_anomaly/dataset.py` | 修改 | 患者/样本标识透传、掩码有效性、三维选片错误与确定性 |
| `text_side_anomaly/metrics.py` | 修改 | 允许传入固定阈值；最优阈值指标显式标为 oracle |
| `text_side_anomaly/model.py` | 尽量复用 | 利用现有 `save_checkpoint(..., **extra)` 保存支持集与协议，不改核心前向逻辑 |
| `text_side_anomaly/prompts.py` | 复用 | 新实验 `brain_mri_sentence`；保留旧版本映射 |
| `brain_fewshot_run.py` | 新增 | 脑部多 K、多 seed、指定训练组调度与汇总 |
| `tests/test_brain_fewshot.py` | 新增 | 抽样、恢复、步数、划分、阈值、三维选片等回归测试 |
| `text_side_anomaly/README.md` | 修改 | 明确 K 的单位、训练命令、数据结构和评估协议 |

不建议直接把脑部逻辑塞进当前胸腺瘤 `fewshot_run.py`：它仍然依赖 `thymoma_slices`、胸腺瘤 NPZ、胸腺瘤提示词和病例命名规则。可以提取与任务无关的配置摘要、JSONL 保存逻辑，但不要为了复用脚本隐式更换医学数据语义。

## 6. 新增 `text_side_anomaly/fewshot.py`

建议拆成五类纯数据函数，便于独立测试：

```python
def sample_support_indices(samples, k, seed, policy):
    """仅从 train 选择索引；samples 含 sample_id、label、可选 case_id。"""

def make_support_manifest(samples, indices, data_root, sampling_config):
    """返回选中样本及抽样元数据，不保存模型。"""

def restore_support_indices(samples, saved_manifest, data_root):
    """按 checkpoint 保存的样本身份恢复，不能重新随机抽取。"""

def validate_splits(train_samples, val_samples, test_samples, require_cases):
    """检查样本/患者隔离。"""

def validate_masks(samples, indices, local_supervision):
    """标注缺失不等同于空病灶。"""
```

### 6.1 抽样算法

```text
输入：已固定的 train 样本、K、seed
  1. 按稳定 sample_id 排序，避免 os.listdir 顺序改变结果。
  2. K=None：返回 train 的全部索引。
  3. 检查 1 <= K <= N。
  4. 按第 3.2 节计算正常和异常配额。
  5. 对每类按 case_id 分组；没有病例信息时以 sample_id 分组，并记录 unknown。
  6. 用局部 RNG(seed) 打乱病例顺序和病例内切片顺序。
  7. 轮转选择各病例的切片，直到满足该类配额。
  8. 检查总数为 K、索引无重复、全部来自 train。
  9. 返回固定索引及计数信息。
```

局部 RNG 不应依赖“模型在抽样前随机初始化了多少个参数”。抽样应尽量在建模前完成。

### 6.2 清单建议格式

```json
{
  "schema_version": 1,
  "task": "brain_mri",
  "support_unit": "slice",
  "requested_k": 5,
  "actual_support": 5,
  "seed": 0,
  "sampling_policy": "total_k_balanced_case_round_robin_v1",
  "class_counts": {"normal": 2, "abnormal": 3},
  "split_id": "<划分清单摘要>",
  "case_disjoint_verified": true,
  "samples": [
    {
      "sample_id": "patient001_slice042",
      "case_id": "patient001",
      "relative_path": "abnormal/patient001_slice042.png",
      "label": 1,
      "mask_relative_path": "abnormal/patient001_slice042.png"
    }
  ]
}
```

这里 `samples` 为格式示例，真实 K=5 必须包含五条。根目录和环境信息单独记录，样本身份优先用相对路径与 sample_id，使数据集整体搬迁后仍可恢复。

恢复时任何支持样本不存在、标签变化或出现重复，都应报错；不能自动寻找“相近名字”替换。

## 7. 修改脑部 `train.py`

### 7.1 新增或调整命令行参数

下表均为**计划实现的接口**，当前并未全部存在：

| 参数 | 建议默认 | 用途 |
|---|---|---|
| `--n-support` / `--k` | `None` | 总支持样本数；恢复时优先继承已有清单 |
| `--seed` | 解析时 `None`，新实验取 0 | 固定支持集、模型与数据顺序；续训继承保存值 |
| `--steps` | `None` | 指定后覆盖 epochs，严格训练指定优化步数 |
| `--num-workers` | 0 | Windows / 小样本调试优先稳定；后续可调大 |
| `--val-data-root` | `None` | 独立验证集 |
| `--val-mask-root` | `None` | 独立验证标注 |
| `--eval-data-root` | `None` | 独立测试集 |
| `--eval-mask-root` | `None` | 独立测试标注 |
| `--split-manifest` | `None` | 患者、样本和划分的显式清单 |
| `--case-id-regex` | `None` | 没有 manifest 时按明确命名规则提取患者编号 |
| `--support-manifest` | `None` | 使用已固定的支持集清单，与随机抽样互斥 |
| `--resample-support` | false | 明确开启新的支持集实验，不能默认续训时重抽 |
| `--eval-only` | false | 按权重元数据重建，只做独立评估 |
| `--ckpt` | `None` | `--eval-only` 时指定权重 |
| `--pixel-threshold` | `None` | 无验证集时可明确给定像素阈值 |
| `--image-threshold` | `None` | 同理，图像级固定阈值 |
| `--oracle-metrics` | false | 显式计算当前评估集最优阈值的附加参考值 |

继续保留 `--inlayer`、`--visual-inlayer`、`--train-stage`、`--init-ckpt`、`--prompt-set`。

新脑部实验默认 `brain_mri_sentence`；不能把字符串 `sentence` 直接当成脑部版本，因为当前 `sentence` 属于胸腺瘤词表。

### 7.2 主流程顺序

建议把当前较长的 `main` 拆成几个清晰步骤：

```text
parse_and_validate_args
  -> 读取 checkpoint 元数据（如果加载已有权重）
  -> 解析提示词、训练阶段、支持集继承规则
  -> 扫描 train / val / test 与患者清单
  -> validate_splits
  -> sample_support 或 restore_support
  -> validate_masks
  -> 保存 support_manifest 和 run_config
  -> 固定模型 RNG，构造/加载模型
  -> 按阶段建立优化器
  -> 仅使用支持集训练，严格计步
  -> 保存完成训练的 checkpoint
  -> 使用 val 选择阈值（如提供）
  -> 在 test 上计算指标
  -> 保存结果、阈值来源和可视化路径
```

将参数解析提取为 `build_parser()`，将核心流程提取为 `run_experiment(args)`，方便批量入口与测试复用，避免通过修改 `sys.argv` 调用训练。

### 7.3 K 必须真正作用于 DataLoader

可使用 PyTorch `Subset`：

```python
full_train_ds = build_dataset(train_cfg)
indices = sample_support_indices(train_samples, k, seed, policy)
support_ds = torch.utils.data.Subset(full_train_ds, indices)
train_loader = DataLoader(support_ds, ...)
```

必须检查后续代码没有又把 `full_train_ds` 传入优化循环。日志同时打印：

```text
train_pool=800, support=5, normal=2, abnormal=3, unit=slice
```

测试时对实际进入模型的 `sample_id` 做计数，而不只检查 `len(indices)`。

### 7.4 精确优化步数

不同 K 的正式对照统一使用显式 `--steps`，例如 2700。2700 只是与现有实验预算衔接的示例，不代表已经验证的脑部最优训练长度。

建议循环结构：

```python
target_steps = args.steps if args.steps is not None else args.epochs * len(train_loader)
if target_steps < 1:
    raise ValueError("训练步数必须为正")

iterator = iter(train_loader)
for step in range(target_steps):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(train_loader)
        batch = next(iterator)

    model.train()
    # 视觉阶段使用固定文本锚点；文本/联合阶段每步重算。
    enc = cached_enc if cached_enc is not None else model.encode_anchors(anchors)
    outputs = model(batch["image"].to(device), enc)
    losses = criterion(enc, outputs, labels, masks)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
```

实际代码补全 labels、masks 和阶段逻辑；这里仅展示预算控制。

记录 `requested_batch_size` 与 `effective_batch_size=min(requested_batch_size, support_count)`，`drop_last=False`。不以复制样本的方式假装 K 个样本形成了更多独立监督。

不要在重建 DataLoader iterator 时反复把 RNG 重置为同一个初始值，否则每轮 shuffle 都可能完全相同。

### 7.5 训练阶段复用现有模型

| 配置 | 层内分支 | 训练阶段 | 文本锚点 |
|---|---|---|---|
| 输出端基线 | 不启用层内 | text | 每步重算 |
| 文本层内 T | 仅文本层内 | text | 每步重算 |
| 视觉层内 V | 视觉层内；文本根据明确基线设置冻结 | visual | 阶段开始编码一次 |
| 双侧联合 TVJ | 文本 + 视觉层内 | joint | 每步重算 |
| 两阶段 TVS | 文本 + 视觉层内 | 先 text，后 visual | 切到 visual 后重新编码一次并固定 |

调用已有 `model.set_train_stage(...)` 和 `model.build_optimizer()`。切换阶段后重建优化器，不能继续使用包含上一阶段参数的旧 optimizer。

从头训练 TVS 时，总步数拆成 `S // 2` 和 `S - S // 2`，要求 S >= 2。两个阶段必须复用同一份支持集清单。

视觉阶段将纯文本分离、多样性损失的权重置零。冻结的 CLIP 主干保持 eval 状态，不能因 `model.train()` 让主干 dropout 改变训练/评估行为。

如果 V 想使用已经训练过的文本层内分支作为初始化，它的文本架构也必须与 checkpoint 相容；不能只传一个文本 checkpoint 给不含文本适配器的 V 模型，然后忽略多余权重。

## 8. 初始化、续训与支持集继承

### 8.1 新训练

没有 `--init-ckpt`：

- 新脑部词表默认 `brain_mri_sentence`。
- seed 默认 0。
- K 为显式值时抽 K 个，否则使用全量 train。
- 保存支持集清单和完整运行配置。

### 8.2 基于已有权重继续训练

若 checkpoint 已保存支持集清单：

1. 默认恢复清单中的同一批样本。
2. 未显式指定 K / seed 时继承；不要把 CLI 默认值误当作用户主动更改。
3. 显式 K、seed 与保存配置冲突时，先报出差异。
4. 只有明确指定 `--resample-support` 才重新抽样，并生成新的实验标识，记录来源 checkpoint。
5. 即使训练目录新增了文件，也不能在普通续训时重新抽样。
6. 当显式切换新旧脑部提示词时，记录为更换锚点的新实验，而不是同一实验的断点恢复。

若旧 checkpoint 没有支持集信息，只能根据本次明确的训练目录与 K 建立新的支持集记录；日志标记“初始化迁移”，不能宣称精确恢复了旧实验。

### 8.3 区分权重初始化与精确断点恢复

当前 `--init-ckpt` 主要是加载模型权重，再建立新的优化器。即使保存了 optimizer 字段，也不能自动称作“从中断处精确恢复”。

建议第一版保持这个语义：`--init-ckpt` 是继续适配/阶段迁移，`--steps` 表示本次额外优化的步数。分别记录 `source_step`、`steps_this_run` 和阶段历史。

若需要精确断点恢复，另设 `--resume`，并同时恢复：

- 模型、优化器、调度器、AMP scaler（若使用）；
- 已完成 step、当前阶段；
- Python / NumPy / CPU / CUDA RNG 状态；
- 支持集清单、当前 epoch 内采样顺序和位置。

阶段切换或换支持集不属于精确断点恢复。第一版可以不实现 `--resume`，但不要在帮助文本中承诺这一能力。

## 9. checkpoint、实验标识与产物

### 9.1 必须保存的元数据

利用已有 `save_checkpoint(..., **extra)` 增加：

```text
task=brain_mri
prompt_set
prompt_snapshot / prompt_digest
support_config
support_set（完整清单）
support_digest
split_manifest_digest
train / val / test 数据版本
case_disjoint_verified
data_format / support_unit
volume_sampling_policy（若使用 volume）
seed / split_seed / support_seed / loader_seed
steps_this_run / completed_step / source_step
requested_batch_size / effective_batch_size
loss_config / optimizer_config
init_checkpoint_sha256
experiment_id
evaluation_protocol
```

提示词名称仍需保存；同时保存实际六句话及摘要，可以防止以后修改同名词表后错误解释旧权重。历史 checkpoint 没有 snapshot 时，按保留的版本映射加载。

### 9.2 实验标识

使用稳定 JSON 序列化后的 SHA256。建议至少纳入：

```text
task、数据版本与划分、sample unit、K、实际支持集摘要、抽样规则、seed、
提示词名称与内容、模型名称、两侧适配器参数、多尺度读取层、训练阶段及步数分配、
总优化步数、batch size、学习率、weight decay、损失权重、初始化权重摘要、
三维选片规则、评估协议与阈值来源
```

同 K、同 seed 但换视觉层数或训练步数，必须生成不同标识。

### 9.3 输出目录

建议按任务和实验分目录：

```text
runs/brain_mri/
  tv_joint_k5_s0_<config_hash>/
    run_config.json
    support_manifest.json
    checkpoint_final.pt
    metrics.json
    thresholds.json
    heatmaps/
```

若同目录已有产物，普通运行应复用已验证的完成记录或另建一个 run 目录。显式 fresh 重跑也应保留旧权重，不能先覆盖 checkpoint 再让旧 JSONL 继续指向它。

保存先写临时文件，再用 `os.replace` 完成同目录原子替换，避免断电后把半个 checkpoint 标记为完成。

模型训练完成后先保存 checkpoint，再做评估和出图。结果状态建议区分 `trained`、`evaluated`、`visualized`；出图失败不能让已经训练完的模型丢失，也不应该强迫重新训练。

## 10. 独立评估与阈值口径

### 10.1 不能再默认把训练集结果当测试结果

当前 `train.py` 在训练结束后调用训练 loader 计算指标。新流程分两种情况：

- 提供独立 test：在完整 test 上评估，不能只测试 K 个样本。
- 未提供 test：可以打印 `support/train diagnostic`，但结果字段必须是 `evaluation_split="support"`，不能写成测试集结果或进入正式对照表。

验证集和测试集使用 `shuffle=False`。文本固定后评估只编码一次锚点，避免每个 batch 重复编码。

### 10.2 阈值选择

当前 `image_metrics` 在被评估数据上找最优 F1 阈值，`pixel_metrics` 在同一份被评估数据上找最优 Dice 阈值。因此历史 F1/ACC/Dice/IoU 是当前集合的最优阈值参考值。

新脑部正式结果建议采用：

1. train：更新模型。
2. val：选择图像阈值与像素阈值，保存数值及来源。
3. test：固定使用上述阈值，只计算一次结果。
4. AUROC/AP 使用连续分数，与分类/分割阈值分开计算。

没有 val 时，可由用户明确传入固定阈值；没有预设阈值就只报告无需阈值的指标。确有需要时，用 `--oracle-metrics` 额外报告 `image_f1_oracle` / `pixel_dice_oracle` 等，不能复用固定阈值字段名。

`anomaly_map` 是正常/异常相似度差，不是概率，因此不能未经定义就统一用 0.5 作为像素阈值。图像 `cls_probs` 和像素相似度差的阈值要分别处理。

### 10.3 建议的指标 API

```python
def choose_image_threshold(scores, labels):
    """仅在 val 上调用。"""

def choose_pixel_threshold(maps, masks):
    """仅在 val 上调用。"""

def image_metrics(scores, labels, threshold=None, include_oracle=False):
    ...

def pixel_metrics(maps, masks, threshold=None, include_oracle=False):
    ...
```

为避免突然改变胸腺瘤历史表，迁移时保留其现有接口或明确标为 legacy；不要只改公共函数默认行为，却不检查所有调用者。

求最优阈值时，可按分数排序并累计 TP/FP，在分数相同的位置成组处理，避免“每个唯一阈值都遍历全部像素”的近似 O(N²) 实现。用含重复分数的数据验证与暴力求解一致。

### 10.4 出图

热图必须使用同一 checkpoint 的提示词、阶段和图像预处理。显示分位数标定与用于指标的阈值分开保存，不能把每张图的 min-max 后热度当成原始预测分数。

如果选择只训练局部损失、关闭全局损失，则出图时不能用未经该任务训练的 CLS 分数把整幅局部热图压暗。当前通用热图函数含这种全局门控，后续应按保存的损失配置决定是否启用；胸腺瘤已有不依赖 CLS 门控的显示路径可供参考。

## 11. 三维 MRI 的处理边界

建议先完成二维切片小样本闭环，再开放三维正式对照。三维不是简单把文件后缀换成 `.nii.gz`。

### 11.1 K 个体积不等于 K 张切片

当前每次 `__getitem__` 都可能从一个体积随机取不同切片。若 K=5 表示五个体积，训练多步可能看过远多于五张二维切片。

支持两种明确命名的协议，不能混用：

- `volume_random_slices`：K 个体积，每次允许随机选片，限制的是可访问体积数。
- `volume_fixed_slices`：K 个体积，每个体积预选 M 张并固定，最多 K×M 张二维支持切片。

建议第一版三维小样本使用后者，增加 `--volume-support-slices M`，默认 M=1，保存每个体积选中的 z 索引。异常训练体积可在训练 mask 标出的病灶切片中抽取；正常体积可按图像前景范围选片。

若某体积有效候选切片少于 M，应明确报错或记录实际数量，并让实验标识体现真实监督量，不能静默重复同一切片充数。

### 11.2 评估不能随机变动，也不能借测试 mask 挑病灶切片

测试选片由图像与固定协议决定，不使用测试掩码决定“挑哪张”。测试 mask 只用于计算真值指标。

正式三维评估建议对预先确定的切片范围逐张预测，再按固定规则汇总体积指标，例如图像级分数取切片最大值；具体聚合规则必须在验证阶段确定并记录。

如果为了快速检查只评估中间切片，应明确输出 `middle_slice_diagnostic`，不能称为完整体积评估。异常体积的中间切片可能没有病灶，因此切片标签和体积标签也不能无条件混用。

### 11.3 三维最少增加的检查

- 图像与 mask 的 H/W/D 一致。
- 深度轴聚合使用 `sum(axis=(0, 1))`。
- 四维多模态数据显式记录所选通道。
- 固定支持切片的 z 索引可随 checkpoint 恢复。
- 测试结果不会因反复调用 `__getitem__` 而随机改变。

## 12. 新增 `brain_fewshot_run.py`

批量脚本只负责调度和汇总，单组训练逻辑复用 `run_experiment(args)`。不要再复制一套训练循环。

建议接口：

```text
--ks 5,10,20
--seeds 0,1,2
--groups T,V,TVJ,TVS
--steps 2700
--batch-size 16
--data-root ...
--val-data-root ...
--eval-data-root ...
--results runs/brain_mri/results.jsonl
--out-root runs/brain_mri
--no-heatmaps
--fresh
```

`--ks` 只枚举支持样本数量，不枚举提示词。一次运行默认只使用 `brain_mri_sentence`，不能顺便把所有提示词对照跑一遍。

实现顺序：

1. 先支持 TVJ 的多 K、多 seed，确认闭环。
2. 再支持 T / V 对照。
3. 最后支持从头 TVS 和基于文本权重的视觉阶段迁移。

所有 K、seed 共用同一 train / val / test 患者划分。汇总按完整实验协议分组，不把不同 prompt、不同步数、不同初始化来源混成一个均值。

输出字段建议：K、unit、seed、两类支持数、患者数、步数、训练阶段、AUROC/AP、固定阈值 F1/Dice/IoU、oracle 附加值、权重路径、支持集清单路径、运行状态。

## 13. 计划实现后的命令示例

**以下命令包含待新增参数，现在不能直接当作已经可用的接口运行。** 目录名为示例，需要换成真实数据路径。

### 13.1 单组：合计 5 张二维切片，双侧联合训练

```powershell
conda activate torch-gpu1
Set-Location 'D:\图神经网络\小样本原型学习\PA-clip新设计2\paclip'

python -m text_side_anomaly.train `
  --data_root 'brain_mri/train' `
  --mask_root 'brain_mri_masks/train' `
  --val-data-root 'brain_mri/val' `
  --val-mask-root 'brain_mri_masks/val' `
  --eval-data-root 'brain_mri/test' `
  --eval-mask-root 'brain_mri_masks/test' `
  --data_format slice `
  --case-id-regex '^(patient[0-9]+)_' `
  --n-support 5 --seed 0 --steps 2700 --batch_size 16 `
  --prompt-set brain_mri_sentence `
  --inlayer --visual-inlayer --train-stage joint `
  --save_dir 'runs/brain_mri'
```

期望日志至少包含：`support=5`、两类数量、患者数、`step=2700/2700`、独立 test 大小、提示词版本和最终权重路径。

### 13.2 保持同一支持集，从文本阶段切换到视觉阶段

```powershell
python -m text_side_anomaly.train `
  --data_root 'brain_mri/train' `
  --mask_root 'brain_mri_masks/train' `
  --data_format slice `
  --init-ckpt 'runs/brain_mri/text_k5_s0/checkpoint_final.pt' `
  --inlayer --visual-inlayer --train-stage visual `
  --steps 1350 `
  --save_dir 'runs/brain_mri'
```

这个例子省略独立评估参数，因此只用于说明阶段迁移；正式评估应添加 val / test 入口。K、seed、提示词和支持样本从来源权重继承；新增视觉分支从零残差开始。

### 13.3 批量：多个 K、三个种子，仍只用一个脑部提示词版本

```powershell
python brain_fewshot_run.py `
  --data-root 'brain_mri/train' `
  --val-data-root 'brain_mri/val' `
  --eval-data-root 'brain_mri/test' `
  --ks 5,10,20 --seeds 0,1,2 --groups TVJ `
  --steps 2700 --batch-size 16 `
  --out-root 'runs/brain_mri' `
  --results 'runs/brain_mri/results.jsonl'
```

批量接口还需透传掩码路径及患者信息参数。此处省略它们以展示调度形式；不提供 mask 时只能按无局部标注的训练/评估协议运行，不能声称完成定位监督实验。

## 14. 必须新增的测试与验收标准

测试分为纯逻辑、入口整合和短 CUDA 检查。不要一开始就跑完整 2700 步训练来发现参数接错。

### 14.1 支持集单元测试

| 测试 | 必须满足 |
|---|---|
| K=5，双类别充足 | 选中五个，不是十个；正常 2、异常 3 |
| K=10 | 两类各 5，总计 10 |
| 某类不足 | 按规则补足，总数严格 K |
| 单类别数据 | 不伪造另一类，记录真实计数 |
| K=0、负数、大于总量 | 报错 |
| 同 seed、同清单 | 支持 sample_id 集合与顺序可复现 |
| 文件枚举顺序改变 | 稳定排序后抽样结果不变 |
| 多切片同病例 | 有病例信息时按病例轮转，记录真实患者数 |
| 数据集根目录搬迁 | 按相对身份恢复同一支持集 |
| 训练目录新增样本 | 普通续训保持原清单 |
| 保存的支持样本缺失/标签改变 | 明确报错，不补抽 |

### 14.2 数据与流程测试

| 测试 | 必须满足 |
|---|---|
| 同一患者出现在 train/test | 训练前报错 |
| 同一文件进入多个 split | 训练前报错 |
| K 个支持样本训练十几步 | 实际进入模型的 sample_id 只来自这 K 个 |
| 训练集 35 个、batch 16、steps 4 | 正好更新四次，不是六次 |
| TVS steps=1 | 报错 |
| TVS 两阶段 | 支持集摘要不变，阶段切换后优化器参数集合正确 |
| 提供 test | 评估 loader 使用 test，不使用 support |
| 不提供 test | 输出明确标记为训练/支持集诊断 |
| 异常支持样本缺 mask | 启用局部监督时拒绝训练 |
| val 选择阈值后修改 test 标签 | 阈值不变，只有测试指标随标签变化 |
| 缺某个类别 | 无法计算的指标明确为空，不崩溃也不伪造满分 |

### 14.3 三维针对性测试

构造 `(H=4, W=5, D=7)` 的全零 mask，只让 `mask[:, :, 3]` 含病灶。病灶选片模式只能返回 z=3；这个测试能直接发现旧 reshape 错误。

继续验证：固定 seed 与保存 z 清单可复现；恢复支持集不会重新选 z；测试选片不依赖测试 mask；数据维度不匹配时给出清晰错误。

### 14.4 真实 BiomedCLIP 的短 CUDA 检查

使用 `torch-gpu1`，先用少量合成二维图像和 mask 运行 2～4 步：

- TVJ：两侧层内适配器能更新，主干无梯度。
- visual：文本参数与锚点不变，视觉参数更新。
- text：视觉分支旁路，文本侧更新。
- 保存再加载，固定输入下输出一致。
- 支持集、K、prompt、阶段元数据与实际运行一致。

这些检查通过后，再用真实数据做一个 K=5、单 seed 的短流程，核对路径、分割标注和实际计数，最后才运行正式多 seed 对照。

## 15. 建议的实施顺序

1. **先固定数据协议。** 确定二维/三维、患者标识来源、train/val/test 目录及 K 单位。
2. **实现纯抽样与清单。** 完成第 6 节及对应单元测试，不加载 CLIP。
3. **接入二维单组训练。** 新增 K、seed、steps、Subset，验证实际只使用 K 个样本。
4. **接入 checkpoint 支持集恢复。** 防止续训时静默换样本、换词表、换预算。
5. **分离独立评估。** 先打通 train/val/test，再加入固定阈值协议。
6. **完善输出与完成状态。** 保存配置、清单、权重、阈值和指标，避免覆盖或错误复用。
7. **增加脑部批量脚本。** 先 TVJ 多 K、多 seed，再扩展其余对照组。
8. **单独处理三维。** 修正深度轴错误，明确体积和切片预算，验证固定选片与评估协议。
9. **更新说明并执行验收。** 运行逻辑测试、短 CUDA 检查，再做真实数据实验。

## 16. 完成标准

只有以下条件全部成立，才可以说“脑部的小样本流程已完善”：

- `--n-support 5` 真正限制训练只访问五个支持样本，并记录正常/异常分配。
- K 的单位明确，切片数、体积数、患者数分别记录。
- 支持集只来自固定的 train，患者划分能被验证。
- 不同 K 的正式对照使用可比较的优化步数预算。
- 续训恢复相同支持集；主动换样本或提示词产生新实验身份。
- 独立测试集不会参与抽样、训练或阈值选择。
- 缺失 mask、缺失权重、不一致版本和非法预算不会被静默接受。
- checkpoint 与结果能够追溯到支持集、提示词、初始化来源和评估协议。
- 二维测试通过；若声称支持三维正式实验，则三维选片与评估测试也必须通过。

本文中的脑部小样本参数、批量脚本和三维补充方案尚待实现。此前已完成的 bug 修复与新提示词可作为实施起点，但不替代上述验收。

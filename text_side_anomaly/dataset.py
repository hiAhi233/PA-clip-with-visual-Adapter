"""brain MRI 数据加载：2D 切片 或 3D 体积（.nii.gz）+ 病灶掩码。

2D 切片结构：
    data_root/
        normal/      *.png|*.jpg   正常切片
        abnormal/    *.png|*.jpg   异常切片
    mask_root/（可选）
        abnormal/    *.png         与 abnormal 同名的病灶掩码（>0 为病灶）

3D 体积结构（如 BraTS）：
    data_root/
        normal/      *.nii.gz
        abnormal/    *.nii.gz
    mask_root/
        abnormal/    *.nii.gz      与 abnormal 同名的病灶掩码

训练时从体积中抽取 2D 轴向切片，缩放到 224×224、灰度转 3 通道、强度归一化。
患者编号只来自显式正则或清单，不从文件名前缀猜测。
"""

import json
import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .roi import roi_from_gray

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_NII_EXT = {".nii", ".nii.gz"}


def _list_files(root: str, exts) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if any(f.lower().endswith(e) for e in exts)
    )


def load_binary_mask(path: str, height: int, width: int, plane=None) -> np.ndarray:
    """原始 0/1 或 0/255 掩码，最近邻对齐到工作尺寸。大于 0 即病灶。"""
    if plane is None:
        array = np.asarray(Image.open(path).convert("L"))
    else:
        array = np.asarray(plane)
    positive = array > 0
    resized = Image.fromarray(positive.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
    return np.asarray(resized) > 127


def _eval_views(gray01: np.ndarray, label: int, mask_path, plane=None, slice_z: int = -1, modality=None):
    height, width = gray01.shape
    has_file = bool(mask_path)
    if has_file or plane is not None:
        gt = load_binary_mask(mask_path, height, width, plane=plane)
        available = True
    elif int(label) == 0:
        gt = np.zeros((height, width), dtype=bool)
        available = True
    else:
        gt = np.zeros((height, width), dtype=bool)
        available = False
    modality_index = -1 if modality in (None, "") else int(modality)
    return {
        "gray01": torch.from_numpy(np.asarray(gray01, dtype=np.float32)),
        "gt_mask_high": torch.from_numpy(gt.astype(np.float32)),
        "gt_available": torch.tensor(bool(available)),
        "slice_z": torch.tensor(int(slice_z), dtype=torch.long),
        "modality_index": torch.tensor(modality_index, dtype=torch.long),
    }


def _gray2clip(x: np.ndarray) -> torch.Tensor:
    """(H, W) → (3, H, W) CLIP 归一化张量。"""
    x = np.stack([x, x, x], axis=0).astype(np.float32)
    x = (x - np.array(_MEAN, dtype=np.float32).reshape(3, 1, 1)) / np.array(
        _STD, dtype=np.float32
    ).reshape(3, 1, 1)
    return torch.from_numpy(x)


def _normalize_slice(s: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(s, low), np.percentile(s, high)
    if hi - lo < 1e-6:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def file_stem(path: str) -> str:
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _posix_rel(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def compile_case_regex(pattern: Optional[str]):
    """患者编号正则必须恰好有一个捕获组。"""
    if not pattern:
        return None
    regex = re.compile(pattern)
    if regex.groups != 1:
        raise ValueError("患者编号正则必须恰好包含一个捕获组")
    return regex


def case_id_from_name(filename: str, regex) -> Optional[str]:
    if regex is None:
        return None
    match = regex.search(os.path.basename(filename))
    if not match:
        raise ValueError(f"文件名不符合患者编号规则：{os.path.basename(filename)}")
    return match.group(1)


def scan_split(
    data_root: Optional[str],
    mask_root: Optional[str] = None,
    data_format: str = "slice",
    case_id_regex: Optional[str] = None,
) -> List[dict]:
    """按 normal/、abnormal/ 扫描一个划分。sample_id 使用相对路径，避免同名文件撞车。"""
    if not data_root or not os.path.isdir(data_root):
        return []
    regex = compile_case_regex(case_id_regex)
    exts = _NII_EXT if data_format == "volume" else _IMG_EXT
    records = []
    mask_by_stem = {}
    if mask_root:
        for path in _list_files(os.path.join(mask_root, "abnormal"), exts):
            mask_by_stem[file_stem(path)] = path
    for label, folder in ((0, "normal"), (1, "abnormal")):
        for path in _list_files(os.path.join(data_root, folder), exts):
            rel = _posix_rel(data_root, path)
            mask_path = mask_by_stem.get(file_stem(path)) if label == 1 else None
            records.append({
                "sample_id": rel,
                "relative_path": rel,
                "label": label,
                "case_id": case_id_from_name(path, regex),
                "mask_relative_path": None if mask_path is None else _posix_rel(mask_root, mask_path),
                "mask_found": True if label == 0 else mask_path is not None,
                "path": path,
                "mask_path": mask_path,
                "modality": None,
                "z_indices": None,
                "z_policy": None,
            })
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{data_root} 中样本标识重复")
    return records


def load_manifest_splits(manifest_path: str) -> dict:
    """JSONL 的图像和掩码路径相对于清单文件所在目录。"""
    root = os.path.dirname(os.path.abspath(manifest_path))
    groups = {"train": [], "val": [], "test": []}
    with open(manifest_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            split = item.get("split")
            if split not in groups:
                raise ValueError(f"清单第 {lineno} 行的 split 必须是 train、val 或 test")
            image = item.get("image") or item.get("relative_path")
            if not image:
                raise ValueError(f"清单第 {lineno} 行缺少图像路径")
            path = image if os.path.isabs(image) else os.path.join(root, image)
            mask = item.get("mask") or item.get("mask_relative_path")
            mask_path = None
            if mask:
                mask_path = mask if os.path.isabs(mask) else os.path.join(root, mask)
            label = int(item["label"])
            if label not in (0, 1):
                raise ValueError(f"清单第 {lineno} 行的标签必须是 0 或 1")
            rel = image.replace("\\", "/")
            groups[split].append({
                "sample_id": item.get("sample_id") or rel,
                "relative_path": rel,
                "label": label,
                "case_id": item.get("case_id"),
                "mask_relative_path": None if not mask else str(mask).replace("\\", "/"),
                "mask_found": True if label == 0 else bool(mask_path and os.path.isfile(mask_path)),
                "path": path,
                "mask_path": mask_path,
                "modality": item.get("modality"),
                "z_indices": item.get("z_indices"),
                "z_policy": item.get("z_policy"),
            })
    return groups


def lesion_z_indices(mask_volume: np.ndarray, depth: int) -> np.ndarray:
    """病灶所在的轴向索引。深度轴是最后一维，不能把 (H, W, D) 摊成 (D, H*W)。"""
    mask_volume = np.asarray(mask_volume)
    if mask_volume.ndim != 3 or int(mask_volume.shape[2]) != int(depth):
        raise ValueError(
            f"掩码形状 {tuple(mask_volume.shape)} 与体积深度 {depth} 不一致，期望 (H, W, {depth})"
        )
    sums = np.asarray(mask_volume).sum(axis=(0, 1))
    return np.flatnonzero(sums > 0)


def pick_axial_slice(depth: int, mask_volume, strategy: str, rng, fixed_z: Optional[int] = None) -> int:
    """评估用的中间切片不读取掩码。fixed_z 只来自已保存的支持集。"""
    if fixed_z is not None:
        z = int(fixed_z)
        if z < 0 or z >= int(depth):
            raise ValueError(f"z={z} 超出深度范围 [0, {depth})")
        return z
    if strategy == "middle":
        return int(depth) // 2
    if strategy == "lesion" and mask_volume is not None:
        indices = lesion_z_indices(mask_volume, depth)
        if len(indices):
            return int(rng.choice(indices))
    return int(rng.integers(0, int(depth)))


def choose_fixed_zs(depth: int, mask_volume, count: int, rng, label: int):
    """每个体积固定抽取 count 张，不重复。异常只在病灶切片中抽，正常使用全部轴向切片。"""
    count = int(count)
    if count < 1:
        raise ValueError("每个体积的支持切片数必须为正")
    if int(label) == 1:
        if mask_volume is None:
            raise ValueError("异常体积没有掩码，不能按病灶切片确定支持切片")
        candidates = lesion_z_indices(mask_volume, depth)
        policy = "lesion_z"
        if len(candidates) < count:
            raise ValueError(
                f"病灶切片只有 {len(candidates)} 张，少于要求的 {count}，不会重复同一张来凑数"
            )
    else:
        if mask_volume is not None:
            lesion_z_indices(mask_volume, depth)
        if int(depth) < count:
            raise ValueError(f"体积深度 {depth} 少于要求的 {count} 张切片，不会重复同一张来凑数")
        candidates = np.arange(int(depth))
        policy = "normal_candidate=all_axial"
    picked = np.atleast_1d(rng.choice(np.asarray(candidates), size=count, replace=False))
    return [int(z) for z in sorted(picked)], policy


def volume_channel(volume: np.ndarray, record: dict, dataset_modality=None) -> int:
    """读取样本自己的通道编号，并检查它落在体积的通道范围内。"""
    recorded = record.get("modality")
    sample_id = record.get("sample_id")
    if volume.ndim < 4:
        if recorded not in (None, ""):
            raise ValueError(f"{sample_id} 是三维体积，没有通道轴，不能指定 modality={recorded!r}")
        return 0
    n_channels = int(volume.shape[3])
    if recorded in (None, ""):
        channel = 0 if dataset_modality in (None, "") else int(dataset_modality)
    else:
        try:
            channel = int(recorded)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample_id} 的 modality 必须是通道编号，收到 {recorded!r}") from exc
        if dataset_modality not in (None, "") and int(dataset_modality) != channel:
            raise ValueError(
                f"{sample_id} 的通道是 {channel}，与数据集通道 {int(dataset_modality)} 不一致"
            )
    if channel < 0 or channel >= n_channels:
        raise ValueError(f"{sample_id} 的通道 {channel} 超出范围 [0, {n_channels})")
    return channel


def validate_modalities(samples: Sequence[dict]) -> None:
    modalities = sorted({
        str(sample.get("modality"))
        for sample in samples
        if sample.get("modality") not in (None, "")
    })
    if len(modalities) > 1:
        raise ValueError("同一次实验混用了多种 MRI 序列或通道：" + ", ".join(modalities))


class SliceAnomalyDataset(Dataset):
    """2D brain MRI 切片数据集。"""

    def __init__(
        self,
        data_root: str,
        mask_root: Optional[str] = None,
        image_size: int = 224,
        grid: int = 14,
        case_id_regex: Optional[str] = None,
        records: Optional[Sequence[dict]] = None,
    ):
        self.image_size = image_size
        self.grid = grid
        self.data_root = data_root
        self.records = list(records) if records is not None else scan_split(
            data_root, mask_root, "slice", case_id_regex
        )
        self.paths: List[Tuple[str, int]] = [(record["path"], int(record["label"])) for record in self.records]
        self.mask_paths: List[Optional[str]] = [record.get("mask_path") for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        path, label = record["path"], int(record["label"])
        img = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        x = np.asarray(img, dtype=np.float32) / 255.0
        image = _gray2clip(x)

        mask = torch.zeros(self.grid, self.grid)
        if record.get("mask_path"):
            m = Image.open(record["mask_path"]).convert("L").resize(
                (self.grid, self.grid), Image.BILINEAR
            )
            mask = torch.from_numpy((np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32))

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "roi": self._roi(x),
            "path": path,
            "sample_id": record["sample_id"],
            "case_id": record.get("case_id") or "",
            "mask_found": bool(record.get("mask_found", False)),
            **_eval_views(x, label, record.get("mask_path"), slice_z=-1, modality=record.get("modality")),
        }

    def _roi(self, x: np.ndarray) -> torch.Tensor:
        """由图像强度给出解剖 ROI（(grid, grid) 布尔），见 roi.py。"""
        return torch.from_numpy(roi_from_gray(x, grid=self.grid))


class VolumeAnomalyDataset(Dataset):
    """3D brain MRI 体积（.nii.gz）数据集，按轴向切片训练。

    Args:
        modality: 4D 体积时选用的通道索引；None 表示 3D 单模态。
        slice_strategy: "lesion"（异常体积优先采含病灶切片）/ "middle" / "random"。
            middle 只按深度取中间张，不读取掩码。
    """

    def __init__(
        self,
        data_root: str,
        mask_root: Optional[str] = None,
        image_size: int = 224,
        grid: int = 14,
        modality: Optional[int] = None,
        slice_strategy: str = "lesion",
        normalize: bool = True,
        case_id_regex: Optional[str] = None,
        records: Optional[Sequence[dict]] = None,
    ):
        self.image_size = image_size
        self.grid = grid
        self.modality = modality
        self.slice_strategy = slice_strategy
        self.normalize = normalize
        self._rng = np.random.default_rng()
        self.records = list(records) if records is not None else scan_split(
            data_root, mask_root, "volume", case_id_regex
        )
        for record in self.records:
            if record.get("modality") is None and modality is not None:
                record["modality"] = modality
        self.paths: List[Tuple[str, int]] = [(record["path"], int(record["label"])) for record in self.records]
        self.mask_paths: List[Optional[str]] = [record.get("mask_path") for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def _pick_slice(self, depth: int, mask_slice) -> int:
        return pick_axial_slice(depth, mask_slice, self.slice_strategy, self._rng)

    def __getitem__(self, idx: int) -> dict:
        import nibabel as nib

        record = self.records[idx]
        vol = np.asarray(nib.load(record["path"]).dataobj, dtype=np.float32)
        depth = vol.shape[2]
        mask_slice = None
        if record.get("mask_path"):
            mv = np.asarray(nib.load(record["mask_path"]).dataobj, dtype=np.float32)
            if mv.shape[:3] != vol.shape[:3]:
                raise ValueError(
                    f"{record['sample_id']} 的掩码形状 {mv.shape} 与图像形状 {vol.shape} 不一致"
                )
            mask_slice = (mv > 0).astype(np.float32)
        z = self._pick_slice(depth, mask_slice)
        return self._tensor_from_volume(record, vol, mask_slice, z)

    def load_indexed(self, idx: int, z: int) -> dict:
        """按已保存的 z 取切片，不再重新选择。"""
        import nibabel as nib

        record = self.records[idx]
        vol = np.asarray(nib.load(record["path"]).dataobj, dtype=np.float32)
        depth = int(vol.shape[2])
        z = int(z)
        if z < 0 or z >= depth:
            raise ValueError(f"{record['sample_id']} 的 z={z} 超出深度 {depth}")
        mask_slice = None
        if record.get("mask_path"):
            mv = np.asarray(nib.load(record["mask_path"]).dataobj, dtype=np.float32)
            if mv.shape[:3] != vol.shape[:3]:
                raise ValueError(
                    f"{record['sample_id']} 的掩码形状 {mv.shape} 与图像形状 {vol.shape} 不一致"
                )
            mask_slice = (mv > 0).astype(np.float32)
        return self._tensor_from_volume(record, vol, mask_slice, z)

    def _tensor_from_volume(self, record, vol, mask_slice, z: int) -> dict:
        if vol.ndim == 4:
            channel = volume_channel(vol, record, self.modality)
            plane = vol[:, :, z, channel]
        else:
            volume_channel(vol, record, self.modality)
            plane = vol[:, :, z]
        plane = _normalize_slice(plane) if self.normalize else plane
        resized = Image.fromarray((plane * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        gray = np.asarray(resized, dtype=np.float32) / 255.0
        image = _gray2clip(gray)
        mask = torch.zeros(self.grid, self.grid)
        if mask_slice is not None:
            resized_mask = Image.fromarray((mask_slice[:, :, z] * 255).astype(np.uint8)).resize(
                (self.grid, self.grid), Image.BILINEAR
            )
            mask = torch.from_numpy(
                (np.asarray(resized_mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            )
        high_plane = None if mask_slice is None else mask_slice[:, :, z]
        return {
            "image": image,
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "mask": mask,
            "roi": torch.from_numpy(roi_from_gray(gray, grid=self.grid)),
            "path": record["path"],
            "sample_id": record["sample_id"],
            "case_id": record.get("case_id") or "",
            "mask_found": bool(record.get("mask_found", False)),
            "z": int(z),
            **_eval_views(
                gray, int(record["label"]), record.get("mask_path"),
                plane=high_plane, slice_z=int(z), modality=record.get("modality"),
            ),
        }


class VolumeFixedSliceDataset(Dataset):
    """K 个体积各自固定的 z，不再在 __getitem__ 里重新抽片。"""

    def __init__(self, base: VolumeAnomalyDataset, pairs: Sequence[Tuple[int, int]]):
        self.base = base
        self.pairs = [(int(index), int(z)) for index, z in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        index, z = self.pairs[idx]
        return self.base.load_indexed(index, z)

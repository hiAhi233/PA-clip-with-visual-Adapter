"""脑 MRI 支持集、固定步数、划分和三维选片回归测试。不加载 CLIP。"""

import functools
import json
import math
import os
import pickle
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

import brain_fewshot_run as batch
from text_side_anomaly.dataset import (
    VolumeAnomalyDataset,
    VolumeFixedSliceDataset,
    choose_fixed_zs,
    compile_case_regex,
    lesion_z_indices,
    pick_axial_slice,
    scan_split,
    volume_channel,
)
from text_side_anomaly.fewshot import (
    class_quotas,
    make_support_manifest,
    restore_support_indices,
    sample_support_indices,
    saved_z_indices,
    stable_digest,
    validate_splits,
    validate_support_held_out,
    validate_support_manifest,
    _support_rows_v1,
)
from text_side_anomaly.metrics import choose_image_threshold, image_metrics, pixel_metrics
from text_side_anomaly.prompts import BRAIN_MRI_PROMPT_SETS
from text_side_anomaly.inference import export_predictions
from text_side_anomaly.train import (
    _data_version, _display_gt, _load_exported_records, _loader, _save_heatmaps, _worker_init,
    make_args, run_experiment,
)


def _records(spec):
    rows = []
    for sample_id, label, case_id in spec:
        rows.append({
            "sample_id": sample_id,
            "label": label,
            "case_id": case_id,
            "relative_path": sample_id,
        })
    return rows


def _ids(samples, indices):
    return [samples[index]["sample_id"] for index in indices]


def _png(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def _fill_split(root, normal, abnormal):
    for name in normal:
        _png(Path(root) / "normal" / name, 20)
    for name in abnormal:
        _png(Path(root) / "abnormal" / name, 220)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_p = nn.Parameter(torch.zeros(4))
        self.vis_p = nn.Parameter(torch.zeros(4))
        self.training_stage = "text"
        self.checkpoint_meta = {}

    def set_train_stage(self, stage, organ=None):
        if stage not in ("text", "visual", "joint"):
            raise ValueError(stage)
        self.training_stage = stage
        self.text_p.requires_grad = stage in ("text", "joint")
        self.vis_p.requires_grad = stage in ("visual", "joint")

    def build_optimizer(self):
        params = [param for param in self.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("没有可训练参数")
        return torch.optim.SGD(params, lr=0.1)

    def encode_anchors(self, prompts):
        vector = self.text_p if self.text_p.requires_grad else self.text_p.detach()
        return {
            level: {"normal": vector, "abnormal": vector + 0.2}
            for level in prompts.levels
        }

    def forward(self, images, encoded):
        del encoded
        batch = images.shape[0]
        bias = self.text_p.sum() + self.vis_p.sum()
        cls = torch.stack([bias, bias + 0.1]).reshape(1, 2).expand(batch, 2)
        patch = cls.reshape(batch, 2, 1, 1).repeat(1, 1, 14, 14)
        return {
            "cls_logits": cls,
            "cls_probs": torch.softmax(cls, dim=-1)[:, 1],
            "patch_logits": patch,
            "anomaly_map": torch.full((batch, 14, 14), 3.0, device=images.device),
        }

    def save_checkpoint(self, path, optimizer=None, **extra):
        payload = {
            "format_version": 2,
            "model_state_dict": self.state_dict(),
            "training_stage": self.training_stage,
            "config": {},
        }
        payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path, map_location=None, allow_missing_visual=False, resume_stage=None, organ=None):
        del allow_missing_visual, organ
        try:
            blob = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location=map_location or "cpu")
        self.load_state_dict(blob["model_state_dict"])
        self.checkpoint_meta = blob
        if resume_stage:
            self.set_train_stage(resume_stage)
        return blob


class SupportTests(unittest.TestCase):
    def test_quota_table(self):
        self.assertEqual(class_quotas(100, 100, 5), (2, 3))
        self.assertEqual(class_quotas(100, 100, 10), (5, 5))
        self.assertEqual(class_quotas(1, 100, 5), (1, 4))
        self.assertEqual(class_quotas(100, 1, 5), (4, 1))
        self.assertEqual(class_quotas(0, 100, 5), (0, 5))
        self.assertEqual(class_quotas(100, 100, 1), (0, 1))
        with self.assertRaises(ValueError):
            class_quotas(2, 2, 5)
        with self.assertRaises(ValueError):
            class_quotas(10, 10, 0)
        with self.assertRaises(ValueError):
            class_quotas(10, 10, -3)

    def test_k5_is_five_samples_not_ten(self):
        samples = _records([(f"n{i}", 0, f"c{i}") for i in range(8)] + [(f"a{i}", 1, f"d{i}") for i in range(8)])
        indices, info = sample_support_indices(samples, 5, 0)
        self.assertEqual(len(indices), 5)
        self.assertEqual(info["normal_count"], 2)
        self.assertEqual(info["abnormal_count"], 3)
        self.assertEqual(info["actual_support"], 5)

    def test_k10_is_balanced(self):
        samples = _records([(f"n{i}", 0, None) for i in range(10)] + [(f"a{i}", 1, None) for i in range(10)])
        _, info = sample_support_indices(samples, 10, 1)
        self.assertEqual((info["normal_count"], info["abnormal_count"]), (5, 5))

    def test_short_class_is_filled_to_exactly_k(self):
        samples = _records([("n0", 0, "n")] + [(f"a{i}", 1, f"a{i}") for i in range(6)])
        indices, info = sample_support_indices(samples, 5, 2)
        self.assertEqual(len(set(indices)), 5)
        self.assertEqual((info["normal_count"], info["abnormal_count"]), (1, 4))

    def test_single_class_does_not_invent_the_other(self):
        samples = _records([(f"a{i}", 1, f"c{i}") for i in range(6)])
        _, info = sample_support_indices(samples, 5, 0)
        self.assertEqual((info["normal_count"], info["abnormal_count"]), (0, 5))

    def test_invalid_k_and_empty_pool(self):
        samples = _records([("a", 1, "c"), ("b", 0, "d")])
        for k in (0, -1, 3):
            with self.assertRaises(ValueError):
                sample_support_indices(samples, k, 0)
        with self.assertRaises(ValueError):
            sample_support_indices([], 1, 0)

    def test_same_seed_is_reproducible_and_ignores_input_order(self):
        samples = _records(
            [(f"n{i:02d}", 0, f"pn{i}") for i in range(6)] + [(f"a{i:02d}", 1, f"pa{i}") for i in range(6)]
        )
        first, _ = sample_support_indices(samples, 5, 4)
        second, _ = sample_support_indices(list(reversed(samples)), 5, 4)
        self.assertEqual(_ids(samples, first), _ids(list(reversed(samples)), second))
        again, _ = sample_support_indices(samples, 5, 4)
        self.assertEqual(_ids(samples, first), _ids(samples, again))

    def test_case_round_robin_counts_a_patient_once(self):
        samples = _records(
            [(f"p1_{i}", 1, "p1") for i in range(4)] + [("p2_0", 1, "p2")]
        )
        indices, info = sample_support_indices(samples, 2, 0)
        chosen = {samples[index]["case_id"] for index in indices}
        self.assertEqual(chosen, {"p1", "p2"})
        self.assertEqual(info["unique_cases"], 2)
        both = _records([("n", 0, "same"), ("a", 1, "same")])
        _, info = sample_support_indices(both, 2, 1)
        self.assertEqual(info["unique_cases"], 1)
        self.assertEqual((info["normal_count"], info["abnormal_count"]), (1, 1))

    def test_restore_survives_relocation_and_rejects_drift(self):
        samples = _records([("abnormal/a.png", 1, "p"), ("normal/b.png", 0, "q"), ("abnormal/c.png", 1, "r")])
        indices, _ = sample_support_indices(samples, 2, 3)
        manifest = make_support_manifest(samples, indices, r"D:\old", {
            "support_unit": "slice", "seed": 3, "requested_k": 2, "unique_cases": 2,
            "sampling_policy": "total_k_balanced_case_round_robin_v1",
            "split_id": "s", "case_disjoint_verified": True, "volume_sampling_policy": None,
        })
        moved = []
        for sample in samples:
            copied = dict(sample)
            copied["path"] = os.path.join(r"E:\moved", sample["relative_path"])
            moved.append(copied)
        extra = dict(moved[0])
        extra["sample_id"] = "abnormal/new.png"
        extra["relative_path"] = "abnormal/new.png"
        restored = restore_support_indices(moved + [extra], manifest, r"E:\moved")
        self.assertEqual(_ids(moved + [extra], restored), _ids(samples, indices))
        missing = [sample for sample in moved if sample["sample_id"] != manifest["samples"][0]["sample_id"]]
        with self.assertRaises(ValueError):
            restore_support_indices(missing, manifest, r"E:\moved")
        changed = [dict(sample) for sample in moved]
        target = manifest["samples"][0]["sample_id"]
        for sample in changed:
            if sample["sample_id"] == target:
                sample["label"] = 1 - int(sample["label"])
        with self.assertRaises(ValueError):
            restore_support_indices(changed, manifest, r"E:\moved")
        duplicated = dict(manifest)
        duplicated["samples"] = list(manifest["samples"]) + [dict(manifest["samples"][0])]
        with self.assertRaises(ValueError):
            restore_support_indices(moved, duplicated, r"E:\moved")

    def test_saved_z_is_not_recomputed(self):
        samples = _records([("vol.nii.gz", 1, "p")])
        samples[0]["z_indices"] = None
        manifest = {"samples": [{
            "sample_id": "vol.nii.gz", "label": 1, "relative_path": "vol.nii.gz", "z_indices": [3],
        }]}
        restore_support_indices(samples, manifest, r"E:\new")
        self.assertEqual(samples[0]["z_indices"], [3])
        self.assertEqual(saved_z_indices(manifest["samples"][0]), [3])
        with self.assertRaises(ValueError):
            saved_z_indices({"sample_id": "vol.nii.gz"})


class MetricsTests(unittest.TestCase):
    def test_fast_threshold_matches_brute_force_with_ties(self):
        scores = np.array([0.2, 0.2, 0.8, 0.8, 0.5, 0.1])
        labels = np.array([0, 1, 1, 0, 1, 0])
        legacy = image_metrics(scores, labels)
        best_f1, best_thr = 0.0, 0.5
        for thr in np.unique(scores):
            f1 = f1_score(labels, scores >= thr, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        acc = accuracy_score(labels, scores >= best_thr)
        self.assertAlmostEqual(legacy["f1"], best_f1)
        self.assertAlmostEqual(legacy["acc"], acc)
        self.assertEqual(set(legacy), {"auroc", "ap", "f1", "acc"})

        maps = np.array([[[0.2, 0.2], [0.9, 0.4]], [[0.9, 0.1], [0.4, 0.4]]], dtype=np.float64)
        masks = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.float64)
        pix = pixel_metrics(maps, masks)
        flat_m, flat_g = maps.reshape(-1), masks.reshape(-1)
        best_dice, best_iou = 0.0, 0.0
        for thr in np.unique(flat_m):
            pred = flat_m >= thr
            inter = (pred & (flat_g > 0)).sum()
            dice = 2 * inter / (pred.sum() + (flat_g > 0).sum() + 1e-6)
            iou = inter / (pred.sum() + (flat_g > 0).sum() - inter + 1e-6)
            if dice > best_dice:
                best_dice, best_iou = dice, iou
        self.assertAlmostEqual(pix["dice"], best_dice)
        self.assertAlmostEqual(pix["iou"], best_iou)

    def test_fixed_threshold_is_not_replaced_by_oracle(self):
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        labels = np.array([0, 1, 0, 1])
        fixed = image_metrics(scores, labels, threshold=0.95, include_oracle=True)
        self.assertLess(fixed["f1"], fixed["f1_oracle"])
        self.assertNotIn("f1_oracle", image_metrics(scores, labels))

    def test_val_threshold_does_not_change_when_test_labels_change(self):
        val_scores = np.array([0.1, 0.2, 0.8, 0.9])
        val_labels = np.array([0, 0, 1, 1])
        threshold = choose_image_threshold(val_scores, val_labels)
        test_scores = np.array([0.05, 0.3, 0.4, 0.95])
        test_labels = np.array([0, 0, 1, 1])
        before = image_metrics(test_scores, test_labels, threshold=threshold)
        flipped = test_labels.copy()
        flipped[2] = 0
        after = image_metrics(test_scores, flipped, threshold=threshold)
        self.assertEqual(before["threshold"], after["threshold"])
        self.assertEqual(before["threshold"], threshold)
        self.assertNotEqual(before["f1"], after["f1"])
        self.assertNotEqual(choose_image_threshold(test_scores, flipped), threshold)

    def test_single_class_is_nan_not_perfect(self):
        result = image_metrics(np.array([0.2, 0.8, 0.1]), np.array([1, 1, 1]), threshold=None)
        self.assertTrue(math.isnan(result["auroc"]))
        self.assertTrue(math.isnan(result["f1"]))
        self.assertNotEqual(result["f1"], 1.0)
        self.assertIn("无法计算", result["reason"])
        legacy = image_metrics(np.array([0.2, 0.8]), np.array([0, 0]))
        self.assertNotIn("reason", legacy)
        self.assertTrue(math.isnan(legacy["auroc"]))


class VolumeTests(unittest.TestCase):
    def test_lesion_axis_and_middle_slice_ignore_mask(self):
        mask = np.zeros((4, 5, 7), dtype=np.float32)
        mask[:, :, 3] = 1
        self.assertEqual(lesion_z_indices(mask, 7).tolist(), [3])
        dataset = VolumeAnomalyDataset.__new__(VolumeAnomalyDataset)
        dataset.slice_strategy = "lesion"
        dataset._rng = np.random.default_rng(0)
        self.assertEqual(dataset._pick_slice(7, mask), 3)

        other = np.zeros((4, 5, 7), dtype=np.float32)
        other[:, :, 1] = 1
        dataset.slice_strategy = "middle"
        self.assertEqual(dataset._pick_slice(7, other), 3)
        rng = np.random.default_rng(0)
        self.assertEqual(pick_axial_slice(7, other, "middle", rng), pick_axial_slice(7, mask, "middle", rng))
        with self.assertRaises(ValueError):
            lesion_z_indices(np.zeros((4, 5, 6)), 7)

    def test_fixed_z_is_reproducible_and_does_not_duplicate(self):
        mask = np.zeros((2, 2, 5), dtype=np.float32)
        mask[:, :, 1] = 1
        mask[:, :, 4] = 1
        first, policy = choose_fixed_zs(5, mask, 1, np.random.RandomState(8), 1)
        second, _ = choose_fixed_zs(5, mask, 1, np.random.RandomState(8), 1)
        self.assertEqual(first, second)
        self.assertEqual(policy, "lesion_z")
        self.assertIn(first[0], (1, 4))
        with self.assertRaises(ValueError):
            choose_fixed_zs(5, mask, 3, np.random.RandomState(0), 1)
        normal, normal_policy = choose_fixed_zs(5, None, 2, np.random.RandomState(1), 0)
        self.assertEqual(normal_policy, "normal_candidate=all_axial")
        self.assertEqual(len(set(normal)), 2)

        class Base:
            def load_indexed(self, index, z):
                return {"index": index, "z": z}

        fixed = VolumeFixedSliceDataset(Base(), [(0, 3)])
        self.assertEqual(fixed[0]["z"], 3)
        self.assertEqual(fixed[0]["z"], 3)

    def test_record_modality_selects_its_channel(self):
        volume = np.zeros((4, 5, 3, 2), dtype=np.float32)
        volume[..., 0] = 0.1
        volume[..., 1] = 0.9
        record = {
            "sample_id": "vol", "label": 1, "modality": 1, "path": "vol.nii.gz",
            "case_id": "", "mask_found": False,
        }
        self.assertEqual(volume_channel(volume, record, None), 1)
        dataset = VolumeAnomalyDataset.__new__(VolumeAnomalyDataset)
        dataset.modality = None
        dataset.normalize = False
        dataset.image_size = 8
        dataset.grid = 2
        from_channel = dataset._tensor_from_volume(record, volume, None, 0)["image"]
        other = dataset._tensor_from_volume(dict(record, modality=0), volume, None, 0)["image"]
        self.assertFalse(torch.allclose(from_channel, other))
        with self.assertRaises(ValueError):
            volume_channel(volume, dict(record, modality=2), None)
        with self.assertRaises(ValueError):
            volume_channel(volume, record, 0)
        with self.assertRaises(ValueError):
            volume_channel(volume[..., 0], record, None)


class FlowTests(unittest.TestCase):
    def _factory(self, made):
        def factory(cfg, inlayer, visual):
            del cfg, inlayer, visual
            made.append(1)
            return TinyModel()
        return factory

    def test_patient_and_file_overlap_stop_before_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["patient001_a.png"], ["patient002_a.png"])
            _fill_split(root / "test", ["patient001_b.png"], ["patient003_a.png"])
            made = []
            args = make_args(
                str(root / "train"), eval_data_root=str(root / "test"),
                case_id_regex=r"^(patient[0-9]+)_", steps=1, save_dir=str(root / "runs"),
                no_heatmaps=True, device="cpu",
            )
            with self.assertRaises(ValueError):
                run_experiment(args, model_factory=self._factory(made))
            self.assertEqual(made, [])

            _png(root / "test" / "abnormal" / "patient002_a.png", 220)
            args = make_args(
                str(root / "train"), eval_data_root=str(root / "test"),
                steps=1, save_dir=str(root / "runs2"), no_heatmaps=True, device="cpu",
            )
            with self.assertRaises(ValueError):
                run_experiment(args, model_factory=self._factory(made))
            self.assertEqual(made, [])

    def test_training_only_sees_the_k_support_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(
                root / "train",
                [f"n{i:02d}.png" for i in range(10)],
                [f"a{i:02d}.png" for i in range(10)],
            )
            made = []
            result = run_experiment(make_args(
                str(root / "train"), n_support=5, seed=0, steps=8, batch_size=4,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            ), model_factory=self._factory(made))
            self.assertEqual(result["train_pool"], 20)
            self.assertEqual(len(result["support_ids"]), 5)
            self.assertEqual((result["normal_count"], result["abnormal_count"]), (2, 3))
            self.assertTrue(result["seen_sample_ids"])
            self.assertTrue(set(result["seen_sample_ids"]).issubset(set(result["support_ids"])))

    def test_exact_step_count_does_not_finish_the_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", [f"n{i:02d}.png" for i in range(20)], [f"a{i:02d}.png" for i in range(15)])
            result = run_experiment(make_args(
                str(root / "train"), steps=4, batch_size=16, save_dir=str(root / "runs"),
                no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            self.assertEqual(result["completed_steps"], 4)
            self.assertEqual(result["train_pool"], 35)

    def test_tvs_rejects_one_step_and_keeps_one_support_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            with self.assertRaises(ValueError) as caught:
                args = make_args(
                    str(root / "train"), train_stage="tvs", steps=1, inlayer=True,
                    visual_inlayer=True, save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                )
                run_experiment(args, model_factory=self._factory(made))
            self.assertIn("2", str(caught.exception))
            self.assertEqual(made, [])
            result = run_experiment(make_args(
                str(root / "train"), train_stage="tvs", steps=4, inlayer=True, visual_inlayer=True,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu", seed=0,
            ), model_factory=self._factory(made))
            self.assertEqual(len(result["stage_optimizer_ids"]), 2)
            self.assertNotEqual(set(result["stage_optimizer_ids"][0]), set(result["stage_optimizer_ids"][1]))
            manifest = json.loads((Path(result["run_dir"]) / "support_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["support_digest"], result["support_digest"])
            self.assertEqual(result["completed_steps"], 4)

    def test_eval_split_is_test_or_explicit_support_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n1.png", "n2.png"], ["a1.png", "a2.png"])
            _fill_split(root / "test", ["tn.png"], ["ta.png"])
            tested = run_experiment(make_args(
                str(root / "train"), eval_data_root=str(root / "test"), n_support=2, seed=0,
                steps=1, save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            self.assertEqual(tested["evaluation_split"], "test")
            self.assertEqual(set(tested["eval_sample_ids"]), {"normal/tn.png", "abnormal/ta.png"})
            self.assertTrue(set(tested["seen_sample_ids"]).isdisjoint(set(tested["eval_sample_ids"])))

            diagnostic = run_experiment(make_args(
                str(root / "train"), n_support=2, seed=0, steps=1, save_dir=str(root / "diag"),
                no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            self.assertEqual(diagnostic["evaluation_split"], "support")
            self.assertEqual(set(diagnostic["eval_sample_ids"]), set(diagnostic["support_ids"]))

    def test_missing_abnormal_mask_stops_before_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            (root / "masks" / "abnormal").mkdir(parents=True)
            made = []
            with self.assertRaises(ValueError):
                run_experiment(make_args(
                    str(root / "train"), mask_root=str(root / "masks"), steps=1,
                    save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                ), model_factory=self._factory(made))
            self.assertEqual(made, [])

    def test_new_files_do_not_change_a_restored_support_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n0.png", "n1.png", "n2.png"], ["a0.png", "a1.png", "a2.png"])
            first = run_experiment(make_args(
                str(root / "train"), n_support=4, seed=0, steps=2, batch_size=4,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            for index in range(12):
                _png(root / "train" / "abnormal" / f"extra{index}.png", 180)
            second = run_experiment(make_args(
                str(root / "train"), init_ckpt=first["checkpoint"], steps=2, batch_size=4,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            self.assertEqual(second["support_ids"], first["support_ids"])
            self.assertTrue(set(second["seen_sample_ids"]).issubset(set(first["support_ids"])))

    def test_completed_run_is_reused_until_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            factory = self._factory(made)
            args = dict(
                n_support=2, seed=1, steps=1, save_dir=str(root / "runs"),
                no_heatmaps=True, device="cpu",
            )
            first = run_experiment(make_args(str(root / "train"), **args), model_factory=factory)
            second = run_experiment(make_args(str(root / "train"), **args), model_factory=factory)
            self.assertTrue(second["reused"])
            self.assertEqual(second["run_dir"], first["run_dir"])
            self.assertEqual(made, [1])
            third = run_experiment(make_args(str(root / "train"), fresh=True, **args), model_factory=factory)
            self.assertNotEqual(third["run_dir"], first["run_dir"])
            self.assertEqual(third["experiment_id"], first["experiment_id"])
            self.assertNotEqual(third["run_id"], first["run_id"])
            self.assertTrue(os.path.isfile(os.path.join(first["run_dir"], "checkpoint_final.pt")))
            self.assertEqual(len(made), 2)

    def test_display_png_does_not_replace_raw_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            result = run_experiment(make_args(
                str(root / "train"), steps=1, save_dir=str(root / "runs"), device="cpu",
            ), model_factory=self._factory([]))
            arrays = list(Path(result["run_dir"], "heatmaps").glob("*.npy"))
            self.assertTrue(arrays)
            self.assertTrue(np.allclose(np.load(arrays[0]), 3.0))
            self.assertTrue(list(Path(result["run_dir"], "heatmaps").glob("*_display.png")))

    def test_display_contour_follows_the_original_mask(self):
        from PIL import ImageDraw

        with tempfile.TemporaryDirectory() as tmp:
            mask = Path(tmp) / "lesion.png"
            canvas = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(canvas).ellipse((16, 18, 46, 50), fill=255)
            canvas.save(mask)
            gt = _display_gt(str(mask), 64, 64)
            ys, xs = np.where(gt > 0)
            box = gt[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
            self.assertLess(float(box.mean()), 0.9)
            self.assertGreater(float(box.mean()), 0.5)

    def test_case_regex_must_have_one_group(self):
        with self.assertRaises(ValueError):
            compile_case_regex("patient")
        with self.assertRaises(ValueError):
            compile_case_regex(r"^(patient[0-9]+)_(slice[0-9]+)")
        with tempfile.TemporaryDirectory() as tmp:
            _png(Path(tmp) / "normal" / "foo.png", 10)
            with self.assertRaises(ValueError):
                scan_split(tmp, case_id_regex=r"^(patient[0-9]+)_")

    def test_split_overlap_message(self):
        train = _records([("a", 1, "p")])
        test = _records([("b", 0, "p")])
        with self.assertRaises(ValueError):
            validate_splits(train, [], test, require_cases=True)
        with self.assertRaises(ValueError):
            validate_splits(train, [], _records([("a", 0, "q")]), require_cases=False)

    def test_same_file_with_different_ids_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "patient001_a.png"
            _png(image, 80)
            alias = Path(tmp) / "nested" / ".." / "patient001_a.png"
            train = [{"sample_id": "left", "label": 1, "case_id": "case-a", "path": str(image)}]
            test = [{"sample_id": "right", "label": 0, "case_id": "case-b", "path": str(alias)}]
            with self.assertRaises(ValueError) as caught:
                validate_splits(train, [], test, require_cases=True)
            self.assertIn("同一文件", str(caught.exception))


    def test_missing_checkpoint_is_not_treated_as_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            args = dict(steps=1, save_dir=str(root / "runs"), no_heatmaps=True, device="cpu", seed=0)
            first = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            config_path = Path(first["run_dir"]) / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["status"] = "trained"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            os.remove(first["checkpoint"])
            with self.assertRaises(FileNotFoundError):
                run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            self.assertEqual(made, [1])

    def test_changed_checkpoint_bytes_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            args = dict(steps=1, save_dir=str(root / "runs"), no_heatmaps=True, device="cpu", seed=0)
            first = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            blob = Path(first["checkpoint"]).read_bytes()
            Path(first["checkpoint"]).write_bytes(b"\x00" + blob[1:])
            with self.assertRaises(RuntimeError):
                run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            self.assertEqual(made, [1])

    def test_eval_checkpoint_identity_and_saved_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            _fill_split(root / "test", ["tn.png"], ["ta.png"])
            first_ckpt = root / "one.pt"
            second_ckpt = root / "two.pt"
            TinyModel().save_checkpoint(first_ckpt, prompt_set="brain_mri_sentence")
            other = TinyModel()
            other.text_p.data.fill_(1)
            other.save_checkpoint(second_ckpt, prompt_set="brain_mri_sentence")
            made = []
            common = dict(
                eval_data_root=str(root / "test"), eval_only=True, steps=1,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            )
            first = run_experiment(
                make_args(str(root / "train"), ckpt=str(first_ckpt), **common),
                model_factory=self._factory(made),
            )
            second = run_experiment(
                make_args(str(root / "train"), ckpt=str(second_ckpt), **common),
                model_factory=self._factory(made),
            )
            self.assertFalse(second["reused"])
            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertEqual(os.path.abspath(first["checkpoint"]), os.path.abspath(first_ckpt))
            self.assertEqual(os.path.abspath(second["checkpoint"]), os.path.abspath(second_ckpt))
            self.assertTrue(os.path.isfile(first["checkpoint"]))
            self.assertFalse(os.path.isfile(os.path.join(first["run_dir"], "checkpoint_final.pt")))
            again = run_experiment(
                make_args(str(root / "train"), ckpt=str(first_ckpt), **common),
                model_factory=self._factory(made),
            )
            self.assertTrue(again["reused"])
            self.assertEqual(len(made), 2)

            spec = {
                "model_name": "saved-spec",
                "text_inlayer_enabled": True,
                "text_inlayer_layers": [8, 9],
                "text_inlayer_positions": ["attn", "ffn"],
                "text_inlayer_bottleneck": 64,
                "visual_inlayer_enabled": True,
                "visual_inlayer_layers": [5, 8, 11],
                "visual_inlayer_positions": ["attn", "ffn"],
                "visual_inlayer_bottleneck": 64,
                "visual_inlayer_lambda": 0.1,
            }
            spec_ckpt = root / "spec.pt"
            TinyModel().save_checkpoint(spec_ckpt, prompt_set="brain_mri_sentence", config=spec)
            seen = []

            def arch_factory(saved, device):
                del device
                seen.append(saved)
                return TinyModel()

            cli_made = []
            result = run_experiment(
                make_args(str(root / "train"), ckpt=str(spec_ckpt), **common),
                model_factory=self._factory(cli_made),
                arch_factory=arch_factory,
            )
            self.assertEqual(cli_made, [])
            self.assertTrue(seen[0]["text_inlayer_enabled"])
            self.assertTrue(seen[0]["visual_inlayer_enabled"])
            self.assertEqual(os.path.abspath(result["checkpoint"]), os.path.abspath(spec_ckpt))

    def test_mask_edit_is_a_different_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            mask = root / "masks" / "abnormal" / "a.png"
            _png(mask, 255)
            made = []
            args = dict(
                mask_root=str(root / "masks"), steps=1, save_dir=str(root / "runs"),
                no_heatmaps=True, device="cpu", seed=0,
            )
            first = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            _png(mask, 20)
            second = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            self.assertFalse(second["reused"])
            self.assertNotEqual(first["experiment_id"], second["experiment_id"])
            self.assertEqual(len(made), 2)

    def test_training_loader_workers_can_start(self):
        function = functools.partial(_worker_init, base_seed=2)
        pickle.loads(pickle.dumps(function))(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root, ["n.png"], ["a.png"])
            dataset = scan_split(str(root))
            from text_side_anomaly.dataset import SliceAnomalyDataset
            loader = _loader(SliceAnomalyDataset(str(root), records=dataset), 1, True, 0, 1)
            batch = next(iter(loader))
            self.assertEqual(batch["image"].shape[0], 1)


    def test_eval_rejects_support_samples_presented_as_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").mkdir()
            _fill_split(root / "test", ["patient001_n.png"], ["patient002_a.png"])
            samples = [
                {"sample_id": "normal/patient001_n.png", "label": 0,
                 "relative_path": "normal/patient001_n.png", "case_id": "patient001"},
                {"sample_id": "abnormal/patient002_a.png", "label": 1,
                 "relative_path": "abnormal/patient002_a.png", "case_id": "patient002"},
            ]
            manifest = make_support_manifest(samples, [0, 1], str(root / "empty"), {
                "support_unit": "slice", "seed": 0, "requested_k": 2, "unique_cases": 2,
                "sampling_policy": "total_k_balanced_case_round_robin_v1",
                "split_id": "s", "case_disjoint_verified": True, "volume_sampling_policy": None,
            })
            ckpt = root / "model.pt"
            TinyModel().save_checkpoint(ckpt, prompt_set="brain_mri_sentence", support_set=manifest)
            made = []
            with self.assertRaises(ValueError) as caught:
                run_experiment(make_args(
                    str(root / "empty"), eval_data_root=str(root / "test"), eval_only=True,
                    ckpt=str(ckpt), case_id_regex=r"^(patient[0-9]+)_",
                    save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                ), model_factory=self._factory(made))
            self.assertIn("支持", str(caught.exception))
            self.assertEqual(made, [])

            other = [
                {"sample_id": "kept-n", "label": 0, "relative_path": "archive/n.png", "case_id": "patient001"},
                {"sample_id": "kept-a", "label": 1, "relative_path": "archive/a.png", "case_id": "patient009"},
            ]
            other_manifest = make_support_manifest(other, [0, 1], str(root / "empty"), {
                "support_unit": "slice", "seed": 0, "requested_k": 2, "unique_cases": 2,
                "sampling_policy": "total_k_balanced_case_round_robin_v1",
                "split_id": "s", "case_disjoint_verified": True, "volume_sampling_policy": None,
            })
            other_ckpt = root / "other.pt"
            TinyModel().save_checkpoint(other_ckpt, prompt_set="brain_mri_sentence", support_set=other_manifest)
            with self.assertRaises(ValueError) as caught:
                run_experiment(make_args(
                    str(root / "empty"), eval_data_root=str(root / "test"), eval_only=True,
                    ckpt=str(other_ckpt), case_id_regex=r"^(patient[0-9]+)_",
                    save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                ), model_factory=self._factory(made))
            self.assertIn("患者", str(caught.exception))

    def test_eval_without_case_ids_is_not_marked_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").mkdir()
            _fill_split(root / "test", ["patient001_n.png"], ["patient002_a.png"])
            samples = [
                {"sample_id": "kept-n", "label": 0, "relative_path": "archive/n.png"},
                {"sample_id": "kept-a", "label": 1, "relative_path": "archive/a.png"},
            ]
            manifest = make_support_manifest(samples, [0, 1], str(root / "empty"), {
                "support_unit": "slice", "seed": 0, "requested_k": 2, "unique_cases": None,
                "sampling_policy": "total_k_balanced_case_round_robin_v1",
                "split_id": "s", "case_disjoint_verified": True, "volume_sampling_policy": None,
            })
            ckpt = root / "model.pt"
            TinyModel().save_checkpoint(ckpt, prompt_set="brain_mri_sentence", support_set=manifest)
            result = run_experiment(make_args(
                str(root / "empty"), eval_data_root=str(root / "test"), eval_only=True,
                ckpt=str(ckpt), case_id_regex=r"^(patient[0-9]+)_",
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            config = json.loads((Path(result["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
            self.assertFalse(config["case_disjoint_verified"])
            held = validate_support_held_out(samples, [], [
                {"sample_id": "normal/patient003_x.png", "case_id": "patient003", "relative_path": "normal/patient003_x.png"}
            ])
            self.assertFalse(held["case_disjoint_verified"])

    def test_restore_keeps_saved_modality_and_rejects_a_conflict(self):
        saved = {
            "sample_id": "abnormal/v.nii.gz", "label": 1, "relative_path": "abnormal/v.nii.gz",
            "case_id": "p", "modality": 1,
        }
        manifest = make_support_manifest([saved], [0], "data", {
            "support_unit": "volume", "seed": 0, "requested_k": 1, "unique_cases": 1,
            "sampling_policy": "total_k_balanced_case_round_robin_v1",
            "split_id": "s", "case_disjoint_verified": False, "volume_sampling_policy": "volume_fixed_slices",
        })
        current = [dict(saved, modality=None, path="moved/v.nii.gz")]
        restore_support_indices(current, manifest, "moved")
        self.assertEqual(current[0]["modality"], 1)
        conflict = [dict(saved, modality=0)]
        with self.assertRaises(ValueError) as caught:
            restore_support_indices(conflict, manifest, "data")
        self.assertIn("通道", str(caught.exception))

    def test_schema1_support_manifest_is_migrated(self):
        samples = [
            {"sample_id": "normal/a.png", "label": 0, "relative_path": "normal/a.png",
             "z_indices": None, "z_policy": None},
            {"sample_id": "abnormal/b.png", "label": 1, "relative_path": "abnormal/b.png",
             "z_indices": None, "z_policy": None},
        ]
        manifest = {
            "schema_version": 1,
            "requested_k": 2,
            "actual_support": 2,
            "class_counts": {"normal": 1, "abnormal": 1},
            "samples": samples,
            "support_digest": stable_digest(_support_rows_v1(samples)),
        }
        validate_support_manifest(manifest)
        self.assertEqual(manifest["schema_version"], 2)
        validate_support_manifest(manifest)
        shortened = dict(manifest)
        shortened["schema_version"] = 1
        shortened["samples"] = samples[:1]
        shortened["support_digest"] = stable_digest(_support_rows_v1(samples))
        with self.assertRaises(ValueError):
            validate_support_manifest(shortened)

    def test_modality_changes_the_experiment(self):
        base = {"sample_id": "v", "label": 1, "relative_path": "abnormal/v.nii.gz", "modality": 0}
        other = dict(base, modality=1)
        self.assertNotEqual(_data_version([base]), _data_version([other]))
        config = {
            "support_unit": "volume", "seed": 0, "requested_k": 1, "unique_cases": 1,
            "sampling_policy": "total_k_balanced_case_round_robin_v1",
            "split_id": "s", "case_disjoint_verified": False, "volume_sampling_policy": "volume_fixed_slices",
        }
        first = make_support_manifest([base], [0], "data", config)
        second = make_support_manifest([other], [0], "data", config)
        self.assertEqual(first["samples"][0]["modality"], 0)
        self.assertNotEqual(first["support_digest"], second["support_digest"])

    def test_shortened_support_manifest_is_rejected(self):
        samples = [
            {"sample_id": f"s{i}", "label": i % 2, "relative_path": f"x/{i}.png", "case_id": f"c{i}"}
            for i in range(5)
        ]
        manifest = make_support_manifest(samples, list(range(5)), "data", {
            "support_unit": "slice", "seed": 0, "requested_k": 5, "unique_cases": 5,
            "sampling_policy": "total_k_balanced_case_round_robin_v1",
            "split_id": "s", "case_disjoint_verified": False, "volume_sampling_policy": None,
        })
        manifest["samples"] = manifest["samples"][:1]
        with self.assertRaises(ValueError):
            validate_support_manifest(manifest)
        with self.assertRaises(ValueError):
            restore_support_indices(samples, manifest, "data")

    def test_eval_only_does_not_need_the_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").mkdir()
            _fill_split(root / "test", ["tn.png"], ["ta.png"])
            samples = [
                {"sample_id": f"s{i}", "label": i % 2, "relative_path": f"missing/{i}.png", "case_id": f"c{i}"}
                for i in range(5)
            ]
            manifest = make_support_manifest(samples, list(range(5)), str(root / "empty"), {
                "support_unit": "slice", "seed": 0, "requested_k": 5, "unique_cases": 5,
                "sampling_policy": "total_k_balanced_case_round_robin_v1",
                "split_id": "s", "case_disjoint_verified": False, "volume_sampling_policy": None,
            })
            ckpt = root / "model.pt"
            TinyModel().save_checkpoint(ckpt, prompt_set="brain_mri_sentence", support_set=manifest)
            result = run_experiment(make_args(
                str(root / "empty"), eval_data_root=str(root / "test"), eval_only=True,
                ckpt=str(ckpt), save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
            ), model_factory=self._factory([]))
            self.assertEqual(result["evaluation_split"], "test")
            self.assertEqual(len(result["support_ids"]), 5)
            self.assertEqual(set(result["eval_sample_ids"]), {"normal/tn.png", "abnormal/ta.png"})

    def test_missing_metrics_are_recomputed_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            args = dict(steps=1, save_dir=str(root / "runs"), no_heatmaps=True, device="cpu", seed=0)
            first = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            os.remove(os.path.join(first["run_dir"], "metrics.json"))
            second = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            self.assertFalse(second["reused"])
            self.assertIn("image", second["metrics"])
            self.assertEqual(second["evaluation_split"], "support")
            self.assertTrue(os.path.isfile(os.path.join(first["run_dir"], "metrics.json")))
            self.assertEqual(len(made), 2)

    def test_postprocess_change_reevaluates_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            common = dict(steps=1, save_dir=str(root / "runs"), no_heatmaps=True, device="cpu", seed=0)
            first = run_experiment(make_args(str(root / "train"), topk_ratio=0.05, **common), model_factory=self._factory(made))
            before = Path(first["checkpoint"]).read_bytes()
            second = run_experiment(make_args(str(root / "train"), topk_ratio=0.2, **common), model_factory=self._factory(made))
            self.assertEqual(first["experiment_id"], second["experiment_id"])
            self.assertNotEqual(first["evaluation_id"], second["evaluation_id"])
            self.assertFalse(second["reused"])
            self.assertEqual(second["run_dir"], first["run_dir"])
            self.assertEqual(second["completed_steps"], 0)
            self.assertEqual(second["seen_sample_ids"], [])
            self.assertEqual(Path(second["checkpoint"]).read_bytes(), before)

    def test_manual_postprocess_thresholds_are_not_left_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            _fill_split(root / "val", ["vn.png"], ["va.png"])
            result = run_experiment(make_args(
                str(root / "train"), val_data_root=str(root / "val"), steps=1,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                image_threshold=0.8, pixel_threshold=0.6,
            ), model_factory=self._factory([]))
            post = result["metrics"]["postprocess"]["thresholds"]
            self.assertEqual(post["image_threshold"], 0.8)
            self.assertEqual(post["pixel_threshold"], 0.6)
            self.assertEqual(post["score_space"], "post_v1_probability_score")
            self.assertNotEqual(result["thresholds"]["raw_margin"]["image_threshold"], 0.8)
            self.assertIsNone(result["thresholds"]["raw_margin"]["pixel_threshold"])
            saved = json.loads(Path(result["run_dir"], "thresholds.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["post_v1_probability_score"]["pixel_threshold"], 0.6)

    def test_floor_alone_does_not_start_the_seed_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            _fill_split(root / "val", ["vn.png"], ["va.png"])
            result = run_experiment(make_args(
                str(root / "train"), val_data_root=str(root / "val"), steps=1,
                save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                mask_mode="adaptive_seeded", grow_floor=0.025,
            ), model_factory=self._factory([]))
            post = result["metrics"]["postprocess"]["thresholds"]
            self.assertIsNone(post["seed_threshold"])
            self.assertEqual(post["grow_floor"], 0.025)
            self.assertEqual(result["evaluation_status"], "ok")

    def test_postprocess_failure_is_not_marked_visualized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            args = dict(
                steps=1, save_dir=str(root / "runs"), device="cpu", seed=0,
                mask_mode="adaptive_seeded", seed_threshold=0.4, grow_floor=0.7,
            )
            first = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            self.assertNotEqual(first["status"], "visualized")
            self.assertEqual(first["evaluation_status"], "failed")
            self.assertNotEqual(first["visualization_status"], "ok")
            config = json.loads(Path(first["run_dir"], "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["evaluation_status"], "failed")
            self.assertTrue(os.path.isfile(first["checkpoint"]))
            second = run_experiment(make_args(str(root / "train"), **args), model_factory=self._factory(made))
            self.assertFalse(second["reused"])
            self.assertEqual(second["evaluation_status"], "failed")
            self.assertEqual(len(made), 2)

    def test_fusion_reads_checkpoint_supervision_not_config_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            _fill_split(root / "test", ["tn.png"], ["ta.png"])
            ckpt = root / "model.pt"
            TinyModel().save_checkpoint(
                ckpt, prompt_set="brain_mri_sentence",
                global_supervision_enabled=False, w_global=0.0,
            )
            with self.assertRaises(ValueError) as caught:
                run_experiment(make_args(
                    str(root / "train"), eval_data_root=str(root / "test"), eval_only=True,
                    ckpt=str(ckpt), save_dir=str(root / "runs"), no_heatmaps=True, device="cpu",
                    score_mode="fusion", global_weight=0.5,
                ), model_factory=self._factory([]))
            self.assertIn("全局监督", str(caught.exception))

    def test_layout_change_rerenders_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            made = []
            common = dict(steps=1, save_dir=str(root / "runs"), device="cpu", seed=0)
            first = run_experiment(make_args(str(root / "train"), panel_size=32, **common), model_factory=self._factory(made))
            before = Path(first["checkpoint"]).read_bytes()
            with Image.open(next(Path(first["run_dir"], "heatmaps").glob("*_display.png"))) as image:
                self.assertEqual(image.size[0], 32 * 4 + 8 * 3)
            second = run_experiment(make_args(str(root / "train"), panel_size=64, **common), model_factory=self._factory(made))
            self.assertFalse(second["reused"])
            self.assertEqual(second["run_dir"], first["run_dir"])
            self.assertNotEqual(second["render_id"], first["render_id"])
            self.assertEqual(len(made), 1)
            self.assertEqual(Path(second["checkpoint"]).read_bytes(), before)
            with Image.open(next(Path(second["run_dir"], "heatmaps").glob("*_display.png"))) as image:
                self.assertEqual(image.size[0], 64 * 4 + 8 * 3)
            third = run_experiment(
                make_args(str(root / "train"), panel_size=64, rows_per_page=1, **common),
                model_factory=self._factory(made),
            )
            self.assertFalse(third["reused"])
            self.assertNotEqual(third["render_id"], second["render_id"])
            self.assertEqual(len(made), 1)
            self.assertTrue(list(Path(third["run_dir"], "heatmaps", "figures").glob("overview_*.png")))
            fourth = run_experiment(
                make_args(str(root / "train"), panel_size=64, rows_per_page=8, **common),
                model_factory=self._factory(made),
            )
            self.assertFalse(fourth["reused"])
            figures = Path(fourth["run_dir"], "heatmaps", "figures")
            self.assertTrue((figures / "overview.png").is_file())
            self.assertFalse(list(figures.glob("overview_*.png")))
            self.assertEqual(len(made), 1)

    def test_rerender_keeps_the_true_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / "run" / "evaluations" / "eval1"
            def one(sample_id, file_id, label):
                return {
                    "sample_id": sample_id,
                    "file_id": file_id,
                    "case_id": "c",
                    "slice_z": 3,
                    "label": label,
                    "pred_label": True,
                    "raw_map": np.zeros((2, 2), np.float32),
                    "score_map_high": np.zeros((4, 4), np.float32),
                    "score_map_refined": np.zeros((4, 4), np.float32),
                    "gray01": np.zeros((4, 4), np.float32),
                    "pred_mask": None,
                    "gt_mask_high": np.zeros((4, 4), dtype=bool),
                    "gt_available": True,
                    "score_global": 0.1,
                    "score_local": 0.2,
                    "score_fused": 0.2,
                    "score_mode": "local_topk",
                    "mask_info": {"mask_status": "unavailable"},
                }
            export_predictions(
                [one("normal/n.png", "abc", 0), one("abnormal/a.png", "def", 1)],
                str(evaluation),
            )
            loaded = _load_exported_records(str(root / "run"), "eval1")
            self.assertEqual([item["label"] for item in loaded], [0, 1])
            self.assertTrue(all(item["pred_label"] is True for item in loaded))
            heatmaps = root / "heatmaps"
            _save_heatmaps(loaded, str(heatmaps), panel_size=32, rows_per_page=1)
            self.assertTrue(list((heatmaps / "figures").glob("overview_*.png")))
            _save_heatmaps(loaded, str(heatmaps), panel_size=32, rows_per_page=8)
            self.assertTrue((heatmaps / "figures" / "overview.png").is_file())
            self.assertFalse(list((heatmaps / "figures").glob("overview_*.png")))
            with Image.open(next(heatmaps.glob("*_display.png"))) as image:
                self.assertEqual(image.size[0], 32 * 4 + 8 * 3)


class BatchTests(unittest.TestCase):
    def test_scheduler_uses_one_prompt_and_all_groups(self):
        parser = batch.build_parser()
        args = parser.parse_args([
            "--data-root", "train", "--ks", "5,10", "--seeds", "0,1",
            "--groups", "T,TVJ", "--steps", "4",
        ])
        jobs = list(batch.iter_jobs(args))
        self.assertEqual(len(jobs), 8)
        self.assertEqual({job.prompt_set for _, job in jobs}, { "brain_mri_sentence" })
        self.assertEqual({job.train_stage for _, job in jobs}, {"text", "joint"})
        self.assertTrue(all(job.n_support in (5, 10) for _, job in jobs))
        args.prompt_set = "brain_mri,brain_mri_sentence"
        with self.assertRaises(ValueError):
            list(batch.iter_jobs(args))
        self.assertIn("brain_mri_sentence", BRAIN_MRI_PROMPT_SETS)

    def test_fresh_run_is_appended_with_a_new_run_id(self):
        self.assertFalse(batch.should_append(
            {"run_id": "r1", "experiment_id": "c1"}, {"r1"}, {"c1"}, False
        ))
        self.assertFalse(batch.should_append(
            {"run_id": "r2", "experiment_id": "c1"}, set(), {"c1"}, True
        ))
        self.assertTrue(batch.should_append(
            {"run_id": "r2", "experiment_id": "c1"}, set(), {"c1"}, False
        ))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fill_split(root / "train", ["n.png"], ["a.png"])
            results = root / "results.jsonl"
            common = [
                "--data-root", str(root / "train"), "--ks", "2", "--seeds", "0",
                "--groups", "T", "--steps", "1", "--batch-size", "2",
                "--out-root", str(root / "runs"), "--results", str(results),
                "--no-heatmaps", "--device", "cpu",
            ]
            answers = [
                {"experiment_id": "same-config", "run_id": "run-a", "run_dir": str(root / "a"),
                 "reused": False, "metrics": {"image": {"auroc": 0.2}, "pixel": None},
                 "checkpoint": str(root / "a.pt"), "support_digest": "d",
                 "evaluation_split": "test", "normal_count": 1, "abnormal_count": 1, "status": "evaluated"},
                {"experiment_id": "same-config", "run_id": "run-b", "run_dir": str(root / "b"),
                 "reused": False, "metrics": {"image": {"auroc": 0.8}, "pixel": None},
                 "checkpoint": str(root / "b.pt"), "support_digest": "d",
                 "evaluation_split": "test", "normal_count": 1, "abnormal_count": 1, "status": "evaluated"},
            ]

            def fake_run(job):
                del job
                return answers.pop(0)

            with unittest.mock.patch.object(batch, "run_experiment", side_effect=fake_run):
                batch.main(common)
                batch.main(common + ["--fresh"])
            lines = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["run_id"] for line in lines], ["run-a", "run-b"])
            self.assertEqual(lines[1]["checkpoint"], str(root / "b.pt"))
            self.assertEqual(lines[0]["experiment_id"], lines[1]["experiment_id"])


if __name__ == "__main__":
    unittest.main()

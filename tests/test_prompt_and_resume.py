"""提示词路由、实验复用、产物保护和精确计步的回归测试（无需数据/下载模型）。"""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import fewshot_run as FS
import generate_heatmaps as GH
import thymoma_local as TL
from text_side_anomaly.prompts import (
    BRAIN_MRI_PROMPT_SETS, DEFAULT_BRAIN_MRI_PROMPTS, resolve_prompt_set,
)


SPLITS = {"train": ["001_a.npz", "002_a.npz"], "val": ["003_a.npz"], "test": ["004_a.npz"]}
METRICS = {"dice": 0.6, "pixel_auroc": 0.9, "iou": 0.4}


def parameters(prompt="sentence"):
    return {
        "text_layers": [8, 9, 10, 11], "text_positions": ["attn", "ffn"], "text_bottleneck": 64,
        "visual_layers": [5, 8, 11], "visual_positions": ["attn", "ffn"],
        "visual_bottleneck": 64, "visual_lambda": 0.1, "loss": dict(FS.LEGACY_LOSS),
        "prompt_set": prompt, "prompt_digest": "synthetic-prompts", "model_config": {},
        "split_id": "synthetic-patient-split",
    }


def fields(steps=2, batch=2, mode="tv_joint", params=None):
    return FS.identity_fields(mode, 1, 0, params or parameters(), FS.step_plan_for(mode, None), "", steps, batch)


class TinyModel(torch.nn.Module):
    """只替换模型计算，真实执行入口的优化循环和预算控制。"""
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.training_stage = cfg.train_stage if cfg else "joint"
        self.inlayer_bank = None
        self.visual_inlayer_bank = None
        self.calls = 0
        self.saved = None

    def build_optimizer(self, **kwargs):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, images, enc):
        self.calls += 1
        return self.weight.square()

    def save_checkpoint(self, path, **kwargs):
        self.saved = kwargs
        Path(path).write_text(str(self.calls), encoding="utf-8")

    def load_checkpoint_blob(self, *args, **kwargs):
        pass


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 35  # batch_size=16 -> 每 epoch 3 步；指定 4 步必须在第二轮中途停止。

    def __getitem__(self, index):
        return {"image": torch.zeros(1), "mask": torch.zeros(1), "label": torch.tensor(1)}


class TinyLoss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, enc, out, labels, masks):
        return {"total": out}


class PromptTests(unittest.TestCase):
    def test_new_defaults_and_checkpoint_inheritance(self):
        self.assertEqual(resolve_prompt_set(None, None, TL.PROMPT_SETS, "sentence"), "sentence")
        for name in TL.PROMPT_SETS:
            with self.subTest(name=name):
                self.assertEqual(resolve_prompt_set(None, name, TL.PROMPT_SETS, "sentence", loading=True), name)
        self.assertIs(DEFAULT_BRAIN_MRI_PROMPTS, BRAIN_MRI_PROMPT_SETS["brain_mri_sentence"])
        self.assertNotEqual(BRAIN_MRI_PROMPT_SETS["brain_mri"].all_texts(), DEFAULT_BRAIN_MRI_PROMPTS.all_texts())

    def test_mismatches_unknown_and_unidentified_checkpoint_rejected(self):
        for requested, saved in [("sentence", "short"), (None, "brain_mri"), (None, None)]:
            with self.subTest(requested=requested, saved=saved), self.assertRaises(ValueError):
                resolve_prompt_set(requested, saved, TL.PROMPT_SETS, "sentence", loading=True)
        self.assertEqual(resolve_prompt_set("sentence", None, TL.PROMPT_SETS, "sentence", loading=True), "sentence")

    def test_heatmap_routes_every_saved_version(self):
        versions = {**TL.PROMPT_SETS, **BRAIN_MRI_PROMPT_SETS, "pneumonia": GH.PNEUMONIA_PROMPTS}
        for name, expected in versions.items():
            model = SimpleNamespace(checkpoint_meta={"prompt_set": name}, cfg=SimpleNamespace(levels=expected.levels),
                                    training_stage="joint")
            with self.subTest(name=name), patch.object(GH, "load_trained_model", return_value=model):
                _, actual = GH.load_model("synthetic.pt", "cpu")
                self.assertIs(actual, expected)
        for name in [None, "unknown"]:
            with patch.object(GH, "load_trained_model", return_value=SimpleNamespace(checkpoint_meta={"prompt_set": name})):
                with self.assertRaises(RuntimeError):
                    GH.load_model("synthetic.pt", "cpu")

    def test_brain_heatmap_requires_explicit_dataset(self):
        model = SimpleNamespace(checkpoint_meta={"prompt_set": "brain_mri_sentence"})
        with patch("sys.argv", ["generate_heatmaps.py", "--ckpt", "brain.pt"]), \
             patch.object(GH, "load_model", return_value=(model, DEFAULT_BRAIN_MRI_PROMPTS)), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            GH.main()
        self.assertEqual(caught.exception.code, 2)

    def test_four_panel_layout_does_not_use_the_single_overlay(self):
        self.assertEqual(GH.heatmap_route(SimpleNamespace(predictions=None, layout="four_panel")), "checkpoint_four_panel")
        self.assertEqual(GH.heatmap_route(SimpleNamespace(predictions="a.jsonl", layout=None)), "replay")
        self.assertEqual(GH.heatmap_route(SimpleNamespace(predictions=None, layout=None)), "single_overlay")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for cls, value in (("normal", 20), ("abnormal", 220)):
                path = root / "data" / cls / "a.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (16, 16), color=value).save(path)
            model = SimpleNamespace(
                checkpoint_meta={"prompt_set": "brain_mri_sentence", "global_supervision_enabled": False},
                cfg=SimpleNamespace(levels=DEFAULT_BRAIN_MRI_PROMPTS.levels, temperature=0.07, image_size=32),
                training_stage="text",
            )

            def fake_map(model, anchors, image_tensor, device):
                del model, anchors, image_tensor, device
                return np.zeros((14, 14), dtype=np.float32), 0.2

            out = root / "out"
            argv = [
                "generate_heatmaps.py", "--ckpt", "brain.pt", "--data-root", str(root / "data"),
                "--layout", "four_panel", "--out-dir", str(out),
            ]
            with patch("sys.argv", argv), \
                 patch.object(GH, "load_model", return_value=(model, DEFAULT_BRAIN_MRI_PROMPTS)), \
                 patch.object(GH, "anomaly_map_for", side_effect=fake_map), \
                 patch.object(GH, "overlay_heatmap", side_effect=AssertionError("single overlay")):
                GH.main()
            with Image.open(out / "overview.png") as overview:
                self.assertGreater(overview.size[0], overview.size[1])

    def test_thymoma_calibration_produces_a_prediction_mask(self):
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = SimpleNamespace(temperature=1.0)
                self.checkpoint_meta = {}

            def encode_anchors(self, anchors):
                del anchors
                return {}

            def forward(self, images, encoded):
                del encoded
                amap = torch.zeros(images.shape[0], 4, 4)
                amap[:, 1:3, 1:3] = 3.0
                return {"anomaly_map": amap, "cls_probs": torch.zeros(images.shape[0])}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gray = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
            mask = np.zeros((16, 16), dtype=np.float32)
            mask[4:12, 4:12] = 1
            val_path = root / "val.npz"
            test_path = root / "test.npz"
            np.savez(val_path, image=gray, mask224=mask)
            np.savez(test_path, image=gray, mask224=mask)
            formal = TL.calibrate_thymoma_predictions(Stub(), None, [str(val_path)], [str(test_path)], "cpu")
            self.assertEqual(formal["metrics"]["calibration"], "ok")
            self.assertIsNotNone(formal["metrics"]["pixel_threshold"])
            mask_info = formal["records"][0]["mask_info"]
            self.assertEqual(mask_info["mask_status"], "predicted")
            self.assertTrue(np.asarray(formal["records"][0]["pred_mask"]).any())
            out = root / "figures"
            TL.save_formal_figures(
                Stub(), None, [str(test_path)], "cpu", str(out), n=1,
                thresholds=formal["thresholds"],
            )
            self.assertTrue((out / "figures" / "rows").exists())


class ResumeTests(unittest.TestCase):
    def test_budget_batch_prompt_and_layers_change_identity_and_filename(self):
        base = fields()
        changes = [fields(steps=2700), fields(batch=16)]
        for key, value in [("prompt_set", "short"), ("prompt_digest", "edited-sentence"),
                           ("visual_layers", [11]), ("split_id", "changed-split")]:
            item = copy.deepcopy(base)
            item[key] = value
            changes.append(item)
        for item in changes:
            self.assertNotEqual(FS.experiment_identity(base), FS.experiment_identity(item))
            self.assertNotEqual(FS.artifact_stem(base), FS.artifact_stem(item))

    def test_init_not_in_nonvisual_identity(self):
        a = FS.identity_fields("output", 1, 0, parameters(), FS.step_plan_for("output", "unused.pt"), "abc", 2, 2)
        self.assertEqual(FS.experiment_identity(a), FS.experiment_identity(fields(mode="output")))

    def test_legacy_missing_and_overwritten_checkpoint_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "weights.pt"
            checkpoint.write_bytes(b"first")
            record = fields()
            record.update(experiment_id=FS.experiment_identity(record), checkpoint_path=str(checkpoint),
                          checkpoint_sha256=FS.ckpt_identity(str(checkpoint)), dice=.6, auroc=.9, iou=.4, n_trainable=1)
            log = str(Path(folder) / "records.jsonl")
            FS.append_result(log, {"mode": "tv_joint", "k": 1, "seed": 0, "steps": 2})
            FS.append_result(log, record)
            self.assertEqual(list(FS.load_done(log)), [record["experiment_id"]])
            checkpoint.write_bytes(b"other")  # 同样大小，也必须识别成不同内容。
            self.assertEqual(FS.load_done(log), {})
            checkpoint.unlink()
            self.assertEqual(FS.load_done(log), {})

    def test_half_written_log_does_not_swallow_next_record(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "records.jsonl"
            log.write_text('{"mode":', encoding="utf-8")
            FS.append_result(str(log), {"complete": True})
            self.assertEqual(json.loads(log.read_text(encoding="utf-8").splitlines()[1]), {"complete": True})

    def test_run_one_keeps_existing_checkpoint_and_records_paths(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(FS, "build_model", side_effect=lambda *a: (None, TinyModel())), \
             patch.object(FS, "sample_support", return_value=SPLITS["train"]), \
             patch.object(FS, "train", return_value=1.0), patch.object(FS, "evaluate", return_value=METRICS):
            old_path = Path(folder) / (FS.artifact_stem(fields()) + ".pt")
            old_path.write_bytes(b"do not overwrite")
            result = FS.run_one("tv_joint", 1, 0, "cpu", SPLITS, TL.PROMPT_SETS["sentence"],
                                2, 2, folder, None, parameters(), None)
            self.assertEqual(old_path.read_bytes(), b"do not overwrite")
            self.assertNotEqual(result["checkpoint_path"], str(old_path))
            self.assertTrue(Path(result["checkpoint_path"]).is_file())
            self.assertEqual(result["checkpoint_sha256"], FS.ckpt_identity(result["checkpoint_path"]))

    def test_zero_step_and_one_step_sequential_rejected(self):
        for mode, steps in [("tv_joint", 0), ("tv_seq", 1)]:
            with self.assertRaises(ValueError):
                FS.run_one(mode, 1, 0, "cpu", SPLITS, TL.PROMPT_SETS["sentence"],
                           steps, 2, "unused", None, parameters(), None)

    def test_cli_resume_uses_actual_budget(self):
        seen = []
        with tempfile.TemporaryDirectory() as folder:
            def fake_run(mode, k, seed, device, splits, prompts, steps, batch, ckpts, heat, hp, init):
                seen.append((steps, batch))
                result = FS.identity_fields(mode, k, seed, hp, FS.step_plan_for(mode, init), "", steps, batch)
                checkpoint = Path(folder) / (FS.artifact_stem(result) + ".pt")
                checkpoint.write_text(str(seen), encoding="utf-8")
                result.update(experiment_id=FS.experiment_identity(result), checkpoint_path=str(checkpoint),
                              checkpoint_sha256=FS.ckpt_identity(str(checkpoint)), dice=.6, auroc=.9, iou=.4, n_trainable=1)
                return result

            argv = ["fewshot_run.py", "--groups", "TVJ", "--ks", "1", "--seeds", "0", "--no-heatmaps",
                    "--cooldown", "0", "--results", str(Path(folder) / "results.jsonl")]
            with patch.object(FS, "make_slice_splits", return_value=SPLITS), patch.object(FS, "run_one", side_effect=fake_run):
                for steps, batch in [(2, 2), (2700, 16), (2700, 16)]:
                    with patch("sys.argv", argv + ["--steps", str(steps), "--batch-size", str(batch)]):
                        FS.main()
            self.assertEqual(seen, [(2, 2), (2700, 16)])


class TrainingEntryTests(unittest.TestCase):
    def run_thymoma(self, folder, extra, blob=None):
        made = []
        def construct(cfg, **kwargs):
            model = TinyModel(cfg)
            made.append(model)
            return model

        argv = ["thymoma_local.py", "--steps", "4", "--legacy-display", "--out", str(Path(folder) / "model.pt"),
                "--heatmap-dir", str(Path(folder) / "heatmaps"), *extra]
        with patch("sys.argv", argv), patch.object(TL, "TextSideAnomalyModel", side_effect=construct), \
             patch.object(TL, "_torch_load", return_value=blob), \
             patch.object(TL, "prepare_heatmap_dir"), patch.object(TL, "make_slice_splits", return_value=SPLITS), \
             patch.object(TL, "ThymomaSliceDataset", TinyDataset), patch.object(TL, "TotalLoss", TinyLoss), \
             patch.object(TL, "pre_tokenize", return_value={}), patch.object(TL, "encode_anchors_cached", return_value={}), \
             patch.object(TL, "evaluate", return_value=METRICS), patch.object(TL, "save_heatmaps"):
            TL.main()
        return made[0]

    def test_exact_steps_not_rounded_to_full_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            model = self.run_thymoma(folder, [])
        self.assertEqual(model.calls, 4)
        self.assertEqual(model.saved["step"], 4)
        self.assertEqual(model.saved["prompt_set"], "sentence")

    def test_init_prompt_resolved_before_model_levels(self):
        blob = {"format_version": 2, "model_state_dict": {}, "prompt_set": "tiers3"}
        with tempfile.TemporaryDirectory() as folder:
            model = self.run_thymoma(folder, ["--init-ckpt", "old.pt"], blob)
        self.assertEqual(model.cfg.levels, TL.PROMPT_SETS["tiers3"].levels)
        self.assertEqual(model.saved["prompt_set"], "tiers3")

    def test_init_prompt_mismatch_fails_before_training(self):
        blob = {"format_version": 2, "model_state_dict": {}, "prompt_set": "short"}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.run_thymoma(folder, ["--init-ckpt", "old.pt", "--prompt-set", "sentence"], blob)
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

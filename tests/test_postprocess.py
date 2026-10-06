"""后处理和四列图不依赖 CLIP，也不允许 GT 改变预测。"""

import unittest

import numpy as np

from text_side_anomaly.postprocess import (
    OperatingThresholds,
    PostprocessConfig,
    adaptive_candidates,
    build_postprocess_config,
    choose_adaptive_operating_point,
    make_seeded_adaptive_mask,
    postprocess_one,
    resolve_score_mode,
    score_adaptive_preview,
    should_search_adaptive_thresholds,
    split_manual_thresholds,
    topk_pool,
)
from text_side_anomaly.visualize import as_binary_mask, binary_contour, render_four_panel


def _config(**kwargs):
    config = PostprocessConfig(**kwargs)
    config.validate()
    return config


class PostprocessTests(unittest.TestCase):
    def test_topk_ratio_is_not_the_fewshot_k(self):
        score = np.arange(4, dtype=np.float32).reshape(2, 2)
        pooled, fallback = topk_pool(score, 1.0)
        self.assertFalse(fallback)
        self.assertAlmostEqual(pooled, float(score.mean()))
        with self.assertRaises(ValueError):
            topk_pool(score, 0)
        pooled, fallback = topk_pool(score, 0.5, roi=np.zeros_like(score, dtype=bool))
        self.assertTrue(fallback)
        with self.assertRaises(ValueError):
            topk_pool(np.array([[np.nan, 0.0]]), 0.5)

    def test_prediction_ignores_ground_truth_and_keeps_raw_map(self):
        raw = np.array([[0.2, -0.1], [0.4, 0.0]], dtype=np.float32)
        before = raw.copy()
        gray = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
        config = _config()
        first = postprocess_one(raw, 0.8, gray, 0.07, config, OperatingThresholds(pixel_threshold=0.6))
        second = postprocess_one(raw, 0.8, gray, 0.07, config, OperatingThresholds(pixel_threshold=0.6))
        self.assertTrue(np.array_equal(raw, before))
        self.assertTrue(np.array_equal(first["pred_mask"], second["pred_mask"]))
        self.assertTrue(np.array_equal(first["raw_map"], before))
        self.assertEqual(first["score_mode"], "local_topk")
        self.assertAlmostEqual(first["score_fused"], first["score_local"])

    def test_seeded_mask_keeps_every_connected_seed_and_rejects_isolated_low_score(self):
        score = np.zeros((8, 8), dtype=np.float32)
        score[1, 1] = 0.9
        score[6, 6] = 0.5
        mask, info = make_seeded_adaptive_mask(score, 0.7, 0.35)
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[6, 6])
        self.assertGreaterEqual(info["kept_component_count"], 1)
        empty, empty_info = make_seeded_adaptive_mask(np.full((4, 4), 0.2), 0.7, 0.35)
        self.assertFalse(empty.any())
        self.assertEqual(empty_info["mask_status"], "no_seed")
        full, full_info = make_seeded_adaptive_mask(np.full((4, 4), 0.8), 0.7, 0.35)
        self.assertTrue(full.all())
        self.assertEqual(full_info["adaptive_status"], "constant_map")

    def test_fusion_requires_global_supervision(self):
        with self.assertRaises(ValueError):
            resolve_score_mode("fusion", {"w_global": 0})
        self.assertEqual(resolve_score_mode(None, {"w_global": 0}), "local_topk")
        self.assertEqual(resolve_score_mode("fusion", {"global_supervision_enabled": True}), "fusion")
        blocked = {"global_supervision_enabled": False, "w_global": 1.0}
        with self.assertRaises(ValueError):
            build_postprocess_config("fusion", blocked, global_weight=0.5)
        config = build_postprocess_config(None, blocked, global_weight=0.5)
        self.assertEqual(config.score_mode, "local_topk")
        self.assertEqual(config.global_weight, 0.0)

    def test_manual_thresholds_stay_in_their_score_space(self):
        spaces = split_manual_thresholds(0.8, 0.6)
        self.assertEqual(spaces["post_v1_probability_score"]["image_threshold"], 0.8)
        self.assertEqual(spaces["post_v1_probability_score"]["pixel_threshold"], 0.6)
        self.assertIsNone(spaces["raw_margin"]["image_threshold"])
        self.assertIsNone(spaces["raw_margin"]["pixel_threshold"])
        raw = split_manual_thresholds(0.8, 0.6, score_space="raw_margin")
        self.assertEqual(raw["raw_margin"]["pixel_threshold"], 0.6)
        self.assertIsNone(raw["post_v1_probability_score"]["pixel_threshold"])
        separate = split_manual_thresholds(
            None, None, raw_pixel_threshold=-0.2, post_pixel_threshold=0.6,
        )
        self.assertEqual(separate["raw_margin"]["pixel_threshold"], -0.2)
        self.assertEqual(separate["post_v1_probability_score"]["pixel_threshold"], 0.6)

    def test_low_scores_enter_the_seed_grid_and_false_positives_count(self):
        normal = np.full((8, 8), 0.001, dtype=np.float32)
        lesion = np.full((8, 8), 0.001, dtype=np.float32)
        lesion[1:3, 1:3] = 0.05
        records = [
            {"score_map_refined": normal, "label": 0, "gt_available": True,
             "gt_mask_high": np.zeros_like(normal, dtype=bool)},
            {"score_map_refined": lesion, "label": 1, "gt_available": True,
             "gt_mask_high": lesion > 0.02},
        ]
        pairs = adaptive_candidates(records)
        seeds = {seed for seed, _ in pairs}
        self.assertTrue(any(seed <= 0.05 for seed in seeds))
        self.assertTrue(seeds - {0.3, 0.5, 0.7})
        self.assertTrue(all(floor <= seed for seed, floor in pairs))
        self.assertFalse(should_search_adaptive_thresholds(None, 0.025))
        self.assertTrue(should_search_adaptive_thresholds(None, None))
        chosen = choose_adaptive_operating_point([
            {"seed": 0.3, "floor": 0.1, "normal_false_positive_rate": 0.0,
             "lesion_detection_rate": 0.0, "dice": 0.0},
            {"seed": 0.02, "floor": 0.005, "normal_false_positive_rate": 0.0,
             "lesion_detection_rate": 1.0, "dice": 0.4},
            {"seed": 0.001, "floor": 0.0, "normal_false_positive_rate": 1.0,
             "lesion_detection_rate": 1.0, "dice": 0.9},
        ])
        self.assertEqual(chosen["seed"], 0.02)
        self.assertTrue(chosen["ok"])

    def test_constant_normal_score_keeps_a_reject_all_seed(self):
        score = np.full((8, 8), 0.5, dtype=np.float32)
        record = {
            "score_map_refined": score,
            "label": 0,
            "gt_available": True,
            "gt_mask_high": np.zeros_like(score, dtype=bool),
        }
        pairs = adaptive_candidates([record])
        self.assertTrue(any(seed > 0.5 for seed, _floor in pairs))
        scored = []
        for seed, floor in pairs:
            mask, _info = make_seeded_adaptive_mask(score, seed, floor)
            quality = score_adaptive_preview([dict(record, pred_mask=mask)])
            scored.append({"seed": seed, "floor": floor, "dice": 0.0 if mask.any() else 1.0, **quality})
        chosen = choose_adaptive_operating_point(scored)
        self.assertTrue(chosen["ok"])
        self.assertGreater(chosen["seed"], 0.5)
        self.assertEqual(chosen["normal_false_positive_rate"], 0.0)
        failed = choose_adaptive_operating_point([
            {"seed": 0.25, "floor": 0.1, "normal_false_positive_rate": 1.0,
             "lesion_detection_rate": 1.0, "dice": 0.9},
        ])
        self.assertFalse(failed["ok"])
        self.assertIn("无法完成校准", failed["reason"])
        missing = choose_adaptive_operating_point([
            {"seed": 0.2, "floor": 0.1, "normal_false_positive_rate": None,
             "lesion_detection_rate": 1.0, "dice": 0.4},
        ])
        self.assertFalse(missing["ok"])
        self.assertIn("验证数据不足", missing["reason"])

    def test_zero_and_255_masks_share_a_contour(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[2:4, 2:4] = 1
        self.assertTrue(np.array_equal(binary_contour(mask), binary_contour(mask * 255)))
        self.assertTrue(np.array_equal(as_binary_mask(mask), as_binary_mask(mask * 255)))
        self.assertFalse(binary_contour(np.zeros((4, 4))).any())

    def test_heatmap_columns_match_except_for_the_contour(self):
        gray = np.zeros((16, 16), dtype=np.float32)
        score = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
        gt = np.zeros((16, 16), dtype=bool)
        gt[4:8, 4:8] = True
        panel = render_four_panel(gray, score, None, gt, True, panel_size=32, mask_status="unavailable")
        heat = panel[36:68, 40:72]
        heat_gt = panel[36:68, 80:112]
        green = np.all(heat_gt == np.array([0, 255, 0]), axis=-1)
        self.assertTrue(green.any())
        self.assertTrue(np.array_equal(heat[~green], heat_gt[~green]))


if __name__ == "__main__":
    unittest.main()

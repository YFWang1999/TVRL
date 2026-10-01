"""CPU regression tests; run with python -m unittest discover -s tests."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("credit_layout", ROOT / "hyvideo/models/reward_models/credit_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
restore_qwen_video_grid = layout.restore_qwen_video_grid

# Execute the actual scorer methods without loading a model or optional training
# dependencies. No duplicated credit conversion implementation in these tests.
tree = ast.parse((ROOT / "hyvideo/models/reward_models/rewards.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLMRewardScorer")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {
    "_compute_credit_map_from_visual_grad", "_normalize_credit_map",
    "_compute_frame_importance_from_visual_grad", "_normalize_frame_importance"}]
exec(compile(ast.Module(body=methods, type_ignores=[]), str(ROOT / "hyvideo/models/reward_models/rewards.py"), "exec"))
tree = ast.parse((ROOT / "hyvideo/utils/grpo_utils.py").read_text())
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
    "_smooth_credit_maps", "_align_credit_maps_to_log_prob_map", "_aggregate_log_prob_map_by_credit_map"}]
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(ROOT / "hyvideo/utils/grpo_utils.py"), "exec"))


class CreditLayoutTest(unittest.TestCase):
    def test_real_processor_and_scorer(self):
        processor = Qwen3VLVideoProcessor()
        scorer = SimpleNamespace(model_family="qwen3_5", processor=SimpleNamespace(video_processor=processor))
        scorer._normalize_credit_map = lambda x: _normalize_credit_map(scorer, x)
        scorer._normalize_frame_importance = lambda x: _normalize_frame_importance(scorer, x)
        for t, h, w in [(1, 2, 4), (3, 4, 8), (10, 10, 18)]:
            with self.subTest(grid=(t, h, w)):
                expected = torch.arange(1, t*h*w+1).reshape(t, h, w).float()
                pixels = expected.repeat_interleave(2, 0)[:, None].repeat(1, 3, 1, 1)
                pixels = pixels.repeat_interleave(16, -2).repeat_interleave(16, -1)[None]
                packed, gt, gh, gw = processor.patchify(pixels, 16, 2, 2)
                scores = packed.mean(-1).flatten()
                restored = restore_qwen_video_grid(scores, (gt, gh, gw), scorer.processor)
                torch.testing.assert_close(restored, expected)
                result = _compute_credit_map_from_visual_grad(scorer, -scores[:, None], {"video_grid_thw": torch.tensor([[t,h,w]])})
                actual = torch.tensor(result["values"]).reshape(t,h,w)
                torch.testing.assert_close(actual, expected/expected.sum())
                self.assertFalse(torch.equal(scores.reshape(t,h,w), expected))
                frame = _compute_frame_importance_from_visual_grad(scorer, scores[:,None], {"video_grid_thw": torch.tensor([[t,h,w]])})
                torch.testing.assert_close(torch.tensor(frame), expected.sum((1,2))/expected.sum())

    def test_merge_size_and_invalid_metadata(self):
        for m in [1, 2, 4]:
            expected = torch.arange(32).reshape(1,4,8)
            packed = expected.reshape(1,4//m,m,8//m,m).permute(0,1,3,2,4).flatten()
            processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=m))
            torch.testing.assert_close(restore_qwen_video_grid(packed, (1,4,8), processor), expected)
        with self.assertRaises(ValueError):
            restore_qwen_video_grid(torch.ones(8), (1,2,4), SimpleNamespace())
        with self.assertRaises(ValueError):
            restore_qwen_video_grid(torch.ones(8), (1,2,4), SimpleNamespace(video_processor=SimpleNamespace(merge_size=3)))

    def test_window3_routing_and_frozen_ratio(self):
        # Packed index4 belongs to top row, column2, not bottom-left.
        packed = torch.zeros(80)
        packed.reshape(10,8)[:,4] = 1
        processor = SimpleNamespace(video_processor=SimpleNamespace(merge_size=2))
        grid = restore_qwen_video_grid(packed, (10,2,4), processor)
        maps = F.interpolate(grid[None,None], (20,16,16), mode="trilinear", align_corners=False)
        maps = maps/maps.sum()
        smoothed = _smooth_credit_maps(maps, 3)
        torch.testing.assert_close(smoothed.sum(), torch.tensor(1.))
        logp = torch.arange(20*16*16).float().reshape(1,1,20,16,16)
        weights = _align_credit_maps_to_log_prob_map(maps, logp, window_size=3)
        self.assertGreater(float(weights[:,:,:,:8,8:].sum()), .5)
        old = _aggregate_log_prob_map_by_credit_map(logp, maps, window_size=3)
        new = _aggregate_log_prob_map_by_credit_map(logp.clone(), maps, window_size=3)
        torch.testing.assert_close(old, (weights*logp).flatten(2).sum(-1))
        torch.testing.assert_close(torch.exp(new-old), torch.ones_like(old))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights>=0).all())


if __name__ == "__main__":
    unittest.main()

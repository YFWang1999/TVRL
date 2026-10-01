"""
Unified reward interface for video generation models.
Provides a standard interface for all reward models including video and image rewards.
All reward functions are self-contained without external flow_grpo dependencies.
"""

import importlib.util
import json
import math
import os
import pkgutil
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode
from .credit_layout import restore_qwen_video_grid

# VideoAlign model path: default to local path
_DEFAULT_VIDEOALIGN_PATH = "./ckpts/VideoReward"
_DEFAULT_VIDEOSCORE2_PATH = os.environ.get("VIDEOSCORE2_MODEL_PATH", "TIGER-Lab/VideoScore2")
REWARD_MODEL_PATH = {
    "videoalign": _DEFAULT_VIDEOALIGN_PATH,
    "video_score2": _DEFAULT_VIDEOSCORE2_PATH,
}

YES_NO_RULE = '- For "yes_no" questions, answer with "Yes" or "No" only.'
LEGACY_YES_NO_UPPERCASE_RULE = '- For "yes_no" questions, answer with uppercase "YES" or "NO" only.'
RATING_1_5_RULE = (
    '- For "rating_1_5" questions, answer with a single integer from "1" to "5" only, '
    "where 5 is best."
)
FT_ANSWER_VALUE_PATTERNS = (
    re.compile(r"""['"]answer['"]\s*:\s*'(?P<answer>(?:\\.|[^'\\])*)'""", re.DOTALL),
    re.compile(r"""['"]answer['"]\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)\"""", re.DOTALL),
)

_DEFAULT_VLM_REWARD_PROMPT_TEMPLATE = (
    '{"id": "q1", "question": "Does the generated video faithfully match this prompt: {prompt}?", '
    '"answer_type": "yes_no"}'
)

_VLM_REWARD_MODEL_ALIASES = frozenset(
    {
        "vlm_reward",
        "vlm_qa_local",
        "vlm_free_generation_yes",
        "qwen_vl_free_generation_yes",
    }
)
_VLM_FREE_GENERATION_ALIASES = frozenset(
    {
        "vlm_free_generation_yes",
        "qwen_vl_free_generation_yes",
    }
)


# ============================================================================
# Video Reward Models
# ============================================================================

def videoalign_local_score(
    device,
    reward_checkpoint_mode="none",
    *,
    gradient_credit: bool = False,
    credit_num_frames: Optional[int] = None,
    credit_metric: str = "weighted",
    credit_metric_weights: Optional[Dict[str, float]] = None,
):
    """Local VideoAlign reward model."""
    from hyvideo.models.reward_models.videoalign.inference import VideoVLMRewardInference

    dtype = torch.bfloat16
    scorer = VideoVLMRewardInference(
        REWARD_MODEL_PATH["videoalign"],
        device=device,
        dtype=dtype,
        reward_checkpoint_mode=reward_checkpoint_mode,
    )

    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        """
        Args:
            video_paths: List of video file paths
            prompts: List of text prompts
            metadata: Optional metadata for each sample

        Returns:
            scores_dict: Dict of {metric_name: List[float]}
            meta_dict: Additional metadata (empty for now)
        """
        if metadata is None:
            metadata = [{}] * len(video_paths)

        scores_dict = {}
        all_entries = []
        all_records = []

        for local_idx, (video_path, prompt) in enumerate(zip(video_paths, prompts)):
            try:
                if gradient_credit:
                    try:
                        credit_frames = int(credit_num_frames or 0)
                    except (TypeError, ValueError):
                        credit_frames = 0
                    reward_entries, records = scorer.reward_with_gradient_credit(
                        [video_path],
                        [prompt],
                        num_frames=credit_frames if credit_frames > 0 else None,
                        metric_weights=credit_metric_weights,
                        credit_metric=credit_metric,
                        sample_indices=[local_idx],
                    )
                    reward_entry = reward_entries[0]
                    all_records.extend(records)
                else:
                    reward_entry = scorer.reward([video_path], [prompt])[0]
                all_entries.append(reward_entry)
            except Exception:
                all_entries.append({})

        if all_entries:
            all_keys = set()
            for entry in all_entries:
                all_keys.update(entry.keys())

            for key in all_keys:
                scores = []
                for entry in all_entries:
                    val = entry.get(key)
                    try:
                        scores.append(float(val) if val is not None else float("nan"))
                    except (TypeError, ValueError):
                        scores.append(float("nan"))
                scores_dict[key] = scores

        return scores_dict, {"records": all_records} if all_records else {}

    for hook_name in ("prepare_for_rollout", "prepare_for_reward", "cleanup_after_reward"):
        hook = getattr(scorer, hook_name, None)
        if callable(hook):
            setattr(_fn, hook_name, hook)

    return _fn


class VideoScore2LocalRewardInference:
    """Local VideoScore2 point-wise reward with optional visual-gradient credit."""

    def __init__(
        self,
        model_name_or_path: str,
        device,
        *,
        infer_fps: float = 2.0,
        max_new_tokens: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

        self.process_vision_info = process_vision_info
        self.device = torch.device(device)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.infer_fps = float(infer_fps)
        self.max_new_tokens = int(max_new_tokens)
        self.offload_device = torch.device("cpu")
        self._move_model_to(self.device)

        self.score_token_ids: List[int] = []
        self.score_values: List[float] = []
        for score in range(1, 6):
            token_ids = self.tokenizer.encode(str(score), add_special_tokens=False)
            if len(token_ids) == 1:
                self.score_token_ids.append(int(token_ids[0]))
                self.score_values.append(float(score))
        if not self.score_token_ids:
            raise ValueError("VideoScore2 tokenizer does not expose single-token score ids for 1..5.")

    def _move_model_to(self, target_device):
        target_device = torch.device(target_device)
        try:
            current_device = next(self.model.parameters()).device
        except StopIteration:
            current_device = target_device
        if current_device == target_device:
            return
        self.model.to(target_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_for_rollout(self):
        self._move_model_to(self.offload_device)

    def prepare_for_reward(self):
        self._move_model_to(self.device)

    def cleanup_after_reward(self):
        self.prepare_for_rollout()

    def _build_user_prompt(self, prompt: str) -> str:
        return (
            "You are an expert for evaluating AI-generated videos from three dimensions:\n"
            "(1) visual quality - clarity, smoothness, artifacts;\n"
            "(2) text-to-video alignment - fidelity to the prompt;\n"
            "(3) physical/common-sense consistency - naturalness and physics plausibility.\n\n"
            f"Video prompt: {prompt}\n\n"
            "Please output in this format:\n"
            "visual quality: <v_score>;\n"
            "text-to-video alignment: <t_score>;\n"
            "physical/common-sense consistency: <p_score>"
        )

    def _build_messages(self, video_path: str, prompt: str):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.infer_fps,
                    },
                    {
                        "type": "text",
                        "text": self._build_user_prompt(prompt),
                    },
                ],
            }
        ]

    def _prepare_processor_inputs(self, text: str, image_inputs, video_inputs):
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=self.infer_fps,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def _parse_scores(self, output_text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        pattern = (
            r"visual quality:\s*(\d+).*?"
            r"text-to-video alignment:\s*(\d+).*?"
            r"physical/common-sense consistency:\s*(\d+)"
        )
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return None, None, None
        return tuple(min(max(int(value), 1), 5) for value in match.groups())

    def _find_visual_input_key(self, inputs: Mapping) -> Optional[str]:
        for key in ("pixel_values_videos", "pixel_values_video", "pixel_values"):
            value = inputs.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                return key
        return None

    def _prepare_inputs_for_gradient(self, inputs: Mapping):
        visual_key = self._find_visual_input_key(inputs)
        prepared = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and key == visual_key:
                prepared[key] = value.detach().clone().requires_grad_(True)
            else:
                prepared[key] = value
        return prepared, visual_key

    def _normalize_frame_importance(self, frame_scores: torch.Tensor) -> List[float]:
        frame_scores = torch.nan_to_num(
            frame_scores.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if frame_scores.numel() == 0:
            return []
        total = frame_scores.sum()
        if not torch.isfinite(total) or total <= 0:
            frame_scores = torch.ones_like(frame_scores) / max(int(frame_scores.numel()), 1)
        else:
            frame_scores = frame_scores / total
        return frame_scores.cpu().tolist()

    def _compute_frame_importance_from_visual_grad(self, visual_grad, model_inputs: Mapping) -> List[float]:
        if visual_grad is None or not torch.is_tensor(visual_grad):
            return []
        grad_abs = torch.nan_to_num(
            visual_grad.detach().abs().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if grad_abs.numel() == 0:
            return []

        video_grid = model_inputs.get("video_grid_thw")
        if torch.is_tensor(video_grid) and video_grid.numel() >= 3:
            grid = video_grid[0] if video_grid.ndim > 1 else video_grid
            try:
                t_dim, h_dim, w_dim = [int(x) for x in grid[:3].tolist()]
            except Exception:
                t_dim = h_dim = w_dim = 0
            if t_dim > 0 and h_dim > 0 and w_dim > 0 and grad_abs.numel() == t_dim * h_dim * w_dim:
                frame_scores = grad_abs.view(t_dim, h_dim, w_dim).sum(dim=(1, 2))
                return self._normalize_frame_importance(frame_scores)
            token_scores = grad_abs
            while token_scores.ndim > 1:
                token_scores = token_scores.sum(dim=-1)
            if t_dim > 0 and h_dim > 0 and w_dim > 0 and token_scores.numel() == t_dim * h_dim * w_dim:
                frame_scores = token_scores.view(t_dim, h_dim, w_dim).sum(dim=(1, 2))
                return self._normalize_frame_importance(frame_scores)

        if grad_abs.ndim == 5:
            if grad_abs.shape[0] == 1 and grad_abs.shape[1] <= grad_abs.shape[2]:
                return self._normalize_frame_importance(grad_abs.sum(dim=(0, 2, 3, 4)))
            if grad_abs.shape[0] == 1:
                return self._normalize_frame_importance(grad_abs.sum(dim=(0, 1, 3, 4)))
        if grad_abs.ndim == 4:
            if grad_abs.shape[0] <= grad_abs.shape[1]:
                return self._normalize_frame_importance(grad_abs.sum(dim=(1, 2, 3)))
            return self._normalize_frame_importance(grad_abs.sum(dim=(0, 2, 3)))
        return self._normalize_frame_importance(grad_abs.flatten())

    def _normalize_credit_map(self, credit_map: torch.Tensor) -> Dict[str, Any]:
        credit_map = torch.nan_to_num(
            credit_map.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if credit_map.ndim != 3 or credit_map.numel() == 0:
            return {}

        max_spatial_bins = 32
        if credit_map.shape[-2] > max_spatial_bins or credit_map.shape[-1] > max_spatial_bins:
            target_h = min(int(credit_map.shape[-2]), max_spatial_bins)
            target_w = min(int(credit_map.shape[-1]), max_spatial_bins)
            credit_map = F.adaptive_avg_pool3d(
                credit_map.view(1, 1, *credit_map.shape),
                output_size=(int(credit_map.shape[0]), target_h, target_w),
            ).view(int(credit_map.shape[0]), target_h, target_w)

        total = credit_map.sum()
        if not torch.isfinite(total) or total <= 0:
            credit_map = torch.full_like(credit_map, 1.0 / float(max(credit_map.numel(), 1)))
        else:
            credit_map = credit_map / total.clamp_min(1e-6)
        return {
            "values": credit_map.flatten().cpu().tolist(),
            "shape": [int(x) for x in credit_map.shape],
        }

    def _compute_credit_map_from_visual_grad(
        self,
        visual_grad: Optional[torch.Tensor],
        model_inputs: Mapping,
    ) -> Dict[str, Any]:
        if visual_grad is None or not torch.is_tensor(visual_grad):
            return {}
        grad_abs = torch.nan_to_num(
            visual_grad.detach().abs().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if grad_abs.numel() == 0:
            return {}

        video_grid = model_inputs.get("video_grid_thw")
        if torch.is_tensor(video_grid) and video_grid.numel() >= 3:
            grid = video_grid[0] if video_grid.ndim > 1 else video_grid
            try:
                t_dim, h_dim, w_dim = [int(x) for x in grid[:3].tolist()]
            except Exception:
                t_dim = h_dim = w_dim = 0
            if t_dim > 0 and h_dim > 0 and w_dim > 0 and grad_abs.numel() == t_dim * h_dim * w_dim:
                return self._normalize_credit_map(grad_abs.view(t_dim, h_dim, w_dim))
            token_scores = grad_abs
            while token_scores.ndim > 1:
                token_scores = token_scores.sum(dim=-1)
            if t_dim > 0 and h_dim > 0 and w_dim > 0 and token_scores.numel() == t_dim * h_dim * w_dim:
                return self._normalize_credit_map(token_scores.view(t_dim, h_dim, w_dim))

        if grad_abs.ndim == 5:
            if grad_abs.shape[0] == 1 and grad_abs.shape[1] <= grad_abs.shape[2]:
                return self._normalize_credit_map(grad_abs[0].sum(dim=1))
            if grad_abs.shape[0] == 1:
                return self._normalize_credit_map(grad_abs[0].sum(dim=0))

        if grad_abs.ndim == 4:
            if grad_abs.shape[0] <= grad_abs.shape[1]:
                return self._normalize_credit_map(grad_abs.sum(dim=1))
            return self._normalize_credit_map(grad_abs.sum(dim=0))

        return {}

    def _answer_digit_offsets(self, answer_text: str) -> List[int]:
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        offsets = []
        for idx, token_id in enumerate(answer_ids):
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=False).strip()
            if decoded in {"1", "2", "3", "4", "5"}:
                offsets.append(idx)
        return offsets[:3]

    def _score_teacher_forced_answer(
        self,
        prepared_inputs: Mapping,
        prompt_len: int,
        answer_text: str,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        try:
            outputs = self.model(**prepared_inputs, return_dict=True, use_cache=False)
        except TypeError:
            outputs = self.model(**prepared_inputs, return_dict=True)
        logits = outputs.logits[0]
        score_token_ids = torch.tensor(
            self.score_token_ids,
            device=logits.device,
            dtype=torch.long,
        )
        score_values = torch.tensor(
            self.score_values,
            device=logits.device,
            dtype=logits.dtype,
        )

        expected_values: List[torch.Tensor] = []
        for offset in self._answer_digit_offsets(answer_text):
            position = prompt_len + offset
            if position <= 0 or position - 1 >= logits.shape[0]:
                continue
            score_logits = logits[position - 1, score_token_ids].float()
            probs = torch.softmax(score_logits, dim=-1).to(dtype=logits.dtype)
            expected_values.append((probs * score_values).sum())
        if len(expected_values) < 3:
            raise ValueError(f"Could not locate all VideoScore2 score tokens in answer: {answer_text!r}")
        selected = expected_values[0] + expected_values[1] + expected_values[2]
        return expected_values[:3], selected

    def _generate_hard_scores(self, prompt_inputs):
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **prompt_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_len = prompt_inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._parse_scores(output_text), output_text

    def score_one(self, video_path: str, prompt: str, sample_idx: int, *, gradient_credit: bool):
        messages = self._build_messages(video_path, prompt)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        prompt_inputs = self._prepare_processor_inputs(prompt_text, image_inputs, video_inputs)

        hard_scores, generated_text = self._generate_hard_scores(prompt_inputs)
        v_hard, t_hard, p_hard = hard_scores
        if v_hard is None or t_hard is None or p_hard is None:
            v_hard = t_hard = p_hard = 1

        answer_text = (
            f"visual quality: {int(v_hard)}; "
            f"text-to-video alignment: {int(t_hard)}; "
            f"physical/common-sense consistency: {int(p_hard)}"
        )
        full_inputs = self._prepare_processor_inputs(
            prompt_text + answer_text,
            image_inputs,
            video_inputs,
        )
        prepared_inputs, visual_key = self._prepare_inputs_for_gradient(full_inputs)

        context = torch.enable_grad() if gradient_credit else torch.inference_mode()
        with context:
            expected_values, selected = self._score_teacher_forced_answer(
                prepared_inputs,
                prompt_len=prompt_inputs["input_ids"].shape[1],
                answer_text=answer_text,
            )
            visual_grad = None
            if gradient_credit and visual_key is not None:
                visual_grad = torch.autograd.grad(
                    selected,
                    prepared_inputs[visual_key],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]

        soft_scores = [float(value.detach().cpu().item()) for value in expected_values]
        selected_value = float(selected.detach().cpu().item())
        frame_importance = (
            self._compute_frame_importance_from_visual_grad(visual_grad, prepared_inputs)
            if gradient_credit
            else []
        )
        credit_map = (
            self._compute_credit_map_from_visual_grad(visual_grad, prepared_inputs)
            if gradient_credit
            else {}
        )
        top_frames = sorted(
            range(len(frame_importance)),
            key=lambda idx: frame_importance[idx],
            reverse=True,
        )[: min(3, len(frame_importance))]

        reward_entry = {
            "VQ": soft_scores[0],
            "TA": soft_scores[1],
            "MQ": soft_scores[2],
            "Overall": selected_value,
        }
        record = {
            "sample_idx": sample_idx,
            "question_id": "videoscore2_overall",
            "question": str(prompt),
            "token_credit_reward": selected_value if gradient_credit else None,
            "selected_reward": selected_value,
            "frame_importance": frame_importance,
            "credit_map": credit_map.get("values", []),
            "credit_map_shape": credit_map.get("shape", []),
            "top_frames": top_frames,
            "generated_text": generated_text,
            "hard_scores": {"VQ": int(v_hard), "TA": int(t_hard), "MQ": int(p_hard)},
            "soft_scores": dict(reward_entry),
        }
        return reward_entry, record


def videoscore2_local_score(
    device,
    *,
    gradient_credit: bool = False,
    credit_metric: str = "weighted",
    credit_metric_weights: Optional[Dict[str, float]] = None,
):
    model_path = os.environ.get("VIDEOSCORE2_MODEL_PATH", REWARD_MODEL_PATH["video_score2"])
    infer_fps = float(os.environ.get("VIDEOSCORE2_INFER_FPS", "2.0"))
    max_new_tokens = int(os.environ.get("VIDEOSCORE2_MAX_NEW_TOKENS", "1024"))
    dtype_name = os.environ.get("VIDEOSCORE2_TORCH_DTYPE", "bf16").lower()
    dtype = torch.float16 if dtype_name in {"fp16", "float16"} else torch.bfloat16
    scorer = VideoScore2LocalRewardInference(
        model_path,
        device=device,
        infer_fps=infer_fps,
        max_new_tokens=max_new_tokens,
        torch_dtype=dtype,
    )

    def _fn(video_paths: List[str], prompts: List[str], metadata: List[Dict] = None):
        if metadata is None:
            metadata = [{}] * len(video_paths)

        all_entries = []
        all_records = []
        for local_idx, (video_path, prompt) in enumerate(zip(video_paths, prompts)):
            sample_idx = local_idx
            if local_idx < len(metadata) and isinstance(metadata[local_idx], dict):
                try:
                    sample_idx = int(metadata[local_idx].get("sample_idx", local_idx))
                except (TypeError, ValueError):
                    sample_idx = local_idx
            try:
                reward_entry, record = scorer.score_one(
                    video_path,
                    prompt,
                    sample_idx,
                    gradient_credit=gradient_credit,
                )
                all_entries.append(reward_entry)
                all_records.append(record)
            except Exception as exc:
                all_entries.append({"VQ": 1.0, "TA": 1.0, "MQ": 1.0, "Overall": 3.0})
                if gradient_credit:
                    all_records.append(
                        {
                            "sample_idx": sample_idx,
                            "question_id": "videoscore2_overall",
                            "question": str(prompt),
                            "token_credit_reward": 3.0,
                            "selected_reward": 3.0,
                            "frame_importance": [],
                            "credit_map": [],
                            "credit_map_shape": [],
                            "top_frames": [],
                            "error": str(exc),
                        }
                    )

        scores_dict = {}
        if all_entries:
            for key in ("VQ", "TA", "MQ", "Overall"):
                scores_dict[key] = [float(entry.get(key, float("nan"))) for entry in all_entries]
        return scores_dict, {"records": all_records} if all_records else {}

    for hook_name in ("prepare_for_rollout", "prepare_for_reward", "cleanup_after_reward"):
        hook = getattr(scorer, hook_name, None)
        if callable(hook):
            setattr(_fn, hook_name, hook)

    return _fn


def video_align_remote_score(server_url: str, model: str = "video_align"):
    """
    Implementing your own remote video reward service here if required.
    Default using local video reward service.
    """
    if not server_url:
        raise ValueError(f"server_url is required for remote reward model {model!r}.")

    class _RemoteVideoRewardClient:
        prefers_tensor_input = False

        def __init__(self, base_url: str, model_name: str):
            self.base_url = base_url.rstrip("/")
            self.model_name = model_name

        def prepare_for_rollout(self) -> None:
            return None

        def prepare_for_reward(self) -> None:
            return None

        def cleanup_after_reward(self) -> None:
            return None

        def __call__(
            self,
            video_paths: List[str],
            prompts: List[str],
            metadata: Optional[List[Dict[str, Any]]] = None,
        ):
            import requests

            if metadata is None:
                metadata = [{} for _ in prompts]
            payload = {
                "model": self.model_name,
                "video_paths": list(video_paths),
                "prompts": list(prompts),
                "metadata": list(metadata),
            }
            timeout = float(os.environ.get("REMOTE_REWARD_TIMEOUT", "3600"))
            response = requests.post(
                f"{self.base_url}/score",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("scores", {}), result.get("meta", {})

    return _RemoteVideoRewardClient(server_url, model)


def remote_vlm_token_credit_score(server_url: str):
    """HTTP client for a separate VLM token-credit server.

    This keeps the main training process free to use the VideoAlign-compatible
    transformers environment while the Qwen3.5 credit model runs in another
    environment/process.
    """
    if not server_url:
        raise ValueError("token_credit_remote_url is required for remote VLM token credit.")

    class _RemoteVLMTokenCreditClient:
        prefers_tensor_input = False

        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")

        def prepare_for_rollout(self) -> None:
            return None

        def prepare_for_reward(self) -> None:
            return None

        def cleanup_after_reward(self) -> None:
            return None

        def __call__(
            self,
            video_paths: List[str],
            prompts: List[str],
            metadata: Optional[List[Dict[str, Any]]] = None,
        ):
            import requests

            if metadata is None:
                metadata = [{} for _ in prompts]
            payload = {
                "video_paths": list(video_paths),
                "prompts": list(prompts),
                "metadata": list(metadata),
            }
            timeout = float(os.environ.get("TOKEN_CREDIT_REMOTE_TIMEOUT", "3600"))
            response = requests.post(
                f"{self.base_url}/score",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("scores", {}), result.get("meta", {})

    return _RemoteVLMTokenCreditClient(server_url)


# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_key(key: str) -> str:
    """Normalize key to lowercase for case-insensitive lookup."""
    return str(key).strip().lower()


def _config_get(config: Any, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_set(config: Any, key: str, value) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def is_vlm_reward_model(model_name: Optional[str]) -> bool:
    if model_name is None:
        return False
    return str(model_name).strip().lower() in _VLM_REWARD_MODEL_ALIASES


def _resolve_vlm_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    looks_like_local_path = path.is_absolute() or model_path.startswith(".")
    if path.exists():
        return str(path)
    if looks_like_local_path:
        raise FileNotFoundError(
            f"VLM reward model path does not exist: {path}. "
            "If you intended to load a Hugging Face repo, pass the repo id instead."
        )
    return model_path


def _normalize_vlm_model_family(model_family: str) -> str:
    normalized = model_family.strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "gemma_3": "gemma3",
        "gemma3_vl": "gemma3",
        "gemma_3_vl": "gemma3",
        "gemma_4": "gemma4",
        "gemma4_vl": "gemma4",
        "gemma_4_vl": "gemma4",
        "internvl3": "internvl",
        "internvl_3": "internvl",
        "internvl3_hf": "internvl",
        "internvl_3_hf": "internvl",
        "unifiedreward2_qwen35": "qwen3_5",
        "unifiedreward_2_qwen35": "qwen3_5",
        "unifiedreward2_qwen3vl": "qwen3_vl",
        "unifiedreward_2_qwen3vl": "qwen3_vl",
        "unifiedreward2_qwen25": "qwen2_5_vl",
        "unifiedreward_2_qwen25": "qwen2_5_vl",
    }
    return aliases.get(normalized, normalized)


def _infer_vlm_model_family(model_path: str, model_family: Optional[str] = None) -> str:
    if model_family is not None:
        return _normalize_vlm_model_family(model_family)

    lowered = model_path.lower()
    if "qwen2.5" in lowered or "qwen2_5" in lowered:
        return "qwen2_5_vl"
    if "qwen3.5" in lowered or "qwen3_5" in lowered or "qwen35" in lowered:
        return "qwen3_5"
    if "qwen3vl" in lowered:
        return "qwen3_vl"
    if "qwen3" in lowered:
        return "qwen3_vl"
    if "gemma-3" in lowered or "gemma_3" in lowered or "gemma3" in lowered:
        return "gemma3"
    if "gemma-4" in lowered or "gemma_4" in lowered or "gemma4" in lowered:
        return "gemma4"
    if "internvl" in lowered:
        return "internvl"

    raise ValueError(
        "Unable to infer VLM reward model family from vlm_reward_model_path. "
        "Please set --vlm_reward_model_family explicitly."
    )


def _transformers_has_model_module(module_name: str) -> bool:
    import transformers

    return any(
        candidate.name == module_name
        for candidate in pkgutil.iter_modules(transformers.models.__path__)
    )


def _python_module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> Tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def _resize_frames(frames: torch.Tensor, max_frame_pixels: int, factor: int = 32) -> torch.Tensor:
    """
    Resize frames using the in-memory VLM preprocessing path.

    Args:
        frames: Tensor with shape [B, C, T, H, W].

    Returns:
        Tensor with shape [B, T, C, H, W].
    """
    batch_size, channels, num_frames, height, width = frames.shape
    frames_btchw = frames.permute(0, 2, 1, 3, 4)
    resized_height, resized_width = _smart_resize(
        num_frames=num_frames,
        height=height,
        width=width,
        factor=factor,
        min_pixels=16384,
        max_pixels=max_frame_pixels,
    )

    resized_videos = []
    for video in frames_btchw:
        resized_video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        resized_videos.append(resized_video)

    return torch.stack(resized_videos, dim=0)


def _parse_json_maybe(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_yes_no_label(value: str) -> str:
    stripped = value.strip()
    if stripped.lower() == "yes":
        return "Yes"
    if stripped.lower() == "no":
        return "No"
    return value


def _normalize_rating_1_5_label(value: str) -> str:
    stripped = value.strip()
    if stripped in {"1", "2", "3", "4", "5"}:
        return stripped
    try:
        score = float(stripped)
    except ValueError:
        return value
    if score.is_integer() and 1 <= int(score) <= 5:
        return str(int(score))
    return value


def _normalize_supervised_answer_label(value: str) -> str:
    normalized = _normalize_yes_no_label(value)
    if normalized in {"Yes", "No"}:
        return normalized
    return _normalize_rating_1_5_label(value)


def _is_supported_supervised_answer(value: str) -> bool:
    return value in {"Yes", "No", "1", "2", "3", "4", "5"}


def _normalize_supervised_answers_in_obj(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key == "answer" and isinstance(item, str):
                normalized[key] = _normalize_supervised_answer_label(item)
            else:
                normalized[key] = _normalize_supervised_answers_in_obj(item)
        return normalized

    if isinstance(value, list):
        return [_normalize_supervised_answers_in_obj(item) for item in value]

    if isinstance(value, tuple):
        return [_normalize_supervised_answers_in_obj(item) for item in value]

    return value


def _normalize_ref_answer_text(raw_answer) -> str:
    answer = raw_answer
    while isinstance(answer, (list, tuple)) and len(answer) == 1:
        answer = answer[0]

    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.lower() == "yes":
            return "Yes"
        if stripped.lower() == "no":
            return "No"
        parsed = _parse_json_maybe(answer)
        if parsed is answer:
            return answer
        if parsed is None:
            return ""
        normalized = _normalize_supervised_answers_in_obj(parsed)
        return json.dumps(normalized, ensure_ascii=False)

    if isinstance(answer, (dict, list, tuple)):
        normalized = _normalize_supervised_answers_in_obj(answer)
        return json.dumps(normalized, ensure_ascii=False)

    if answer is None:
        return ""
    return str(answer)


def _normalize_yes_no_question_text(question_text: str) -> str:
    if '"answer_type": "yes_no"' not in question_text:
        return question_text
    
    if LEGACY_YES_NO_UPPERCASE_RULE in question_text:
        question_text = question_text.replace(LEGACY_YES_NO_UPPERCASE_RULE, YES_NO_RULE)

    if YES_NO_RULE in question_text:
        return question_text
    
    anchor = "- Do not infer unseen details."
    if anchor in question_text:
        return question_text.replace(
            anchor,
            f"{anchor}\n{YES_NO_RULE}",
            1,
        )
    
    return f"{question_text}\n{YES_NO_RULE}"


def _normalize_question_item(item) -> Optional[str]:
    if item is None:
        return None
    if isinstance(item, str):
        stripped = item.strip()
        return stripped or None
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return None
        if len(item) == 1:
            return _normalize_question_item(item[0])
        return _normalize_question_item(item[0])
    return str(item)


def _normalize_questions(raw_questions) -> List[str]:
    questions = _parse_json_maybe(raw_questions)
    if questions is None:
        return []

    if isinstance(questions, dict):
        if "questions" in questions:
            questions = questions["questions"]
        elif "sub_questions" in questions:
            questions = questions["sub_questions"]
        else:
            item = _normalize_question_item(questions)
            return [item] if item else []

    if isinstance(questions, (list, tuple)):
        normalized = []
        for item in questions:
            normalized_item = _normalize_question_item(item)
            if normalized_item:
                normalized.append(normalized_item)
        return normalized

    if isinstance(questions, str):
        stripped = questions.strip()
        return [stripped] if stripped else []

    return [str(questions)]


def _normalize_ref_answers(raw_ref_answers) -> List[Any]:
    ref_answers = _parse_json_maybe(raw_ref_answers)
    if ref_answers is None:
        return []

    if isinstance(ref_answers, dict) and "ref_answers" in ref_answers:
        ref_answers = ref_answers["ref_answers"]

    if isinstance(ref_answers, (list, tuple)):
        return list(ref_answers)

    return [ref_answers]


def _extract_atomic_sub_questions(raw_sub_questions) -> List[Dict[str, str]]:
    parsed = _parse_json_maybe(raw_sub_questions)
    if parsed is None:
        return []

    if isinstance(parsed, dict) and "sub_questions" in parsed:
        parsed = parsed["sub_questions"]

    if not isinstance(parsed, (list, tuple)):
        return []

    items: List[Dict[str, str]] = []
    for idx, item in enumerate(parsed):
        if isinstance(item, dict):
            question_text = (
                item.get("question")
                or item.get("text")
                or item.get("prompt")
            )
            if not isinstance(question_text, str) or not question_text.strip():
                continue
            items.append(
                {
                    "id": str(item.get("id") or f"q{idx + 1}"),
                    "question": question_text.strip(),
                    "answer_type": str(item.get("answer_type") or "").strip().lower(),
                }
            )
            continue

        normalized_item = _normalize_question_item(item)
        if normalized_item:
            items.append(
                {
                    "id": f"q{idx + 1}",
                    "question": normalized_item,
                    "answer_type": "",
                }
            )

    return items


def _extract_sub_answer_targets(raw_ref_answers) -> Tuple[Dict[str, str], List[str]]:
    parsed = _parse_json_maybe(raw_ref_answers)
    if isinstance(parsed, list) and len(parsed) == 1:
        nested = _parse_json_maybe(parsed[0])
        if nested is not None:
            parsed = nested

    if isinstance(parsed, dict) and "ref_answers" in parsed:
        parsed = _parse_json_maybe(parsed["ref_answers"])

    answer_map: Dict[str, str] = {}
    ordered_answers: List[str] = []

    if isinstance(parsed, dict) and "sub_answers" in parsed:
        sub_answers = parsed.get("sub_answers", [])
        if isinstance(sub_answers, list):
            for idx, item in enumerate(sub_answers):
                if not isinstance(item, dict):
                    continue
                answer = item.get("answer")
                if answer is None:
                    continue
                normalized = _normalize_supervised_answer_label(str(answer))
                ordered_answers.append(normalized)
                answer_map[str(item.get("id") or f"q{idx + 1}")] = normalized
        return answer_map, ordered_answers

    fallback_answers = _normalize_ref_answers(parsed)
    ordered_answers = [
        _normalize_supervised_answer_label(str(answer))
        for answer in fallback_answers
        if answer is not None
    ]
    return answer_map, ordered_answers


def _format_atomic_yes_no_question(question_text: str) -> str:
    stripped = str(question_text).strip()
    if not stripped:
        return stripped
    if re.search(r"\byes\s+or\s+no\b", stripped, flags=re.IGNORECASE):
        return stripped
    return f'{stripped}\nAnswer with "Yes" or "No" only.'


def _format_atomic_rating_1_5_question(question_text: str) -> str:
    stripped = str(question_text).strip()
    if not stripped:
        return stripped
    if "1" in stripped and "5" in stripped and re.search(r"\brate\b|\bscore\b", stripped, flags=re.IGNORECASE):
        return stripped
    return (
        "Rate from 1 to 5 how well the video satisfies this visual criterion, "
        f"where 5 is best: {stripped}\n"
        'Answer with a single integer from "1" to "5" only.'
    )


def _format_atomic_supervised_question(question_text: str, answer_type: str, target_answer: str) -> str:
    normalized_answer_type = str(answer_type or "").strip().lower()
    if normalized_answer_type in {"rating_1_5", "rating", "score_1_5"} or target_answer in {"1", "2", "3", "4", "5"}:
        return _format_atomic_rating_1_5_question(question_text)
    return _format_atomic_yes_no_question(question_text)


class _SafeTemplateDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class VLMRewardScorer:
    """In-memory VLM reward scorer (teacher-forced QA likelihood and token credit)."""

    prefers_tensor_input = True

    def __init__(self, config: Any, device: torch.device, logger=None):
        self.config = config
        self.device = torch.device(device)
        self.runtime_device = torch.device(device)
        self.offload_device = torch.device("cpu")
        self.logger = logger
        self.is_model_offloaded = False

        model_path = _config_get(config, "vlm_reward_model_path")
        if not model_path:
            raise ValueError(
                "VLM reward model requested, but vlm_reward_model_path is empty."
            )

        self.model_path = _resolve_vlm_model_path(model_path)
        self.model_family = _infer_vlm_model_family(
            self.model_path,
            _config_get(config, "vlm_reward_model_family"),
        )
        self.prompt_template = (
            _config_get(config, "vlm_reward_prompt_template")
            or _DEFAULT_VLM_REWARD_PROMPT_TEMPLATE
        )
        self.num_frames = max(1, int(_config_get(config, "vlm_reward_num_frames", 8)))
        self.max_pixels = int(_config_get(config, "vlm_reward_max_pixels", 65536) or 65536)
        self.max_new_tokens = max(
            1, int(_config_get(config, "vlm_reward_max_new_tokens", 128))
        )
        self.eval_batch_size = max(
            0, int(_config_get(config, "vlm_reward_batch_size", 0) or 0)
        )
        self.missing_score = float(_config_get(config, "vlm_reward_missing_score", 0.0))

        reward_model_name = str(_config_get(config, "reward_model", "")).lower()
        raw_score_type = str(
            _config_get(
                config,
                "vlm_reward_score_type",
                "free_generation_yes"
                if reward_model_name in _VLM_FREE_GENERATION_ALIASES
                else "logprob",
            )
        ).lower()
        if raw_score_type in {"count", "ratio"}:
            self.score_mode = "free_generation_yes"
            self.free_generation_score_type = raw_score_type
        elif raw_score_type in {
            "logprob",
            "yes_ratio",
            "yes_no_margin",
            "free_generation_yes",
            "token_margin",
            "token_credit",
        }:
            self.score_mode = raw_score_type
            self.free_generation_score_type = "ratio"
        else:
            raise ValueError(
                f"Unsupported vlm_reward_score_type={raw_score_type!r}. "
                "Expected one of: 'logprob', 'yes_ratio', 'free_generation_yes', "
                "'count', 'ratio', 'yes_no_margin', 'token_margin', or 'token_credit'."
            )

        self.model, self.processor = self._load_model_and_processor()

    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if callable(log_fn):
            log_fn(message)

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def _move_model_to(self, target_device: torch.device) -> None:
        target_device = torch.device(target_device)
        current_device = self._get_model_device()
        if current_device == target_device:
            return
        self.model.to(target_device)
        if target_device.type == "cpu":
            self.is_model_offloaded = True
        else:
            self.is_model_offloaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_for_rollout(self) -> None:
        self._move_model_to(self.offload_device)
        self._log("info", f"[RewardDebug] Offloaded VLM reward model to {self.offload_device} before rollout.")

    def prepare_for_reward(self) -> None:
        self._move_model_to(self.runtime_device)
        self._log("info", f"[RewardDebug] Loaded VLM reward model onto {self.runtime_device} for reward scoring.")

    def cleanup_after_reward(self) -> None:
        self._move_model_to(self.offload_device)
        self._log("info", f"[RewardDebug] Offloaded VLM reward model back to {self.offload_device} after reward scoring.")

    def _iter_row_chunks(self, num_rows: int):
        if num_rows <= 0:
            return
        chunk_size = self.eval_batch_size if self.eval_batch_size > 0 else num_rows
        chunk_size = max(1, min(chunk_size, num_rows))
        for start in range(0, num_rows, chunk_size):
            yield start, min(start + chunk_size, num_rows)

    def _slice_model_inputs(
        self,
        inputs: Dict[str, Any],
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        chunk_inputs: Dict[str, Any] = {}
        video_grid = inputs.get("video_grid_thw")
        image_grid = inputs.get("image_grid_thw")

        video_slice = None
        if torch.is_tensor(video_grid):
            video_grid_cpu = video_grid.detach().cpu()
            video_start = (
                int(video_grid_cpu[:start].prod(dim=1).sum().item()) if start > 0 else 0
            )
            video_count = int(video_grid_cpu[start:end].prod(dim=1).sum().item())
            video_slice = slice(video_start, video_start + video_count)

        image_slice = None
        if torch.is_tensor(image_grid):
            image_grid_cpu = image_grid.detach().cpu()
            image_start = (
                int(image_grid_cpu[:start].prod(dim=1).sum().item()) if start > 0 else 0
            )
            image_count = int(image_grid_cpu[start:end].prod(dim=1).sum().item())
            image_slice = slice(image_start, image_start + image_count)

        for key, value in inputs.items():
            if torch.is_tensor(value):
                if (
                    key == "pixel_values"
                    and self._should_use_image_frame_processor()
                    and video_slice is None
                    and image_slice is None
                    and value.ndim >= 4
                ):
                    images_per_row = max(1, int(self.num_frames))
                    chunk_inputs[key] = value[
                        start * images_per_row : end * images_per_row
                    ]
                elif key == "pixel_values_videos" and video_slice is not None:
                    chunk_inputs[key] = value[video_slice]
                elif key == "video_grid_thw":
                    chunk_inputs[key] = value[start:end]
                elif key == "pixel_values" and image_slice is not None:
                    chunk_inputs[key] = value[image_slice]
                elif key == "image_grid_thw":
                    chunk_inputs[key] = value[start:end]
                else:
                    chunk_inputs[key] = value[start:end]
            else:
                chunk_inputs[key] = value
        return chunk_inputs

    def _load_model_and_processor(self):
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        load_kwargs = {
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }

        if self.model_family == "qwen2_5_vl":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(self.model_path)
        elif self.model_family == "qwen3_vl":
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(self.model_path)
        elif self.model_family == "qwen3_5":
            if not _transformers_has_model_module("qwen3_5"):
                raise ImportError(
                    "vlm_reward_model_family='qwen3_5' requires a newer transformers build "
                    "that exposes transformers.models.qwen3_5."
                )
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        elif self.model_family == "gemma3":
            if not _transformers_has_model_module("gemma3"):
                raise ImportError(
                    "vlm_reward_model_family='gemma3' requires a transformers build "
                    "with the gemma3 model module."
                )
            if not _python_module_exists("transformers.models.gemma3.processing_gemma3"):
                raise ImportError(
                    "vlm_reward_model_family='gemma3' requires the Gemma3 processor module "
                    "in transformers."
                )
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(self.model_path)
        elif self.model_family == "gemma4":
            if not _transformers_has_model_module("gemma4"):
                raise ImportError(
                    "vlm_reward_model_family='gemma4' requires a transformers build "
                    "with the gemma4 model module."
                )
            if not _python_module_exists("transformers.models.gemma4.processing_gemma4"):
                raise ImportError(
                    "vlm_reward_model_family='gemma4' requires the Gemma4 processor module "
                    "in transformers."
                )
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(self.model_path)
        elif self.model_family == "internvl":
            if not _transformers_has_model_module("internvl"):
                raise ImportError(
                    "vlm_reward_model_family='internvl' requires a transformers build "
                    "with the internvl model module."
                )
            if not _python_module_exists("transformers.models.internvl.processing_internvl"):
                raise ImportError(
                    "vlm_reward_model_family='internvl' requires the InternVL processor module "
                    "in transformers. Use the HF-format checkpoint such as OpenGVLab/InternVL3-1B-hf."
                )
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            raise ValueError(
                f"Unsupported vlm_reward_model_family={self.model_family!r}. "
                "Supported families: qwen2_5_vl, qwen3_vl, qwen3_5, gemma3, gemma4, internvl."
            )

        model.eval()
        model.requires_grad_(False)
        model.to(self.runtime_device)
        return model, processor

    def _normalize_reward_tensor(self, reward_inputs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(reward_inputs):
            raise TypeError(
                "VLM reward expects an in-memory tensor input "
                "with shape [B, C, T, H, W] or [B, C, H, W]."
            )

        frames = reward_inputs.detach().float().cpu()
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError(
                f"Expected reward tensor with 4 or 5 dims, got shape {tuple(frames.shape)}"
            )

        if frames.min().item() < 0.0:
            frames = torch.clamp((frames + 1.0) * 0.5, 0.0, 1.0)
        else:
            frames = torch.clamp(frames, 0.0, 1.0)
        return frames

    def _normalize_metadata_list(
        self,
        batch_size: int,
        metadata: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if metadata is None:
            return [{} for _ in range(batch_size)]

        normalized = []
        for idx in range(batch_size):
            current = metadata[idx] if idx < len(metadata) else None
            normalized.append(dict(current) if isinstance(current, dict) else {})
        return normalized

    def _build_default_questions_and_answers(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Tuple[List[str], List[Any]]:
        format_values = _SafeTemplateDict(prompt=prompt)
        for key, value in metadata.items():
            if value is None:
                continue
            format_values[str(key)] = value
        question_text = str(self.prompt_template).format_map(format_values)
        return [question_text], ["Yes"]

    def _get_sample_questions_and_answers(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Tuple[List[str], List[Any]]:
        questions = _normalize_questions(metadata.get("questions"))
        ref_answers = _normalize_ref_answers(metadata.get("ref_answers"))

        if not questions:
            return self._build_default_questions_and_answers(prompt, metadata)
        if not ref_answers:
            ref_answers = ["Yes"]
        return questions, ref_answers

    def _get_sample_atomic_questions_and_answers(
        self,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        raw_sub_questions = metadata.get("sub_questions")
        if raw_sub_questions is None:
            raw_sub_questions = metadata.get("questions")

        question_items = _extract_atomic_sub_questions(raw_sub_questions)
        answer_map, ordered_answers = _extract_sub_answer_targets(metadata.get("ref_answers"))

        if not question_items:
            questions, answers = self._build_default_questions_and_answers(prompt, metadata)
            normalized_answers = [
                _normalize_supervised_answer_label(str(answer))
                for answer in (answers or ["Yes"])
            ]
            question_ids = [f"q{idx + 1}" for idx in range(len(questions))]
            return questions, normalized_answers, question_ids

        questions: List[str] = []
        answers: List[str] = []
        question_ids: List[str] = []

        for idx, item in enumerate(question_items):
            answer = answer_map.get(item["id"])
            if answer is None and idx < len(ordered_answers):
                answer = ordered_answers[idx]
            if answer is None and len(ordered_answers) == 1:
                answer = ordered_answers[0]
            if answer is None:
                answer = "Yes"

            normalized_answer = _normalize_supervised_answer_label(str(answer))
            if not _is_supported_supervised_answer(normalized_answer):
                continue

            questions.append(
                _format_atomic_supervised_question(
                    item["question"],
                    item.get("answer_type", ""),
                    normalized_answer,
                )
            )
            answers.append(normalized_answer)
            question_ids.append(item["id"])

        if not questions:
            fallback_questions, fallback_answers = self._build_default_questions_and_answers(
                prompt, metadata
            )
            normalized_answers = [
                _normalize_supervised_answer_label(str(answer))
                for answer in (fallback_answers or ["Yes"])
            ]
            question_ids = [f"q{idx + 1}" for idx in range(len(fallback_questions))]
            return fallback_questions, normalized_answers, question_ids

        return questions, answers, question_ids

    def _uniform_sample_frames(self, sample_video_cthw: torch.Tensor) -> torch.Tensor:
        total_frames = sample_video_cthw.shape[1]
        if total_frames <= 0:
            raise ValueError("Reward input video contains zero frames.")

        frame_indices = (
            torch.linspace(0, total_frames - 1, self.num_frames).round().long()
        )
        sample_video_tchw = sample_video_cthw.permute(1, 0, 2, 3)
        return sample_video_tchw[frame_indices]

    def _load_reference_image_tensor(
        self,
        ref_image_path: Optional[str],
        target_height: int,
        target_width: int,
    ) -> Optional[torch.Tensor]:
        if not isinstance(ref_image_path, str) or not ref_image_path:
            return None
        if not os.path.exists(ref_image_path):
            return None

        with Image.open(ref_image_path) as image:
            ref_image = image.convert("RGB")
        ref_tensor = transforms.functional.pil_to_tensor(ref_image).float() / 255.0
        if ref_tensor.shape[-2:] != (target_height, target_width):
            ref_tensor = transforms.functional.resize(
                ref_tensor,
                [target_height, target_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
        return ref_tensor

    def _resize_single_video(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        frames_bcthw = frames_tchw.permute(1, 0, 2, 3).unsqueeze(0)
        return _resize_frames(
            frames_bcthw,
            self.max_pixels,
            factor=self._get_video_resize_factor(),
        )[0]

    def _get_video_resize_factor(self) -> int:
        image_processor = getattr(self.processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 0) or 0)
        merge_size = int(getattr(image_processor, "merge_size", 0) or 0)
        if patch_size > 0 and merge_size > 0:
            return patch_size * merge_size
        if self.model_family == "qwen2_5_vl":
            return 28
        return 32

    def _should_use_image_frame_processor(self) -> bool:
        return self.model_family in {"gemma3", "internvl"}

    def _should_use_native_video_processor_resize(self) -> bool:
        return self.model_family == "gemma4"

    def _get_processor_video_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "do_rescale": False,
            "do_sample_frames": False,
        }
        if self._should_use_native_video_processor_resize():
            kwargs["do_resize"] = True
        else:
            kwargs["do_resize"] = False
        return kwargs

    def _get_processor_image_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "do_rescale": False,
            "do_resize": True,
        }
        if self.model_family == "internvl":
            kwargs["do_resize"] = False
        return kwargs

    def _get_processor_image_size(self) -> Tuple[int, int]:
        image_processor = getattr(self.processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        if isinstance(size, Mapping):
            height = int(size.get("height", 448) or 448)
            width = int(size.get("width", 448) or 448)
            return height, width
        height = int(getattr(size, "height", 448) or 448)
        width = int(getattr(size, "width", 448) or 448)
        return height, width

    def _prepare_sample_image_frames(
        self,
        sampled_frames_tchw: torch.Tensor,
    ) -> List[torch.Tensor]:
        if self.model_family == "internvl":
            target_height, target_width = self._get_processor_image_size()
            if sampled_frames_tchw.shape[-2:] != (target_height, target_width):
                sampled_frames_tchw = transforms.functional.resize(
                    sampled_frames_tchw,
                    [target_height, target_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
        return [
            frame.detach().contiguous()
            for frame in sampled_frames_tchw
        ]

    def _build_gemma3_frame_content(
        self,
        question_text: str,
        num_frames: int,
    ) -> List[Dict[str, str]]:
        content: List[Dict[str, str]] = [{"type": "image"} for _ in range(num_frames)]
        content.append(
            {
                "type": "text",
                "text": (
                    "The images are uniformly sampled frames from the generated video "
                    "in chronological order. "
                    f"{question_text}"
                ),
            }
        )
        return content

    def _prepare_sample_videos(
        self,
        sampled_frames_tchw: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._should_use_native_video_processor_resize():
            base_video = sampled_frames_tchw.contiguous()
            aux_frames = torch.cat(
                [sampled_frames_tchw[:1], sampled_frames_tchw[-1:]],
                dim=0,
            )
            ref_frame = self._load_reference_image_tensor(
                metadata.get("ref_image_path"),
                target_height=sampled_frames_tchw.shape[-2],
                target_width=sampled_frames_tchw.shape[-1],
            )
            if ref_frame is not None:
                aux_frames = torch.cat([aux_frames, ref_frame.unsqueeze(0)], dim=0)
            return base_video, aux_frames.contiguous()

        low_res_video = self._resize_single_video(sampled_frames_tchw)

        high_res_frames = torch.cat(
            [sampled_frames_tchw[:1], sampled_frames_tchw[-1:]],
            dim=0,
        )
        ref_frame = self._load_reference_image_tensor(
            metadata.get("ref_image_path"),
            target_height=sampled_frames_tchw.shape[-2],
            target_width=sampled_frames_tchw.shape[-1],
        )
        if ref_frame is not None:
            high_res_frames = torch.cat(
                [high_res_frames, ref_frame.unsqueeze(0)],
                dim=0,
            )

        return low_res_video, self._resize_single_video(high_res_frames)

    def _choose_video_for_question(
        self,
        question_text: str,
        low_res_video: torch.Tensor,
        physics_video: torch.Tensor,
    ) -> torch.Tensor:
        if "physics-related" in question_text.lower():
            return physics_video
        return low_res_video

    def _find_last_subsequence_start(self, sequence, pattern) -> int:
        if len(pattern) == 0 or len(pattern) > len(sequence):
            return -1
        for start in range(len(sequence) - len(pattern), -1, -1):
            if sequence[start : start + len(pattern)] == pattern:
                return start
        return -1

    def _find_subsequence_start(self, sequence, pattern) -> int:
        if len(pattern) == 0 or len(pattern) > len(sequence):
            return -1
        for start in range(len(sequence) - len(pattern) + 1):
            if sequence[start : start + len(pattern)] == pattern:
                return start
        return -1

    def _fill_labels_from_answer_text(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sample_indices,
        answer_text_by_sample,
    ) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        filled_labels = labels.clone()

        for sample_idx in sample_indices:
            answer_text = answer_text_by_sample.get(sample_idx)
            if not answer_text:
                continue

            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            if len(answer_ids) == 0:
                continue

            if attention_mask is not None:
                valid_positions = torch.nonzero(
                    attention_mask[sample_idx] != 0
                ).squeeze(-1)
            else:
                valid_positions = torch.arange(
                    input_ids.shape[1], device=input_ids.device, dtype=torch.long
                )
            if valid_positions.numel() == 0:
                continue

            sample_input_ids = input_ids[sample_idx, valid_positions].tolist()
            answer_start = self._find_last_subsequence_start(
                sample_input_ids,
                answer_ids,
            )
            if answer_start < 0:
                continue

            answer_end = answer_start + len(answer_ids)
            keep_positions = valid_positions[answer_start:answer_end]
            filled_labels[sample_idx, keep_positions] = input_ids[sample_idx, keep_positions]

        return filled_labels

    def _extract_ft_answer_value_char_spans(self, answer_text: str):
        spans = []
        for pattern in FT_ANSWER_VALUE_PATTERNS:
            spans.extend(match.span("answer") for match in pattern.finditer(answer_text))
        spans.sort(key=lambda span: span[0])
        return spans

    def _select_token_offsets_from_char_spans(self, text: str, char_spans):
        tokenizer = self.processor.tokenizer
        try:
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_offsets = encoding["offset_mapping"]
            return {
                token_offset
                for token_offset, (char_start, char_end) in enumerate(token_offsets)
                for value_start, value_end in char_spans
                if char_start < value_end and char_end > value_start
            }
        except (NotImplementedError, TypeError, KeyError):
            selected_offsets = set()
            for value_start, value_end in char_spans:
                prefix_ids = tokenizer(
                    text[:value_start],
                    add_special_tokens=False,
                )["input_ids"]
                prefix_value_ids = tokenizer(
                    text[:value_end],
                    add_special_tokens=False,
                )["input_ids"]
                selected_offsets.update(range(len(prefix_ids), len(prefix_value_ids)))
            return selected_offsets

    def _keep_ft_answer_value_tokens_only(
        self,
        labels: torch.Tensor,
        row_indices,
        answer_text_by_row,
    ) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        filtered_labels = labels.clone()

        for row_idx in row_indices:
            answer_text = answer_text_by_row.get(row_idx)
            if not answer_text or "sub_answers" not in answer_text:
                continue

            value_char_spans = self._extract_ft_answer_value_char_spans(answer_text)
            if not value_char_spans:
                continue

            answer_positions = torch.nonzero(filtered_labels[row_idx] != -100).squeeze(-1)
            if answer_positions.numel() == 0:
                continue

            answer_token_ids = filtered_labels[row_idx, answer_positions].tolist()
            full_answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            if len(full_answer_ids) == 0:
                continue

            answer_start = self._find_subsequence_start(answer_token_ids, full_answer_ids)
            if answer_start < 0:
                continue

            selected_offsets = self._select_token_offsets_from_char_spans(
                text=answer_text,
                char_spans=value_char_spans,
            )
            selected_offsets = {
                answer_start + token_offset
                for token_offset in selected_offsets
                if answer_start + token_offset < answer_positions.numel()
            }
            if not selected_offsets:
                continue

            keep_positions = answer_positions[sorted(selected_offsets)]
            filtered_labels[row_idx] = -100
            filtered_labels[row_idx, keep_positions] = labels[row_idx, keep_positions]

        return filtered_labels

    def _build_teacher_forced_reward_inputs(
        self,
        reward_inputs: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict[str, Any]],
        prefer_atomic_questions: bool = False,
    ):
        batch_frames = self._normalize_reward_tensor(reward_inputs)

        texts = []
        video_batch = []
        row_sample_indices = []
        prompt_keys = []
        ref_answer_texts = []
        answer_text_by_row = {}
        question_ids = []

        for sample_idx, prompt in enumerate(prompts):
            sampled_frames_tchw = self._uniform_sample_frames(batch_frames[sample_idx])
            image_frames = None
            if self._should_use_image_frame_processor():
                image_frames = self._prepare_sample_image_frames(sampled_frames_tchw)
                low_res_video = physics_video = None
            else:
                low_res_video, physics_video = self._prepare_sample_videos(
                    sampled_frames_tchw,
                    metadata[sample_idx],
                )
            if prefer_atomic_questions:
                questions, ref_answers, sample_question_ids = (
                    self._get_sample_atomic_questions_and_answers(
                        prompt,
                        metadata[sample_idx],
                    )
                )
            else:
                questions, ref_answers = self._get_sample_questions_and_answers(
                    prompt,
                    metadata[sample_idx],
                )
                sample_question_ids = [f"q{idx + 1}" for idx in range(len(questions))]

            for question_idx, raw_question in enumerate(questions):
                ref_answer = (
                    ref_answers[question_idx]
                    if question_idx < len(ref_answers)
                    else ref_answers[0]
                )
                ref_answer_text = _normalize_ref_answer_text(ref_answer)
                question_text = _normalize_yes_no_question_text(raw_question)
                if self._should_use_image_frame_processor():
                    user_content = self._build_gemma3_frame_content(
                        question_text=question_text,
                        num_frames=len(image_frames or []),
                    )
                    visual_input = image_frames
                else:
                    user_content = [
                        {"type": "video", "video": "<video>"},
                        {"type": "text", "text": question_text},
                    ]
                    visual_input = self._choose_video_for_question(
                        question_text=question_text,
                        low_res_video=low_res_video,
                        physics_video=physics_video,
                    )
                msgs = [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": ref_answer_text},
                        ],
                    },
                ]
                texts.append(
                    self.processor.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                video_batch.append(visual_input)
                row_sample_indices.append(sample_idx)
                prompt_keys.append(question_text)
                ref_answer_texts.append(ref_answer_text)
                answer_text_by_row[len(texts) - 1] = ref_answer_text
                question_ids.append(
                    sample_question_ids[question_idx]
                    if question_idx < len(sample_question_ids)
                    else f"q{question_idx + 1}"
                )

        if len(texts) == 0:
            return (
                None,
                None,
                row_sample_indices,
                prompt_keys,
                ref_answer_texts,
                answer_text_by_row,
                question_ids,
            )

        if self._should_use_image_frame_processor():
            inputs = self.processor(
                text=texts,
                images=video_batch,
                padding=True,
                return_tensors="pt",
                **self._get_processor_image_kwargs(),
            ).to(self._get_model_device())
        else:
            inputs = self.processor(
                text=texts,
                images=None,
                videos=video_batch,
                padding=True,
                return_tensors="pt",
                **self._get_processor_video_kwargs(),
            ).to(self._get_model_device())

        labels = torch.full_like(inputs["input_ids"], -100)
        row_indices = list(range(len(texts)))
        labels = self._fill_labels_from_answer_text(
            labels=labels,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            sample_indices=row_indices,
            answer_text_by_sample=answer_text_by_row,
        )
        labels = self._keep_ft_answer_value_tokens_only(
            labels=labels,
            row_indices=row_indices,
            answer_text_by_row=answer_text_by_row,
        )

        return (
            inputs,
            labels,
            row_sample_indices,
            prompt_keys,
            ref_answer_texts,
            answer_text_by_row,
            question_ids,
        )

    def _build_free_generation_reward_inputs(
        self,
        reward_inputs: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict[str, Any]],
    ):
        batch_frames = self._normalize_reward_tensor(reward_inputs)

        texts = []
        video_batch = []
        row_sample_indices = []
        prompt_keys = []

        for sample_idx, prompt in enumerate(prompts):
            sampled_frames_tchw = self._uniform_sample_frames(batch_frames[sample_idx])
            image_frames = None
            if self._should_use_image_frame_processor():
                image_frames = self._prepare_sample_image_frames(sampled_frames_tchw)
                low_res_video = physics_video = None
            else:
                low_res_video, physics_video = self._prepare_sample_videos(
                    sampled_frames_tchw,
                    metadata[sample_idx],
                )
            questions, _ = self._get_sample_questions_and_answers(
                prompt,
                metadata[sample_idx],
            )

            for raw_question in questions:
                question_text = _normalize_yes_no_question_text(raw_question)
                if self._should_use_image_frame_processor():
                    user_content = self._build_gemma3_frame_content(
                        question_text=question_text,
                        num_frames=len(image_frames or []),
                    )
                    visual_input = image_frames
                else:
                    user_content = [
                        {"type": "video", "video": "<video>"},
                        {"type": "text", "text": question_text},
                    ]
                    visual_input = self._choose_video_for_question(
                        question_text=question_text,
                        low_res_video=low_res_video,
                        physics_video=physics_video,
                    )
                msgs = [
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ]
                texts.append(
                    self.processor.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                video_batch.append(visual_input)
                row_sample_indices.append(sample_idx)
                prompt_keys.append(question_text)

        return texts, video_batch, row_sample_indices, prompt_keys

    def _normalize_keyword_groups(self, keyword_groups):
        normalized = []
        for keyword_text in keyword_groups or []:
            matches = re.findall(r"\b(Yes|No)\b", str(keyword_text), flags=re.IGNORECASE)
            if matches:
                normalized.extend(_normalize_yes_no_label(match) for match in matches)
                continue

            stripped = str(keyword_text).strip().strip("\"'.:,;[]{}()")
            if stripped:
                normalized.append(_normalize_yes_no_label(stripped))
        return normalized

    def _extract_free_generation_keywords(self, generated_text: str):
        text = generated_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

        json_candidates = []
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidates.append(text[start_idx : end_idx + 1])
        json_candidates.append(text)

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                sub_answers = parsed.get("sub_answers", [])
                if isinstance(sub_answers, list):
                    keywords = []
                    for item in sub_answers:
                        if not isinstance(item, dict):
                            continue
                        answer = item.get("answer")
                        if isinstance(answer, str):
                            keywords.append(_normalize_yes_no_label(answer))
                    if keywords:
                        return keywords

        answer_matches = re.findall(
            r"""['"]answer['"]\s*:\s*['"]?(Yes|No)['"]?""",
            text,
            flags=re.IGNORECASE,
        )
        if answer_matches:
            return [_normalize_yes_no_label(match) for match in answer_matches]

        fallback_matches = re.findall(r"\b(Yes|No)\b", text, flags=re.IGNORECASE)
        return [_normalize_yes_no_label(match) for match in fallback_matches]

    def _decode_prediction_keywords(self, shift_logits, shift_labels):
        tokenizer = self.processor.tokenizer
        pred_token_ids = shift_logits.argmax(dim=-1)
        token_mask = shift_labels != -100
        pred_sentences = []
        pred_keywords = []

        for row_idx in range(pred_token_ids.shape[0]):
            row_ids = pred_token_ids[row_idx][token_mask[row_idx]].tolist()
            if row_ids:
                pred_text = tokenizer.decode(row_ids, skip_special_tokens=True).strip()
            else:
                pred_text = ""
            pred_sentences.append(pred_text)
            pred_keywords.append(self._extract_free_generation_keywords(pred_text))

        return pred_sentences, pred_keywords

    def _find_visual_input_key(self, model_inputs: Dict[str, Any]) -> Optional[str]:
        for key in ("pixel_values_videos", "pixel_values_video", "pixel_values"):
            value = model_inputs.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                return key
        return None

    def _prepare_row_inputs_for_gradient(
        self,
        row_inputs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        visual_key = self._find_visual_input_key(row_inputs)
        prepared_inputs: Dict[str, Any] = {}

        for key, value in row_inputs.items():
            if not torch.is_tensor(value):
                prepared_inputs[key] = value
                continue

            if key == visual_key:
                prepared_inputs[key] = value.detach().clone().requires_grad_(True)
            else:
                prepared_inputs[key] = value

        return prepared_inputs, visual_key

    def _normalize_frame_importance(self, frame_scores: torch.Tensor) -> List[float]:
        frame_scores = torch.nan_to_num(
            frame_scores.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if frame_scores.numel() == 0:
            return []

        total = frame_scores.sum()
        if not torch.isfinite(total) or total <= 0:
            frame_scores = torch.ones_like(frame_scores) / max(int(frame_scores.numel()), 1)
        else:
            frame_scores = frame_scores / total
        return frame_scores.cpu().tolist()

    def _compute_frame_importance_from_visual_grad(
        self,
        visual_grad: Optional[torch.Tensor],
        model_inputs: Dict[str, Any],
    ) -> List[float]:
        if visual_grad is None or not torch.is_tensor(visual_grad):
            return []

        grad_abs = torch.nan_to_num(
            visual_grad.detach().abs().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if grad_abs.numel() == 0:
            return []

        video_grid = model_inputs.get("video_grid_thw")
        if torch.is_tensor(video_grid) and video_grid.numel() >= 3:
            grid = video_grid[0] if video_grid.ndim > 1 else video_grid
            try:
                t_dim, h_dim, w_dim = [int(x) for x in grid[:3].tolist()]
            except Exception:
                t_dim = h_dim = w_dim = 0
            token_scores = grad_abs
            while token_scores.ndim > 1:
                token_scores = token_scores.sum(dim=-1)
            if t_dim > 0 and h_dim > 0 and w_dim > 0 and token_scores.numel() == t_dim * h_dim * w_dim:
                frame_scores = token_scores.view(t_dim, h_dim, w_dim).sum(dim=(1, 2))
                return self._normalize_frame_importance(frame_scores)

        if grad_abs.ndim == 5:
            if grad_abs.shape[0] == 1 and grad_abs.shape[1] <= grad_abs.shape[2]:
                frame_scores = grad_abs.sum(dim=(0, 2, 3, 4))
                return self._normalize_frame_importance(frame_scores)
            if grad_abs.shape[0] == 1:
                frame_scores = grad_abs.sum(dim=(0, 1, 3, 4))
                return self._normalize_frame_importance(frame_scores)

        if grad_abs.ndim == 4:
            if self._should_use_image_frame_processor():
                frame_scores = grad_abs.sum(dim=(1, 2, 3))
                return self._normalize_frame_importance(frame_scores)
            if grad_abs.shape[0] <= grad_abs.shape[1]:
                frame_scores = grad_abs.sum(dim=(1, 2, 3))
                return self._normalize_frame_importance(frame_scores)
            frame_scores = grad_abs.sum(dim=(0, 2, 3))
            return self._normalize_frame_importance(frame_scores)

        return []

    def _normalize_credit_map(self, credit_map: torch.Tensor) -> Dict[str, Any]:
        credit_map = torch.nan_to_num(
            credit_map.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if credit_map.ndim != 3 or credit_map.numel() == 0:
            return {}

        # Keep reward records bounded. Qwen video-token maps are normally small;
        # raw image-frame gradients can be 448x448 per frame, which is too large
        # to serialize for every question.
        max_spatial_bins = 32
        if credit_map.shape[-2] > max_spatial_bins or credit_map.shape[-1] > max_spatial_bins:
            target_h = min(int(credit_map.shape[-2]), max_spatial_bins)
            target_w = min(int(credit_map.shape[-1]), max_spatial_bins)
            credit_map = F.adaptive_avg_pool3d(
                credit_map.view(1, 1, *credit_map.shape),
                output_size=(int(credit_map.shape[0]), target_h, target_w),
            ).view(int(credit_map.shape[0]), target_h, target_w)

        total = credit_map.sum()
        if not torch.isfinite(total) or total <= 0:
            credit_map = torch.full_like(credit_map, 1.0 / float(max(credit_map.numel(), 1)))
        else:
            credit_map = credit_map / total.clamp_min(1e-6)
        return {
            "values": credit_map.flatten().cpu().tolist(),
            "shape": [int(x) for x in credit_map.shape],
        }

    def _compute_credit_map_from_visual_grad(
        self,
        visual_grad: Optional[torch.Tensor],
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if visual_grad is None or not torch.is_tensor(visual_grad):
            return {}

        grad_abs = torch.nan_to_num(
            visual_grad.detach().abs().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if grad_abs.numel() == 0:
            return {}

        video_grid = model_inputs.get("video_grid_thw")
        if torch.is_tensor(video_grid) and video_grid.numel() >= 3:
            grid = video_grid[0] if video_grid.ndim > 1 else video_grid
            try:
                t_dim, h_dim, w_dim = [int(x) for x in grid[:3].tolist()]
            except Exception:
                t_dim = h_dim = w_dim = 0
            token_scores = grad_abs
            while token_scores.ndim > 1:
                token_scores = token_scores.sum(dim=-1)
            if t_dim > 0 and h_dim > 0 and w_dim > 0 and token_scores.numel() == t_dim * h_dim * w_dim:
                # Input gradients retain Qwen's block-major patch order. Restore
                # coordinates BEFORE normalization/pooling and window3 routing.
                if self.model_family in {"qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
                    spatial_map = restore_qwen_video_grid(
                        token_scores, (t_dim, h_dim, w_dim), self.processor
                    )
                else:
                    spatial_map = token_scores.view(t_dim, h_dim, w_dim)
                return self._normalize_credit_map(spatial_map)

        if grad_abs.ndim == 5:
            if grad_abs.shape[0] == 1 and grad_abs.shape[1] <= grad_abs.shape[2]:
                return self._normalize_credit_map(grad_abs[0].sum(dim=1))
            if grad_abs.shape[0] == 1:
                return self._normalize_credit_map(grad_abs[0].sum(dim=0))

        if grad_abs.ndim == 4:
            if self._should_use_image_frame_processor():
                return self._normalize_credit_map(grad_abs.sum(dim=1))
            if grad_abs.shape[0] <= grad_abs.shape[1]:
                return self._normalize_credit_map(grad_abs.sum(dim=1))

        return {}

    def _get_supervised_answer_token_ids(
        self,
        answer_text: str,
    ) -> Tuple[Optional[str], Optional[str], List[int], List[int]]:
        target_answer = _normalize_supervised_answer_label(str(answer_text))
        if not _is_supported_supervised_answer(target_answer):
            return None, None, [], []

        opposite_answer = None
        if target_answer in {"Yes", "No"}:
            opposite_answer = "No" if target_answer == "Yes" else "Yes"
        tokenizer = self.processor.tokenizer
        target_ids = tokenizer(target_answer, add_special_tokens=False)["input_ids"]
        opposite_ids = (
            tokenizer(opposite_answer, add_special_tokens=False)["input_ids"]
            if opposite_answer is not None
            else []
        )
        return target_answer, opposite_answer, target_ids, opposite_ids

    def _extract_ft_answer_value_slots(self, answer_text: str) -> List[Dict[str, Any]]:
        spans = self._extract_ft_answer_value_char_spans(answer_text)
        if not spans:
            normalized = _normalize_supervised_answer_label(str(answer_text))
            if _is_supported_supervised_answer(normalized):
                return [{"id": "q1", "answer": normalized, "span": (0, len(answer_text))}]
            return []

        parsed = _parse_json_maybe(answer_text)
        sub_answers = []
        if isinstance(parsed, dict):
            maybe_sub_answers = parsed.get("sub_answers")
            if isinstance(maybe_sub_answers, list):
                sub_answers = maybe_sub_answers

        slots = []
        for idx, span in enumerate(spans):
            answer = answer_text[span[0] : span[1]]
            slot_id = f"q{idx + 1}"
            if idx < len(sub_answers) and isinstance(sub_answers[idx], dict):
                slot_id = str(sub_answers[idx].get("id") or slot_id)
                answer = sub_answers[idx].get("answer", answer)
            normalized_answer = _normalize_supervised_answer_label(str(answer))
            if not _is_supported_supervised_answer(normalized_answer):
                continue
            slots.append({"id": slot_id, "answer": normalized_answer, "span": span})
        return slots

    def _get_ft_answer_value_token_slots(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        row_idx: int,
        answer_text: str,
    ) -> List[Dict[str, Any]]:
        slots = self._extract_ft_answer_value_slots(answer_text)
        if not slots:
            return []

        tokenizer = self.processor.tokenizer
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if len(answer_ids) == 0:
            return []

        if attention_mask is not None:
            valid_positions = torch.nonzero(attention_mask[row_idx] != 0).squeeze(-1)
        else:
            valid_positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
        if valid_positions.numel() == 0:
            return []

        sample_input_ids = input_ids[row_idx, valid_positions].tolist()
        answer_start = self._find_last_subsequence_start(sample_input_ids, answer_ids)
        if answer_start < 0:
            return []

        token_slots: List[Dict[str, Any]] = []
        for slot in slots:
            selected_offsets = self._select_token_offsets_from_char_spans(
                text=answer_text,
                char_spans=[slot["span"]],
            )
            positions = []
            for token_offset in sorted(selected_offsets):
                position_idx = answer_start + token_offset
                if 0 <= position_idx < valid_positions.numel():
                    positions.append(valid_positions[position_idx])
            if not positions:
                continue
            token_slots.append(
                {
                    "id": slot["id"],
                    "answer": slot["answer"],
                    "positions": torch.stack(positions).to(device=input_ids.device),
                }
            )
        return token_slots

    def _score_answer_slot_from_log_probs(
        self,
        row_log_probs: torch.Tensor,
        slot_positions: torch.Tensor,
        target_answer: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[float], List[float]]:
        target_answer, _, target_ids, opposite_ids = self._get_supervised_answer_token_ids(target_answer)
        if target_answer is None or not target_ids:
            return None, None, [], []

        shift_positions = (slot_positions.to(row_log_probs.device).long() - 1).clamp_min(0)
        shift_positions = shift_positions[shift_positions < row_log_probs.shape[0]]
        if opposite_ids:
            use_count = min(int(shift_positions.numel()), len(target_ids), len(opposite_ids))
        else:
            use_count = min(int(shift_positions.numel()), len(target_ids))
        if use_count <= 0:
            return None, None, [], []

        shift_positions = shift_positions[:use_count]
        selected_log_probs = row_log_probs[shift_positions]
        target_id_tensor = torch.tensor(
            target_ids[:use_count],
            device=row_log_probs.device,
            dtype=torch.long,
        ).unsqueeze(-1)
        target_values = selected_log_probs.gather(-1, target_id_tensor).squeeze(-1)
        logprob_score = target_values.mean()
        opposite_values = None
        margin_score = logprob_score
        if opposite_ids:
            opposite_id_tensor = torch.tensor(
                opposite_ids[:use_count],
                device=row_log_probs.device,
                dtype=torch.long,
            ).unsqueeze(-1)
            opposite_values = selected_log_probs.gather(-1, opposite_id_tensor).squeeze(-1)
            margin_score = logprob_score - opposite_values.mean()
        return (
            logprob_score,
            margin_score,
            target_values.detach().cpu().tolist(),
            opposite_values.detach().cpu().tolist() if opposite_values is not None else [],
        )

    def _decode_slot_prediction(
        self,
        shift_logits: torch.Tensor,
        slot_positions: torch.Tensor,
    ) -> str:
        shift_positions = (slot_positions.to(shift_logits.device).long() - 1).clamp_min(0)
        shift_positions = shift_positions[shift_positions < shift_logits.shape[0]]
        if shift_positions.numel() == 0:
            return ""
        pred_ids = torch.argmax(shift_logits[shift_positions], dim=-1)
        return self.processor.tokenizer.decode(
            pred_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        ).strip()

    def _compute_structured_slot_rewards(
        self,
        reward_inputs: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict[str, Any]],
        *,
        selected_score: str,
        compute_gradient: bool,
    ):
        build_outputs = self._build_teacher_forced_reward_inputs(
            reward_inputs=reward_inputs,
            prompts=prompts,
            metadata=metadata,
            prefer_atomic_questions=False,
        )
        (
            inputs,
            labels,
            row_sample_indices,
            prompt_keys,
            ref_answer_texts,
            _,
            _,
        ) = build_outputs

        batch_size = len(prompts)
        if inputs is None or labels is None:
            zeros = [0.0] * batch_size
            return {
                "avg": zeros,
                "vlm_logprob": zeros,
                "vlm_yes_no_margin": zeros,
                "vlm_token_credit": zeros,
                "vlm_yes_ratio": zeros,
                "vlm_num_questions": [0.0] * batch_size,
            }, {"records": []}

        slot_reward_values: List[float] = []
        slot_logprob_values: List[float] = []
        slot_margin_values: List[float] = []
        slot_yes_ratio_values: List[float] = []
        slot_sample_indices: List[int] = []
        records: List[Dict[str, Any]] = []

        for row_idx in range(len(row_sample_indices)):
            row_inputs = self._slice_model_inputs(inputs, row_idx, row_idx + 1)
            row_labels = labels[row_idx : row_idx + 1]
            prepared_inputs, visual_key = (
                self._prepare_row_inputs_for_gradient(row_inputs)
                if compute_gradient
                else (row_inputs, None)
            )

            context = torch.enable_grad() if compute_gradient else torch.inference_mode()
            try:
                with context:
                    outputs = self.model.forward(
                        **prepared_inputs,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
                    logits = outputs.logits
                    row_labels = row_labels.to(logits.device)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = row_labels[..., 1:].contiguous()
                    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
                    row_log_probs = log_probs[0]

                    token_slots = self._get_ft_answer_value_token_slots(
                        input_ids=row_inputs["input_ids"],
                        attention_mask=row_inputs.get("attention_mask"),
                        row_idx=0,
                        answer_text=ref_answer_texts[row_idx],
                    )
                    if not token_slots:
                        token_positions = torch.nonzero(shift_labels[0] != -100).squeeze(-1) + 1
                        target_answer = _normalize_supervised_answer_label(ref_answer_texts[row_idx])
                        if _is_supported_supervised_answer(target_answer) and token_positions.numel() > 0:
                            token_slots = [
                                {
                                    "id": "q1",
                                    "answer": target_answer,
                                    "positions": token_positions.to(row_inputs["input_ids"].device),
                                }
                            ]

                    for slot_idx, slot in enumerate(token_slots):
                        logprob_score, margin_score, target_tokens, opposite_tokens = (
                            self._score_answer_slot_from_log_probs(
                                row_log_probs=row_log_probs,
                                slot_positions=slot["positions"],
                                target_answer=slot["answer"],
                            )
                        )
                        if logprob_score is None or margin_score is None:
                            continue

                        selected_tensor = (
                            margin_score if selected_score == "margin" else logprob_score
                        )
                        selected_value = float(selected_tensor.detach().cpu().item())
                        logprob_value = float(logprob_score.detach().cpu().item())
                        margin_value = float(margin_score.detach().cpu().item())
                        pred_answer = self._decode_slot_prediction(
                            shift_logits=shift_logits[0],
                            slot_positions=slot["positions"],
                        )
                        normalized_pred = _normalize_supervised_answer_label(pred_answer)
                        yes_ratio_value = 1.0 if normalized_pred == slot["answer"] else 0.0

                        frame_importance: List[float] = []
                        credit_map: Dict[str, Any] = {}
                        top_frames: List[int] = []
                        if compute_gradient and visual_key is not None:
                            visual_grad = torch.autograd.grad(
                                selected_tensor,
                                prepared_inputs[visual_key],
                                retain_graph=slot_idx < len(token_slots) - 1,
                                create_graph=False,
                                allow_unused=True,
                            )[0]
                            frame_importance = self._compute_frame_importance_from_visual_grad(
                                visual_grad=visual_grad,
                                model_inputs=prepared_inputs,
                            )
                            credit_map = self._compute_credit_map_from_visual_grad(
                                visual_grad=visual_grad,
                                model_inputs=prepared_inputs,
                            )
                            if frame_importance:
                                top_frames = sorted(
                                    range(len(frame_importance)),
                                    key=lambda idx: frame_importance[idx],
                                    reverse=True,
                                )[: min(3, len(frame_importance))]

                        slot_sample_idx = row_sample_indices[row_idx]
                        slot_sample_indices.append(slot_sample_idx)
                        slot_reward_values.append(selected_value)
                        slot_logprob_values.append(logprob_value)
                        slot_margin_values.append(margin_value)
                        slot_yes_ratio_values.append(yes_ratio_value)
                        records.append(
                            {
                                "sample_idx": slot_sample_idx,
                                "question_id": slot["id"],
                                "question": prompt_keys[row_idx],
                                "ref_answer": ref_answer_texts[row_idx],
                                "target_answer": slot["answer"],
                                "pred_sentence": pred_answer,
                                "normalized_pred_keywords": [normalized_pred] if _is_supported_supervised_answer(normalized_pred) else [],
                                "target_logprob_tokens": target_tokens,
                                "opposite_logprob_tokens": opposite_tokens,
                                "logprob_reward": logprob_value,
                                "yes_no_margin_reward": margin_value,
                                "token_credit_reward": logprob_value if compute_gradient else None,
                                "yes_ratio_reward": yes_ratio_value,
                                "frame_importance": frame_importance,
                                "credit_map": credit_map.get("values", []),
                                "credit_map_shape": credit_map.get("shape", []),
                                "top_frames": top_frames,
                                "selected_reward": selected_value,
                            }
                        )
            except Exception as exc:
                self._log(
                    "warning",
                    f"[RewardDebug] structured slot reward fallback on row {row_idx}: {exc}",
                )

        sample_rewards, sample_counts = self._reduce_rows_to_sample_level(
            slot_reward_values,
            slot_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_logprobs, _ = self._reduce_rows_to_sample_level(
            slot_logprob_values,
            slot_sample_indices,
            batch_size,
            default_value=0.0,
        )
        sample_margins, _ = self._reduce_rows_to_sample_level(
            slot_margin_values,
            slot_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_yes_ratios, _ = self._reduce_rows_to_sample_level(
            slot_yes_ratio_values,
            slot_sample_indices,
            batch_size,
            default_value=0.0,
        )

        return {
            "avg": sample_rewards,
            "vlm_logprob": sample_logprobs,
            "vlm_yes_no_margin": sample_margins,
            "vlm_token_credit": sample_rewards if compute_gradient else [0.0] * batch_size,
            "vlm_yes_ratio": sample_yes_ratios,
            "vlm_num_questions": [float(count) for count in sample_counts],
        }, {"records": records}

    def _compute_token_margin_rewards(
        self,
        reward_inputs: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict[str, Any]],
    ):
        build_outputs = self._build_teacher_forced_reward_inputs(
            reward_inputs=reward_inputs,
            prompts=prompts,
            metadata=metadata,
            prefer_atomic_questions=True,
        )
        (
            inputs,
            labels,
            row_sample_indices,
            prompt_keys,
            ref_answer_texts,
            _,
            question_ids,
        ) = build_outputs

        batch_size = len(prompts)
        if inputs is None or labels is None:
            zeros = [0.0] * batch_size
            return {
                "avg": zeros,
                "vlm_logprob": zeros,
                "vlm_token_margin": zeros,
                "vlm_yes_ratio": zeros,
                "vlm_num_questions": [0.0] * batch_size,
            }, {"records": []}

        row_reward_values = []
        row_logprob_values = []
        row_margin_values = []
        row_yes_ratio_values = []
        pred_sentences = []
        pred_keywords = []
        normalized_pred_keywords = []
        records = []

        for row_idx in range(len(row_sample_indices)):
            row_inputs = self._slice_model_inputs(inputs, row_idx, row_idx + 1)
            row_labels = labels[row_idx : row_idx + 1]
            prepared_inputs, visual_key = self._prepare_row_inputs_for_gradient(row_inputs)

            target_answer, opposite_answer, target_ids, opposite_ids = (
                self._get_supervised_answer_token_ids(ref_answer_texts[row_idx])
            )

            margin_value = float(self.missing_score)
            logprob_value = float(self.missing_score)
            yes_ratio_value = 0.0
            pred_sentence = ""
            pred_keyword_values: List[str] = []
            normalized_keyword_values: List[str] = []
            frame_importance: List[float] = []
            top_frames: List[int] = []
            target_logprob_tokens: List[float] = []
            opposite_logprob_tokens: List[float] = []

            try:
                with torch.enable_grad():
                    outputs = self.model.forward(
                        **prepared_inputs,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )

                    logits = outputs.logits
                    row_labels = row_labels.to(logits.device)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = row_labels[..., 1:].contiguous()

                    token_mask = shift_labels != -100
                    safe_labels = shift_labels.masked_fill(~token_mask, 0)
                    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
                    token_log_probs = log_probs.gather(
                        -1,
                        safe_labels.unsqueeze(-1),
                    ).squeeze(-1)
                    token_log_probs = torch.where(
                        token_mask,
                        token_log_probs,
                        torch.zeros_like(token_log_probs),
                    )

                    token_counts = token_mask.sum(dim=1).clamp_min(1)
                    logprob_reward_scores = token_log_probs.sum(dim=1) / token_counts
                    logprob_value = float(logprob_reward_scores[0].detach().cpu().item())

                    pred_sentence_list, pred_keyword_list = self._decode_prediction_keywords(
                        shift_logits=shift_logits,
                        shift_labels=shift_labels,
                    )
                    pred_sentence = pred_sentence_list[0]
                    pred_keyword_values = pred_keyword_list[0]
                    yes_ratio_reward_scores, normalized_keyword_batch = (
                        self._compute_yes_ratio_reward_scores(
                            pred_keywords=pred_keyword_list,
                            device=logits.device,
                        )
                    )
                    yes_ratio_value = float(yes_ratio_reward_scores[0].detach().cpu().item())
                    normalized_keyword_values = normalized_keyword_batch[0]

                    token_positions = torch.nonzero(token_mask[0]).squeeze(-1)
                    if opposite_ids:
                        use_count = min(
                            int(token_positions.numel()),
                            len(target_ids),
                            len(opposite_ids),
                        )
                    else:
                        use_count = min(
                            int(token_positions.numel()),
                            len(target_ids),
                        )
                    if use_count > 0 and target_answer is not None:
                        row_log_probs = log_probs[0, token_positions[:use_count]]
                        target_id_tensor = torch.tensor(
                            target_ids[:use_count],
                            device=row_log_probs.device,
                            dtype=torch.long,
                        ).unsqueeze(-1)

                        target_values = row_log_probs.gather(-1, target_id_tensor).squeeze(-1)
                        margin_score = target_values.mean()
                        opposite_values = None
                        if opposite_ids:
                            opposite_id_tensor = torch.tensor(
                                opposite_ids[:use_count],
                                device=row_log_probs.device,
                                dtype=torch.long,
                            ).unsqueeze(-1)
                            opposite_values = row_log_probs.gather(-1, opposite_id_tensor).squeeze(-1)
                            margin_score = margin_score - opposite_values.mean()

                        target_logprob_tokens = target_values.detach().cpu().tolist()
                        opposite_logprob_tokens = (
                            opposite_values.detach().cpu().tolist()
                            if opposite_values is not None
                            else []
                        )
                        margin_value = float(margin_score.detach().cpu().item())

                        if visual_key is not None:
                            visual_grad = torch.autograd.grad(
                                margin_score,
                                prepared_inputs[visual_key],
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=True,
                            )[0]
                            frame_importance = self._compute_frame_importance_from_visual_grad(
                                visual_grad=visual_grad,
                                model_inputs=prepared_inputs,
                            )
                            if frame_importance:
                                top_frames = sorted(
                                    range(len(frame_importance)),
                                    key=lambda idx: frame_importance[idx],
                                    reverse=True,
                                )[: min(3, len(frame_importance))]
                    else:
                        margin_value = logprob_value
            except Exception as exc:
                self._log(
                    "warning",
                    f"[RewardDebug] token_margin reward fallback on row {row_idx}: {exc}",
                )
                margin_value = logprob_value

            row_reward_values.append(float(margin_value))
            row_logprob_values.append(float(logprob_value))
            row_margin_values.append(float(margin_value))
            row_yes_ratio_values.append(float(yes_ratio_value))
            pred_sentences.append(pred_sentence)
            pred_keywords.append(pred_keyword_values)
            normalized_pred_keywords.append(normalized_keyword_values)
            records.append(
                {
                    "sample_idx": row_sample_indices[row_idx],
                    "question_id": question_ids[row_idx] if row_idx < len(question_ids) else f"q{row_idx + 1}",
                    "question": prompt_keys[row_idx],
                    "ref_answer": ref_answer_texts[row_idx],
                    "target_answer": target_answer,
                    "opposite_answer": opposite_answer,
                    "pred_sentence": pred_sentence,
                    "pred_keywords": pred_keyword_values,
                    "normalized_pred_keywords": normalized_keyword_values,
                    "target_logprob_tokens": target_logprob_tokens,
                    "opposite_logprob_tokens": opposite_logprob_tokens,
                    "logprob_reward": float(logprob_value),
                    "margin_reward": float(margin_value),
                    "yes_ratio_reward": float(yes_ratio_value),
                    "frame_importance": frame_importance,
                    "top_frames": top_frames,
                    "selected_reward": float(margin_value),
                }
            )

        sample_rewards, sample_counts = self._reduce_rows_to_sample_level(
            row_reward_values,
            row_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_logprobs, _ = self._reduce_rows_to_sample_level(
            row_logprob_values,
            row_sample_indices,
            batch_size,
            default_value=0.0,
        )
        sample_margins, _ = self._reduce_rows_to_sample_level(
            row_margin_values,
            row_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_yes_ratios, _ = self._reduce_rows_to_sample_level(
            row_yes_ratio_values,
            row_sample_indices,
            batch_size,
            default_value=0.0,
        )

        return {
            "avg": sample_rewards,
            "vlm_logprob": sample_logprobs,
            "vlm_token_margin": sample_margins,
            "vlm_yes_ratio": sample_yes_ratios,
            "vlm_num_questions": [float(count) for count in sample_counts],
        }, {"records": records}

    def _compute_yes_ratio_reward_scores(self, pred_keywords, device: torch.device):
        reward_scores = []
        normalized_pred_keywords = []

        for keyword_groups in pred_keywords:
            normalized_keywords = self._normalize_keyword_groups(keyword_groups)
            yes_no_keywords = [
                keyword for keyword in normalized_keywords if keyword in ("Yes", "No")
            ]
            normalized_pred_keywords.append(yes_no_keywords)

            if yes_no_keywords:
                yes_count = sum(keyword == "Yes" for keyword in yes_no_keywords)
                reward_scores.append(float(yes_count) / float(len(yes_no_keywords)))
            else:
                reward_scores.append(0.0)

        return torch.tensor(
            reward_scores,
            device=device,
            dtype=torch.float32,
        ), normalized_pred_keywords

    def _reduce_rows_to_sample_level(
        self,
        row_values: List[float],
        row_sample_indices: List[int],
        batch_size: int,
        default_value: float,
    ) -> Tuple[List[float], List[int]]:
        sample_sums = [0.0] * batch_size
        sample_counts = [0] * batch_size

        for value, sample_idx in zip(row_values, row_sample_indices):
            if not 0 <= sample_idx < batch_size:
                continue
            sample_sums[sample_idx] += float(value)
            sample_counts[sample_idx] += 1

        sample_values = []
        for sample_idx in range(batch_size):
            if sample_counts[sample_idx] == 0:
                sample_values.append(float(default_value))
            else:
                sample_values.append(sample_sums[sample_idx] / sample_counts[sample_idx])
        return sample_values, sample_counts

    def _compute_free_generation_rewards(
        self,
        reward_inputs: torch.Tensor,
        prompts: List[str],
        metadata: List[Dict[str, Any]],
    ):
        texts, video_batch, row_sample_indices, prompt_keys = (
            self._build_free_generation_reward_inputs(
                reward_inputs=reward_inputs,
                prompts=prompts,
                metadata=metadata,
            )
        )
        batch_size = len(prompts)
        if len(texts) == 0:
            zeros = [0.0] * batch_size
            return {"avg": zeros, "vlm_yes_ratio": zeros, "vlm_yes_count": zeros}, {"records": []}

        tokenizer = self.processor.tokenizer
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        row_rewards = []
        row_yes_ratios = []
        row_yes_counts = []
        records = []

        try:
            tokenizer.padding_side = "left"
            with torch.inference_mode():
                for start, end in self._iter_row_chunks(len(texts)):
                    chunk_texts = texts[start:end]
                    chunk_videos = video_batch[start:end]
                    if self._should_use_image_frame_processor():
                        inputs = self.processor(
                            text=chunk_texts,
                            images=chunk_videos,
                            padding=True,
                            return_tensors="pt",
                            **self._get_processor_image_kwargs(),
                        ).to(self._get_model_device())
                    else:
                        inputs = self.processor(
                            text=chunk_texts,
                            images=None,
                            videos=chunk_videos,
                            padding=True,
                            return_tensors="pt",
                            **self._get_processor_video_kwargs(),
                        ).to(self._get_model_device())
                    prompt_token_count = inputs["input_ids"].shape[1]
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    generated_suffix_ids = generated_ids[:, prompt_token_count:]
                    for chunk_offset in range(end - start):
                        row_idx = start + chunk_offset
                        generated_text = tokenizer.decode(
                            generated_suffix_ids[chunk_offset],
                            skip_special_tokens=True,
                        ).strip()
                        keywords = self._extract_free_generation_keywords(generated_text)
                        normalized_keywords = self._normalize_keyword_groups(keywords)
                        yes_no_keywords = [
                            keyword for keyword in normalized_keywords if keyword in ("Yes", "No")
                        ]
                        yes_count = sum(keyword == "Yes" for keyword in yes_no_keywords)
                        yes_no_count = len(yes_no_keywords)

                        if yes_no_count == 0:
                            yes_ratio = self.missing_score
                            reward_value = self.missing_score
                        else:
                            yes_ratio = float(yes_count) / float(yes_no_count)
                            if self.free_generation_score_type == "count":
                                reward_value = float(yes_count)
                            else:
                                reward_value = yes_ratio

                        row_rewards.append(float(reward_value))
                        row_yes_ratios.append(float(yes_ratio))
                        row_yes_counts.append(float(yes_count))
                        records.append(
                            {
                                "sample_idx": row_sample_indices[row_idx],
                                "question": prompt_keys[row_idx],
                                "generated_text": generated_text,
                                "normalized_keywords": yes_no_keywords,
                                "row_reward": float(reward_value),
                            }
                        )
        finally:
            tokenizer.padding_side = original_padding_side

        sample_rewards, sample_counts = self._reduce_rows_to_sample_level(
            row_rewards,
            row_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_yes_ratios, _ = self._reduce_rows_to_sample_level(
            row_yes_ratios,
            row_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_yes_counts, _ = self._reduce_rows_to_sample_level(
            row_yes_counts,
            row_sample_indices,
            batch_size,
            default_value=0.0,
        )

        return {
            "avg": sample_rewards,
            "vlm_yes_ratio": sample_yes_ratios,
            "vlm_yes_count": sample_yes_counts,
            "vlm_num_questions": [float(count) for count in sample_counts],
        }, {"records": records}

    def __call__(
        self,
        reward_inputs: torch.Tensor,
        prompts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        prompts = list(prompts)
        batch_size = int(reward_inputs.shape[0])
        metadata = self._normalize_metadata_list(batch_size, metadata)

        if self.score_mode == "free_generation_yes":
            # import pdb
            # pdb.set_trace()
            # x = self._compute_free_generation_rewards(reward_inputs=reward_inputs,prompts=prompts,metadata=metadata,)
            return self._compute_free_generation_rewards(
                reward_inputs=reward_inputs,
                prompts=prompts,
                metadata=metadata,
            )
        if self.score_mode == "token_margin":
            return self._compute_token_margin_rewards(
                reward_inputs=reward_inputs,
                prompts=prompts,
                metadata=metadata,
            )
        if self.score_mode == "yes_no_margin":
            return self._compute_structured_slot_rewards(
                reward_inputs=reward_inputs,
                prompts=prompts,
                metadata=metadata,
                selected_score="margin",
                compute_gradient=False,
            )
        if self.score_mode == "token_credit":
            return self._compute_structured_slot_rewards(
                reward_inputs=reward_inputs,
                prompts=prompts,
                metadata=metadata,
                selected_score="logprob",
                compute_gradient=True,
            )

        build_outputs = self._build_teacher_forced_reward_inputs(
            reward_inputs=reward_inputs,
            prompts=prompts,
            metadata=metadata,
        )
        inputs, labels, row_sample_indices, prompt_keys, ref_answer_texts, _, question_ids = build_outputs

        if inputs is None or labels is None:
            zeros = [0.0] * batch_size
            return {
                "avg": zeros,
                "vlm_logprob": zeros,
                "vlm_yes_ratio": zeros,
                "vlm_num_questions": [0.0] * batch_size,
            }, {"records": []}

        row_reward_values = []
        row_logprob_values = []
        row_yes_ratio_values = []
        pred_sentences = []
        pred_keywords = []
        normalized_pred_keywords = []

        for start, end in self._iter_row_chunks(len(row_sample_indices)):
            chunk_inputs = self._slice_model_inputs(inputs, start, end)
            chunk_labels = labels[start:end]

            with torch.inference_mode():
                outputs = self.model.forward(
                    **chunk_inputs,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )

            logits = outputs.logits
            chunk_labels = chunk_labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk_labels[..., 1:].contiguous()

            token_mask = shift_labels != -100
            safe_labels = shift_labels.masked_fill(~token_mask, 0)
            token_log_probs = torch.log_softmax(shift_logits.float(), dim=-1).gather(
                -1,
                safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_log_probs = torch.where(
                token_mask,
                token_log_probs,
                torch.zeros_like(token_log_probs),
            )

            token_counts = token_mask.sum(dim=1).clamp_min(1)
            logprob_reward_scores = token_log_probs.sum(dim=1) / token_counts

            chunk_pred_sentences, chunk_pred_keywords = self._decode_prediction_keywords(
                shift_logits=shift_logits,
                shift_labels=shift_labels,
            )
            yes_ratio_reward_scores, chunk_normalized_pred_keywords = (
                self._compute_yes_ratio_reward_scores(
                    pred_keywords=chunk_pred_keywords,
                    device=logits.device,
                )
            )

            if self.score_mode == "yes_ratio":
                row_reward_scores = yes_ratio_reward_scores
            else:
                row_reward_scores = logprob_reward_scores

            row_reward_values.extend(row_reward_scores.detach().cpu().tolist())
            row_logprob_values.extend(logprob_reward_scores.detach().cpu().tolist())
            row_yes_ratio_values.extend(yes_ratio_reward_scores.detach().cpu().tolist())
            pred_sentences.extend(chunk_pred_sentences)
            pred_keywords.extend(chunk_pred_keywords)
            normalized_pred_keywords.extend(chunk_normalized_pred_keywords)

            # import pdb
            # pdb.set_trace()

        sample_rewards, sample_counts = self._reduce_rows_to_sample_level(
            row_reward_values,
            row_sample_indices,
            batch_size,
            default_value=self.missing_score,
        )
        sample_logprobs, _ = self._reduce_rows_to_sample_level(
            row_logprob_values,
            row_sample_indices,
            batch_size,
            default_value=0.0,
        )
        sample_yes_ratios, _ = self._reduce_rows_to_sample_level(
            row_yes_ratio_values,
            row_sample_indices,
            batch_size,
            default_value=0.0,
        )

        records = []
        for row_idx in range(len(row_sample_indices)):
            records.append(
                {
                    "sample_idx": row_sample_indices[row_idx],
                    "question_id": question_ids[row_idx] if row_idx < len(question_ids) else f"q{row_idx + 1}",
                    "question": prompt_keys[row_idx],
                    "ref_answer": ref_answer_texts[row_idx],
                    "pred_sentence": pred_sentences[row_idx],
                    "pred_keywords": pred_keywords[row_idx],
                    "normalized_pred_keywords": normalized_pred_keywords[row_idx],
                    "logprob_reward": float(row_logprob_values[row_idx]),
                    "yes_ratio_reward": float(row_yes_ratio_values[row_idx]),
                    "selected_reward": float(row_reward_values[row_idx]),
                }
            )

        return {
            "avg": sample_rewards,
            "vlm_logprob": sample_logprobs,
            "vlm_yes_ratio": sample_yes_ratios,
            "vlm_num_questions": [float(count) for count in sample_counts],
        }, {"records": records}


_VIDEO_MODELS = frozenset(["videoalign_local", "video_align", "video_score2"])


def _infer_reward_input_kind(video_paths_or_images) -> str:
    if isinstance(video_paths_or_images, torch.Tensor):
        if video_paths_or_images.ndim == 5:
            return "video_tensor"
        return "image_tensor"
    return "video_path"


def _init_reward_metadata(video_paths_or_images, metadata, reward_input_kind: str) -> List[Dict]:
    if metadata is not None:
        return metadata

    if reward_input_kind in {"image_tensor", "video_tensor"}:
        batch_size = int(video_paths_or_images.shape[0])
    else:
        batch_size = len(video_paths_or_images)
    return [{}] * batch_size


def _should_skip_model(model_name: str, reward_input_kind: str) -> bool:
    is_video_model = model_name in _VIDEO_MODELS
    if is_video_model and reward_input_kind != "video_path":
        print(f"Warning: Skipping path-based video model '{model_name}' for tensor input")
        return True
    if not is_video_model and reward_input_kind == "video_path":
        print(f"Warning: Skipping tensor-based model '{model_name}' for video path input")
        return True
    return False


def _store_model_scores(model_name: str, scores, all_scores: Dict, model_raw_scores: Dict) -> None:
    if isinstance(scores, dict):
        model_raw_scores[model_name] = scores
        for key, values in scores.items():
            prefixed_key = f"{model_name}_{_normalize_key(key)}"
            all_scores[prefixed_key] = values
    elif isinstance(scores, (list, np.ndarray)):
        scores_list = list(scores) if isinstance(scores, np.ndarray) else scores
        model_raw_scores[model_name] = scores_list
        all_scores[model_name] = scores_list
    else:
        raise ValueError(f"Unexpected score format from {model_name}: {type(scores)}")


def _compute_video_model_weighted_scores(raw_scores: Dict, per_model_metric_weights, model_weight: float) -> List[float]:
    if per_model_metric_weights:
        model_metric_scores = []
        for metric_name, weight in per_model_metric_weights.items():
            metric_key = _normalize_key(metric_name)
            metric_values = raw_scores.get(metric_key, raw_scores.get(metric_name))
            if metric_values is not None:
                model_metric_scores.append([weight * s for s in metric_values])
        if model_metric_scores:
            model_sum = [sum(col) for col in zip(*model_metric_scores)]
            return [model_weight * s for s in model_sum]
    else:
        all_metric_values = list(raw_scores.values())
        if all_metric_values:
            model_avg = [sum(col) / len(all_metric_values) for col in zip(*all_metric_values)]
            return [model_weight * s for s in model_avg]
    return None


def _compute_weighted_avg(
    weighted_scores_list: List[List[float]],
    output_scores: Dict,
    video_paths_or_images,
    reward_input_kind: str,
) -> None:
    if weighted_scores_list:
        output_scores["avg"] = [sum(col) for col in zip(*weighted_scores_list)]
        return

    all_values = [v for k, v in output_scores.items() if k != "avg"]
    if all_values:
        output_scores["avg"] = [sum(col) / len(all_values) for col in zip(*all_values)]
        return

    if reward_input_kind in {"image_tensor", "video_tensor"}:
        batch_size = int(video_paths_or_images.shape[0])
    else:
        batch_size = len(video_paths_or_images)
    output_scores["avg"] = [0.0] * batch_size


# ============================================================================
# Multi-Reward Interface
# ============================================================================

def multi_video_score(device, reward_config: Dict[str, Any]):
    """
    Unified multi-reward interface with flexible weighting.
    Only the standard format is supported.
    """
    if "models" not in reward_config:
        raise ValueError(
            "reward_config must contain a 'models' field with the standard format: "
            '{"models": {"model_name": {"weight": float, "sub_reward": {...}}}}'
        )

    model_weights = reward_config["models"]
    model_configs: Dict[str, Dict[str, Any]] = {}
    for model_name, weight_config in model_weights.items():
        if not isinstance(weight_config, dict):
            raise ValueError(
                f"Model config for '{model_name}' must be a dict with at least 'weight' key, "
                f"got {type(weight_config)}"
            )
        if "weight" not in weight_config:
            raise ValueError(
                f"Model config for '{model_name}' must contain 'weight', "
                f"optionally with 'sub_reward', got keys: {list(weight_config.keys())}"
            )

        model_weight = float(weight_config["weight"])
        metric_weights = weight_config.get("sub_reward", None)
        if metric_weights is not None and not isinstance(metric_weights, dict):
            raise ValueError(
                f"'sub_reward' for model '{model_name}' must be a dict of metric_name -> weight, "
                f"got {type(metric_weights)}"
            )

        model_configs[model_name] = {
            "model_weight": model_weight,
            "metric_weights": metric_weights,
        }

    if len(model_configs) == 1:
        only_model_name = next(iter(model_configs.keys()))
        only_model_weight = model_configs[only_model_name]["model_weight"]
        if abs(only_model_weight - 1.0) > 1e-6:
            raise ValueError(
                f"When only one reward model is used ('{only_model_name}'), "
                f"its weight must be 1.0, got {only_model_weight}"
            )

    reward_fns = {}
    for model_name in model_configs.keys():
        if model_name == "videoalign_local":
            checkpoint_mode = reward_config.get("reward_checkpoint_mode", "none")
            reward_fns[model_name] = videoalign_local_score(
                device,
                checkpoint_mode,
                gradient_credit=bool(reward_config.get("reward_gradient_credit", False)),
                credit_num_frames=reward_config.get("reward_credit_num_frames", None),
                credit_metric=reward_config.get("reward_credit_metric", "weighted"),
                credit_metric_weights=model_configs[model_name]["metric_weights"],
            )
        elif model_name == "video_score2":
            reward_fns[model_name] = videoscore2_local_score(
                device,
                gradient_credit=bool(reward_config.get("reward_gradient_credit", False)),
                credit_metric=reward_config.get("reward_credit_metric", "weighted"),
                credit_metric_weights=model_configs[model_name]["metric_weights"],
            )
        elif model_name == "video_align":
            server_url = reward_config.get("server_url", reward_config.get("remote_reward_url"))
            if server_url is None:
                raise ValueError(f"server_url required for {model_name}")
            reward_fns[model_name] = video_align_remote_score(server_url, model=model_name)
        else:
            raise ValueError(
                f"Unknown reward model: {model_name}. "
                "Only videoalign_local and video_align/video_score2 are supported here."
            )

    def _fn(video_paths_or_images: Union[List[str], torch.Tensor], prompts: List[str], metadata: List[Dict] = None):
        reward_input_kind = _infer_reward_input_kind(video_paths_or_images)
        metadata_local = _init_reward_metadata(video_paths_or_images, metadata, reward_input_kind)

        all_scores = {}
        all_meta = {}
        model_raw_scores = {}

        for model_name in model_configs.keys():
            if _should_skip_model(model_name, reward_input_kind):
                continue
            reward_fn = reward_fns[model_name]
            scores, meta = reward_fn(video_paths_or_images, prompts, metadata_local)
            _store_model_scores(model_name, scores, all_scores, model_raw_scores)
            all_meta.update(meta)

        output_scores = all_scores.copy()
        weighted_scores_list = []

        for model_name, config in model_configs.items():
            if model_name not in model_raw_scores:
                continue
            model_weight = config["model_weight"]
            per_model_metric_weights = config["metric_weights"]
            raw_scores = model_raw_scores[model_name]

            if isinstance(raw_scores, dict):
                weighted = _compute_video_model_weighted_scores(
                    raw_scores, per_model_metric_weights, model_weight
                )
                if weighted is not None:
                    weighted_scores_list.append(weighted)
            else:
                weighted_scores_list.append([model_weight * s for s in raw_scores])

        _compute_weighted_avg(
            weighted_scores_list,
            output_scores,
            video_paths_or_images,
            reward_input_kind,
        )
        assert "avg" in output_scores, f"avg not in output_scores: {output_scores.keys()}"
        return output_scores, all_meta

    return _fn


def _build_vlm_reward_fn(config: Any, device, logger=None):
    reward_model = str(_config_get(config, "reward_model", "")).lower()
    if not reward_model:
        reward_model = "vlm_reward"
        _config_set(config, "reward_model", reward_model)

    scorer = VLMRewardScorer(config=config, device=device, logger=logger)
    if logger is not None:
        logger.info(
            "Creating VLM reward scorer: "
            f"reward_model={reward_model}, "
            f"model_path={_config_get(config, 'vlm_reward_model_path')}, "
            f"model_family={_config_get(config, 'vlm_reward_model_family')}, "
            f"score_type={_config_get(config, 'vlm_reward_score_type')}"
        )
    return scorer


def _has_single_vlm_reward_model(reward_config: Dict[str, Any]) -> Optional[str]:
    models = reward_config.get("models", {})
    if not isinstance(models, dict) or not models:
        return None

    vlm_model_names = [name for name in models if is_vlm_reward_model(name)]
    if not vlm_model_names:
        return None
    if len(vlm_model_names) != len(models):
        raise ValueError(
            "Mixing VLM reward with other reward models is not supported."
        )
    if len(vlm_model_names) != 1:
        raise ValueError(
            "Only a single VLM reward model is supported at a time."
        )
    return vlm_model_names[0]


def get_reward_fn(args, device, logger):
    """
    Factory function to create reward function from args.
    """
    reward_model = getattr(args, "reward_model", "videoalign_local")
    if is_vlm_reward_model(reward_model):
        if logger is not None:
            logger.info(
                "Using direct VLM reward config from args: "
                f"reward_model={reward_model}, "
                f"model_path={getattr(args, 'vlm_reward_model_path', None)}, "
                f"model_family={getattr(args, 'vlm_reward_model_family', None)}, "
                f"score_type={getattr(args, 'vlm_reward_score_type', None)}"
            )
        return _build_vlm_reward_fn(args, device, logger=logger)

    if hasattr(args, "reward_config") and isinstance(args.reward_config, dict) and args.reward_config:
        reward_config = args.reward_config.copy()
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)
        if hasattr(args, "remote_reward_url"):
            reward_config.setdefault("server_url", args.remote_reward_url)
        logger.info(f"Using external reward_config: {reward_config}")
    else:
        logger.warning(
            "reward_config not provided, falling back to default config with reward_model only"
        )

        if reward_model == "auto":
            reward_model = "videoalign_local"

        rw = getattr(args, "reward_weights", None)
        if isinstance(rw, dict) and rw:
            sub_reward = rw
        elif isinstance(rw, str) and rw.strip():
            try:
                parsed = json.loads(rw.strip())
                sub_reward = (
                    parsed
                    if isinstance(parsed, dict) and parsed
                    else {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}
                )
            except (json.JSONDecodeError, TypeError):
                sub_reward = {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}
        else:
            sub_reward = {"VQ": 1.0, "MQ": 1.0, "TA": 1.0}

        reward_config = {
            "models": {
                reward_model: {
                    "weight": 1.0,
                    "sub_reward": sub_reward,
                }
            }
        }

        if reward_model == "video_align":
            reward_config["server_url"] = getattr(args, "remote_reward_url", None)
        elif reward_model == "videoalign_local":
            reward_config["reward_checkpoint_mode"] = getattr(
                args, "reward_checkpoint_mode", "v2"
            )

        if hasattr(args, "remote_reward_url"):
            reward_config.setdefault("server_url", args.remote_reward_url)
        if hasattr(args, "reward_checkpoint_mode"):
            reward_config.setdefault("reward_checkpoint_mode", args.reward_checkpoint_mode)

    for attr in (
        "reward_gradient_credit",
        "reward_credit_num_frames",
        "reward_credit_metric",
    ):
        if hasattr(args, attr):
            reward_config.setdefault(attr, getattr(args, attr))

    vlm_model_name = _has_single_vlm_reward_model(reward_config)
    if vlm_model_name is not None:
        _config_set(args, "reward_model", vlm_model_name)
        return _build_vlm_reward_fn(args, device, logger=logger)

    logger.info(f"Creating reward function with config: {reward_config}")
    return multi_video_score(device, reward_config)


def create_reward_fn_from_config(config: Dict[str, Any], device, logger=None):
    """
    Create reward function from config dict.
    """
    if logger:
        logger.info(f"Creating reward function from config: {config}")

    vlm_model_name = _has_single_vlm_reward_model(config)
    if vlm_model_name is not None:
        namespace = SimpleNamespace(
            reward_model=vlm_model_name,
            vlm_reward_model_path=config.get("vlm_reward_model_path"),
            vlm_reward_model_family=config.get("vlm_reward_model_family"),
            vlm_reward_prompt_template=config.get("vlm_reward_prompt_template"),
            vlm_reward_num_frames=config.get("vlm_reward_num_frames", 8),
            vlm_reward_max_pixels=config.get("vlm_reward_max_pixels", 65536),
            vlm_reward_max_new_tokens=config.get("vlm_reward_max_new_tokens", 128),
            vlm_reward_score_type=config.get("vlm_reward_score_type", "logprob"),
            vlm_reward_missing_score=config.get("vlm_reward_missing_score", 0.0),
        )
        return _build_vlm_reward_fn(namespace, device, logger=logger)

    return multi_video_score(device, config)

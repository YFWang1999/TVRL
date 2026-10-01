import ast
import json
import os
import pdb
from collections.abc import Mapping
from typing import Dict, List, Optional
import pandas as pd

import torch
from hyvideo.models.reward_models.videoalign.vision_process import process_vision_info

from hyvideo.models.reward_models.videoalign.data import DataConfig
from hyvideo.models.reward_models.videoalign.utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from hyvideo.models.reward_models.videoalign.utils import load_model_from_checkpoint, create_model_and_processor
from hyvideo.models.reward_models.videoalign.prompt_template import build_prompt


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16, reward_checkpoint_mode="v1"):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step, reward_checkpoint_mode)
        model.eval()
        model.requires_grad_(False)

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config
        self.fallback_reward = {"VQ": -1.0, "MQ": -1.0, "TA": -1.0, "Overall": -1.0}
        self.offload_device = torch.device("cpu")

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

    def _normalize_reward_logits(self, logits: torch.Tensor) -> torch.Tensor:
        reward_values = logits.float()
        if self.inference_config is None:
            return reward_values

        columns = []
        for idx in range(reward_values.shape[-1]):
            column = reward_values[:, idx]
            if idx < 3:
                key = ("VQ", "MQ", "TA")[idx]
                mean = float(self.inference_config[f"{key}_mean"])
                std = max(float(self.inference_config[f"{key}_std"]), 1e-6)
                column = (column - mean) / std
            columns.append(column)
        if not columns:
            return reward_values
        return torch.stack(columns, dim=-1)

    def _reward_entry_from_values(self, reward_values: torch.Tensor) -> Dict[str, float]:
        values = reward_values.detach().float().cpu().flatten()
        vq = float(values[0].item()) if values.numel() > 0 else float("nan")
        mq = float(values[1].item()) if values.numel() > 1 else float("nan")
        ta = float(values[2].item()) if values.numel() > 2 else float("nan")
        return {"VQ": vq, "MQ": mq, "TA": ta, "Overall": vq + mq + ta}

    def _select_reward_tensor(
        self,
        reward_values: torch.Tensor,
        metric_weights: Optional[Dict[str, float]] = None,
        metric: str = "weighted",
    ) -> torch.Tensor:
        metric_tensors = {
            "vq": reward_values[:, 0],
            "mq": reward_values[:, 1],
            "ta": reward_values[:, 2],
        }
        overall = reward_values[:, :3].sum(dim=-1)

        if isinstance(metric_weights, dict) and metric_weights:
            selected = torch.zeros_like(overall)
            used_weight = False
            for key, weight in metric_weights.items():
                try:
                    weight_value = float(weight)
                except (TypeError, ValueError):
                    continue
                key_norm = str(key).strip().lower()
                if key_norm in metric_tensors:
                    selected = selected + weight_value * metric_tensors[key_norm]
                    used_weight = True
                elif key_norm in {"overall", "avg"}:
                    selected = selected + weight_value * overall
                    used_weight = True
            if used_weight:
                return selected

        metric_norm = str(metric or "weighted").strip().lower()
        if metric_norm in metric_tensors:
            return metric_tensors[metric_norm]
        return overall

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

        return []

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        # Newer transformers/Qwen processors may return auxiliary fields that
        # this VideoAlign reward head does not consume.
        batch.pop("second_per_grid_ts", None)
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards

    def reward_with_gradient_credit(
        self,
        video_paths,
        prompts,
        fps=None,
        num_frames=None,
        max_pixels=None,
        use_norm=True,
        metric_weights: Optional[Dict[str, float]] = None,
        credit_metric: str = "weighted",
        sample_indices: Optional[List[int]] = None,
    ):
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."

        rewards = []
        records = []
        for local_idx, (video_path, prompt) in enumerate(zip(video_paths, prompts)):
            sample_idx = (
                int(sample_indices[local_idx])
                if sample_indices is not None and local_idx < len(sample_indices)
                else local_idx
            )
            try:
                batch = self.prepare_batch(
                    [video_path],
                    [prompt],
                    fps=fps,
                    num_frames=num_frames,
                    max_pixels=max_pixels,
                )
                prepared_batch, visual_key = self._prepare_inputs_for_gradient(batch)
                with torch.enable_grad():
                    outputs = self.model(return_dict=True, **prepared_batch)
                    raw_logits = outputs["logits"]
                    reward_values = (
                        self._normalize_reward_logits(raw_logits)
                        if use_norm
                        else raw_logits.float()
                    )
                    selected_rewards = self._select_reward_tensor(
                        reward_values,
                        metric_weights=metric_weights,
                        metric=credit_metric,
                    )
                    selected_reward = selected_rewards[0]

                    visual_grad = None
                    if visual_key is not None:
                        visual_grad = torch.autograd.grad(
                            selected_reward,
                            prepared_batch[visual_key],
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )[0]

                reward_entry = self._reward_entry_from_values(reward_values[0])
                selected_value = float(selected_reward.detach().cpu().item())
                frame_importance = self._compute_frame_importance_from_visual_grad(
                    visual_grad,
                    prepared_batch,
                )
                top_frames = sorted(
                    range(len(frame_importance)),
                    key=lambda idx: frame_importance[idx],
                    reverse=True,
                )[: min(3, len(frame_importance))]
                records.append(
                    {
                        "sample_idx": sample_idx,
                        "question_id": "videoalign_overall",
                        "question": str(prompt),
                        "token_credit_reward": selected_value,
                        "selected_reward": selected_value,
                        "frame_importance": frame_importance,
                        "top_frames": top_frames,
                    }
                )
                rewards.append(reward_entry)
            except Exception:
                rewards.append(dict(self.fallback_reward))

        return rewards, records


if __name__ == "__main__":
    load_from_pretrained = "./checkpoints"
    device = "cuda:0"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

    video_paths = [
        "datasets/train/videos/example_1_A.mp4",
        "datasets/train/videos/example_1_B.mp4",
        "datasets/train/videos/example_2_A.mp4",
    ]

    prompts = [
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        "The camera follows a young explorer through an abandoned urban building at night, exploring hidden corridors and forgotten spaces, with a mix of light and shadow creating a mysterious atmosphere.",
    ]

    with torch.no_grad():
        rewards = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards)


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import ModelConfig, TrainingConfig
    from .diffusion import SimpleDiffusion
    from .model import UNet
except ImportError:
    from config import ModelConfig, TrainingConfig
    from diffusion import SimpleDiffusion
    from model import UNet


def build_cadm(device: Optional[torch.device | str] = None) -> UNet:

    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()
    model = UNet(
        input_channels=train_cfg.IMG_SHAPE[0],
        output_channels=train_cfg.IMG_SHAPE[0],
        base_channels=model_cfg.BASE_CH,
        base_channels_multiples=model_cfg.BASE_CH_MULT,
        apply_attention=model_cfg.APPLY_ATTENTION,
        dropout_rate=model_cfg.DROPOUT_RATE,
        time_multiple=model_cfg.TIME_EMB_MULT,
    )
    return model.to(device) if device is not None else model


def build_diffusion(device: torch.device | str = "cpu") -> SimpleDiffusion:
    cfg = TrainingConfig()
    return SimpleDiffusion(
        num_diffusion_timesteps=cfg.TIMESTEPS,
        img_shape=cfg.IMG_SHAPE,
        device=device,
    )


@torch.no_grad()
def ddim_teacher_target(
    teacher: nn.Module,
    diffusion: SimpleDiffusion,
    condition: torch.Tensor,
    initial_noise: torch.Tensor,
    steps: int = 20,
) -> torch.Tensor:

    if steps < 1:
        raise ValueError("steps must be positive")

    batch = condition.shape[0]
    total = diffusion.num_diffusion_timesteps
    alpha_bar = diffusion.alpha_cumulative.to(condition.device)
    times = np.rint(np.linspace(total - 1, 0, steps + 1)).astype(np.int64)
    times = np.clip(times, 0, total - 1)
    state = initial_noise.clone()

    for current, previous in zip(times[:-1], times[1:]):
        t = torch.full(
            (batch,), int(current), dtype=torch.long, device=condition.device
        )
        predicted_noise = teacher(condition, state, t)
        ab_t = alpha_bar[int(current)]
        predicted_clean = (
            state - torch.sqrt(1.0 - ab_t) * predicted_noise
        ) / torch.sqrt(ab_t)
        predicted_clean = predicted_clean.clamp(-1.0, 1.0)

        if int(previous) == 0:
            state = predicted_clean
        else:
            ab_previous = alpha_bar[int(previous)]
            state = (
                torch.sqrt(ab_previous) * predicted_clean
                + torch.sqrt(1.0 - ab_previous) * predicted_noise
            )
    return state


def _batch_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    return F.cosine_similarity(left, right, dim=1).mean()


def _extract(schedule: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    return schedule.gather(0, timesteps).reshape(-1, 1, 1, 1)


@dataclass(frozen=True)
class DistillationWeights:
    reconstruction: float = 1.0
    teacher: float = 0.3
    score: float = 0.1
    physical: float = 0.5


class TeacherStudentDistillation(nn.Module):







    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        diffusion: SimpleDiffusion,
        weights: DistillationWeights = DistillationWeights(),
        teacher_steps: int = 20,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.diffusion = diffusion
        self.weights = weights
        self.teacher_steps = teacher_steps
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def student_output(
        self, condition: torch.Tensor, initial_noise: torch.Tensor
    ) -> torch.Tensor:
        flag = torch.full(
            (condition.shape[0],),
            self.diffusion.num_diffusion_timesteps - 1,
            dtype=torch.long,
            device=condition.device,
        )
        return self.student(condition, initial_noise, flag)

    def compute_losses(
        self,
        alice: torch.Tensor,
        jack: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
        score_timesteps: Optional[torch.Tensor] = None,
        score_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        if alice.shape != jack.shape:
            raise ValueError("alice and jack must have identical CSI tensor shapes")
        if initial_noise is None:
            initial_noise = torch.randn_like(alice)

        teacher_target = ddim_teacher_target(
            self.teacher,
            self.diffusion,
            jack,
            initial_noise,
            steps=self.teacher_steps,
        )
        generated = self.student_output(jack, initial_noise)

        reconstruction = F.l1_loss(generated, alice)
        teacher_consistency = F.mse_loss(generated, teacher_target)

        if score_timesteps is None:
            score_timesteps = torch.randint(
                50, 500, (alice.shape[0],), device=alice.device
            )
        shared_noise = torch.randn_like(alice)
        sqrt_ab = _extract(
            self.diffusion.sqrt_alpha_cumulative.to(alice.device), score_timesteps
        )
        sqrt_one_minus_ab = _extract(
            self.diffusion.sqrt_one_minus_alpha_cumulative.to(alice.device),
            score_timesteps,
        )
        noisy_student = sqrt_ab * generated + sqrt_one_minus_ab * shared_noise
        noisy_target = sqrt_ab * alice + sqrt_one_minus_ab * shared_noise



        student_score = self.teacher(jack, noisy_student, score_timesteps)
        with torch.no_grad():
            target_score = self.teacher(jack, noisy_target, score_timesteps)
        score_consistency = 1.0 - _batch_cosine(student_score, target_score)

        nmse = (generated - alice).square().sum() / alice.square().sum().clamp_min(1e-8)
        physical_consistency = nmse + 1.0 - _batch_cosine(generated, alice)
        active_score_weight = (
            self.weights.score if score_weight is None else score_weight
        )
        total = (
            self.weights.reconstruction * reconstruction
            + self.weights.teacher * teacher_consistency
            + active_score_weight * score_consistency
            + self.weights.physical * physical_consistency
        )
        return {
            "total": total,
            "reconstruction": reconstruction,
            "teacher_consistency": teacher_consistency,
            "score_consistency": score_consistency,
            "physical_consistency": physical_consistency,
            "student_output": generated,
            "teacher_target": teacher_target,
        }


def score_weight_schedule(
    step: int,
    total_steps: int,
    target: float = 0.1,
    warmup_fraction: float = 0.1,
    ramp_fraction: float = 0.7,
) -> float:

    warmup_end = warmup_fraction * total_steps
    ramp_end = ramp_fraction * total_steps
    if step < warmup_end:
        return 0.0
    if step < ramp_end:
        return target * (step - warmup_end) / max(1.0, ramp_end - warmup_end)
    return target

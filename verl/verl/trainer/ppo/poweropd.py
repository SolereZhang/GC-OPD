# Copyright 2026 GC-OPD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PowerOPD bounded power-transform rewards.

Implements Eq. (4) from "PowerOPD: Stabilizing On-Policy Distillation with
Bounded Power Transformation": r_t = sg[p_teacher**alpha - p_student**alpha],
alpha > 0. The paper lists github.com/EIT-NLP/PowerOPD as its code release,
but that repository was unavailable when this implementation was added.
"""

from __future__ import annotations

from typing import Any

import torch


def _cfg(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return selected.float().mean().item()


def compute_poweropd_advantages(
    *,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    policy_loss_config: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Compute PowerOPD sampled-token policy-gradient rewards.

    ``old_log_prob`` is the sampled-policy probability term used by the PPO
    batch. In the on-policy single-epoch case the actor replaces it with the
    current log probability detached, matching the stop-gradient student term.
    """

    alpha = float(_cfg(policy_loss_config, "poweropd_alpha", 100.0))
    if alpha <= 0:
        raise ValueError(f"poweropd_alpha must be > 0, got {alpha}")

    dtype = old_log_prob.dtype
    mask = response_mask.to(device=old_log_prob.device, dtype=dtype)
    valid = mask.bool()

    teacher_prob = torch.exp(torch.clamp(ref_log_prob, max=0.0))
    student_prob = torch.exp(torch.clamp(old_log_prob, max=0.0))
    reward = (teacher_prob.pow(alpha) - student_prob.pow(alpha)).detach()
    advantages = reward * mask

    metrics: dict[str, float] = {
        "actor/poweropd_alpha": alpha,
        "actor/poweropd_advantage_abs_mean": _masked_mean_or_zero(advantages.abs(), valid),
        "actor/poweropd_teacher_prob_mean": _masked_mean_or_zero(teacher_prob, valid),
        "actor/poweropd_student_prob_mean": _masked_mean_or_zero(student_prob, valid),
    }
    if valid.any():
        valid_reward = reward[valid].float()
        metrics["actor/poweropd_reward_min"] = valid_reward.min().item()
        metrics["actor/poweropd_reward_max"] = valid_reward.max().item()
        metrics["actor/poweropd_reward_mean"] = valid_reward.mean().item()
    else:
        metrics["actor/poweropd_reward_min"] = 0.0
        metrics["actor/poweropd_reward_max"] = 0.0
        metrics["actor/poweropd_reward_mean"] = 0.0

    return {"poweropd_advantages": advantages}, metrics

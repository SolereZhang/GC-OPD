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
"""GOPD/ExOPD reference-baseline tensors.

This keeps the ExOPD baseline objective isolated from the other OPD-family
methods. The trainer still supplies teacher ``ref_log_prob`` and optional
student-base ``base_ref_log_prob``; actor workers only need to construct the
distillation advantage.
"""

from __future__ import annotations

from typing import Any

import torch


def _cfg(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return selected.float().mean().item()


def _expand_lambda(
    *,
    policy_loss_config: Any,
    old_log_prob: torch.Tensor,
    ep_gopd_lambda: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    position_aware = bool(
        _cfg(policy_loss_config, "gopd_position_aware", False)
        or _cfg(policy_loss_config, "position_aware_exopd", False)
    )
    if position_aware and ep_gopd_lambda is not None:
        lambda_tensor = ep_gopd_lambda.to(device=old_log_prob.device, dtype=old_log_prob.dtype)
        if lambda_tensor.dim() == 0:
            lambda_tensor = lambda_tensor.reshape(1, 1)
        if lambda_tensor.dim() == 1:
            lambda_tensor = lambda_tensor[:, None]
        if lambda_tensor.shape[-1] == 1:
            lambda_tensor = lambda_tensor.expand_as(old_log_prob)
        if lambda_tensor.shape != old_log_prob.shape:
            raise ValueError(
                "ep_gopd_lambda must be shaped as (batch,), (batch, 1), or "
                f"{tuple(old_log_prob.shape)}, got {tuple(lambda_tensor.shape)}"
            )
        metrics = {
            "actor/gopd_position_aware": 1.0,
            "actor/gopd_lambda": lambda_tensor.float().mean().item(),
            "actor/gopd_lambda_min": lambda_tensor.float().min().item(),
            "actor/gopd_lambda_max": lambda_tensor.float().max().item(),
        }
        return lambda_tensor, metrics

    lambda_value = float(_cfg(policy_loss_config, "gopd_lambda", 1.0))
    lambda_tensor = old_log_prob.new_tensor(lambda_value)
    metrics = {
        "actor/gopd_position_aware": 0.0,
        "actor/gopd_lambda": lambda_value,
        "actor/gopd_lambda_min": lambda_value,
        "actor/gopd_lambda_max": lambda_value,
    }
    return lambda_tensor, metrics


def compute_gopd_advantages(
    *,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    policy_loss_config: Any,
    base_ref_log_prob: torch.Tensor | None = None,
    ep_gopd_lambda: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Return GOPD advantages and diagnostics.

    ExOPD computes an extrapolated teacher target:

        lambda * teacher_log_prob + (1 - lambda) * base_ref_log_prob

    If no base reference is supplied, the old policy log-prob is used as the
    baseline reference, matching the legacy fallback behavior.
    """

    dtype = old_log_prob.dtype
    device = old_log_prob.device
    ref_log_prob = ref_log_prob.to(device=device, dtype=dtype)
    response_mask = response_mask.to(device=device, dtype=dtype)
    reference_log_prob = (
        base_ref_log_prob.to(device=device, dtype=dtype) if base_ref_log_prob is not None else old_log_prob
    )

    lambda_tensor, metrics = _expand_lambda(
        policy_loss_config=policy_loss_config,
        old_log_prob=old_log_prob,
        ep_gopd_lambda=ep_gopd_lambda,
    )
    target_log_prob = lambda_tensor * ref_log_prob + (1.0 - lambda_tensor) * reference_log_prob
    advantages = target_log_prob - old_log_prob

    delta_clip = _optional_float(_cfg(policy_loss_config, "gopd_delta_clip", None))
    if delta_clip is None:
        delta_clip = _optional_float(_cfg(policy_loss_config, "opd_delta_clip", None))
    if delta_clip is not None and delta_clip > 0.0:
        advantages = advantages.clamp(min=-delta_clip, max=delta_clip)
        metrics["actor/gopd_delta_clip"] = float(delta_clip)
    else:
        metrics["actor/gopd_delta_clip"] = 0.0

    valid = response_mask.bool()
    metrics.update(
        {
            "actor/gopd_enabled": 1.0,
            "actor/gopd_base_ref_available": float(base_ref_log_prob is not None),
            "actor/gopd_advantage_abs_mean": _masked_mean_or_zero(advantages.abs(), valid),
            "actor/gopd_advantage_mean": _masked_mean_or_zero(advantages, valid),
            "actor/gopd_teacher_delta_mean": _masked_mean_or_zero(ref_log_prob - old_log_prob, valid),
            "actor/gopd_base_delta_mean": _masked_mean_or_zero(reference_log_prob - old_log_prob, valid),
        }
    )
    tensors = {
        "gopd_advantages": advantages.detach(),
        "gopd_target_log_prob": target_log_prob.detach(),
    }
    return tensors, metrics

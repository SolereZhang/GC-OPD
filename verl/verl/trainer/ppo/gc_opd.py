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
"""Base-free residual-calibrated on-policy distillation tensors.

GC-OPD keeps vanilla teacher-student OPD as a dense prior, then applies token
credit to a group-relative reward-teacher residual. The full rollout batch is
required so sibling rollouts remain intact before actor mini-batch splitting.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto


def _cfg(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    values_f = values.float()
    mask_f = mask.float()
    denom = mask_f.sum(dim=dim).clamp(min=1.0)
    return ((values_f * mask_f).sum(dim=dim) / denom).to(dtype=values.dtype)


def _group_ids_from_uids(
    uids: Any, batch_size: int, group_size: int
) -> tuple[list[list[int]], float, float]:
    groups: dict[Any, list[int]] = defaultdict(list)
    source_uid = 0.0
    contiguous_fallback = 1.0
    if uids is not None and len(uids) == batch_size:
        source_uid = 1.0
        contiguous_fallback = 0.0
        for i, uid in enumerate(uids):
            groups[uid].append(i)
    else:
        safe_group_size = max(int(group_size), 1)
        for i in range(batch_size):
            groups[i // safe_group_size].append(i)
    return list(groups.values()), source_uid, contiguous_fallback


def _group_zscore(
    values: torch.Tensor,
    groups: list[list[int]],
    *,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    zscores = torch.zeros_like(values)
    identifiable = torch.zeros_like(values)
    for group in groups:
        if len(group) < 2:
            continue
        idx = torch.as_tensor(group, device=values.device, dtype=torch.long)
        group_values = values[idx]
        group_values_f = group_values.float()
        std_f = group_values_f.std(unbiased=False)
        if std_f <= float(min_std):
            continue
        zscores[idx] = (
            (group_values_f - group_values_f.mean()) / std_f.clamp(min=float(min_std))
        ).to(dtype=values.dtype)
        identifiable[idx] = 1.0
    return zscores, identifiable


def _batch_zscore(
    values: torch.Tensor, *, min_std: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() < 2:
        return torch.zeros_like(values), torch.zeros_like(values)
    values_f = values.float()
    std_f = values_f.std(unbiased=False)
    if std_f <= float(min_std):
        return torch.zeros_like(values), torch.zeros_like(values)
    zscores = ((values_f - values_f.mean()) / std_f.clamp(min=float(min_std))).to(
        dtype=values.dtype
    )
    return zscores, torch.ones_like(values)


def _token_zscore(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values_f = values.float()
    mask_f = response_mask.float()
    valid_count = mask_f.sum(dim=-1, keepdim=True)
    denom = valid_count.clamp(min=1.0)
    mean = (values_f * mask_f).sum(dim=-1, keepdim=True) / denom
    centered = (values_f - mean) * mask_f
    std = (centered.square().sum(dim=-1, keepdim=True) / denom).sqrt()
    identifiable = (valid_count >= 2.0) & (std > float(min_std))
    zscores = torch.where(
        identifiable,
        centered / std.clamp(min=float(min_std)),
        torch.zeros_like(centered),
    )
    return zscores.to(dtype=values.dtype) * response_mask, identifiable.squeeze(-1)


def _credit_weights(
    mode: str,
    opd_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    credit_cap: float,
    min_token_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode).strip().lower().replace("-", "_")
    response_mask = response_mask.to(dtype=opd_advantages.dtype)
    if credit_cap < 1.0:
        raise ValueError("gc_opd_credit_cap must be at least 1 for uniform fallback")
    if mode == "uniform":
        valid_rows = response_mask.sum(dim=-1) > 0
        return response_mask, valid_rows

    if mode == "teacher_abs":
        raw_f = opd_advantages.float().abs() * response_mask.float()
        valid_count = response_mask.float().sum(dim=-1, keepdim=True)
        credit_mass = raw_f.sum(dim=-1, keepdim=True)
        identifiable = (valid_count > 0.0) & (credit_mass > float(min_token_std))
        normalized = raw_f * valid_count.clamp(min=1.0) / credit_mass.clamp(
            min=float(min_token_std)
        )
        bounded = normalized.clamp(min=0.0, max=float(credit_cap)).to(
            dtype=opd_advantages.dtype
        )
        credit = torch.where(identifiable, bounded, response_mask)
        return credit * response_mask, identifiable.squeeze(-1)

    zscores, identifiable = _token_zscore(
        opd_advantages,
        response_mask,
        min_std=min_token_std,
    )
    if mode == "raca":
        raw = (1.0 + torch.tanh(zscores.float() / 2.0)).to(
            dtype=opd_advantages.dtype
        )
    elif mode == "relative_teacher_support":
        raw = 1.0 + zscores
    elif mode == "relative_opd_leverage":
        raw = zscores.abs()
    else:
        raise ValueError(f"Unsupported gc_opd_credit_mode={mode!r}")

    bounded = raw.clamp(min=0.0, max=float(credit_cap)) * response_mask
    credit = torch.where(identifiable[:, None], bounded, response_mask)
    return credit * response_mask, identifiable


def _pairwise_order_accuracy(
    rewards: torch.Tensor,
    scores: torch.Tensor,
    groups: list[list[int]],
    *,
    eps: float,
) -> torch.Tensor:
    correct = rewards.new_tensor(0.0)
    comparable = rewards.new_tensor(0.0)
    for group in groups:
        if len(group) < 2:
            continue
        idx = torch.as_tensor(group, device=rewards.device, dtype=torch.long)
        reward_diff = rewards[idx][:, None] - rewards[idx][None, :]
        score_diff = scores[idx][:, None] - scores[idx][None, :]
        upper = torch.triu(torch.ones_like(reward_diff, dtype=torch.bool), diagonal=1)
        valid = upper & (reward_diff.abs() > eps) & (score_diff.abs() > eps)
        if not valid.any():
            continue
        signed = torch.sign(reward_diff[valid]) * torch.sign(score_diff[valid])
        correct = correct + (signed > 0).to(dtype=rewards.dtype).sum()
        comparable = comparable + valid.to(dtype=rewards.dtype).sum()
    if comparable <= rewards.new_tensor(0.0):
        return rewards.new_tensor(0.5)
    return correct / comparable.clamp(min=1.0)


def compute_gc_opd_tensors(
    *,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor,
    policy_loss_config: Any,
    base_ref_log_prob: torch.Tensor | None = None,
    uids: Any = None,
    rollout_n: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Return base-free GC-OPD advantages and diagnostics."""

    device = old_log_prob.device
    dtype = old_log_prob.dtype
    ref_log_prob = ref_log_prob.to(device=device, dtype=dtype)
    response_mask = response_mask.to(device=device, dtype=dtype)
    token_level_scores = token_level_scores.to(device=device, dtype=dtype)

    # Kept in the public signature so callers can pass historical batches.
    # GC-OPD intentionally never consumes teacher-base log probabilities.
    base_ref_input_present = base_ref_log_prob is not None

    agg = (
        str(_cfg(policy_loss_config, "gc_opd_teacher_score_agg", "mean"))
        .strip()
        .lower()
    )
    if agg != "mean":
        raise ValueError(
            f"Unsupported gc_opd_teacher_score_agg={agg!r}; only 'mean' is implemented"
        )

    beta = float(_cfg(policy_loss_config, "gc_opd_residual_beta", 0.05))
    credit_cap = float(_cfg(policy_loss_config, "gc_opd_credit_cap", 5.0))
    min_group_std = float(_cfg(policy_loss_config, "gc_opd_min_group_std", 1e-6))
    min_token_std = float(_cfg(policy_loss_config, "gc_opd_min_token_std", 1e-6))
    opd_advantages = (ref_log_prob - old_log_prob) * response_mask
    teacher_score = _masked_mean(opd_advantages, response_mask, dim=-1)
    seq_rewards = (token_level_scores * response_mask).sum(dim=-1)

    group_size = max(int(_cfg(policy_loss_config, "gc_opd_group_size", rollout_n)), 1)
    groups, source_uid, contiguous_fallback = _group_ids_from_uids(
        uids,
        batch_size=old_log_prob.size(0),
        group_size=group_size,
    )
    residual_norm = (
        str(_cfg(policy_loss_config, "gc_opd_residual_norm", "group_zscore"))
        .strip()
        .lower()
    )
    if residual_norm == "group_zscore":
        reward_z, reward_identifiable = _group_zscore(
            seq_rewards, groups, min_std=min_group_std
        )
        teacher_z, teacher_identifiable = _group_zscore(
            teacher_score, groups, min_std=min_group_std
        )
    elif residual_norm == "batch_zscore":
        reward_z, reward_identifiable = _batch_zscore(
            seq_rewards, min_std=min_group_std
        )
        teacher_z, teacher_identifiable = _batch_zscore(
            teacher_score, min_std=min_group_std
        )
    else:
        raise ValueError(f"Unsupported gc_opd_residual_norm={residual_norm!r}")

    residual_identifiable = reward_identifiable * teacher_identifiable
    residual = ((reward_z - teacher_z) * residual_identifiable).detach()
    credit, token_credit_identifiable = _credit_weights(
        str(_cfg(policy_loss_config, "gc_opd_credit_mode", "relative_opd_leverage")),
        opd_advantages=opd_advantages.detach(),
        response_mask=response_mask,
        credit_cap=credit_cap,
        min_token_std=min_token_std,
    )
    credit = credit.detach()

    residual_advantages = beta * residual[:, None] * credit
    raw_advantages = (opd_advantages + residual_advantages) * response_mask

    adv_clip = float(_cfg(policy_loss_config, "gc_opd_adv_clip", 10.0))
    valid = response_mask.bool()
    if adv_clip > 0.0:
        clip_ratio = (
            (raw_advantages[valid].abs() > adv_clip)
            .to(dtype=torch.float32)
            .mean()
            .item()
            if valid.any()
            else 0.0
        )
        advantages = raw_advantages.clamp(min=-adv_clip, max=adv_clip) * response_mask
    else:
        clip_ratio = 0.0
        advantages = raw_advantages
    advantages = advantages.detach()

    corrected_proxy = teacher_z + beta * residual
    order_eps = max(min_group_std, 1e-6)
    metrics: dict[str, float] = {
        "gc_opd/enabled": 1.0,
        "gc_opd/base_ref_input_present_but_unused": float(base_ref_input_present),
        "gc_opd/base_correction_used": 0.0,
        "gc_opd/group_source_uid": source_uid,
        "gc_opd/group_source_contiguous_fallback": contiguous_fallback,
        "gc_opd/group_size": float(group_size),
        "gc_opd/reward_mean": seq_rewards.float().mean().item(),
        "gc_opd/teacher_score_mean": teacher_score.float().mean().item(),
        "gc_opd/reward_z_mean": reward_z.float().mean().item(),
        "gc_opd/teacher_score_z_mean": teacher_z.float().mean().item(),
        "gc_opd/residual_mean": residual.float().mean().item(),
        "gc_opd/residual_abs_mean": residual.float().abs().mean().item(),
        "gc_opd/residual_std": residual.float().std(unbiased=False).item(),
        "gc_opd/group_identifiable_ratio": residual_identifiable.float().mean().item(),
        "gc_opd/token_credit_identifiable_ratio": token_credit_identifiable.float()
        .mean()
        .item(),
        "gc_opd/order_acc_before": _pairwise_order_accuracy(
            seq_rewards,
            teacher_z,
            groups,
            eps=order_eps,
        ).item(),
        "gc_opd/order_acc_after_proxy": _pairwise_order_accuracy(
            seq_rewards,
            corrected_proxy,
            groups,
            eps=order_eps,
        ).item(),
        "gc_opd/residual_beta": beta,
        "gc_opd/credit_cap": credit_cap,
        "gc_opd/final_adv_clip_ratio": float(clip_ratio),
    }

    if valid.any():
        residual_per_token = residual[:, None].expand_as(opd_advantages)
        opposed = (
            valid & (opd_advantages.abs() > 1e-8) & (residual_per_token.abs() > 1e-8)
        )
        sign_flip_valid = valid & (opd_advantages.abs() > 1e-8)
        metrics.update(
            {
                "gc_opd/credit_weight_mean": credit[valid].float().mean().item(),
                "gc_opd/credit_weight_max": credit[valid].float().max().item(),
                "gc_opd/credit_zero_ratio": (credit[valid] <= 0).float().mean().item(),
                "gc_opd/credit_cap_ratio": (
                    credit[valid] >= max(credit_cap - 1e-6, 0.0)
                )
                .float()
                .mean()
                .item(),
                "gc_opd/opd_adv_mean": opd_advantages[valid].float().mean().item(),
                "gc_opd/opd_adv_abs_mean": opd_advantages[valid]
                .float()
                .abs()
                .mean()
                .item(),
                "gc_opd/residual_adv_abs_mean": residual_advantages[valid]
                .float()
                .abs()
                .mean()
                .item(),
                "gc_opd/opd_residual_opposition_ratio": (
                    (opd_advantages[opposed] * residual_per_token[opposed]) < 0
                )
                .float()
                .mean()
                .item()
                if opposed.any()
                else 0.0,
                "gc_opd/final_sign_flip_ratio": (
                    torch.sign(advantages[sign_flip_valid])
                    != torch.sign(opd_advantages[sign_flip_valid])
                )
                .float()
                .mean()
                .item()
                if sign_flip_valid.any()
                else 0.0,
                "gc_opd/final_adv_mean": advantages[valid].float().mean().item(),
                "gc_opd/final_adv_abs_mean": advantages[valid]
                .float()
                .abs()
                .mean()
                .item(),
            }
        )
    else:
        metrics.update(
            {
                "gc_opd/credit_weight_mean": 0.0,
                "gc_opd/credit_weight_max": 0.0,
                "gc_opd/credit_zero_ratio": 0.0,
                "gc_opd/credit_cap_ratio": 0.0,
                "gc_opd/opd_adv_mean": 0.0,
                "gc_opd/opd_adv_abs_mean": 0.0,
                "gc_opd/residual_adv_abs_mean": 0.0,
                "gc_opd/opd_residual_opposition_ratio": 0.0,
                "gc_opd/final_sign_flip_ratio": 0.0,
                "gc_opd/final_adv_mean": 0.0,
                "gc_opd/final_adv_abs_mean": 0.0,
            }
        )

    return {
        "gc_opd_advantages": advantages,
        "gc_opd_token_weight": credit * response_mask,
        "gc_opd_residual": residual,
        "gc_opd_teacher_score": teacher_score.detach(),
    }, metrics


def add_gc_opd_to_batch(
    batch: DataProto,
    *,
    policy_loss_config: Any,
    rollout_n: int,
) -> tuple[DataProto, dict[str, float]]:
    """Compute GC-OPD tensors once before actor mini-batch splitting."""

    required = ["old_log_probs", "ref_log_prob", "response_mask", "token_level_scores"]
    missing = [key for key in required if key not in batch.batch]
    if missing:
        raise KeyError(
            f"GC-OPD requires batch tensors {missing}, available={list(batch.batch.keys())}"
        )

    tensors, metrics = compute_gc_opd_tensors(
        old_log_prob=batch.batch["old_log_probs"],
        ref_log_prob=batch.batch["ref_log_prob"],
        response_mask=batch.batch["response_mask"],
        token_level_scores=batch.batch["token_level_scores"],
        base_ref_log_prob=batch.batch.get("base_ref_log_prob", None),
        uids=batch.non_tensor_batch.get("uid", None),
        rollout_n=rollout_n,
        policy_loss_config=policy_loss_config,
    )
    for key, value in tensors.items():
        batch.batch[key] = value
    return batch, metrics

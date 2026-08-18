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
"""Uni-OPD-lite outcome-guided margin calibration.

The public Uni-OPD recipe combines data balancing, online rollout filtering,
and teacher reliability calibration. This module wires the calibration part into
the clean OPD-family matrix: compute per-sample margin shifts from outcome
correctness, then let actor workers add those shifts to current OPD advantages.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch


def _cfg(config: Any, name: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_uid(uid: Any) -> Any:
    if hasattr(uid, "item"):
        try:
            return uid.item()
        except Exception:
            pass
    return uid


def _groups_from_uids(
    *,
    batch_size: int,
    uids: Sequence[Any] | None,
    rollout_n: int,
    scope: str,
) -> tuple[list[list[int]], dict[str, float]]:
    if scope == "global":
        groups = [list(range(batch_size))]
        source = 3.0
    elif uids is not None and len(uids) == batch_size:
        uid_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid_to_indices[_normalize_uid(uid)].append(idx)
        groups = list(uid_to_indices.values())
        source = 1.0
    elif rollout_n > 1 and batch_size % rollout_n == 0:
        groups = [list(range(start, start + rollout_n)) for start in range(0, batch_size, rollout_n)]
        source = 2.0
    else:
        groups = [list(range(batch_size))]
        source = 0.0

    sizes = torch.tensor([len(group) for group in groups], dtype=torch.float32)
    metrics = {
        "uni_opd/group_source": source,  # 3=global, 1=uid, 2=rollout_n fallback, 0=global fallback
        "uni_opd/group_count": float(len(groups)),
        "uni_opd/group_size_mean": sizes.mean().item() if sizes.numel() else 0.0,
    }
    return groups, metrics


def _masked_seq_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    seq_len = mask.sum(dim=-1).clamp(min=1.0)
    return (values * mask).sum(dim=-1) / seq_len


def _group_shift(
    *,
    mean_advantages: torch.Tensor,
    correct_mask: torch.Tensor,
    group: list[int],
    mode: str,
    delta: float,
) -> tuple[float, float, float]:
    group_idx = torch.tensor(group, device=mean_advantages.device, dtype=torch.long)
    group_correct = group_idx[correct_mask[group_idx]]
    group_incorrect = group_idx[~correct_mask[group_idx]]
    if group_correct.numel() == 0 or group_incorrect.numel() == 0:
        return 0.0, 0.0, 0.0

    correct_adv = mean_advantages[group_correct]
    incorrect_adv = mean_advantages[group_incorrect]
    if mode == "mean":
        correct_stat = correct_adv.mean()
        incorrect_stat = incorrect_adv.mean()
    elif mode == "minmax":
        correct_stat = correct_adv.min()
        incorrect_stat = incorrect_adv.max()
    else:
        raise ValueError(f"uni_opd_margin_mode must be 'mean' or 'minmax', got {mode!r}")

    gap = (correct_stat - incorrect_stat).item()
    if gap >= delta:
        return 0.0, gap, gap
    shift = (incorrect_stat - correct_stat).item() + delta
    return shift, gap, gap + shift


def compute_uni_opd_margin_shifts(
    *,
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor,
    policy_loss_config: Any,
    uids: Sequence[Any] | None = None,
    rollout_n: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Return per-sample margin shifts for Uni-OPD-lite.

    Official scripts enable margin shift with group scope, mean mode, both-side
    direction, and delta=0.4. The shift is computed from sequence-mean OPD
    advantages and outcome correctness ``sum(token_level_scores) > 0``.
    """

    use_margin_shift = _as_bool(_cfg(policy_loss_config, "uni_opd_use_margin_shift", True))
    scope = str(_cfg(policy_loss_config, "uni_opd_margin_scope", "group")).strip().lower()
    mode = str(_cfg(policy_loss_config, "uni_opd_margin_mode", "mean")).strip().lower()
    direction = str(_cfg(policy_loss_config, "uni_opd_margin_direction", "both")).strip().lower()
    delta = float(_cfg(policy_loss_config, "uni_opd_margin_delta", 0.4))

    if scope not in {"global", "group"}:
        raise ValueError(f"uni_opd_margin_scope must be 'global' or 'group', got {scope!r}")
    if direction not in {"correct_up", "incorrect_down", "both"}:
        raise ValueError(
            "uni_opd_margin_direction must be 'correct_up', 'incorrect_down', or 'both', "
            f"got {direction!r}"
        )
    if delta < 0.0:
        raise ValueError(f"uni_opd_margin_delta must be >= 0, got {delta}")

    dtype = old_log_probs.dtype
    device = old_log_probs.device
    response_mask = response_mask.to(device=device, dtype=dtype)
    ref_log_prob = ref_log_prob.to(device=device, dtype=dtype)
    token_level_scores = token_level_scores.to(device=device, dtype=dtype)

    sample_shift = torch.zeros(old_log_probs.shape[0], device=device, dtype=dtype)
    rewards = token_level_scores.sum(dim=-1)
    correct_mask = rewards > 0
    groups, group_metrics = _groups_from_uids(
        batch_size=old_log_probs.shape[0],
        uids=uids,
        rollout_n=rollout_n,
        scope=scope,
    )

    metrics: dict[str, float] = {
        "uni_opd/enabled": 1.0,
        "uni_opd/use_margin_shift": float(use_margin_shift),
        "uni_opd/margin_delta": delta,
        "uni_opd/margin_scope_group": float(scope == "group"),
        "uni_opd/margin_mode_mean": float(mode == "mean"),
        "uni_opd/margin_direction_both": float(direction == "both"),
        "uni_opd/correct_ratio": correct_mask.float().mean().item(),
        "uni_opd/shifted_group_count": 0.0,
        "uni_opd/shift_mean": 0.0,
        "uni_opd/gap_before_mean": 0.0,
        "uni_opd/gap_after_mean": 0.0,
    }
    metrics.update(group_metrics)

    if not use_margin_shift:
        return {"uni_opd_sample_shift": sample_shift.detach()}, metrics

    advantages = (ref_log_prob - old_log_probs) * response_mask
    mean_advantages = _masked_seq_mean(advantages, response_mask)
    applied_shifts: list[float] = []
    gaps_before: list[float] = []
    gaps_after: list[float] = []

    for group in groups:
        shift, gap_before, gap_after = _group_shift(
            mean_advantages=mean_advantages,
            correct_mask=correct_mask,
            group=group,
            mode=mode,
            delta=delta,
        )
        if shift <= 0.0:
            continue
        group_idx = torch.tensor(group, device=device, dtype=torch.long)
        group_correct = group_idx[correct_mask[group_idx]]
        group_incorrect = group_idx[~correct_mask[group_idx]]
        if direction == "correct_up":
            sample_shift[group_correct] += shift
        elif direction == "incorrect_down":
            sample_shift[group_incorrect] -= shift
        else:
            sample_shift[group_correct] += shift / 2.0
            sample_shift[group_incorrect] -= shift / 2.0
        applied_shifts.append(shift)
        gaps_before.append(gap_before)
        gaps_after.append(gap_after)

    if applied_shifts:
        shifts = torch.tensor(applied_shifts, dtype=torch.float32)
        metrics["uni_opd/shifted_group_count"] = float(len(applied_shifts))
        metrics["uni_opd/shift_mean"] = shifts.mean().item()
        metrics["uni_opd/shift_max"] = shifts.max().item()
        metrics["uni_opd/gap_before_mean"] = float(sum(gaps_before) / len(gaps_before))
        metrics["uni_opd/gap_after_mean"] = float(sum(gaps_after) / len(gaps_after))
    else:
        metrics["uni_opd/shift_max"] = 0.0

    return {"uni_opd_sample_shift": sample_shift.detach()}, metrics

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

from __future__ import annotations

from typing import Any

import torch


def _teacher_type_at(opd_teacher: Any, index: int) -> Any:
    if opd_teacher is None or isinstance(opd_teacher, (str, bytes)):
        return opd_teacher
    if isinstance(opd_teacher, (list, tuple)):
        return opd_teacher[index]
    if hasattr(opd_teacher, "__len__") and hasattr(opd_teacher, "__getitem__"):
        try:
            return opd_teacher[index]
        except Exception:
            return opd_teacher
    return opd_teacher


def compute_entropy_aware_distill_weights(
    *,
    policy_loss_config: Any,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    student_entropys: torch.Tensor,
    ref_entropys: torch.Tensor,
    base_ref_log_prob: torch.Tensor | None = None,
    base_ref_entropys: torch.Tensor | None = None,
    opd_teacher: Any = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Compute FiRe-OPD trajectory filtering and entropy-aware token weights."""

    bsz, _ = ref_log_prob.shape
    response_mask = response_mask.to(dtype=ref_log_prob.dtype)
    multi_teacher = bool(getattr(policy_loss_config, "multi_teacher_distill", False))
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)

    if multi_teacher and opd_teacher is not None and base_ref_log_prob is not None:
        normalized_teacher_logprob = torch.zeros(bsz, device=ref_log_prob.device, dtype=ref_log_prob.dtype)
        for i in range(bsz):
            selected_log_prob = base_ref_log_prob[i] if _teacher_type_at(opd_teacher, i) == "code" else ref_log_prob[i]
            normalized_teacher_logprob[i] = (selected_log_prob * response_mask[i]).sum() / seq_lengths[i]
    else:
        normalized_teacher_logprob = (ref_log_prob * response_mask).sum(dim=-1) / seq_lengths

    skip_percentile = float(getattr(policy_loss_config, "traj_skip_percentile", 20.0))
    logprob_threshold = torch.quantile(normalized_teacher_logprob.float(), skip_percentile / 100.0)
    traj_keep_mask = normalized_teacher_logprob >= logprob_threshold.to(dtype=normalized_teacher_logprob.dtype)

    if multi_teacher and opd_teacher is not None and base_ref_entropys is not None:
        teacher_entropys = ref_entropys.clone()
        for i in range(bsz):
            if _teacher_type_at(opd_teacher, i) == "code":
                teacher_entropys[i] = base_ref_entropys[i]
    else:
        teacher_entropys = ref_entropys

    valid_response_mask = response_mask.bool()
    valid_teacher_entropys = teacher_entropys[valid_response_mask]
    if valid_teacher_entropys.numel() > 0:
        teacher_entropy_max = valid_teacher_entropys.max().clamp(min=1e-6)
    else:
        teacher_entropy_max = torch.tensor(1.0, device=teacher_entropys.device, dtype=teacher_entropys.dtype)
    teacher_confidence = (1.0 - teacher_entropys / teacher_entropy_max).clamp(min=0.0, max=1.0)

    valid_student_entropys = student_entropys[valid_response_mask]
    if valid_student_entropys.numel() > 0:
        student_entropy_max = valid_student_entropys.max().clamp(min=1e-6)
    else:
        student_entropy_max = torch.tensor(1.0, device=student_entropys.device, dtype=student_entropys.dtype)
    student_confusion = (student_entropys / student_entropy_max).clamp(min=0.0, max=1.0)

    alpha = float(getattr(policy_loss_config, "entropy_alpha", 1.0))
    beta = float(getattr(policy_loss_config, "entropy_beta", 1.0))
    token_weight = ((1.0 + alpha * teacher_confidence) * (1.0 + beta * student_confusion)).detach()

    valid_weight_sum = (token_weight * response_mask).sum()
    valid_token_count = response_mask.sum().clamp(min=1)
    token_weight = token_weight / (valid_weight_sum / valid_token_count).clamp(min=1e-6)

    final_keep_mask = traj_keep_mask
    traj_weight = torch.ones_like(normalized_teacher_logprob)

    traj_keep_expanded = final_keep_mask[:, None].to(dtype=response_mask.dtype).expand_as(response_mask)
    effective_mask = response_mask * traj_keep_expanded

    metrics: dict[str, float] = {
        "fire_opd/traj_keep_ratio": traj_keep_mask.float().mean().item(),
        "fire_opd/logprob_threshold": logprob_threshold.item(),
        "fire_opd/normalized_teacher_logprob_mean": normalized_teacher_logprob.mean().item(),
        "fire_opd/final_keep_ratio": final_keep_mask.float().mean().item(),
    }
    valid_effective_mask = effective_mask.bool()
    valid_weights = token_weight[valid_effective_mask]
    if valid_weights.numel() > 0:
        metrics["fire_opd/token_weight_mean"] = valid_weights.mean().item()
        metrics["fire_opd/token_weight_max"] = valid_weights.max().item()
        metrics["fire_opd/token_weight_min"] = valid_weights.min().item()

    valid_tc = teacher_confidence[valid_effective_mask]
    valid_sc = student_confusion[valid_effective_mask]
    if valid_tc.numel() > 0:
        metrics["fire_opd/teacher_confidence_mean"] = valid_tc.mean().item()
        metrics["fire_opd/student_confusion_mean"] = valid_sc.mean().item()

    if valid_teacher_entropys.numel() > 0:
        metrics["fire_opd/teacher_entropy_mean"] = valid_teacher_entropys.mean().item()
    if valid_student_entropys.numel() > 0:
        metrics["fire_opd/student_entropy_mean"] = valid_student_entropys.mean().item()

    tensors = {
        "token_weight": token_weight,
        "traj_keep_mask": traj_keep_mask,
        "final_keep_mask": final_keep_mask,
        "traj_weight": traj_weight,
        "effective_mask": effective_mask,
        "final_token_weight": token_weight,
        "teacher_confidence": teacher_confidence,
        "student_confusion": student_confusion,
        "teacher_entropys": teacher_entropys,
        "normalized_teacher_logprob": normalized_teacher_logprob,
    }
    return tensors, metrics

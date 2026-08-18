#!/usr/bin/env python3
"""Analyze frozen OPD rollouts as counterfactual GC-OPD updates.

This script is deliberately training-free.  It consumes trusted-local replay
packets containing sibling rollouts, verifier rewards, student log-probability,
and teacher log-probability. Every GC-OPD credit mode and beta value is then
computed on exactly the same rollouts.

The output directory contains:

* ``summary.json`` and ``report.md`` for paper-facing conclusions;
* group, length, task, beta, and credit-mode CSV files;
* ``mechanism_analysis.{png,pdf}``;
* decoded high-teacher/low-reward cases when ``--tokenizer`` is supplied;
* ``provenance.json`` with packet hashes and all analysis arguments.

Replay packets are pickle-backed through ``DataProto.load_from_disk``.  Only
analyze bundles produced by a trusted local training job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from verl import DataProto


PROMPT_BUCKETS = (
    ("<8k", 0, 8_192),
    ("8–16k", 8_192, 16_384),
    ("16–24k", 16_384, 24_576),
    ("24–32k", 24_576, None),
)
RESPONSE_BUCKETS = (
    ("<1k", 0, 1_024),
    ("1–4k", 1_024, 4_096),
    ("4–8k", 4_096, 8_192),
    ("≥8k", 8_192, None),
)
DEFAULT_BETAS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0)
DEFAULT_CREDIT_MODES = ("uniform", "teacher_abs", "rol", "rts", "raca")


@dataclass
class GroupRow:
    packet: str
    global_step: int
    uid: str
    task: str
    num_rollouts: int
    prompt_length: float
    response_length: float
    reward_std: float
    teacher_std: float
    reward_variable: int
    identifiable: int
    pair_correct: int
    pair_total: int
    top1_correct: int
    top1_regret: float
    mean_abs_residual: float


@dataclass
class PairRow:
    prompt_bucket: str
    response_bucket: str
    task: str
    reward_diff: float
    teacher_diff: float
    teacher_correct: int
    residual_flip_threshold: float | None
    direct_flip_threshold: float | None


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bucket(value: float, buckets: Iterable[tuple[str, int, int | None]]) -> str:
    for name, lower, upper in buckets:
        if value >= lower and (upper is None or value < upper):
            return name
    raise ValueError(f"no bucket for value={value}")


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _safe_json(value: Any) -> Any:
    value = _safe_scalar(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _task_name(data: DataProto, index: int) -> str:
    non_tensor = data.non_tensor_batch
    candidates: list[Any] = []
    for key in ("data_source", "ability", "task", "dataset"):
        if key in non_tensor:
            candidates.append(non_tensor[key][index])
    if "extra_info" in non_tensor:
        extra = non_tensor["extra_info"][index]
        if isinstance(extra, dict):
            candidates.extend(
                extra.get(key) for key in ("ability", "task", "data_source")
            )
    if "reward_model" in non_tensor:
        reward_model = non_tensor["reward_model"][index]
        if isinstance(reward_model, dict):
            candidates.extend(
                reward_model.get(key) for key in ("style", "type", "name")
            )
    for candidate in candidates:
        candidate = _safe_scalar(candidate)
        if candidate not in (None, "", "None"):
            return str(candidate)
    return "unknown"


def _groups_from_uid(data: DataProto, group_size: int) -> list[tuple[str, list[int]]]:
    uids = data.non_tensor_batch.get("uid")
    if uids is not None and len(uids) == len(data):
        groups: dict[str, list[int]] = {}
        for index, uid in enumerate(uids):
            groups.setdefault(str(_safe_scalar(uid)), []).append(index)
        return list(groups.items())
    return [
        (
            f"contiguous-{start // group_size}",
            list(range(start, min(start + group_size, len(data)))),
        )
        for start in range(0, len(data), group_size)
    ]


def _zscore(values: torch.Tensor, eps: float) -> tuple[torch.Tensor, bool]:
    values = values.float()
    if values.numel() < 2:
        return torch.zeros_like(values), False
    std = values.std(unbiased=False)
    if float(std) <= eps:
        return torch.zeros_like(values), False
    return (values - values.mean()) / std.clamp(min=eps), True


def _token_zscore(
    values: torch.Tensor, mask: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    mask = mask.float()
    counts = mask.sum(dim=-1, keepdim=True)
    mean = (values * mask).sum(dim=-1, keepdim=True) / counts.clamp(min=1.0)
    centered = (values - mean) * mask
    std = (centered.square().sum(dim=-1, keepdim=True) / counts.clamp(min=1.0)).sqrt()
    identifiable = (counts >= 2) & (std > eps)
    zscore = torch.where(
        identifiable, centered / std.clamp(min=eps), torch.zeros_like(centered)
    )
    return zscore * mask, identifiable.squeeze(-1)


def _credit_weights(
    mode: str,
    opd: torch.Tensor,
    mask: torch.Tensor,
    *,
    cap: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the GC-OPD credit definitions without importing trainer state."""

    mode = mode.lower().replace("-", "_")
    aliases = {
        "relative_opd_leverage": "rol",
        "relative_teacher_support": "rts",
    }
    mode = aliases.get(mode, mode)
    mask = mask.float()
    if mode == "uniform":
        return mask, mask.sum(dim=-1) > 0
    if mode == "teacher_abs":
        raw = opd.float().abs() * mask
        count = mask.sum(dim=-1, keepdim=True)
        mass = raw.sum(dim=-1, keepdim=True)
        identifiable = (count > 0) & (mass > eps)
        normalized = raw * count.clamp(min=1.0) / mass.clamp(min=eps)
        credit = torch.where(identifiable, normalized.clamp(0.0, cap), mask)
        return credit * mask, identifiable.squeeze(-1)

    zscore, identifiable = _token_zscore(opd, mask, eps)
    if mode == "rol":
        raw = zscore.abs()
    elif mode == "rts":
        raw = 1.0 + zscore
    elif mode == "raca":
        raw = 1.0 + torch.tanh(zscore / 2.0)
    else:
        raise ValueError(f"unsupported credit mode: {mode}")
    credit = torch.where(identifiable[:, None], raw.clamp(0.0, cap), mask)
    return credit * mask, identifiable


def _gini(values: torch.Tensor, eps: float) -> float:
    values = values.float().flatten().clamp(min=0)
    if values.numel() == 0 or float(values.sum()) <= eps:
        return 0.0
    values = values.sort().values
    n = values.numel()
    ranks = torch.arange(1, n + 1, dtype=values.dtype)
    return float(
        2 * (ranks * values.cpu()).sum() / (n * values.sum().cpu()) - (n + 1) / n
    )


def _top_mass(values: torch.Tensor, fraction: float, eps: float) -> float:
    values = values.float().flatten().clamp(min=0)
    total = values.sum()
    if values.numel() == 0 or float(total) <= eps:
        return 0.0
    k = max(1, math.ceil(values.numel() * fraction))
    return float(values.topk(k).values.sum() / total.clamp(min=eps))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    denominator = denominators.sum()
    point = float(numerators.sum() / denominator) if denominator else float("nan")
    if len(numerators) == 0 or denominator == 0 or reps <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        indices = rng.integers(0, len(numerators), size=len(numerators))
        sampled_denominator = denominators[indices].sum()
        draws[rep] = (
            numerators[indices].sum() / sampled_denominator
            if sampled_denominator
            else np.nan
        )
    valid = draws[np.isfinite(draws)]
    if not len(valid):
        return point, float("nan"), float("nan")
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return point, float(lower), float(upper)


def _summarize_groups(
    rows: list[GroupRow],
    *,
    bucket_attr: str | None,
    bucket_order: tuple[str, ...] | None,
    bootstrap_reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[GroupRow]] = defaultdict(list)
    if bucket_attr is None:
        grouped["all"] = rows
    else:
        for row in rows:
            grouped[str(getattr(row, bucket_attr))].append(row)

    names = list(grouped)
    if bucket_order:
        ordered = [name for name in bucket_order if name in grouped]
        names = ordered + sorted(set(names) - set(ordered))

    output = []
    for offset, name in enumerate(names):
        subset = grouped[name]
        pair = _bootstrap_ratio(
            np.asarray([row.pair_correct for row in subset], dtype=float),
            np.asarray([row.pair_total for row in subset], dtype=float),
            reps=bootstrap_reps,
            seed=seed + offset,
        )
        top1 = _bootstrap_ratio(
            np.asarray(
                [
                    row.top1_correct if row.reward_variable else 0
                    for row in subset
                ],
                dtype=float,
            ),
            np.asarray([row.reward_variable for row in subset], dtype=float),
            reps=bootstrap_reps,
            seed=seed + 1000 + offset,
        )
        output.append(
            {
                "bucket": name,
                "groups": len(subset),
                "rollouts": sum(row.num_rollouts for row in subset),
                "reward_variable_groups": sum(row.reward_variable for row in subset),
                "identifiable_groups": sum(row.identifiable for row in subset),
                "pairwise_accuracy": pair[0],
                "pairwise_ci_low": pair[1],
                "pairwise_ci_high": pair[2],
                "pairwise_mismatch": 1.0 - pair[0]
                if math.isfinite(pair[0])
                else float("nan"),
                "top1_accuracy": top1[0],
                "top1_ci_low": top1[1],
                "top1_ci_high": top1[2],
                "top1_regret_mean": float(
                    np.mean(
                        [
                            row.top1_regret
                            for row in subset
                            if row.reward_variable
                        ]
                    )
                )
                if any(row.reward_variable for row in subset)
                else float("nan"),
                "prompt_length_mean": float(
                    np.mean([row.prompt_length for row in subset])
                )
                if subset
                else float("nan"),
                "response_length_mean": float(
                    np.mean([row.response_length for row in subset])
                )
                if subset
                else float("nan"),
                "residual_abs_mean": float(
                    np.mean([row.mean_abs_residual for row in subset])
                )
                if subset
                else float("nan"),
            }
        )
    return output


def _pair_proxy_sweep(
    pair_rows: list[PairRow], betas: tuple[float, ...]
) -> list[dict[str, Any]]:
    output = []
    for beta in betas:
        for method in ("teacher", "direct_reward", "residual"):
            correct = 0
            total = 0
            repaired = 0
            misranked = 0
            for row in pair_rows:
                reward_diff = row.reward_diff
                teacher_diff = row.teacher_diff
                if abs(reward_diff) <= 1e-12 or abs(teacher_diff) <= 1e-12:
                    continue
                if method == "teacher":
                    score_diff = teacher_diff
                elif method == "direct_reward":
                    score_diff = teacher_diff + beta * reward_diff
                else:
                    score_diff = (1.0 - beta) * teacher_diff + beta * reward_diff
                is_correct = reward_diff * score_diff > 0
                correct += int(is_correct)
                total += 1
                if not row.teacher_correct:
                    misranked += 1
                    repaired += int(is_correct)
            output.append(
                {
                    "beta": beta,
                    "sequence_proxy": method,
                    "pairwise_accuracy": correct / total if total else float("nan"),
                    "repaired_misrank_rate": repaired / misranked
                    if misranked
                    else float("nan"),
                    "pairs": total,
                    "misranked_pairs": misranked,
                }
            )
    return output


def _decode_case(
    candidate: dict[str, Any],
    tokenizer: Any | None,
    *,
    max_prompt_chars: int,
    max_response_chars: int,
) -> dict[str, Any]:
    output = {
        key: value for key, value in candidate.items() if not key.endswith("_ids")
    }
    if tokenizer is None:
        output["prompt_token_ids_head"] = candidate["prompt_ids"][:64]
        output["prompt_token_ids_tail"] = candidate["prompt_ids"][-64:]
        output["teacher_choice_token_ids_head"] = candidate["teacher_response_ids"][
            :128
        ]
        output["reward_choice_token_ids_head"] = candidate["reward_response_ids"][:128]
        return output
    prompt = tokenizer.decode(candidate["prompt_ids"], skip_special_tokens=True)
    teacher_response = tokenizer.decode(
        candidate["teacher_response_ids"], skip_special_tokens=True
    )
    reward_response = tokenizer.decode(
        candidate["reward_response_ids"], skip_special_tokens=True
    )
    output["prompt"] = prompt[-max_prompt_chars:] if max_prompt_chars else prompt
    output["teacher_choice_response"] = (
        teacher_response[:max_response_chars]
        if max_response_chars
        else teacher_response
    )
    output["reward_choice_response"] = (
        reward_response[:max_response_chars] if max_response_chars else reward_response
    )
    return output


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    bundle = Path(args.bundle).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_paths = sorted(bundle.glob(args.packets_glob))
    if args.max_packets:
        packet_paths = packet_paths[: args.max_packets]
    if not packet_paths:
        raise FileNotFoundError(f"no packets matched {bundle / args.packets_glob}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )

    group_rows: list[GroupRow] = []
    pair_rows: list[PairRow] = []
    case_heap: list[tuple[float, int, dict[str, Any]]] = []
    case_counter = 0
    credit_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "valid_tokens": 0,
            "identifiable_sequences": 0,
            "sequences": 0,
            "credit_sum": 0.0,
            "credit_max": 0.0,
            "zero_count": 0,
            "cap_count": 0,
            "gini_sum": 0.0,
            "top10_sum": 0.0,
            "positive_mass": 0.0,
            "negative_mass": 0.0,
            "segments": np.zeros(4, dtype=np.float64),
        }
    )
    beta_acc: dict[tuple[str, float], dict[str, float]] = defaultdict(
        lambda: {
            "valid_tokens": 0.0,
            "base_abs_sum": 0.0,
            "correction_abs_sum": 0.0,
            "sign_flip_count": 0.0,
            "sign_flip_den": 0.0,
            "opposition_count": 0.0,
            "opposition_den": 0.0,
            "clip_count": 0.0,
        }
    )
    packet_meta: list[dict[str, Any]] = []
    seen_non_tensor_keys: set[str] = set()

    for packet_path in packet_paths:
        data = DataProto.load_from_disk(packet_path)
        required = (
            "attention_mask",
            "response_mask",
            "old_log_probs",
            "ref_log_prob",
            "token_level_scores",
        )
        missing = [key for key in required if key not in data.batch]
        if missing:
            raise KeyError(f"{packet_path}: missing required tensors {missing}")

        mask = data.batch["response_mask"].float().cpu()
        old = data.batch["old_log_probs"].float().cpu()
        ref = data.batch["ref_log_prob"].float().cpu()
        opd = (ref - old) * mask
        rewards = (data.batch["token_level_scores"].float().cpu() * mask).sum(dim=-1)
        response_lengths = mask.sum(dim=-1)
        prompt_lengths = (
            data.batch["attention_mask"].float().cpu().sum(dim=-1) - response_lengths
        ).clamp(min=0)
        teacher_scores = opd.sum(dim=-1) / response_lengths.clamp(min=1)
        groups = _groups_from_uid(data, args.group_size)
        seen_non_tensor_keys.update(data.non_tensor_batch.keys())

        packet_meta.append(
            {
                "path": str(packet_path.relative_to(bundle)),
                "sha256": _sha256(packet_path),
                "num_rollouts": len(data),
                "num_groups": len(groups),
                "global_step": int(data.meta_info.get("global_step", -1)),
                "tensor_keys": sorted(data.batch.keys()),
                "non_tensor_keys": sorted(data.non_tensor_batch.keys()),
            }
        )

        residual_by_row = torch.zeros(len(data), dtype=torch.float32)
        for uid, indices in groups:
            idx = torch.as_tensor(indices, dtype=torch.long)
            group_rewards = rewards[idx]
            group_teacher = teacher_scores[idx]
            reward_z, reward_ok = _zscore(group_rewards, args.eps)
            teacher_z, teacher_ok = _zscore(group_teacher, args.eps)
            identifiable = reward_ok and teacher_ok
            residual = (
                reward_z - teacher_z if identifiable else torch.zeros_like(reward_z)
            )
            residual_by_row[idx] = residual

            prompt_length = float(prompt_lengths[idx].mean())
            response_length = float(response_lengths[idx].mean())
            tasks = [_task_name(data, item) for item in indices]
            task = Counter(tasks).most_common(1)[0][0]
            pair_correct = 0
            pair_total = 0
            for left_pos, left_local in enumerate(range(len(indices))):
                for right_local in range(left_local + 1, len(indices)):
                    reward_diff = float(
                        group_rewards[left_local] - group_rewards[right_local]
                    )
                    teacher_diff = float(
                        group_teacher[left_local] - group_teacher[right_local]
                    )
                    if abs(reward_diff) <= args.eps or abs(teacher_diff) <= args.eps:
                        continue
                    teacher_correct = int(reward_diff * teacher_diff > 0)
                    pair_correct += teacher_correct
                    pair_total += 1
                    residual_threshold = None
                    direct_threshold = None
                    if not teacher_correct:
                        oriented_reward = abs(reward_diff)
                        oriented_teacher = -abs(teacher_diff)
                        residual_threshold = -oriented_teacher / (
                            oriented_reward - oriented_teacher
                        )
                        direct_threshold = -oriented_teacher / oriented_reward
                    pair_rows.append(
                        PairRow(
                            prompt_bucket=_bucket(prompt_length, PROMPT_BUCKETS),
                            response_bucket=_bucket(response_length, RESPONSE_BUCKETS),
                            task=task,
                            reward_diff=reward_diff,
                            teacher_diff=teacher_diff,
                            teacher_correct=teacher_correct,
                            residual_flip_threshold=residual_threshold,
                            direct_flip_threshold=direct_threshold,
                        )
                    )

            teacher_best_local = int(torch.argmax(group_teacher))
            reward_best_local = int(torch.argmax(group_rewards))
            teacher_best_reward = float(group_rewards[teacher_best_local])
            max_reward = float(group_rewards.max())
            top1_regret = max_reward - teacher_best_reward
            top1_correct = int(top1_regret <= args.eps)
            group_rows.append(
                GroupRow(
                    packet=str(packet_path.relative_to(bundle)),
                    global_step=int(data.meta_info.get("global_step", -1)),
                    uid=uid,
                    task=task,
                    num_rollouts=len(indices),
                    prompt_length=prompt_length,
                    response_length=response_length,
                    reward_std=float(group_rewards.std(unbiased=False)),
                    teacher_std=float(group_teacher.std(unbiased=False)),
                    reward_variable=int(reward_ok),
                    identifiable=int(identifiable),
                    pair_correct=pair_correct,
                    pair_total=pair_total,
                    top1_correct=top1_correct,
                    top1_regret=top1_regret,
                    mean_abs_residual=float(residual.abs().mean()),
                )
            )

            if top1_regret > args.eps and "responses" in data.batch:
                teacher_global = indices[teacher_best_local]
                reward_global = indices[reward_best_local]
                response_ids = data.batch["responses"].cpu()
                input_ids = data.batch["input_ids"].cpu()
                attention_mask = data.batch["attention_mask"].bool().cpu()
                teacher_len = int(response_lengths[teacher_global])
                reward_len = int(response_lengths[reward_global])
                if "prompts" in data.batch:
                    prompt_tensor = data.batch["prompts"].cpu()
                    prompt_width = prompt_tensor.shape[1]
                    prompt_valid = attention_mask[teacher_global, :prompt_width]
                    prompt_ids = prompt_tensor[teacher_global][prompt_valid].tolist()
                else:
                    prompt_width = input_ids.shape[1] - response_ids.shape[1]
                    prompt_valid = attention_mask[teacher_global, :prompt_width]
                    prompt_ids = input_ids[teacher_global, :prompt_width][
                        prompt_valid
                    ].tolist()
                candidate = {
                    "packet": str(packet_path.relative_to(bundle)),
                    "global_step": int(data.meta_info.get("global_step", -1)),
                    "uid": uid,
                    "task": task,
                    "prompt_length": prompt_length,
                    "teacher_choice_reward": teacher_best_reward,
                    "reward_choice_reward": max_reward,
                    "reward_regret": top1_regret,
                    "teacher_choice_score": float(group_teacher[teacher_best_local]),
                    "reward_choice_teacher_score": float(
                        group_teacher[reward_best_local]
                    ),
                    "teacher_score_gap": float(
                        group_teacher[teacher_best_local]
                        - group_teacher[reward_best_local]
                    ),
                    "prompt_ids": prompt_ids,
                    "teacher_response_ids": response_ids[
                        teacher_global, :teacher_len
                    ].tolist(),
                    "reward_response_ids": response_ids[
                        reward_global, :reward_len
                    ].tolist(),
                    "teacher_choice_extra_info": _safe_json(
                        data.non_tensor_batch.get("extra_info", [None] * len(data))[
                            teacher_global
                        ]
                    ),
                    "reward_choice_extra_info": _safe_json(
                        data.non_tensor_batch.get("extra_info", [None] * len(data))[
                            reward_global
                        ]
                    ),
                }
                priority = top1_regret * (
                    1.0 + max(candidate["teacher_score_gap"], 0.0)
                )
                heapq.heappush(case_heap, (priority, case_counter, candidate))
                case_counter += 1
                if len(case_heap) > args.max_cases:
                    heapq.heappop(case_heap)

        valid = mask.bool()
        residual_token = residual_by_row[:, None].expand_as(opd)
        for mode in args.credit_modes:
            credit, credit_identifiable = _credit_weights(
                mode, opd, mask, cap=args.credit_cap, eps=args.eps
            )
            stats = credit_acc[mode]
            selected = credit[valid]
            selected_opd = opd[valid]
            stats["valid_tokens"] += int(selected.numel())
            stats["identifiable_sequences"] += int(credit_identifiable.sum())
            stats["sequences"] += len(data)
            stats["credit_sum"] += float(selected.sum())
            stats["credit_max"] = max(
                stats["credit_max"], float(selected.max()) if selected.numel() else 0.0
            )
            stats["zero_count"] += int((selected <= args.eps).sum())
            stats["cap_count"] += int((selected >= args.credit_cap - args.eps).sum())
            stats["positive_mass"] += float(selected[selected_opd > 0].sum())
            stats["negative_mass"] += float(selected[selected_opd < 0].sum())
            for row_credit, row_mask in zip(credit, valid, strict=True):
                values = row_credit[row_mask]
                if values.numel() == 0:
                    continue
                stats["gini_sum"] += _gini(values, args.eps)
                stats["top10_sum"] += _top_mass(values, 0.10, args.eps)
                positions = torch.arange(values.numel())
                segment_ids = (positions * 4 // values.numel()).clamp(max=3)
                segment_mass = torch.zeros(4)
                segment_mass.scatter_add_(0, segment_ids, values.float())
                if float(segment_mass.sum()) > args.eps:
                    stats["segments"] += (segment_mass / segment_mass.sum()).numpy()

            base = opd[valid]
            residual_selected = residual_token[valid]
            credit_selected = credit[valid]
            for beta in args.betas:
                correction = beta * residual_selected * credit_selected
                raw = base + correction
                clipped = (
                    raw.clamp(-args.adv_clip, args.adv_clip)
                    if args.adv_clip > 0
                    else raw
                )
                acc = beta_acc[(mode, beta)]
                acc["valid_tokens"] += float(base.numel())
                acc["base_abs_sum"] += float(base.abs().sum())
                acc["correction_abs_sum"] += float(correction.abs().sum())
                nonzero_base = base.abs() > args.eps
                acc["sign_flip_count"] += float(
                    (
                        torch.sign(clipped[nonzero_base])
                        != torch.sign(base[nonzero_base])
                    ).sum()
                )
                acc["sign_flip_den"] += float(nonzero_base.sum())
                comparable = nonzero_base & (correction.abs() > args.eps)
                acc["opposition_count"] += float(
                    (base[comparable] * correction[comparable] < 0).sum()
                )
                acc["opposition_den"] += float(comparable.sum())
                if args.adv_clip > 0:
                    acc["clip_count"] += float((raw.abs() > args.adv_clip).sum())

    group_dicts = [asdict(row) for row in group_rows]
    for row in group_dicts:
        row["prompt_bucket"] = _bucket(row["prompt_length"], PROMPT_BUCKETS)
        row["response_bucket"] = _bucket(row["response_length"], RESPONSE_BUCKETS)
    _write_csv(output_dir / "groups.csv", group_dicts)

    overall_rows = _summarize_groups(
        group_rows,
        bucket_attr=None,
        bucket_order=None,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    prompt_group_rows = [
        GroupRow(**{**asdict(row), "task": _bucket(row.prompt_length, PROMPT_BUCKETS)})
        for row in group_rows
    ]
    response_group_rows = [
        GroupRow(
            **{**asdict(row), "task": _bucket(row.response_length, RESPONSE_BUCKETS)}
        )
        for row in group_rows
    ]
    length_rows = []
    for axis, transformed, order in (
        ("prompt", prompt_group_rows, tuple(name for name, _, _ in PROMPT_BUCKETS)),
        (
            "response",
            response_group_rows,
            tuple(name for name, _, _ in RESPONSE_BUCKETS),
        ),
    ):
        rows = _summarize_groups(
            transformed,
            bucket_attr="task",
            bucket_order=order,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed + (0 if axis == "prompt" else 100),
        )
        for row in rows:
            row["axis"] = axis
        length_rows.extend(rows)
    _write_csv(output_dir / "length_buckets.csv", length_rows)

    task_rows = _summarize_groups(
        group_rows,
        bucket_attr="task",
        bucket_order=None,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed + 200,
    )
    _write_csv(output_dir / "task_buckets.csv", task_rows)

    credit_rows = []
    for mode in args.credit_modes:
        stats = credit_acc[mode]
        token_count = max(stats["valid_tokens"], 1)
        sequence_count = max(stats["sequences"], 1)
        total_signed_mass = stats["positive_mass"] + stats["negative_mass"]
        row = {
            "credit_mode": mode,
            "sequences": stats["sequences"],
            "valid_tokens": stats["valid_tokens"],
            "identifiable_sequence_ratio": stats["identifiable_sequences"]
            / sequence_count,
            "credit_mean": stats["credit_sum"] / token_count,
            "credit_max": stats["credit_max"],
            "zero_ratio": stats["zero_count"] / token_count,
            "cap_ratio": stats["cap_count"] / token_count,
            "gini_mean": stats["gini_sum"] / sequence_count,
            "top10pct_mass_mean": stats["top10_sum"] / sequence_count,
            "positive_teacher_credit_mass_ratio": (
                stats["positive_mass"] / total_signed_mass
                if total_signed_mass
                else float("nan")
            ),
        }
        for index, value in enumerate(stats["segments"] / sequence_count):
            row[f"segment_{index + 1}_mass"] = float(value)
        credit_rows.append(row)
    _write_csv(output_dir / "credit_modes.csv", credit_rows)

    beta_rows = []
    for mode in args.credit_modes:
        for beta in args.betas:
            stats = beta_acc[(mode, beta)]
            beta_rows.append(
                {
                    "credit_mode": mode,
                    "beta": beta,
                    "valid_tokens": int(stats["valid_tokens"]),
                    "correction_to_opd_l1_ratio": stats["correction_abs_sum"]
                    / max(stats["base_abs_sum"], args.eps),
                    "sign_flip_ratio": stats["sign_flip_count"]
                    / max(stats["sign_flip_den"], 1.0),
                    "opposition_ratio": stats["opposition_count"]
                    / max(stats["opposition_den"], 1.0),
                    "clip_ratio": stats["clip_count"] / max(stats["valid_tokens"], 1.0),
                }
            )
    _write_csv(output_dir / "beta_token_sweep.csv", beta_rows)

    proxy_rows = _pair_proxy_sweep(pair_rows, args.betas)
    _write_csv(output_dir / "beta_sequence_proxy.csv", proxy_rows)
    _write_csv(
        output_dir / "pairwise_rows.csv",
        [asdict(row) for row in pair_rows],
    )

    misrank_pairs = [row for row in pair_rows if not row.teacher_correct]
    residual_thresholds = np.asarray(
        [
            row.residual_flip_threshold
            for row in misrank_pairs
            if row.residual_flip_threshold is not None
        ],
        dtype=float,
    )
    direct_thresholds = np.asarray(
        [
            row.direct_flip_threshold
            for row in misrank_pairs
            if row.direct_flip_threshold is not None
        ],
        dtype=float,
    )
    threshold_summary = {
        "misranked_pairs": len(misrank_pairs),
        "residual_beta_threshold_median": float(np.median(residual_thresholds))
        if len(residual_thresholds)
        else float("nan"),
        "direct_reward_beta_threshold_median": float(np.median(direct_thresholds))
        if len(direct_thresholds)
        else float("nan"),
        "residual_threshold_lower_ratio": float(
            np.mean(residual_thresholds < direct_thresholds)
        )
        if len(residual_thresholds)
        else float("nan"),
        "median_threshold_reduction": float(
            1.0 - np.median(residual_thresholds) / np.median(direct_thresholds)
        )
        if len(residual_thresholds) and np.median(direct_thresholds) > 0
        else float("nan"),
    }

    selected_cases = [
        item[2] for item in sorted(case_heap, key=lambda item: item[0], reverse=True)
    ]
    decoded_cases: list[dict[str, Any]] = []
    with (output_dir / "misrank_cases.jsonl").open("w", encoding="utf-8") as handle:
        for rank, candidate in enumerate(selected_cases, start=1):
            decoded = _decode_case(
                candidate,
                tokenizer,
                max_prompt_chars=args.max_prompt_chars,
                max_response_chars=args.max_response_chars,
            )
            decoded["case_rank"] = rank
            decoded_cases.append(decoded)
            handle.write(json.dumps(decoded, ensure_ascii=False) + "\n")
    case_summary_rows = []
    for decoded in decoded_cases:
        case_summary_rows.append(
            {
                "case_rank": decoded["case_rank"],
                "global_step": decoded["global_step"],
                "uid": decoded["uid"],
                "task": decoded["task"],
                "prompt_length": decoded["prompt_length"],
                "teacher_choice_reward": decoded["teacher_choice_reward"],
                "reward_choice_reward": decoded["reward_choice_reward"],
                "reward_regret": decoded["reward_regret"],
                "teacher_choice_score": decoded["teacher_choice_score"],
                "reward_choice_teacher_score": decoded[
                    "reward_choice_teacher_score"
                ],
                "teacher_score_gap": decoded["teacher_score_gap"],
                "teacher_choice_response": " ".join(
                    decoded.get("teacher_choice_response", "").split()
                ),
                "reward_choice_response": " ".join(
                    decoded.get("reward_choice_response", "").split()
                ),
            }
        )
    _write_csv(output_dir / "misrank_cases.csv", case_summary_rows)

    manifest_path = bundle / "manifest.json"
    public_arguments = {
        key: _safe_json(value)
        for key, value in vars(args).items()
        if key not in {"bundle", "output_dir", "tokenizer"}
    }
    provenance = {
        "format": "gc_opd_frozen_replay_analysis_v1",
        "bundle_name": bundle.name,
        "analysis_script": script_path.name,
        "analysis_script_sha256": _sha256(script_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "arguments": public_arguments,
        "bundle_manifest_sha256": _sha256(manifest_path)
        if manifest_path.exists()
        else None,
        "packets": packet_meta,
        "seen_non_tensor_keys": sorted(seen_non_tensor_keys),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    default_credit = next(
        (row for row in credit_rows if row["credit_mode"] == args.default_credit_mode),
        credit_rows[0],
    )
    default_beta = next(
        (
            row
            for row in beta_rows
            if row["credit_mode"] == args.default_credit_mode
            and abs(row["beta"] - args.default_beta) < 1e-12
        ),
        beta_rows[0],
    )
    summary = {
        "format": "gc_opd_frozen_replay_summary_v1",
        "num_packets": len(packet_paths),
        "num_groups": len(group_rows),
        "num_rollouts": sum(meta["num_rollouts"] for meta in packet_meta),
        "num_valid_tokens": int(default_beta["valid_tokens"]),
        "overall": overall_rows[0],
        "prompt_length_buckets": [
            row for row in length_rows if row["axis"] == "prompt"
        ],
        "response_length_buckets": [
            row for row in length_rows if row["axis"] == "response"
        ],
        "task_buckets": task_rows,
        "misrank_thresholds": threshold_summary,
        "default_credit": default_credit,
        "default_beta": default_beta,
        "credit_modes": credit_rows,
        "beta_sequence_proxy": proxy_rows,
        "case_count": len(selected_cases),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    _write_report(output_dir / "report.md", summary, args)
    _plot(output_dir, summary, proxy_rows, credit_rows, args)
    return summary


def _pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100 * value:.1f}%"


def _write_report(
    path: Path, summary: dict[str, Any], args: argparse.Namespace
) -> None:
    overall = summary["overall"]
    thresholds = summary["misrank_thresholds"]
    prompt_rows = summary["prompt_length_buckets"]
    default_beta = summary["default_beta"]
    credit_by_mode = {
        row["credit_mode"]: row for row in summary.get("credit_modes", [])
    }
    shortest = prompt_rows[0] if prompt_rows else None
    longest = prompt_rows[-1] if prompt_rows else None
    lines = [
        "# Frozen-rollout GC-OPD mechanism analysis",
        "",
        "## Protocol",
        "",
        (
            f"A fixed checkpoint produced {summary['num_rollouts']} rollouts from "
            f"{summary['num_groups']} prompts in {summary['num_packets']} seeded frozen batches. "
            "All teacher, verifier, credit, and beta comparisons reuse these exact rollouts; "
            "no parameter update is needed for the counterfactual analyses. "
            "Teacher preference is the response-mean teacher–student log-ratio used by OPD."
        ),
        "",
        "## Main observations",
        "",
        (
            f"- Teacher preference agrees with verifier ordering on "
            f"{_pct(overall['pairwise_accuracy'])} of comparable sibling pairs "
            f"(prompt-bootstrap 95% CI "
            f"{_pct(overall['pairwise_ci_low'])}–{_pct(overall['pairwise_ci_high'])}); "
            f"teacher top-1 selects a maximum-reward sibling in "
            f"{_pct(overall['top1_accuracy'])} of the "
            f"{overall['reward_variable_groups']} prompts with non-constant verifier reward."
        ),
        (
            f"- Among {thresholds['misranked_pairs']} misranked pairs, the median beta needed "
            f"to reverse the ordering is {thresholds['residual_beta_threshold_median']:.3f} "
            f"for residual correction versus "
            f"{thresholds['direct_reward_beta_threshold_median']:.3f} for direct reward addition "
            f"({thresholds['median_threshold_reduction'] * 100:.1f}% lower)."
        ),
        (
            f"- At beta={args.default_beta:g} with {args.default_credit_mode.upper()} credit, "
            f"the correction carries {default_beta['correction_to_opd_l1_ratio']:.3f}× "
            f"the OPD L1 mass, flips {_pct(default_beta['sign_flip_ratio'])} of nonzero token "
            f"advantages, and clips {_pct(default_beta['clip_ratio'])} of valid tokens."
        ),
    ]
    if all(mode in credit_by_mode for mode in ("teacher_abs", "rol", "raca")):
        teacher_abs = credit_by_mode["teacher_abs"]
        rol = credit_by_mode["rol"]
        raca = credit_by_mode["raca"]
        lines.append(
            f"- RACA assigns {_pct(raca['top10pct_mass_mean'])} of its credit mass to "
            f"the top 10% of tokens, versus {_pct(teacher_abs['top10pct_mass_mean'])} "
            f"for teacher-absolute and {_pct(rol['top10pct_mass_mean'])} for ROL; "
            f"only {_pct(raca['zero_ratio'])} of valid tokens receive zero credit."
        )
    if shortest and longest:
        direction = (
            "increases"
            if longest["pairwise_mismatch"] > shortest["pairwise_mismatch"]
            else "does not increase"
        )
        lines.append(
            f"- Pairwise teacher–verifier mismatch {direction} from "
            f"{_pct(shortest['pairwise_mismatch'])} in {shortest['bucket']} prompts to "
            f"{_pct(longest['pairwise_mismatch'])} in {longest['bucket']} prompts "
            "(descriptive frozen-batch comparison; see bucket CIs in the CSV)."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "These results diagnose the signal geometry on frozen rollouts. They support "
                "mechanism claims about mismatch, correction selectivity, and token routing; "
                "they do not replace trained-checkpoint benchmark evaluation."
            ),
            "",
            "## Reproduction",
            "",
            "```bash",
            (
                "python -m examples.gc_opd.analyze_gc_opd_frozen_replay "
                f"--bundle {Path(args.bundle)} --output-dir {Path(args.output_dir)} "
                f"--betas {','.join(str(value) for value in args.betas)} "
                f"--credit-modes {','.join(args.credit_modes)} "
                f"--bootstrap-reps {args.bootstrap_reps} --seed {args.seed}"
            ),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(
    output_dir: Path,
    summary: dict[str, Any],
    proxy_rows: list[dict[str, Any]],
    credit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.25))

    length_rows = summary["prompt_length_buckets"]
    labels = [row["bucket"] for row in length_rows]
    mismatch = [100 * row["pairwise_mismatch"] for row in length_rows]
    low = [
        100 * (row["pairwise_accuracy"] - row["pairwise_ci_low"]) for row in length_rows
    ]
    high = [
        100 * (row["pairwise_ci_high"] - row["pairwise_accuracy"])
        for row in length_rows
    ]
    axes[0].bar(labels, mismatch, color="#7B61A8", alpha=0.9)
    for index, row in enumerate(length_rows):
        axes[0].text(
            index,
            4,
            f"n={row['reward_variable_groups']}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    axes[0].errorbar(
        range(len(labels)),
        mismatch,
        yerr=[high, low],
        fmt="none",
        ecolor="#24283B",
        capsize=3,
        linewidth=1,
    )
    axes[0].set_ylabel("Teacher–verifier mismatch (%)")
    axes[0].set_xlabel("Prompt length")
    axes[0].set_title("(a) Mismatch across context lengths", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    for method, color, marker, label in (
        ("teacher", "#24283B", "o", "Teacher only"),
        ("direct_reward", "#D8A62A", "^", "Direct reward"),
        ("residual", "#E56B6F", "s", "Residual"),
    ):
        rows = [row for row in proxy_rows if row["sequence_proxy"] == method]
        axes[1].plot(
            [row["beta"] for row in rows],
            [100 * row["pairwise_accuracy"] for row in rows],
            marker=marker,
            color=color,
            linewidth=2,
            markersize=4,
            label=label,
        )
    axes[1].axvline(args.default_beta, color="#888888", linestyle=":", linewidth=1)
    axes[1].set_xlabel(r"Residual/reward weight $\beta$")
    axes[1].set_ylabel("Pairwise alignment (%)")
    axes[1].set_title("(b) Correction efficiency", fontweight="bold")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False, fontsize=8)

    mode_labels = {
        "uniform": "Uniform",
        "teacher_abs": r"Teacher-$|A|$",
        "rol": "ROL",
        "rts": "RTS",
        "raca": "RACA",
    }
    modes = [mode_labels.get(row["credit_mode"], row["credit_mode"]) for row in credit_rows]
    concentration = [100 * row["top10pct_mass_mean"] for row in credit_rows]
    zero_credit = [100 * row["zero_ratio"] for row in credit_rows]
    x = np.arange(len(modes))
    width = 0.38
    axes[2].bar(
        x - width / 2, concentration, width, color="#4C78A8", label="Top-10% mass"
    )
    axes[2].bar(
        x + width / 2,
        zero_credit,
        width,
        color="#F28E2B",
        label="Zero-credit tokens",
    )
    axes[2].set_xticks(x, modes, rotation=20)
    axes[2].set_ylabel("Share (%)")
    axes[2].set_title("(c) Token credit geometry", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"mechanism_analysis.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packets-glob", default="packets/*.pt")
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--betas", type=_parse_floats, default=DEFAULT_BETAS)
    parser.add_argument(
        "--credit-modes", type=_parse_strings, default=DEFAULT_CREDIT_MODES
    )
    parser.add_argument("--credit-cap", type=float, default=5.0)
    parser.add_argument("--adv-clip", type=float, default=10.0)
    parser.add_argument("--default-credit-mode", default="raca")
    parser.add_argument("--default-beta", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-prompt-chars", type=int, default=6_000)
    parser.add_argument("--max-response-chars", type=int, default=12_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = analyze(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

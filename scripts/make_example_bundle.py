#!/usr/bin/env python3
"""Create a deterministic synthetic replay bundle for the analysis smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from verl import DataProto


def build_packet() -> DataProto:
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    old = torch.zeros_like(mask)
    teacher_gap = torch.tensor(
        [
            [2.0, 2.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ]
    )
    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    response_width = mask.shape[1]
    prompt_width = 8
    tensors = {
        "input_ids": torch.arange(4 * (prompt_width + response_width)).reshape(
            4, prompt_width + response_width
        ),
        "prompts": torch.arange(4 * prompt_width).reshape(4, prompt_width),
        "attention_mask": torch.ones(4, prompt_width + response_width),
        "position_ids": torch.arange(prompt_width + response_width).repeat(4, 1),
        "responses": torch.arange(4 * response_width).reshape(4, response_width),
        "response_mask": mask,
        "old_log_probs": old,
        "advantages": teacher_gap,
        "ref_log_prob": old + teacher_gap,
        "token_level_scores": scores,
        "token_level_rewards": scores,
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "uid": np.asarray(["p0", "p0", "p1", "p1"], dtype=object),
            "data_source": np.asarray(["qa", "qa", "math", "math"], dtype=object),
            "extra_info": np.asarray([{}, {}, {}, {}], dtype=object),
        },
        meta_info={
            "temperature": 1.0,
            "global_step": 1,
            "rollout_n": 2,
            "policy_loss_method": "opd",
            "source_run_id": "synthetic-example",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet_dir = args.output / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet_path = packet_dir / "step_000001_rank_000.pt"
    build_packet().save_to_disk(packet_path)
    manifest = {
        "format": "gc_opd_synthetic_bundle_v1",
        "seed": 0,
        "packets": [packet_path.name],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

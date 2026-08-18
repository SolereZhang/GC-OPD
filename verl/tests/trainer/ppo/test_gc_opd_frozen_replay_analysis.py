from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from examples.gc_opd.analyze_gc_opd_frozen_replay import (
    DEFAULT_BETAS,
    DEFAULT_CREDIT_MODES,
    _credit_weights,
    analyze,
)
from verl import DataProto


def _packet() -> DataProto:
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
    # Group p0 is teacher-misranked; group p1 is aligned.
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
            "global_token_num": [12, 12, 12, 12],
            "global_step": 1,
            "rollout_n": 2,
            "policy_loss_method": "opd",
            "source_run_id": "frozen-test",
        },
    )


def _args(bundle, output_dir):
    return Namespace(
        bundle=str(bundle),
        output_dir=str(output_dir),
        packets_glob="packets/*.pt",
        max_packets=0,
        group_size=2,
        betas=DEFAULT_BETAS,
        credit_modes=DEFAULT_CREDIT_MODES,
        credit_cap=5.0,
        adv_clip=10.0,
        default_credit_mode="raca",
        default_beta=0.05,
        eps=1e-6,
        bootstrap_reps=50,
        seed=7,
        max_cases=4,
        tokenizer=None,
        max_prompt_chars=1000,
        max_response_chars=1000,
    )


def test_credit_modes_are_bounded_and_masked():
    opd = torch.tensor([[-2.0, 0.0, 2.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    raca, identifiable = _credit_weights("raca", opd, mask, cap=5.0, eps=1e-6)
    rol, _ = _credit_weights("rol", opd, mask, cap=5.0, eps=1e-6)

    assert identifiable.tolist() == [True]
    assert raca[0, 0] < raca[0, 1] < raca[0, 2]
    assert raca[0, 3] == 0
    assert rol[0, 1] == pytest.approx(0.0)
    assert rol[0, 0] == pytest.approx(rol[0, 2])


def test_analysis_writes_reproducible_outputs_and_residual_needs_less_beta(tmp_path):
    bundle = tmp_path / "bundle"
    packet_dir = bundle / "packets"
    packet_dir.mkdir(parents=True)
    _packet().save_to_disk(packet_dir / "step_000001_rank_000.pt")

    output_dir = tmp_path / "analysis"
    summary = analyze(_args(bundle, output_dir))

    assert summary["num_groups"] == 2
    assert summary["overall"]["rollouts"] == 4
    assert summary["overall"]["reward_variable_groups"] == 2
    assert summary["overall"]["pairwise_accuracy"] == pytest.approx(0.5)
    assert summary["misrank_thresholds"]["misranked_pairs"] == 1
    assert (
        summary["misrank_thresholds"]["residual_beta_threshold_median"]
        < summary["misrank_thresholds"]["direct_reward_beta_threshold_median"]
    )
    assert summary["case_count"] == 1
    for filename in (
        "summary.json",
        "report.md",
        "groups.csv",
        "length_buckets.csv",
        "task_buckets.csv",
        "credit_modes.csv",
        "beta_token_sweep.csv",
        "beta_sequence_proxy.csv",
        "pairwise_rows.csv",
        "misrank_cases.csv",
        "misrank_cases.jsonl",
        "provenance.json",
        "mechanism_analysis.png",
        "mechanism_analysis.pdf",
    ):
        assert (output_dir / filename).is_file(), filename

    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["bundle_name"] == "bundle"
    assert provenance["analysis_script"] == "analyze_gc_opd_frozen_replay.py"
    assert "hostname" not in provenance
    assert "environment" not in provenance
    assert not any("git" in key for key in provenance)
    assert "bundle" not in provenance["arguments"]
    assert "output_dir" not in provenance["arguments"]

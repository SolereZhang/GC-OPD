import pytest
import torch

from verl.trainer.ppo.gc_opd import _credit_weights, compute_gc_opd_tensors
from verl.workers.config.actor import PolicyLossConfig


def _config(**overrides):
    values = {
        "method": "gc_opd",
        "gc_opd_group_size": 2,
        "gc_opd_adv_clip": 0.0,
    }
    values.update(overrides)
    return PolicyLossConfig(**values)


def _base_inputs():
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return {
        "old_log_prob": torch.zeros_like(response_mask),
        "ref_log_prob": torch.tensor(
            [
                [1.0, 1.0, 9.0],
                [3.0, 3.0, 9.0],
            ]
        ),
        "response_mask": response_mask,
        "token_level_scores": torch.tensor(
            [
                [2.0, 0.0, 7.0],
                [0.0, 0.0, 7.0],
            ]
        ),
        "uids": ["prompt", "prompt"],
        "rollout_n": 2,
    }


def test_gc_opd_defaults_are_base_free_rol_credit():
    config = PolicyLossConfig(method="gc_opd")

    assert config.gc_opd_residual_beta == pytest.approx(0.05)
    assert config.gc_opd_credit_mode == "relative_opd_leverage"
    assert config.gc_opd_credit_cap == pytest.approx(5.0)
    assert config.gc_opd_residual_norm == "group_zscore"
    assert config.gc_opd_group_size == 8


def test_credit_modes_have_distinct_relative_behavior():
    opd = torch.tensor([[-3.0, -2.0, -1.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    std = torch.tensor(2.0 / 3.0).sqrt()
    z = torch.tensor([-1.0, 0.0, 1.0]) / std

    uniform, uniform_identifiable = _credit_weights(
        "uniform",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )
    support, support_identifiable = _credit_weights(
        "relative_teacher_support",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )
    raca, raca_identifiable = _credit_weights(
        "raca",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )
    teacher_abs, teacher_abs_identifiable = _credit_weights(
        "teacher_abs",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )
    leverage, leverage_identifiable = _credit_weights(
        "relative_opd_leverage",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )

    torch.testing.assert_close(uniform, mask)
    torch.testing.assert_close(support[0, :3], (1.0 + z).clamp(min=0.0, max=5.0))
    torch.testing.assert_close(
        raca[0, :3],
        1.0 + torch.tanh(z / 2.0),
    )
    torch.testing.assert_close(teacher_abs[0, :3], torch.tensor([1.5, 1.0, 0.5]))
    torch.testing.assert_close(leverage[0, :3], z.abs())
    assert support[0, 3] == 0
    assert raca[0, 3] == 0
    assert teacher_abs[0, 3] == 0
    assert leverage[0, 3] == 0
    assert uniform_identifiable.tolist() == [True]
    assert support_identifiable.tolist() == [True]
    assert raca_identifiable.tolist() == [True]
    assert teacher_abs_identifiable.tolist() == [True]
    assert leverage_identifiable.tolist() == [True]


def test_raca_is_monotonic_bounded_and_not_mean_renormalized():
    opd = torch.tensor([[-3.0, -2.0, -1.0, 99.0], [100.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    credit, identifiable = _credit_weights(
        "raca",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )

    expected = torch.tensor([0.45420527, 1.0, 1.5457947])
    torch.testing.assert_close(credit[0, :3], expected)
    assert credit[0, 0] < credit[0, 1] < credit[0, 2]
    assert (credit[mask.bool()] >= 0.0).all()
    assert (credit[mask.bool()] <= 2.0).all()
    assert credit[0, 3] == 0
    assert credit[1].mean() != pytest.approx(1.0)
    assert identifiable.tolist() == [True, True]


def test_raca_is_translation_and_positive_scale_invariant():
    opd = torch.tensor([[-4.0, -2.0, -1.0, 25.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    expected, _ = _credit_weights(
        "raca",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )
    shifted, _ = _credit_weights(
        "raca",
        opd + 100.0,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )
    scaled, _ = _credit_weights(
        "raca",
        opd * 7.0,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )

    torch.testing.assert_close(shifted, expected)
    torch.testing.assert_close(scaled, expected)


def test_teacher_abs_is_base_free_and_falls_back_when_credit_mass_is_zero():
    opd = torch.tensor([[-3.0, -2.0, -1.0, 99.0], [0.0, 0.0, 0.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

    credit, identifiable = _credit_weights(
        "teacher_abs",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )

    torch.testing.assert_close(credit[0], torch.tensor([1.5, 1.0, 0.5, 0.0]))
    torch.testing.assert_close(credit[1], mask[1])
    assert credit[0, :3].mean() == pytest.approx(1.0)
    assert identifiable.tolist() == [True, False]


def test_credit_cap_is_applied_without_mean_renormalization():
    opd = torch.zeros(1, 27)
    opd[0, 0] = 100.0
    mask = torch.ones_like(opd)

    credit, identifiable = _credit_weights(
        "relative_opd_leverage",
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )

    assert identifiable.tolist() == [True]
    assert credit.max() == pytest.approx(5.0)
    assert credit.mean() != pytest.approx(1.0)


@pytest.mark.parametrize(
    "credit_mode",
    [
        "relative_opd_leverage",
        "relative_teacher_support",
        "raca",
    ],
)
def test_credit_falls_back_to_uniform_when_token_scale_is_not_identifiable(
    credit_mode,
):
    opd = torch.tensor([[2.0, 2.0, 2.0, 9.0], [4.0, 9.0, 9.0, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    credit, identifiable = _credit_weights(
        credit_mode,
        opd,
        mask,
        credit_cap=5.0,
        min_token_std=1e-6,
    )

    torch.testing.assert_close(credit, mask)
    assert identifiable.tolist() == [False, False]


def test_gc_opd_formula_combines_base_opd_and_residual():
    inputs = _base_inputs()
    tensors, metrics = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="uniform",
            gc_opd_residual_beta=0.25,
        ),
    )

    # Teacher group z-score is [-1, 1], reward group z-score is [1, -1],
    # so the sequence residual is [2, -2].
    expected = torch.tensor(
        [
            [1.5, 1.5, 0.0],
            [2.5, 2.5, 0.0],
        ]
    )
    torch.testing.assert_close(tensors["gc_opd_residual"], torch.tensor([2.0, -2.0]))
    torch.testing.assert_close(tensors["gc_opd_advantages"], expected)
    assert metrics["gc_opd/residual_adv_abs_mean"] == pytest.approx(0.5)
    assert metrics["gc_opd/base_correction_used"] == 0.0


def test_negative_beta_reverses_raca_residual_correction():
    inputs = _base_inputs()
    base = (inputs["ref_log_prob"] - inputs["old_log_prob"]) * inputs[
        "response_mask"
    ]
    positive, _ = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="raca",
            gc_opd_residual_beta=0.1,
        ),
    )
    negative, metrics = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="raca",
            gc_opd_residual_beta=-0.1,
        ),
    )

    torch.testing.assert_close(
        negative["gc_opd_advantages"] - base,
        -(positive["gc_opd_advantages"] - base),
    )
    torch.testing.assert_close(
        negative["gc_opd_token_weight"],
        positive["gc_opd_token_weight"],
    )
    assert metrics["gc_opd/residual_beta"] == pytest.approx(-0.1)


def test_gc_opd_is_invariant_to_base_reference_inputs():
    inputs = _base_inputs()
    config = _config(
        gc_opd_credit_mode="raca",
        gc_opd_residual_beta=0.1,
    )
    base_a = torch.full_like(inputs["old_log_prob"], -100.0)
    base_b = torch.full_like(inputs["old_log_prob"], 100.0)

    tensors_without_base, _ = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=config,
    )
    tensors_with_base_a, _ = compute_gc_opd_tensors(
        **inputs,
        base_ref_log_prob=base_a,
        policy_loss_config=config,
    )
    tensors_with_base_b, metrics = compute_gc_opd_tensors(
        **inputs,
        base_ref_log_prob=base_b,
        policy_loss_config=config,
    )

    for key, expected in tensors_without_base.items():
        torch.testing.assert_close(tensors_with_base_a[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors_with_base_b[key], expected, rtol=0, atol=0)
    assert metrics["gc_opd/base_ref_input_present_but_unused"] == 1.0
    assert metrics["gc_opd/base_correction_used"] == 0.0


@pytest.mark.parametrize(
    "credit_mode",
    [
        "uniform",
        "teacher_abs",
        "relative_teacher_support",
        "raca",
        "relative_opd_leverage",
    ],
)
def test_zero_beta_reduces_exactly_to_vanilla_opd(credit_mode):
    inputs = _base_inputs()
    tensors, _ = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode=credit_mode,
            gc_opd_residual_beta=0.0,
        ),
    )

    expected = (inputs["ref_log_prob"] - inputs["old_log_prob"]) * inputs[
        "response_mask"
    ]
    torch.testing.assert_close(tensors["gc_opd_advantages"], expected, rtol=0, atol=0)


def test_unidentifiable_group_residual_is_zero():
    inputs = _base_inputs()
    inputs["token_level_scores"] = torch.zeros_like(inputs["token_level_scores"])
    tensors, metrics = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="uniform",
            gc_opd_residual_beta=10.0,
        ),
    )

    expected = (inputs["ref_log_prob"] - inputs["old_log_prob"]) * inputs[
        "response_mask"
    ]
    torch.testing.assert_close(tensors["gc_opd_residual"], torch.zeros(2))
    torch.testing.assert_close(tensors["gc_opd_advantages"], expected)
    assert metrics["gc_opd/group_identifiable_ratio"] == 0.0


def test_opposed_residual_can_flip_the_full_advantage():
    inputs = _base_inputs()
    tensors, metrics = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="uniform",
            gc_opd_residual_beta=1.0,
        ),
    )

    expected = torch.tensor(
        [
            [3.0, 3.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(tensors["gc_opd_advantages"], expected)
    assert metrics["gc_opd/opd_residual_opposition_ratio"] == pytest.approx(0.5)
    assert metrics["gc_opd/final_sign_flip_ratio"] == 0.0

    flipped, flipped_metrics = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="uniform",
            gc_opd_residual_beta=2.0,
        ),
    )
    assert (flipped["gc_opd_advantages"][1, :2] < 0).all()
    assert flipped_metrics["gc_opd/final_sign_flip_ratio"] > 0.0


def test_padding_and_final_advantage_clip_are_enforced():
    inputs = _base_inputs()
    tensors, metrics = compute_gc_opd_tensors(
        **inputs,
        policy_loss_config=_config(
            gc_opd_credit_mode="uniform",
            gc_opd_residual_beta=10.0,
            gc_opd_adv_clip=0.5,
        ),
    )

    assert torch.isfinite(tensors["gc_opd_advantages"]).all()
    assert tensors["gc_opd_advantages"].abs().max() == pytest.approx(0.5)
    assert (tensors["gc_opd_advantages"][:, 2] == 0).all()
    assert (tensors["gc_opd_token_weight"][:, 2] == 0).all()
    assert metrics["gc_opd/final_adv_clip_ratio"] > 0.0


def test_invalid_credit_cap_is_rejected():
    inputs = _base_inputs()
    with pytest.raises(ValueError, match="gc_opd_credit_cap"):
        compute_gc_opd_tensors(
            **inputs,
            policy_loss_config=_config(gc_opd_credit_cap=0.5),
        )

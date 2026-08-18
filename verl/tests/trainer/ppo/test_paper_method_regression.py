import torch

from verl.trainer.ppo.fire_opd import compute_entropy_aware_distill_weights
from verl.trainer.ppo.gopd import compute_gopd_advantages
from verl.trainer.ppo.poweropd import compute_poweropd_advantages
from verl.trainer.ppo.uni_opd import compute_uni_opd_margin_shifts
from verl.workers.config.actor import PolicyLossConfig


def test_exopd_reference_target_regression():
    old = torch.tensor([[-2.0, -1.0]])
    teacher = torch.tensor([[-1.5, -1.2]])
    base = torch.tensor([[-2.2, -0.8]])

    tensors, _ = compute_gopd_advantages(
        old_log_prob=old,
        ref_log_prob=teacher,
        base_ref_log_prob=base,
        response_mask=torch.ones_like(old),
        policy_loss_config=PolicyLossConfig(method="gopd", gopd_lambda=1.5),
    )

    torch.testing.assert_close(tensors["gopd_target_log_prob"], torch.tensor([[-1.15, -1.4]]))
    torch.testing.assert_close(tensors["gopd_advantages"], torch.tensor([[0.85, -0.4]]))


def test_poweropd_bounded_power_reward_regression():
    old = torch.log(torch.tensor([[0.2, 0.4, 0.8]]))
    teacher = torch.log(torch.tensor([[0.5, 0.1, 0.8]]))
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    tensors, _ = compute_poweropd_advantages(
        old_log_prob=old,
        ref_log_prob=teacher,
        response_mask=mask,
        policy_loss_config=PolicyLossConfig(method="poweropd", poweropd_alpha=2.0),
    )

    expected = (teacher.exp().square() - old.exp().square()) * mask
    torch.testing.assert_close(tensors["poweropd_advantages"], expected)


def test_uni_opd_group_margin_shift_regression():
    old = torch.zeros(2, 2)
    teacher = torch.tensor([[-0.1, -0.1], [-0.2, -0.2]])
    mask = torch.ones_like(old)
    scores = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    tensors, metrics = compute_uni_opd_margin_shifts(
        old_log_probs=old,
        ref_log_prob=teacher,
        response_mask=mask,
        token_level_scores=scores,
        policy_loss_config=PolicyLossConfig(method="uni_opd_lite"),
        uids=["same-prompt", "same-prompt"],
        rollout_n=2,
    )

    torch.testing.assert_close(tensors["uni_opd_sample_shift"], torch.tensor([0.15, -0.15]))
    assert metrics["uni_opd/gap_after_mean"] == 0.4


def test_fire_opd_entropy_weight_regression():
    ref_log_prob = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    mask = torch.ones_like(ref_log_prob)
    teacher_entropy = torch.tensor([[0.0, 2.0], [1.0, 2.0]])
    student_entropy = torch.tensor([[0.0, 2.0], [1.0, 2.0]])

    tensors, _ = compute_entropy_aware_distill_weights(
        policy_loss_config=PolicyLossConfig(method="fire", traj_skip_percentile=0.0),
        ref_log_prob=ref_log_prob,
        response_mask=mask,
        student_entropys=student_entropy,
        ref_entropys=teacher_entropy,
    )

    expected = torch.tensor([[2.0, 2.0], [2.25, 2.0]]) / 2.0625
    torch.testing.assert_close(tensors["token_weight"], expected)
    torch.testing.assert_close(tensors["effective_mask"], mask)

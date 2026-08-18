import pytest

from verl.trainer.ppo.opd_methods import SUPPORTED_METHODS, resolve_policy_loss_method
from verl.workers.config.actor import PolicyLossConfig


def test_default_method_is_grpo():
    assert resolve_policy_loss_method(PolicyLossConfig()) == "grpo"


def test_supported_methods_match_the_paper_table():
    assert SUPPORTED_METHODS == (
        "grpo",
        "opd",
        "gopd",
        "gc_opd",
        "fire",
        "poweropd",
        "uni_opd_lite",
    )


@pytest.mark.parametrize(
    ("configured", "canonical"),
    [
        ("grpo", "grpo"),
        ("baseline", "opd"),
        ("opd", "opd"),
        ("exopd", "gopd"),
        ("g-opd", "gopd"),
        ("gc-opd", "gc_opd"),
        ("fire-opd", "fire"),
        ("power-opd", "poweropd"),
        ("uni-opd", "uni_opd_lite"),
        ("uni-opd-lite", "uni_opd_lite"),
    ],
)
def test_paper_method_names_are_canonicalized(configured, canonical):
    assert resolve_policy_loss_method(PolicyLossConfig(method=configured)) == canonical


def test_upstream_switches_map_to_retained_methods():
    assert resolve_policy_loss_method(PolicyLossConfig(only_reverse_kl_advantages=True)) == "opd"
    assert resolve_policy_loss_method(PolicyLossConfig(gopd_enabled=True)) == "gopd"
    assert resolve_policy_loss_method(PolicyLossConfig(entropy_aware_distill=True)) == "fire"
    assert (
        resolve_policy_loss_method(
            PolicyLossConfig(only_reverse_kl_advantages=True, entropy_aware_distill=True)
        )
        == "fire"
    )


def test_switch_conflicts_are_rejected():
    config = PolicyLossConfig(only_reverse_kl_advantages=True, gopd_enabled=True)

    with pytest.raises(ValueError, match="Multiple OPD-family legacy switches"):
        resolve_policy_loss_method(config)


def test_explicit_method_conflicting_with_switch_is_rejected():
    config = PolicyLossConfig(method="opd", gopd_enabled=True)

    with pytest.raises(ValueError, match="conflicts with legacy switches"):
        resolve_policy_loss_method(config)


def test_unknown_method_is_rejected_without_fallback():
    with pytest.raises(ValueError, match="Unknown policy_loss.method"):
        resolve_policy_loss_method(PolicyLossConfig(method="unsupported"))

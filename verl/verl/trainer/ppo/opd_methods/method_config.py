"""Single-source policy-loss method selection.

This module keeps the paper methods separate from their tensor builders and
returns one canonical method so trainer/actor branches cannot overlap.
"""

from __future__ import annotations

from typing import Any, Literal

PolicyLossMethod = Literal[
    "grpo",
    "opd",
    "gopd",
    "gc_opd",
    "fire",
    "poweropd",
    "uni_opd_lite",
]

SUPPORTED_METHODS: tuple[str, ...] = (
    "grpo",
    "opd",
    "gopd",
    "gc_opd",
    "fire",
    "poweropd",
    "uni_opd_lite",
)

_ALIASES = {
    "": "auto",
    "auto": "auto",
    "none": "grpo",
    "ppo": "grpo",
    "grpo": "grpo",
    "baseline": "opd",
    "vanilla_opd": "opd",
    "opd": "opd",
    "exopd": "gopd",
    "g_opd": "gopd",
    "gopd": "gopd",
    "gc_opd": "gc_opd",
    "gcopd": "gc_opd",
    "fire_opd": "fire",
    "fire": "fire",
    "power_opd": "poweropd",
    "poweropd": "poweropd",
    "uni_opd": "uni_opd_lite",
    "uni-opd-lite": "uni_opd_lite",
    "uni_opd_lite": "uni_opd_lite",
}


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_method(value: Any) -> str:
    if value is None:
        raw = "auto"
    else:
        raw = str(value).strip().lower().replace("-", "_")
    method = _ALIASES.get(raw)
    if method is None:
        valid = ", ".join(sorted(set(SUPPORTED_METHODS + ("auto",))))
        raise ValueError(f"Unknown policy_loss.method={value!r}. Valid methods: {valid}")
    return method


def _legacy_enabled_methods(config: Any) -> list[str]:
    methods: list[str] = []
    if _as_bool(_cfg(config, "gopd_enabled", False)):
        methods.append("gopd")
    entropy_aware = _as_bool(_cfg(config, "entropy_aware_distill", False))
    if entropy_aware:
        methods.append("fire")
    if _as_bool(_cfg(config, "only_reverse_kl_advantages", False)) and not entropy_aware:
        methods.append("opd")
    return methods


def resolve_policy_loss_method(config: Any) -> PolicyLossMethod:
    """Resolve policy-loss config to exactly one canonical method.

    The two historical switches still used by the upstream trainer are
    accepted only when they identify a single retained method.
    """

    explicit_method = _normalize_method(_cfg(config, "method", "auto"))
    legacy_methods = _legacy_enabled_methods(config)
    unique_legacy_methods = list(dict.fromkeys(legacy_methods))

    if explicit_method != "auto":
        conflicting = [method for method in unique_legacy_methods if method != explicit_method]
        if conflicting:
            raise ValueError(
                "policy_loss.method conflicts with legacy switches: "
                f"method={explicit_method!r}, legacy={conflicting!r}"
            )
        return explicit_method  # type: ignore[return-value]

    if not unique_legacy_methods:
        return "grpo"
    if len(unique_legacy_methods) > 1:
        raise ValueError(f"Multiple OPD-family legacy switches are enabled: {unique_legacy_methods!r}")
    return unique_legacy_methods[0]  # type: ignore[return-value]

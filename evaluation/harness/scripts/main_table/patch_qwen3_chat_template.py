#!/usr/bin/env python3
"""Create a model overlay with a Qwen3 no-think chat template."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


NO_THINK_LITERAL = r"{{- '<think>\n\n</think>\n\n' }}"


def normalize_mode(mode: str) -> str:
    mode = mode.strip().lower().replace("-", "_")
    if mode in {"nothink", "no_think", "non_thinking", "disable", "disabled", "false", "0", "no"}:
        return "nothink"
    if mode in {"think", "thinking", "enable", "enabled", "true", "1", "yes"}:
        return "think"
    if mode in {"", "auto"}:
        return "auto"
    raise SystemExit(f"invalid thinking mode: {mode}")


def patch_template(text: str) -> str:
    if "<think>\\n\\n</think>\\n\\n" not in text:
        raise SystemExit("template does not look like a Qwen3 thinking template")
    pattern = (
        r"\n\s*\{%- if enable_thinking is defined and enable_thinking is false %\}"
        r"\n\s*\{\{- '<think>\\n\\n</think>\\n\\n' \}\}"
        r"\n\s*\{%- endif %\}"
    )
    patched, count = re.subn(pattern, lambda _: "\n    " + NO_THINK_LITERAL, text, count=1)
    if count:
        return patched
    if "{%- if add_generation_prompt %}" in text and NO_THINK_LITERAL in text:
        return text
    raise SystemExit("could not patch Qwen3 enable_thinking branch")


def link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item)


def sanitize_tokenizer_config(cfg: dict) -> bool:
    """Make tokenizer_config compatible with current transformers/vLLM."""
    extra = cfg.get("extra_special_tokens")
    if not isinstance(extra, list):
        return False

    additional = cfg.get("additional_special_tokens")
    if isinstance(additional, list):
        merged = list(additional)
    else:
        merged = []
    for token in extra:
        if token not in merged:
            merged.append(token)
    if merged:
        cfg["additional_special_tokens"] = merged
    cfg.pop("extra_special_tokens", None)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    mode = normalize_mode(args.mode)
    model_path = Path(args.model_path).resolve()
    if mode != "nothink":
        print(model_path)
        return 0

    overlay = Path(args.out) / "model" / f"{model_path.name}_chat_nothink"
    link_tree(model_path, overlay)

    patched_any_template = False
    src_template = model_path / "chat_template.jinja"
    if src_template.is_file():
        template_text = src_template.read_text(encoding="utf-8")
        patched_template = patch_template(template_text)
        template_dst = overlay / "chat_template.jinja"
        if template_dst.exists() or template_dst.is_symlink():
            template_dst.unlink()
        template_dst.write_text(patched_template, encoding="utf-8")
        patched_any_template = True

    tokenizer_config = model_path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        cfg = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        changed = sanitize_tokenizer_config(cfg)
        if isinstance(cfg.get("chat_template"), str):
            cfg["chat_template"] = patch_template(cfg["chat_template"])
            changed = True
            patched_any_template = True
        if changed:
            cfg_dst = overlay / "tokenizer_config.json"
            if cfg_dst.exists() or cfg_dst.is_symlink():
                cfg_dst.unlink()
            cfg_dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not patched_any_template:
        raise SystemExit(f"missing Qwen3 chat template in {model_path}")

    print(overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

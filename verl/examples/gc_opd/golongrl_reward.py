"""Rule-based GoLongRL reward for the bundled training entrypoint.

The metric dispatches by ability, data source, and reward mode, then returns a
raw verifiable score in ``[0, 1]`` without an LLM judge.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from typing import Any

ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
THINK_BLOCK_RE = re.compile(r"</think>\s*(.*)$", re.IGNORECASE | re.DOTALL)
ANSWER_MARKER_RE = re.compile(r"\[(?:Answer|答案)\]", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"</?answer>", re.IGNORECASE)
CODE_FENCE_ONLY_RE = re.compile(r"^\s*`{3,}\s*[\w+-]*\s*(?:`{3,})?\s*$", re.IGNORECASE)
MARKER_ONLY_RE = re.compile(r"^\s*(?:\[Answer\]|\[答案\]|</?answer>)+\s*$", re.IGNORECASE)
CHOICE_FULL_RE = re.compile(r"^([A-Ja-j])$")
CHOICE_TOKEN_RE = re.compile(r"(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])")
YESNO_RE = re.compile(r"^(yes|no|true|false|是|否|对|错)$", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


FORMAT_BONUS = _env_float("GOLONGRL_FORMAT_BONUS", 0.0)
NDCG_K = int(_env_float("GOLONGRL_NDCG_K", 0))
ROUGE_MAX_TOKENS = int(_env_float("GOLONGRL_ROUGE_MAX_TOKENS", 2000))


ABILITY_TO_METRIC = {
    "precise long-range information retrieval": "em",
    "evidence-grounded comprehension and reasoning": "accuracy",
    "high-recall exhaustive retrieval and verification": "f1",
    "numerical extraction and quantitative reasoning": "math",
    "multi-table structured extraction": "iou",
    "fragment-level structured matching and induction": "subem",
    "dimension-quantified retrieval and graded ranking": "ndcg",
    "sequence reconstruction and ordering": "pairwise",
    "long document summarization": "rougel",
}

DATA_SOURCE_TO_METRIC = {
    "em": "em",
    "selection": "accuracy",
    "math_longcot_math_verify": "math",
    "multitableqa_pretraining": "iou",
}


def resolve_metric(ability: str, data_source: str, reward_mode: str) -> str:
    ability_norm = (ability or "").strip().casefold()
    if ability_norm in ABILITY_TO_METRIC:
        return ABILITY_TO_METRIC[ability_norm]
    if "summar" in ability_norm:
        return "rougel"
    if "ordering" in ability_norm or "sequence reconstruction" in ability_norm:
        return "pairwise"
    if "graded ranking" in ability_norm or "ndcg" in ability_norm:
        return "ndcg"
    if "multi-table" in ability_norm or "structured extraction" in ability_norm:
        return "iou"
    if "fragment" in ability_norm or "induction" in ability_norm:
        return "subem"
    if "numerical" in ability_norm or "quantitative" in ability_norm:
        return "math"
    if "high-recall" in ability_norm or "verification" in ability_norm:
        return "f1"
    if "comprehension" in ability_norm or "reasoning" in ability_norm:
        return "accuracy"
    if "retrieval" in ability_norm:
        return "em"

    data_source_norm = (data_source or "").strip().casefold()
    if data_source_norm in DATA_SOURCE_TO_METRIC:
        return DATA_SOURCE_TO_METRIC[data_source_norm]

    reward_mode_norm = (reward_mode or "").strip().casefold()
    if "math" in reward_mode_norm:
        return "math"
    if "selection" in reward_mode_norm:
        return "accuracy"
    if reward_mode_norm == "em":
        return "em"
    if "multitable" in reward_mode_norm:
        return "iou"
    if "summary" in reward_mode_norm or "summar" in reward_mode_norm:
        return "rougel"
    return "f1"


def extract_answer_text(solution_str: str) -> tuple[str, bool, bool]:
    answer_blocks = ANSWER_BLOCK_RE.findall(solution_str or "")
    has_single_answer_tag = len(answer_blocks) == 1
    if answer_blocks:
        answer_text = answer_blocks[-1]
    else:
        think_match = THINK_BLOCK_RE.search(solution_str or "")
        answer_text = think_match.group(1) if think_match else (solution_str or "")

    marker_match = ANSWER_MARKER_RE.search(answer_text)
    has_marker = marker_match is not None
    if marker_match:
        answer_text = answer_text[marker_match.end() :]
    return answer_text.strip(), has_single_answer_tag, has_marker


def _norm(text: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if text is None else str(text))
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n,，;；。.").casefold()


def _strip_marker(text: Any) -> str:
    text = "" if text is None else str(text)
    marker = ANSWER_MARKER_RE.search(text)
    if marker:
        text = text[marker.end() :]
    return ANSWER_TAG_RE.sub("", text).strip()


def _items(text: Any) -> list[str]:
    if text is None:
        return []
    items: list[str] = []
    for raw in str(text).splitlines():
        line = ANSWER_TAG_RE.sub("", ANSWER_MARKER_RE.sub("", raw))
        normalized = _norm(line)
        if normalized:
            items.append(normalized)
    return items


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def is_degenerate_answer_text(text: Any) -> bool:
    text = ("" if text is None else str(text)).strip()
    return not text or bool(CODE_FENCE_ONLY_RE.fullmatch(text)) or bool(MARKER_ONLY_RE.fullmatch(text))


def em_score(answer_text: str, gold_str: str) -> float:
    gold = _norm(_strip_marker(gold_str))
    if not gold:
        return 0.0
    pred_full = _norm(_strip_marker(answer_text))
    if pred_full == gold:
        return 1.0
    lines = _items(answer_text)
    if lines and lines[-1] == gold:
        return 1.0
    return 1.0 if len(lines) == 1 and lines[0] == gold else 0.0


def _choice_token(text: str) -> str | None:
    text = _strip_marker(text).strip().strip(" .)、:：")
    match = CHOICE_FULL_RE.fullmatch(text)
    if match:
        return match.group(1).upper()
    if YESNO_RE.fullmatch(text):
        return text.casefold()
    if len(text) <= 6:
        match = CHOICE_TOKEN_RE.search(text)
        if match:
            return match.group(1).upper()
    return None


def accuracy_score(answer_text: str, gold_str: str) -> float:
    gold_choice = _choice_token(gold_str)
    if gold_choice is not None:
        pred_choice = _choice_token(answer_text)
        return 1.0 if pred_choice is not None and pred_choice == gold_choice else 0.0
    return em_score(answer_text, gold_str)


def f1_score(pred_items: list[str], gold_items: list[str]) -> float:
    pred, gold = set(_dedupe(pred_items)), set(_dedupe(gold_items))
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    precision = inter / len(pred)
    recall = inter / len(gold)
    return 2 * precision * recall / (precision + recall)


def _last_segment(text: str) -> str:
    text = str(text).strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _normalize_number(text: str) -> str | None:
    matches = list(NUMERIC_RE.finditer(str(text)))
    if not matches:
        return None
    token = matches[-1].group(0).replace(",", "").rstrip("%")
    try:
        value = float(token)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else repr(value)


def math_score(answer_text: str, gold_str: str) -> float:
    candidate = _last_segment(answer_text)
    try:
        from math_verify import parse, verify

        gold_parsed, pred_parsed = parse(str(gold_str)), parse(candidate)
        if gold_parsed and pred_parsed and verify(gold_parsed, pred_parsed):
            return 1.0
    except Exception:
        pass
    pred_number, gold_number = _normalize_number(candidate), _normalize_number(str(gold_str))
    if pred_number is not None and pred_number == gold_number:
        return 1.0
    return 1.0 if _norm(_strip_marker(candidate)) == _norm(_strip_marker(gold_str)) else 0.0


def _extract_json_obj(text: str) -> Any:
    match = JSON_OBJ_RE.search(text or "")
    candidates = [match.group(0)] if match else []
    candidates.append((text or "").strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _json_cells(obj: Any) -> set[str]:
    cells: set[str] = set()
    if isinstance(obj, dict):
        cols, data = obj.get("columns"), obj.get("data")
        if isinstance(cols, list):
            cells.update("col::" + _norm(col) for col in cols)
        if isinstance(data, list):
            for row in data:
                if isinstance(row, list):
                    cells.update(_norm(value) for value in row)
                else:
                    cells.add(_norm(row))
        if cols is None and data is None:
            cells.update(_norm(key) + "::" + _norm(value) for key, value in obj.items())
    elif isinstance(obj, list):
        cells.update(_norm(value) for value in obj)
    elif obj is not None:
        cells.add(_norm(obj))
    cells.discard("")
    return cells


def iou_score(answer_text: str, gold_str: str) -> float:
    pred_obj, gold_obj = _extract_json_obj(answer_text), _extract_json_obj(gold_str)
    if gold_obj is None or pred_obj is None:
        return 0.0
    pred_cells, gold_cells = _json_cells(pred_obj), _json_cells(gold_obj)
    if not pred_cells and not gold_cells:
        return 1.0
    if not pred_cells or not gold_cells:
        return 0.0
    return len(pred_cells & gold_cells) / len(pred_cells | gold_cells)


def subem_score(answer_text: str, gold_items: list[str]) -> float:
    gold = [item for item in _dedupe(gold_items) if item]
    if not gold:
        return 0.0
    pred_norm = _norm(_strip_marker(answer_text))
    return sum(1 for item in gold if item in pred_norm) / len(gold)


def ndcg_score(pred_items: list[str], gold_items: list[str], k: int = 0) -> float:
    gold = _dedupe(gold_items)
    if not gold:
        return 0.0
    relevance = {item: len(gold) - idx for idx, item in enumerate(gold)}
    pred = _dedupe(pred_items)
    cutoff = k if k and k > 0 else max(len(pred), len(gold))
    dcg = sum(relevance.get(item, 0) / math.log2(i + 2) for i, item in enumerate(pred[:cutoff]))
    ideal = sorted(relevance.values(), reverse=True)
    idcg = sum(value / math.log2(i + 2) for i, value in enumerate(ideal[:cutoff]))
    return dcg / idcg if idcg > 0 else 0.0


def pairwise_score(pred_items: list[str], gold_items: list[str]) -> float:
    gold = _dedupe(gold_items)
    if len(gold) <= 1:
        return 1.0 if gold and _dedupe(pred_items)[:1] == gold else 0.0
    pos = {item: idx for idx, item in enumerate(_dedupe(pred_items))}
    correct = total = 0
    for i, left in enumerate(gold):
        for right in gold[i + 1 :]:
            total += 1
            if left in pos and right in pos and pos[left] < pos[right]:
                correct += 1
    return correct / total if total else 0.0


def _tokenize_rouge(text: str) -> list[str]:
    normalized = _norm(text)
    if CJK_RE.search(normalized):
        return [ch for ch in normalized if not ch.isspace()]
    return normalized.split()


def _lcs_len(left: list[str], right: list[str]) -> int:
    left, right = left[:ROUGE_MAX_TOKENS], right[:ROUGE_MAX_TOKENS]
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_item in left:
        cur = [0] * (len(right) + 1)
        for j, right_item in enumerate(right, 1):
            cur[j] = prev[j - 1] + 1 if left_item == right_item else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rougel_score(answer_text: str, gold_items: list[str], gold_str: str) -> float:
    gold_candidates = gold_items or ([gold_str] if gold_str.strip() else [])
    pred_tokens = _tokenize_rouge(_strip_marker(answer_text))
    if not pred_tokens or not gold_candidates:
        return 0.0
    best = 0.0
    for gold in gold_candidates:
        gold_tokens = _tokenize_rouge(_strip_marker(gold))
        if not gold_tokens:
            continue
        lcs = _lcs_len(pred_tokens, gold_tokens)
        if lcs == 0:
            continue
        precision, recall = lcs / len(pred_tokens), lcs / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(item) for item in value if str(item).strip()]
    except TypeError:
        return []


def parse_ground_truth(ground_truth: Any) -> tuple[list[str], str, str]:
    if isinstance(ground_truth, dict):
        return (
            _as_list(ground_truth.get("doc_ids")),
            str(ground_truth.get("golden_label") or ""),
            str(ground_truth.get("summary") or ""),
        )
    return [], str(ground_truth or ""), ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, float]:
    extra_info = extra_info or {}
    ability = str(extra_info.get("ability", "") or "")
    reward_mode = str(extra_info.get("reward_mode", "") or "")

    answer_text, has_answer_tag, has_marker = extract_answer_text(solution_str)
    doc_ids, golden_label, _lang = parse_ground_truth(ground_truth)
    gold_list_raw = doc_ids if doc_ids else [line for line in golden_label.splitlines() if line.strip()]
    gold_items = [_norm(item) for item in gold_list_raw if _norm(item)]
    gold_str = golden_label if golden_label.strip() else "\n".join(doc_ids)
    pred_items = _items(answer_text)

    metric = resolve_metric(ability=ability, data_source=data_source, reward_mode=reward_mode)
    degenerate = is_degenerate_answer_text(answer_text) or (not gold_items and not gold_str.strip())

    if degenerate:
        content = 0.0
    elif metric == "em":
        content = em_score(answer_text, gold_str)
    elif metric == "accuracy":
        content = accuracy_score(answer_text, gold_str)
    elif metric == "f1":
        content = f1_score(pred_items, gold_items)
    elif metric == "math":
        content = math_score(answer_text, gold_str)
    elif metric == "iou":
        content = iou_score(answer_text, gold_str)
    elif metric == "subem":
        content = subem_score(answer_text, gold_items)
    elif metric == "ndcg":
        content = ndcg_score(pred_items, gold_items, k=NDCG_K)
    elif metric == "pairwise":
        content = pairwise_score(pred_items, gold_items)
    elif metric == "rougel":
        content = rougel_score(answer_text, gold_items, gold_str)
    else:
        content = f1_score(pred_items, gold_items)

    content = float(max(0.0, min(1.0, content)))
    format_ok = float(has_answer_tag or has_marker)
    if degenerate:
        score = 0.0
    elif FORMAT_BONUS <= 0.0:
        score = content
    elif content >= 1.0:
        score = 1.0 if format_ok else (1.0 - FORMAT_BONUS)
    elif content > 0.0:
        score = min(content + (FORMAT_BONUS if format_ok else 0.0), 1.0 - FORMAT_BONUS - 0.01)
    else:
        score = FORMAT_BONUS if format_ok else 0.0

    return {
        "score": float(score),
        "content_score": content,
        "metric_em": float(metric == "em"),
        "metric_accuracy": float(metric == "accuracy"),
        "metric_f1": float(metric == "f1"),
        "metric_math": float(metric == "math"),
        "metric_iou": float(metric == "iou"),
        "metric_subem": float(metric == "subem"),
        "metric_ndcg": float(metric == "ndcg"),
        "metric_pairwise": float(metric == "pairwise"),
        "metric_rougel": float(metric == "rougel"),
        "exact_match": float(content >= 1.0),
        "format_score": format_ok,
        "invalid_response": float(degenerate),
        "num_gt_items": float(len(gold_items)),
        "num_pred_items": float(len(pred_items)),
    }

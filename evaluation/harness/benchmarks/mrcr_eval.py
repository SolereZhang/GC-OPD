#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


def strip_thinking(text: str) -> str:
    """
    Remove hidden reasoning content from model responses.
    Handles two formats:
      1) <think>...</think>  — standard format
      2) raw text...</think> — vLLM strips the opening <think> tag,
                               leaving only the closing </think> in the output
    """
    # Standard format: <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # vLLM format: only </think> remains
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """
    Official MRCR grading function (openai/mrcr):
      1) if response does not start with the hash prefix -> 0
      2) strip hash prefix from both response and answer
      3) SequenceMatcher ratio
    """
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_completed_ids(path: str) -> set:
    """Load IDs already present in the output file for resume support."""
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def get_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)


def n_tokens_messages(messages, tok) -> int:
    return len(tok.apply_chat_template(messages, add_generation_prompt=True))


def middle_truncate_messages(messages, max_tokens: int, tok):
    """
    Middle truncation when exceeding context window.
    Keeps the last message intact; truncates earlier context in the middle.
    """
    if max_tokens <= 0:
        return messages

    total = n_tokens_messages(messages, tok)
    if total <= max_tokens or not messages:
        return messages

    query = messages[-1]
    query_tokens = len(tok.encode(query.get("content", ""), add_special_tokens=False))

    budget = max_tokens - query_tokens - 1024
    if budget <= 0:
        return [query]

    prefix_text = "\n".join(m.get("content", "") for m in messages[:-1])
    prefix_tokens = tok.encode(prefix_text, add_special_tokens=False)

    if len(prefix_tokens) > budget:
        half = budget // 2
        prefix_tokens = prefix_tokens[:half] + prefix_tokens[-half:]
        prefix_text = tok.decode(prefix_tokens, skip_special_tokens=True)

    return [{"role": "user", "content": prefix_text}, query]


def parse_prompt_field(prompt_field):
    if isinstance(prompt_field, str):
        return json.loads(prompt_field)
    return prompt_field


async def run_inference(args):
    timeout = httpx.Timeout(
        connect=30.0,
        read=args.timeout,
        write=30.0,
        pool=30.0,
    )
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_url,
        timeout=timeout,
        max_retries=0,
    )
    tok = get_tokenizer(args)

    data = load_jsonl(args.data_file)
    if not data:
        print("[ERROR] Empty dataset.")
        return

    # --- Resume: skip already-completed samples ---
    done_ids = load_completed_ids(args.infer_output)
    pending = [x for x in data if x["id"] not in done_ids]

    print(f"Inference: {len(data)} total, {len(done_ids)} done, {len(pending)} pending")
    print(f"  model={args.model}, temperature=0.7, top_p=0.95, max_tokens={args.max_tokens}")
    print(f"  tokenizer={args.model_path}")
    print(f"  timeout={args.timeout}s, max_retries={args.max_retries}, concurrency={args.concurrency}")
    if args.max_input_tokens > 0:
        print(f"  max_input_tokens={args.max_input_tokens} (middle truncation if needed)")

    if not pending:
        print("All samples already completed, skipping inference.")
        return

    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs(os.path.dirname(args.infer_output) or ".", exist_ok=True)

    async def infer_one(item):
        async with sem:
            messages = parse_prompt_field(item["prompt"])
            messages = middle_truncate_messages(messages, args.max_input_tokens, tok)

            for attempt in range(args.max_retries):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        stream=False,
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=args.max_tokens,
                    )
                    text = strip_thinking((resp.choices[0].message.content or "").strip())

                    score = grade(text, item["answer"], item["random_string_to_prepend"])
                    return {
                        "id": item["id"],
                        "needle_count": item.get("needle_count"),
                        "n_tokens_qwen": item.get("n_tokens_qwen"),
                        "bin": item.get("bin"),
                        "answer": item["answer"],
                        "random_string_to_prepend": item["random_string_to_prepend"],
                        "response": text,
                        "score": score,
                        "model": args.model,
                    }
                except Exception as e:
                    err_type = type(e).__name__
                    print(f"  [RETRY {attempt+1}/{args.max_retries}] {item['id']}: {err_type}: {str(e)[:120]}")
                    if attempt < args.max_retries - 1:
                        if 'timeout' in str(e).lower() or 'Timeout' in err_type:
                            await asyncio.sleep(5)
                        else:
                            await asyncio.sleep(10 * (attempt + 1))
                    else:
                        print(f"  [FAIL] {item['id']}: {err_type}: {str(e)[:200]}")
                        return None

    tasks = [infer_one(x) for x in pending]

    # Append mode for resume safety
    with open(args.infer_output, "a", encoding="utf-8") as f:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
            out = await fut
            if out:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()


def run_rescore(args):
    """
    Re-score an existing inference output file without re-running inference.

    For each record, strip_thinking() is applied to the raw `response` field,
    the score is recomputed, and the cleaned response + new score are written
    to args.rescore_output (defaults to <infer_output>_rescored.jsonl).
    The original file is never modified.
    """
    if not os.path.exists(args.infer_output):
        print(f"[ERROR] Missing inference file: {args.infer_output}")
        return

    records = load_jsonl(args.infer_output)
    if not records:
        print("[ERROR] Inference file is empty.")
        return

    output_path = args.rescore_output or args.infer_output.replace(".jsonl", "_rescored.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    improved = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in tqdm(records, desc="Rescoring"):
            raw_response = r.get("response", "")
            clean_response = strip_thinking(raw_response)
            new_score = grade(
                clean_response,
                r["answer"],
                r["random_string_to_prepend"],
            )
            if new_score > float(r.get("score", 0.0)):
                improved += 1
            out = {**r, "response": clean_response, "score": new_score}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Rescoring complete: {len(records)} records, {improved} scores improved.")
    print(f"Output: {output_path}")

    # Point the report at the rescored file and generate it
    args.infer_output = output_path
    run_report(args)


def run_report(args):
    if not os.path.exists(args.infer_output):
        print(f"[ERROR] Missing result file: {args.infer_output}")
        return

    results = load_jsonl(args.infer_output)
    if not results:
        print("[ERROR] No results.")
        return

    overall = []
    by_needle = defaultdict(list)
    by_bin = defaultdict(list)
    by_needle_bin = defaultdict(list)

    for r in results:
        s = float(r.get("score", 0.0))
        overall.append(s)
        nc = r.get("needle_count", "?")
        bn = r.get("bin", "?")
        by_needle[nc].append(s)
        by_bin[bn].append(s)
        by_needle_bin[f"{nc}needle_{bn}"].append(s)

    def avg_pct(lst):
        return (sum(lst) / len(lst) * 100.0) if lst else 0.0

    print("\n" + "=" * 70)
    print(f"MRCR RESULTS — {args.model}")
    print("=" * 70)
    print(f"\n  Overall: {avg_pct(overall):.2f}  ({len(overall)} samples)")

    print("\n  Per needle_count:")
    for nc in sorted(by_needle, key=lambda x: int(x) if str(x).isdigit() else 10**9):
        scores = by_needle[nc]
        print(f"    {nc}-needle: {avg_pct(scores):.2f}  (n={len(scores)})")

    def bin_key(x):
        x = str(x)
        if "-" in x:
            try:
                return int(x.split("-")[0])
            except Exception:
                return 10**9
        return 10**9

    print("\n  Per token bin:")
    for bn in sorted(by_bin, key=bin_key):
        scores = by_bin[bn]
        print(f"    {bn}: {avg_pct(scores):.2f}  (n={len(scores)})")

    print("\n  Per needle_count × bin:")
    for key in sorted(by_needle_bin):
        scores = by_needle_bin[key]
        print(f"    {key}: {avg_pct(scores):.2f}  (n={len(scores)})")

    print("=" * 70)

    summary = {
        "model": args.model,
        "overall": round(avg_pct(overall), 2),
        "n_samples": len(overall),
        "per_needle": {str(k): round(avg_pct(v), 2) for k, v in sorted(by_needle.items(), key=lambda kv: str(kv[0]))},
        "per_bin": {str(k): round(avg_pct(v), 2) for k, v in sorted(by_bin.items(), key=lambda kv: bin_key(kv[0]))},
    }
    summary_path = args.infer_output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MRCR Evaluation (Official OpenAI Method)")
    p.add_argument("--data_file", help="JSONL data file from download_mrcr_data.py")
    p.add_argument("--model", default="main-table-candidate")
    p.add_argument("--model_path", help="Local path to the eval model (for tokenizer)")
    p.add_argument("--api_url", default=os.getenv("LLM_URL", "http://<LLM_HOST>:<PORT>/v1"))
    p.add_argument("--api_key", default=os.getenv("API_KEY", "<API_KEY>"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=51200, help="Max generation tokens")
    p.add_argument("--timeout", type=float, default=3600,
                   help="Read timeout in seconds per request (default 3600=1hr)")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Max retries per sample on failure")
    p.add_argument("--infer_output", default="runs/mrcr_infer.jsonl",
                   help="Inference+score output JSONL (also used as input for --rescore / --skip_infer)")
    p.add_argument("--rescore_output", default=None,
                   help="Output path for rescored JSONL. Defaults to <infer_output>_rescored.jsonl")
    p.add_argument("--max_input_tokens", type=int, default=131072,
                   help="Max input tokens (paper: 131072). 0=disable")
    p.add_argument("--skip_infer", action="store_true",
                   help="Skip inference and only run the report on an existing infer_output file")
    p.add_argument("--rescore", action="store_true",
                   help="Re-apply strip_thinking + re-grade an existing infer_output without re-running inference")
    args = p.parse_args()

    if args.rescore:
        # Rescore mode: no inference, no tokenizer, no data_file needed
        run_rescore(args)
    else:
        if not args.skip_infer:
            if not args.data_file:
                p.error("--data_file is required unless --skip_infer or --rescore is set")
            if not args.model_path:
                p.error("--model_path is required unless --skip_infer or --rescore is set")
            asyncio.run(run_inference(args))
        run_report(args)

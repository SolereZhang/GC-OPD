"""
LongBench v1 QA Evaluation Pipeline
- Step 1: Download data from THUDM/LongBench (8 QA subsets)
- Step 2: Inference via local vLLM (OpenAI-compatible API)
- Step 3: F1 score evaluation (official LongBench metric)
Implements the fixed LongBench v1 QA evaluation used by this artifact.
"""

import json, os, re, string, argparse, asyncio
from collections import Counter
from openai import AsyncOpenAI
from openai import OpenAI
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
lock = threading.Lock()


QA_SUBSETS = [
    # "test"
    "narrativeqa",
    "qasper",
    # #"multifieldqa_en", "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    #"dureader",
]

PROMPT_TEMPLATE = """Please read the following text and answer the question below.
<text>
{context}
</text>
{question}
Format your response as follows: "Therefore, the answer is (insert answer here)"."""


# Let’s think step by step: $COT$


def normalize_zh(s):
    """Chinese: character-level, remove whitespace/punctuation."""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("P", "Z", "C"))
    return s.lower()

def normalize_en(s):
    """English: word-level, remove articles/punctuation/extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def is_chinese(text):
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def tokenize(text):
    if is_chinese(text):
        return list(normalize_zh(text))
    else:
        return normalize_en(text).split()

def f1_score(prediction, ground_truth):
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def token_accuracy(prediction, ground_truth):
    # pred_tokens = tokenize(prediction)
    # gold_tokens = tokenize(ground_truth)
    pred_tokens = normalize_answer(prediction)
    gold_tokens = normalize_answer(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    return num_same / len(gold_tokens)

def compute_f1(prediction, answers):
    """Max F1 over all reference answers (official LongBench protocol)."""
    return max(f1_score(prediction, ans) for ans in answers) if answers else 0.0

def compute_acc(prediction, answers):
    return max(token_accuracy(prediction, ans) for ans in answers) if answers else 0.0
# ========================= Helpers =========================
def strip_think(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()

def extract_answer(text):
    # Strip brackets: (answer) → answer
    def clean(s):
        s = s.strip().strip('"').strip("'").rstrip(".")
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        return s

    # Layer 1: find the LAST line matching "Therefore, the answer is ..."
    # Using last match to handle self-correction in CoT outputs
    pattern = re.compile(r"[Tt]herefore,?\s*the answer is:?\s*(.+)", re.IGNORECASE)
    match = None
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            match = m
    if match:
        return clean(match.group(1))

    # Layer 2: fallback — return the last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return clean(lines[-1])

    # Layer 3: final fallback
    return text.strip()

def middle_truncate(text, max_chars):
    """Truncate from middle, preserving beginning and end (official LongBench protocol)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_done_ids(path):
    ids = set()
    if os.path.exists(path):
        for item in load_jsonl(path):
            ids.add(item.get("id", ""))
    return ids

def download_data(data_dir):
    from datasets import load_dataset
    os.makedirs(data_dir, exist_ok=True)
    for subset in QA_SUBSETS:
        out_path = os.path.join(data_dir, f"{subset}.jsonl")
        if os.path.exists(out_path):
            print(f"  {subset}: exists ({sum(1 for _ in open(out_path))} samples)")
            continue
        print(f"  {subset}: downloading...")
        ds = load_dataset("THUDM/LongBench", subset, split="test", trust_remote_code=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                record = {
                    "id": f"{subset}_{i}",
                    "dataset": subset,
                    "input": item["input"],
                    "context": item["context"],
                    "answers": item["answers"],
                    "length": item.get("length", 0),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {subset}: {len(ds)} samples saved")

async def run_inference(args):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_url)
    os.makedirs(args.runs_dir, exist_ok=True)

    for subset in QA_SUBSETS:
        data_path = os.path.join(args.data_dir, f"{subset}.jsonl")
        if not os.path.exists(data_path):
            print(f"  [SKIP] {subset}: data not found")
            continue

        out_path = os.path.join(args.runs_dir, f"{args.model}_{subset}.jsonl")
        data = load_jsonl(data_path)
        done = load_done_ids(out_path)
        todo = [x for x in data if x["id"] not in done]

        if not todo:
            print(f"  {subset}: all {len(data)} done, skipping")
            continue

        print(f"  {subset}: {len(done)} done, {len(todo)} remaining")
        sem = asyncio.Semaphore(args.concurrency)

        async def infer_one(item):
            async with sem:
                ctx = middle_truncate(item["context"], args.max_context_chars)
                prompt = PROMPT_TEMPLATE.format(context=ctx, question=item["input"])
                msgs = [{"role": "user", "content": prompt}]
                for attempt in range(5):
                    try:
                        resp = await client.chat.completions.create(
                            model=args.model, messages=msgs, stream=False,
                            temperature=0.7, top_p=0.95, max_tokens=51200)
                        raw = resp.choices[0].message.content.strip()
                        return {**item, "response": strip_think(raw), "raw_response": raw}
                    except Exception as e:
                        if attempt < 4:
                            await asyncio.sleep(10 * (attempt + 1))
                        else:
                            print(f"    [FAIL] {item['id']}: {e}")
                            return None

        tasks = [infer_one(x) for x in todo]
        with open(out_path, "a", encoding="utf-8") as f:
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"  {subset}"):
                r = await fut
                if r:
                    out = {"id": r["id"], "dataset": r["dataset"], "answers": r["answers"],
                           "input": r["input"], "response": r["response"], "model": args.model}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()

def llm_score(text: str) -> int:
    pattern = r"\[\[\s*(yes|no)\s*\]\]"
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        print(f"解析失败: {text}")
        return 0.5

    answer = match.group(1).lower()
    return 1 if answer == "yes" else 0

def max_llm_judge(question, pred, answers, judge_api_key, judge_base_url, judge_model):
    return max(llm_judge(question, pred, ans, judge_api_key, judge_base_url, judge_model) for ans in answers) if answers else 0.0

def llm_judge(question, pred, answer, judge_api_key, judge_base_url, judge_model):
    prompt = f'''You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer1 and Answer2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.
Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
Problem: {question}
Answer 1: {pred}
Answer 2: {answer}'''
    import time
    client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            score = llm_score(response.choices[0].message.content)
            if score == 0.5:
                raise ValueError("Score is ambiguous, retrying...")
            return score
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError("Judge failed after maximum retries") from e
            time.sleep(3)


def process_item(item, judge_api_key, judge_base_url, judge_model):
    pred = extract_answer(item.get("response", ""))
    answers = item.get("answers", [])
    question = item.get("input", "")
    if isinstance(answers, str):
        answers = [answers]
    score = max_llm_judge(question=question, pred=pred, answers=answers,
                          judge_api_key=judge_api_key, judge_base_url=judge_base_url,
                          judge_model=judge_model)
    item["pred"] = pred
    item["llm_score"] = score
    return item, score


def run_eval(args):
    os.makedirs(args.evals_dir, exist_ok=True)
    all_scores = {}

    for subset in QA_SUBSETS:
        infer_path = os.path.join(args.runs_dir, f"{args.model}_{subset}.jsonl")
        eval_path = os.path.join(args.evals_dir, f"{args.model}_{subset}.jsonl")
        if not os.path.exists(infer_path):
            print(f"  [SKIP] {subset}: no inference results")
            continue

        data = load_jsonl(infer_path)
        scores = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_item, item,
                                       args.judge_api_key, args.judge_base_url, args.judge_model)
                       for item in data]

            with open(eval_path, "a", encoding="utf-8") as f:
                for future in as_completed(futures):
                    item, score = future.result()

                    scores.append(score)

                    with lock:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")


        avg = sum(scores) / len(scores) * 100 if scores else 0.0
        all_scores[subset] = avg
        print(f"  {subset:20s}: F1 = {avg:.2f}  ({len(scores)} samples)")

    # Final report
    print("\n" + "=" * 55)
    print("LONGBENCH V1 QA RESULTS")
    print("=" * 55)

    single_doc = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"]
    multi_doc   = ["hotpotqa", "2wikimqa", "musique", "dureader"]

    sd_scores = [all_scores[s] for s in single_doc if s in all_scores]
    md_scores = [all_scores[s] for s in multi_doc  if s in all_scores]

    for subset in QA_SUBSETS:
        if subset in all_scores:
            print(f"  {subset:20s}: {all_scores[subset]:6.2f}")

    print("-" * 55)
    if sd_scores:
        print(f"  {'Single-Doc QA avg':20s}: {sum(sd_scores)/len(sd_scores):6.2f}")
    if md_scores:
        print(f"  {'Multi-Doc QA avg':20s}: {sum(md_scores)/len(md_scores):6.2f}")
    if all_scores:
        print(f"  {'Overall QA avg':20s}: {sum(all_scores.values())/len(all_scores):6.2f}")
    print("=" * 55)

    summary_path = os.path.join(args.evals_dir, f"{args.model}_lbv1qa_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

# ========================= Main =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="main-table-candidate")
    p.add_argument("--api_url", default=os.getenv("LLM_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api_key", default=os.getenv("API_KEY", "EMPTY"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--evals_dir", required=True)
    p.add_argument("--max_context_chars", type=int, default=500000,
                    help="Max context chars before middle truncation (~128K tokens)")
    p.add_argument("--judge_api_key", default=os.getenv("ARK_API_KEY", ""))
    p.add_argument("--judge_base_url", default=os.getenv("JUDGE_BASE_URL", "http://127.0.0.1:1/v1"))
    p.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", "main-table-judge"))
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_infer", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    if not args.skip_download:
        print("[Step 1] Downloading LongBench v1 QA data...")
        download_data(args.data_dir)

    if not args.skip_infer:
        print("\n[Step 2] Running inference...")
        asyncio.run(run_inference(args))

    if not args.skip_eval:
        print("\n[Step 3] Evaluating (F1 score)...")
        run_eval(args)

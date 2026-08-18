"""
CorpusQA Evaluation Pipeline
- Step 1: Inference via local vLLM (OpenAI-compatible API)
- Step 2: LLM-as-judge evaluation via a local API
Implements the fixed CorpusQA evaluation used by this artifact.
"""

import json, os, re, argparse, asyncio, threading
import httpx
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= Official ORM Prompt (Paper Table 6) =========================
JUDGE_SYSTEM = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]"""

JUDGE_USER = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

# ========================= Helpers =========================
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def load_done_ids(path):
    ids = set()
    if os.path.exists(path):
        for item in load_jsonl(path):
            if 'id' in item: ids.add(item['id'])
    return ids

def strip_think(text):
    return text.split('</think>')[-1].strip() if '</think>' in text else text.strip()

def extract_answer(text):
    m = re.search(r'[Tt]he answer is:?\s*(.*)', text)
    return m.group(1).strip() if m else text.strip()

def get_domain(qid):
    for d in ['financial_zh', 'financial_en', 'education_en', 'real_estate_en']:
        if qid.startswith(d): return d
    return None

# ========================= Step 1: Inference =========================
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
    data = load_jsonl(args.data_file)
    done = load_done_ids(args.infer_output)
    todo = [x for x in data if x['id'] not in done]

    if not todo:
        print(f"Inference: all {len(data)} samples done, skipping.")
        return

    print(f"Inference: {len(done)} done, {len(todo)} remaining")
    print(f"Timeout: read={args.timeout}s, max_tokens={args.max_tokens}, concurrency={args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)

    async def infer_one(item):
        async with sem:
            msgs = [{"role": m["role"], "content": m["content"]} for m in item['prompt']]
            for attempt in range(args.max_retries):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model, messages=msgs, stream=False,
                        temperature=0.7, top_p=0.95, max_tokens=args.max_tokens)
                    raw = resp.choices[0].message.content.strip()
                    return {**item, "response": strip_think(raw), "raw_response": raw}
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

    tasks = [infer_one(x) for x in todo]
    os.makedirs(os.path.dirname(args.infer_output) or '.', exist_ok=True)
    with open(args.infer_output, 'a', encoding='utf-8') as f:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
            r = await fut
            if r:
                out = {"id": r["id"], "question": r["question"], "answer": r["answer"],
                       "response": r["response"], "model": args.model}
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
                f.flush()

# ========================= Step 2: Evaluation =========================
def run_eval(args):
    client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    data = load_jsonl(args.infer_output)
    done = load_done_ids(args.eval_output)
    todo = [x for x in data if x['id'] not in done]

    if not todo:
        print(f"Eval: all {len(data)} samples done.")
    else:
        print(f"Eval: {len(done)} done, {len(todo)} remaining (judge: {args.judge_model})")

    stats = {'correct': 0, 'total': 0}
    domain_stats = {d: {'correct': 0, 'total': 0}
                    for d in ['financial_zh', 'financial_en', 'education_en', 'real_estate_en']}

    # Restore stats from existing results
    for item in load_jsonl(args.eval_output) if os.path.exists(args.eval_output) else []:
        stats['total'] += 1
        if item.get('correct'): stats['correct'] += 1
        d = get_domain(item['id'])
        if d:
            domain_stats[d]['total'] += 1
            if item.get('correct'): domain_stats[d]['correct'] += 1

    def judge_one(item):
        extracted = extract_answer(item.get('response', ''))
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(
                    problem=item['question'], answer_1=extracted, answer_2=item['answer'])}]
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=args.judge_model, messages=msgs, temperature=0.0)
                content = resp.choices[0].message.content
                correct = "[[YES]]" in content and "[[NO]]" not in content
                return {"id": item["id"], "question": item["question"],
                        "gold": item["answer"], "pred": extracted,
                        "correct": correct, "reason": content.strip()}
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Judge failed for {item['id']}: {e}") from e
                import time; time.sleep(5 * (attempt + 1))

    lock = threading.Lock()
    os.makedirs(os.path.dirname(args.eval_output) or '.', exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool, \
         open(args.eval_output, 'a', encoding='utf-8') as f:
        futs = {pool.submit(judge_one, x): x['id'] for x in todo}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                with lock:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
                    stats['total'] += 1
                    if r['correct']: stats['correct'] += 1
                    d = get_domain(r['id'])
                    if d:
                        domain_stats[d]['total'] += 1
                        if r['correct']: domain_stats[d]['correct'] += 1
                    if stats['total'] % 20 == 0:
                        print(f"  [{stats['total']}] acc={stats['correct']/stats['total']:.2%}")

    # Report
    print("\n" + "=" * 50)
    print("CORPUSQA RESULTS")
    print("=" * 50)
    if stats['total']:
        print(f"Overall: {stats['correct']}/{stats['total']} = {stats['correct']/stats['total']:.2%}")
        for d in ['financial_zh', 'financial_en', 'education_en', 'real_estate_en']:
            s = domain_stats[d]
            if s['total']:
                print(f"  {d:20s}: {s['correct']:3d}/{s['total']:3d} = {s['correct']/s['total']:.2%}")
    print("=" * 50)

# ========================= Main =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True, help="128k_4domains.jsonl path")
    p.add_argument("--model", default="main-table-candidate")
    p.add_argument("--api_url", default=os.getenv("LLM_URL", "http://<LLM_HOST>:<PORT>/v1"))
    p.add_argument("--api_key", default=os.getenv("API_KEY", "<API_KEY>"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=51200, help="Max generation tokens")
    p.add_argument("--timeout", type=float, default=3600, help="Read timeout in seconds (default 3600=1hr)")
    p.add_argument("--max_retries", type=int, default=3, help="Max retries per sample")
    p.add_argument("--infer_output", default="runs/corpusqa_infer.jsonl")
    p.add_argument("--eval_output", default="evals/corpusqa_eval.jsonl")
    p.add_argument("--judge_model", default="main-table-judge")
    p.add_argument("--judge_api_key", required=True)
    p.add_argument("--judge_base_url", default="http://127.0.0.1:1/v1")
    p.add_argument("--skip_infer", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    if not args.skip_infer:
        asyncio.run(run_inference(args))
    if not args.skip_eval:
        run_eval(args)

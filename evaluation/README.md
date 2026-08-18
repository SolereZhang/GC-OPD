# Evaluation

This directory contains the evaluation protocol used for the main table.
The entrypoint first generates responses with the candidate checkpoint, stops
that server, evaluates judge-based tasks with a local judge model, and reports
the unweighted mean over five benchmarks.

## Benchmarks

| Benchmark | Evaluation split | Scoring |
|---|---|---|
| DocMath | Four `testmini` subsets | EvalScope task score |
| Frames | Test split | EvalScope task score |
| MRCR | Examples up to 128K | Official sequence-matching score |
| CorpusQA | 128K four-domain split | Local LLM judge |
| LongBench v1 QA | NarrativeQA, Qasper, HotpotQA, 2WikiMultihopQA, and MuSiQue | Mean local-judge score over five subsets |

All benchmark scores are normalized to `[0, 1]` before computing the final
unweighted arithmetic mean.

## Environment

Install PyTorch and vLLM for the local CUDA runtime, then install the recorded
evaluation dependencies:

```bash
pip install -r evaluation/requirements-eval.txt
```

The reported environment used Python 3.12, PyTorch 2.9.1, vLLM 0.11.2,
Transformers 4.57.1, and EvalScope 1.8.1.

## Data Layout

Place the evaluation data under one root directory:

```text
DATA_ROOT/
|-- CorpusQA/128k_4domains.jsonl
|-- DocMath/DocMath-Eval/data/
|   |-- complong_testmini-00000-of-00001.parquet
|   |-- compshort_testmini-00000-of-00001.parquet
|   |-- simplong_testmini-00000-of-00001.parquet
|   `-- simpshort_testmini-00000-of-00001.parquet
|-- Frames/test.jsonl
|-- MRCR/mrcr_0_128K.jsonl
`-- LongBench/Longbench/data/
    |-- narrativeqa.jsonl
    |-- qasper.jsonl
    |-- hotpotqa.jsonl
    |-- 2wikimqa.jsonl
    `-- musique.jsonl
```

The candidate and judge model directories must be local Hugging Face
checkpoints containing `config.json`.

## Run

Set the four paths and launch the entrypoint:

```bash
export MODEL_PATH=/path/to/huggingface-checkpoint
export JUDGE_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export DATA_ROOT=/path/to/evaluation-data
export OUTPUT_DIR=/path/to/evaluation-output

bash evaluation/run_main_table_evaluation.sh
```

The entrypoint accepts no positional arguments or benchmark overrides. Only
hardware placement can be adjusted:

```bash
MODEL_TP=4 JUDGE_TP=4 bash evaluation/run_main_table_evaluation.sh
```

Use `DRY_RUN=1` to print the resolved protocol without loading either model.
Use a separate `OUTPUT_DIR` for each candidate checkpoint; rerunning with the
same inputs resumes completed benchmarks.

## Protocol

| Setting | Value |
|---|---|
| Candidate thinking mode | disabled |
| Maximum input tokens | 120,000 |
| Maximum output tokens | 8,192 |
| vLLM maximum model length | 131,072 |
| YaRN factor | 4 |
| Sampling | temperature 0.7, top-p 0.95 |
| Local judge | Qwen3-30B-A3B-Instruct-2507 |
| Judge temperature | 0 |
| Judge maximum tokens | 2,048 |
| Judge maximum model length | 32,768 |

## Outputs

```text
OUTPUT_DIR/
|-- predictions/main_table_five_task/
`-- judge/<judge-run-directory>/
    |-- judge_summary.json
    |-- main_table_scores.json
    `-- main_table_scores.csv
```

`main_table_scores.json` records each normalized benchmark score together with
`main_table_average` and `main_table_average_percent`. The standalone
aggregator can regenerate these files from completed prediction and judge
artifacts:

```bash
export PRED_OUT="$OUTPUT_DIR/predictions/main_table_five_task"
export JUDGE_OUT="$OUTPUT_DIR/judge/<judge-run-directory>"

python evaluation/aggregate_main_table.py \
  --pred-out "$PRED_OUT" \
  --judge-out "$JUDGE_OUT" \
  --model-name main_table_candidate
```

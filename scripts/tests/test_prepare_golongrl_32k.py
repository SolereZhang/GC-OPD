from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import prepare_golongrl_32k as preparation


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs = None
        self.tokenize_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "|".join(message["content"] for message in messages)

    def __call__(self, text, **kwargs):
        self.tokenize_kwargs = kwargs
        return {"input_ids": list(range(len(text)))}


class PrepareGoLongRL32KTest(unittest.TestCase):
    def tearDown(self) -> None:
        preparation._TOKENIZER = None

    def test_instruction_is_appended_to_last_user_message_once(self) -> None:
        prompt = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "question"},
        ]

        preparation.append_instruction(prompt)
        preparation.append_instruction(prompt)

        self.assertEqual(prompt[0]["content"], "first")
        self.assertEqual(prompt[2]["content"].count("Important output format"), 1)

    def test_prompt_length_matches_training_tokenization_contract(self) -> None:
        tokenizer = FakeTokenizer()
        preparation._TOKENIZER = tokenizer

        result = preparation.prompt_length(
            {"prompt": [{"role": "user", "content": "abc"}]}
        )

        self.assertEqual(result, {"_prompt_len": 3})
        self.assertEqual(
            tokenizer.template_kwargs,
            {
                "add_generation_prompt": True,
                "tokenize": False,
                "enable_thinking": False,
            },
        )
        self.assertEqual(
            tokenizer.tokenize_kwargs,
            {"add_special_tokens": False, "truncation": False},
        )

    def test_split_reserves_first_256_rows_before_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shard = root / "train-00000.parquet"
            output_dir = root / "prepared"
            rows = [
                {
                    "prompt": [{"role": "user", "content": f"question-{index}"}],
                    "ability": "test",
                }
                for index in range(260)
            ]
            pq.write_table(pa.Table.from_pylist(rows), shard)

            train_path, val_path, train_count, val_count = preparation.prepare_grpo_split(
                [shard], output_dir, batch_size=17
            )

            self.assertEqual((train_count, val_count), (4, 256))
            train_rows = pq.read_table(train_path).to_pylist()
            val_rows = pq.read_table(val_path).to_pylist()
            self.assertTrue(val_rows[0]["prompt"][0]["content"].startswith("question-0"))
            self.assertTrue(train_rows[0]["prompt"][0]["content"].startswith("question-256"))

    def test_paper_size_check_rejects_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "do not match the paper split"):
            preparation.verify_counts(
                preparation.EXPECTED_RAW_ROWS,
                {
                    "train": {"kept": preparation.EXPECTED_TRAIN_ROWS - 1},
                    "val": {"kept": preparation.EXPECTED_VAL_ROWS},
                },
            )


if __name__ == "__main__":
    unittest.main()

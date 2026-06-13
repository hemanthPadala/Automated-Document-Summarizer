"""
Block 2 — Data Pipeline

Loads the CNN/DailyMail dataset, subsamples a training set, tokenizes
articles/highlights with the BART tokenizer, and saves the resulting
train/validation/test splits to disk for use by training/train.py.

Usage:
    python training/prepare_data.py
    python training/prepare_data.py --train-size 30000 --model-name facebook/bart-large-cnn
"""

import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer

INPUT_MAX_LENGTH = 1024  # BART's hard limit
TARGET_MAX_LENGTH = 128  # ~300 words


def build_preprocess_fn(tokenizer):
    def preprocess(batch):
        inputs = tokenizer(
            batch["article"],
            max_length=INPUT_MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            batch["highlights"],
            max_length=TARGET_MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        # Replace pad token ids in labels with -100 so the loss ignores them.
        label_ids = labels["input_ids"]
        inputs["labels"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in label_ids
        ]
        return inputs

    return preprocess


def main():
    parser = argparse.ArgumentParser(description="Prepare CNN/DailyMail data for BART fine-tuning")
    parser.add_argument("--model-name", default="facebook/bart-large-cnn")
    parser.add_argument("--train-size", type=int, default=30000, help="Number of training examples to subsample")
    parser.add_argument("--val-size", type=int, default=2000, help="Number of validation examples to subsample")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test examples to subsample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--num-proc", type=int, default=1, help="Parallel workers for tokenization")
    args = parser.parse_args()

    print(f"Loading CNN/DailyMail (3.0.0)...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    print({split: len(dataset[split]) for split in dataset})

    print(f"Subsampling: train={args.train_size}, val={args.val_size}, test={args.test_size} (seed={args.seed})")
    train_subset = dataset["train"].shuffle(seed=args.seed).select(range(min(args.train_size, len(dataset["train"]))))
    val_subset = dataset["validation"].shuffle(seed=args.seed).select(range(min(args.val_size, len(dataset["validation"]))))
    test_subset = dataset["test"].shuffle(seed=args.seed).select(range(min(args.test_size, len(dataset["test"]))))

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    preprocess = build_preprocess_fn(tokenizer)

    columns_to_remove = train_subset.column_names

    print("Tokenizing splits (this can take a while for the train split)...")
    tokenized_train = train_subset.map(
        preprocess, batched=True, remove_columns=columns_to_remove, num_proc=args.num_proc
    )
    tokenized_val = val_subset.map(
        preprocess, batched=True, remove_columns=columns_to_remove, num_proc=args.num_proc
    )
    tokenized_test = test_subset.map(
        preprocess, batched=True, remove_columns=columns_to_remove, num_proc=args.num_proc
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving tokenized splits to {args.output_dir}")
    tokenized_train.save_to_disk(os.path.join(args.output_dir, "train"))
    tokenized_val.save_to_disk(os.path.join(args.output_dir, "validation"))
    tokenized_test.save_to_disk(os.path.join(args.output_dir, "test"))

    # Keep an untokenized copy of the test split for ROUGE evaluation (Block 5),
    # which needs the raw reference text rather than token ids.
    raw_test_path = os.path.join(args.output_dir, "test_raw")
    test_subset.save_to_disk(raw_test_path)

    print("Done.")
    print(f"  train:      {len(tokenized_train)} examples -> {os.path.join(args.output_dir, 'train')}")
    print(f"  validation: {len(tokenized_val)} examples -> {os.path.join(args.output_dir, 'validation')}")
    print(f"  test:       {len(tokenized_test)} examples -> {os.path.join(args.output_dir, 'test')}")
    print(f"  test_raw:   {len(test_subset)} examples -> {raw_test_path}")


if __name__ == "__main__":
    main()

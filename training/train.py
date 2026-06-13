"""
Block 3 — Fine-Tuning

Fine-tunes facebook/bart-large-cnn on the tokenized CNN/DailyMail subset
produced by training/prepare_data.py, evaluating with ROUGE each epoch and
saving the best checkpoint.

Usage:
    python training/train.py
    python training/train.py --epochs 3 --batch-size 4

Note: fine-tuning BART-large is GPU-bound. On CPU this will be extremely
slow (BART-large has ~400M parameters); use a GPU (e.g. Google Colab) for
the actual training run. This script auto-detects CUDA and adjusts
fp16/batch size accordingly.
"""

import argparse
import os

import numpy as np
import torch
from datasets import load_from_disk
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def build_compute_metrics(tokenizer):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_rouge(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 with pad_token_id before decoding.
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = [
            scorer.score(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds)
        ]
        return {
            "rouge1": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
            "rouge2": sum(s["rouge2"].fmeasure for s in scores) / len(scores),
            "rougeL": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
        }

    return compute_rouge


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART on CNN/DailyMail")
    parser.add_argument("--model-name", default="facebook/bart-large-cnn")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/bart-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print(
            "WARNING: No GPU detected. Fine-tuning BART-large on CPU is impractical "
            "(expect many hours per epoch). Consider running this on a GPU instance."
        )

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    print(f"Loading tokenized datasets from {args.data_dir}")
    tokenized_train = load_from_disk(os.path.join(args.data_dir, "train"))
    tokenized_val = load_from_disk(os.path.join(args.data_dir, "validation"))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        predict_with_generate=True,
        generation_max_length=128,
        fp16=cuda_available,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()

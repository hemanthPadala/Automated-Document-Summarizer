"""
Block 5 — Evaluation

Measures ROUGE-1, ROUGE-2, and ROUGE-L of the summarizer model on a subset
of the CNN/DailyMail test split.

Usage:
    python evaluation/eval.py
    python evaluation/eval.py --num-samples 1000 --model-path models/bart-finetuned/final

Target benchmarks (from build-plan.md):
    ROUGE-1 >= 0.42
    ROUGE-2 >= 0.20
    ROUGE-L >= 0.44  (primary metric)
"""

import argparse
import json
import os
import sys

import torch
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.summarizer import Stage, get_device, load_model  # noqa: E402

TARGETS = {"rouge1": 0.42, "rouge2": 0.20, "rougeL": 0.44}


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    return {
        "rouge1": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
        "rouge2": sum(s["rouge2"].fmeasure for s in scores) / len(scores),
        "rougeL": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
    }


def load_test_samples(num_samples: int, processed_dir: str):
    raw_test_path = os.path.join(processed_dir, "test_raw")
    if os.path.isdir(raw_test_path):
        from datasets import load_from_disk

        dataset = load_from_disk(raw_test_path)
    else:
        print(f"{raw_test_path} not found; loading CNN/DailyMail test split directly "
              f"(run training/prepare_data.py to cache a local subset).")
        from datasets import load_dataset

        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    num_samples = min(num_samples, len(dataset))
    return dataset.select(range(num_samples))


def generate_summaries(model, tokenizer, articles, device, batch_size=4, num_beams=4):
    predictions = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", max_length=1024, truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=num_beams,
                max_length=128,
                min_length=30,
                length_penalty=2.0,
                early_stopping=True,
            )
        predictions.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(articles))}/{len(articles)}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarizer with ROUGE on CNN/DailyMail test set")
    parser.add_argument("--model-path", default=None, help="Model path/name (default: auto-detect fine-tuned, else facebook/bart-large-cnn)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    with Stage("Load model"):
        model, tokenizer = load_model(args.model_path)

    with Stage(f"Load {args.num_samples} test samples"):
        samples = load_test_samples(args.num_samples, args.processed_dir)
    articles = samples["article"]
    references = samples["highlights"]

    with Stage("Generate summaries (inference)") as gen_stage:
        predictions = generate_summaries(model, tokenizer, articles, device, args.batch_size, args.num_beams)
    elapsed = gen_stage.elapsed

    with Stage("Compute ROUGE (evaluation)"):
        scores = compute_rouge(predictions, references)

    results = {
        "model_path": args.model_path or "auto",
        "device": device,
        "num_samples": len(articles),
        "num_beams": args.num_beams,
        "elapsed_seconds": round(elapsed, 2),
        "rouge1": round(scores["rouge1"], 4),
        "rouge2": round(scores["rouge2"], 4),
        "rougeL": round(scores["rougeL"], 4),
        "targets": TARGETS,
        "meets_targets": {k: scores[k] >= TARGETS[k] for k in TARGETS},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

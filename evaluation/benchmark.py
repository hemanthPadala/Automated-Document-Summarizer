"""
Block 7 — End-to-End Test & Performance Check

Times the full PDF -> summary pipeline on one or more PDFs and reports
elapsed time and summary word count against the targets:
    - Under 8 seconds per document
    - Summary word count between 250-350 words

Usage:
    python evaluation/benchmark.py path/to/file1.pdf path/to/file2.pdf
    python evaluation/benchmark.py --length long *.pdf
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.summarizer import DEFAULT_NUM_BEAMS, get_device, load_model, summarize_pdf  # noqa: E402

WORD_COUNT_RANGE = (250, 350)
TIME_TARGET_SECONDS = 8


def main():
    parser = argparse.ArgumentParser(description="Benchmark the PDF -> summary pipeline")
    parser.add_argument("pdfs", nargs="+", help="One or more PDF file paths")
    parser.add_argument("--length", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--num-beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="evaluation/benchmark_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer = load_model(args.model_path)

    results = []
    for pdf_path in args.pdfs:
        if not os.path.isfile(pdf_path):
            print(f"Skipping (not found): {pdf_path}")
            continue

        print(f"\nSummarizing {pdf_path} (length={args.length}, num_beams={args.num_beams})...")
        t0 = time.time()
        summary = summarize_pdf(pdf_path, model, tokenizer, length=args.length, num_beams=args.num_beams)
        elapsed = time.time() - t0

        word_count = len(summary.split())
        meets_time = elapsed <= TIME_TARGET_SECONDS
        meets_words = WORD_COUNT_RANGE[0] <= word_count <= WORD_COUNT_RANGE[1]

        print(f"  Time: {elapsed:.2f}s ({'OK' if meets_time else 'OVER TARGET'})")
        print(f"  Words: {word_count} ({'OK' if meets_words else 'OUTSIDE 250-350 RANGE'})")
        print(f"  Summary: {summary}")

        results.append(
            {
                "pdf": pdf_path,
                "device": device,
                "length": args.length,
                "num_beams": args.num_beams,
                "elapsed_seconds": round(elapsed, 2),
                "word_count": word_count,
                "meets_time_target": meets_time,
                "meets_word_count_target": meets_words,
                "summary": summary,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

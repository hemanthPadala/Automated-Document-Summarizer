"""
Block 4 — Inference Pipeline (PDF -> Summary)

Given any PDF file, extracts text, splits it into overlapping token chunks,
summarizes each chunk, then hierarchically reduces the chunk summaries into
a single final summary.

Public API:
    load_model(model_path=None) -> (model, tokenizer)
    extract_text(pdf_path) -> str
    chunk_text(text, tokenizer, max_tokens=900, overlap=50) -> list[str]
    summarize_chunks(chunks, model, tokenizer, ...) -> list[str]
    summarize_pdf(pdf_path, model, tokenizer, length="medium") -> str
"""

import os

import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

# facebook/bart-large-cnn is BART-large already fine-tuned on CNN/DailyMail,
# so it works out of the box. If a further fine-tuned checkpoint exists at
# FINETUNED_MODEL_PATH (see training/train.py), that is used instead.
DEFAULT_MODEL_PATH = "facebook/bart-large-cnn"
FINETUNED_MODEL_PATH = os.path.join("models", "bart-finetuned", "final")

# Generation params used for summarizing individual chunks (and intermediate
# reduction rounds). Kept compact so multiple summaries can be re-combined
# and still fit within the model's 1024-token input limit.
CHUNK_GEN_PARAMS = {"max_length": 128, "min_length": 30}

# Generation params for the FINAL output, selectable via the Streamlit
# sidebar ("summary length: short / medium / long").
LENGTH_PRESETS = {
    "short": {"max_length": 90, "min_length": 25},
    "medium": {"max_length": 180, "min_length": 60},
    "long": {"max_length": 350, "min_length": 120},
}

CHUNK_MAX_TOKENS = 900  # leaves headroom below BART's 1024-token limit
CHUNK_OVERLAP = 50

# Beam search width default. On CPU, beam search dominates latency almost
# linearly with num_beams, so default to a narrower beam there; GPUs can
# afford the higher-quality default of 4.
DEFAULT_NUM_BEAMS = 4 if torch.cuda.is_available() else 2


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str | None = None):
    """Load a BART model + tokenizer for summarization.

    If model_path is not given, prefers a locally fine-tuned checkpoint at
    models/bart-finetuned/final, falling back to facebook/bart-large-cnn.
    """
    if model_path is None:
        model_path = FINETUNED_MODEL_PATH if os.path.isdir(FINETUNED_MODEL_PATH) else DEFAULT_MODEL_PATH

    device = get_device()
    if device == "cpu":
        # PyTorch defaults to using only half the available cores; using all
        # of them roughly halves CPU generation time.
        torch.set_num_threads(os.cpu_count())

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF, page by page."""
    doc = fitz.open(pdf_path)
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    return text


def chunk_text(text: str, tokenizer, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of at most max_tokens tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def summarize_chunks(
    chunks: list[str],
    model,
    tokenizer,
    device: str | None = None,
    num_beams: int = 4,
    max_length: int = 128,
    min_length: int = 30,
    batch_size: int = 4,
) -> list[str]:
    """Summarize each chunk, processing in batches for efficiency."""
    device = device or get_device()
    summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                early_stopping=True,
            )
        summaries.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return summaries


def _group_by_token_budget(texts: list[str], tokenizer, max_tokens: int) -> list[list[str]]:
    """Greedily group texts so each group's combined token count stays under max_tokens."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if current and current_tokens + text_tokens > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += text_tokens
    if current:
        groups.append(current)
    return groups


def final_summary(chunk_summaries: list[str], model, tokenizer, device: str | None = None, **gen_kwargs) -> str:
    """Summarize the concatenation of chunk summaries into one final summary."""
    combined = " ".join(chunk_summaries)
    return summarize_chunks([combined], model, tokenizer, device=device, **gen_kwargs)[0]


def summarize_pdf(
    pdf_path: str,
    model,
    tokenizer,
    length: str = "medium",
    num_beams: int = DEFAULT_NUM_BEAMS,
    device: str | None = None,
) -> str:
    """Summarize a PDF end-to-end: extract -> chunk -> summarize -> hierarchically reduce.

    `length` selects the target length of the FINAL summary ("short", "medium",
    or "long"); see LENGTH_PRESETS.
    """
    device = device or get_device()
    final_preset = LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"])

    text = extract_text(pdf_path)
    if not text.strip():
        return ""

    chunks = chunk_text(text, tokenizer, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP)

    if len(chunks) == 1:
        # Single chunk: generate directly at the requested output length.
        return summarize_chunks(
            chunks, model, tokenizer, device=device, num_beams=num_beams, **final_preset
        )[0]

    summaries = summarize_chunks(
        chunks, model, tokenizer, device=device, num_beams=num_beams, **CHUNK_GEN_PARAMS
    )

    # Hierarchical reduction: repeatedly combine + re-summarize until one
    # summary remains. Intermediate rounds stay compact (CHUNK_GEN_PARAMS) so
    # they keep fitting the encoder; the last round applies the requested
    # output length.
    while len(summaries) > 1:
        groups = _group_by_token_budget(summaries, tokenizer, max_tokens=CHUNK_MAX_TOKENS)
        is_final_round = len(groups) == 1
        gen_params = final_preset if is_final_round else CHUNK_GEN_PARAMS
        combined_texts = [" ".join(group) for group in groups]
        summaries = summarize_chunks(
            combined_texts, model, tokenizer, device=device, num_beams=num_beams, **gen_params
        )

    return summaries[0]


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python -m inference.summarizer <pdf_path> [short|medium|long]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    length = sys.argv[2] if len(sys.argv) > 2 else "medium"

    print(f"Device: {get_device()}")
    print("Loading model...")
    model, tokenizer = load_model()

    print(f"Summarizing {pdf_path} (length={length})...")
    t0 = time.time()
    summary = summarize_pdf(pdf_path, model, tokenizer, length=length)
    elapsed = time.time() - t0

    print("\n--- Summary ---")
    print(summary)
    print(f"\nTime: {elapsed:.2f}s | Words: {len(summary.split())}")

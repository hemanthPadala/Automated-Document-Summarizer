# Automated Document Summarizer — Build Plan

**Approach:** BART-large fine-tuned on CNN/DailyMail  
**Target:** 50-page PDFs → ~300-word summary in under 8 seconds  
**UI:** Streamlit  
**Starting point:** Scratch (environment setup included)

---

## Block 1 — Environment Setup

**Goal:** Reproducible Python environment with all dependencies installed.

**Steps:**

1. Install Python 3.10+ (via [python.org](https://python.org) or `pyenv`)
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install core dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets rouge-score accelerate
   pip install pymupdf streamlit
   ```
4. Verify GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True if GPU present
   ```
5. Create project structure:
   ```j
   document-summarizer/
   ├── data/           # raw + processed datasets
   ├── training/       # fine-tuning scripts
   ├── inference/      # chunker + summarizer
   ├── app/            # Streamlit app
   ├── evaluation/     # ROUGE scoring
   └── models/         # saved checkpoints
   ```

**Deliverable:** `requirements.txt` and working Python environment.

---

## Block 2 — Data Pipeline

**Goal:** Load CNN/DailyMail dataset and preprocess into tokenized train/val/test splits.

**Steps:**

1. Load dataset via HuggingFace `datasets`:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("cnn_dailymail", "3.0.0")
   # train: ~287K, validation: ~13K, test: ~11K
   # We use a 30K subset for faster fine-tuning
   ```
2. Subsample 30K training pairs:
   ```python
   train_subset = dataset["train"].shuffle(seed=42).select(range(30000))
   ```
3. Tokenize with BART tokenizer:
   ```python
   from transformers import BartTokenizer
   tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

   def preprocess(batch):
       inputs = tokenizer(batch["article"], max_length=1024,
                          truncation=True, padding="max_length")
       labels = tokenizer(batch["highlights"], max_length=128,
                          truncation=True, padding="max_length")
       inputs["labels"] = labels["input_ids"]
       return inputs
   ```
4. Map tokenization across the dataset (use `batched=True` for speed)
5. Save processed splits to disk with `dataset.save_to_disk()`

**Key decisions:**
- Input max length: **1024 tokens** (BART's hard limit)
- Target max length: **128 tokens** (~300 words maps to ~128 BART tokens for highlights)
- Truncation strategy: truncate from the tail (news articles front-load key info)

**Deliverable:** Tokenized dataset saved to `data/processed/`.

---

## Block 3 — Fine-Tuning

**Goal:** Fine-tune `facebook/bart-large-cnn` on the 30K subset and save the best checkpoint.

**Steps:**

1. Load model:
   ```python
   from transformers import BartForConditionalGeneration
   model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
   ```
2. Configure training with `Seq2SeqTrainer`:
   ```python
   from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

   args = Seq2SeqTrainingArguments(
       output_dir="models/bart-finetuned",
       num_train_epochs=3,
       per_device_train_batch_size=4,       # reduce to 2 if OOM on GPU
       per_device_eval_batch_size=4,
       warmup_steps=500,
       weight_decay=0.01,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
       predict_with_generate=True,
       fp16=True,                           # mixed precision — faster on GPU
       logging_dir="logs",
   )
   ```
3. Define ROUGE compute metric (see Block 5)
4. Launch training:
   ```python
   trainer = Seq2SeqTrainer(
       model=model,
       args=args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_val,
       compute_metrics=compute_rouge,
   )
   trainer.train()
   ```
5. Save final model and tokenizer:
   ```python
   model.save_pretrained("models/bart-finetuned/final")
   tokenizer.save_pretrained("models/bart-finetuned/final")
   ```

**Expected training time:**
- ~4–6 hours on a single A100 (Google Colab Pro+)
- ~12–18 hours on a consumer RTX 3080

**Deliverable:** Checkpoint saved to `models/bart-finetuned/final/`.

---

## Block 4 — Inference Pipeline (PDF → Summary)

**Goal:** Given any PDF file, extract text, chunk it, run inference, and return a ~300-word summary.

**Steps:**

### 4a. PDF Text Extraction
```python
import fitz  # PyMuPDF

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text
```

### 4b. Chunking (handle >1024 token documents)
```python
def chunk_text(text: str, tokenizer, max_tokens=900, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += max_tokens - overlap  # sliding window with overlap
    return chunks
```

Use `max_tokens=900` (not 1024) to leave headroom for special tokens.

### 4c. Chunk Summarization
```python
def summarize_chunks(chunks, model, tokenizer, device="cuda"):
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt",
                           max_length=1024, truncation=True).to(device)
        ids = model.generate(inputs["input_ids"],
                             num_beams=4,
                             max_length=128,
                             min_length=30,
                             length_penalty=2.0,
                             early_stopping=True)
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return summaries
```

### 4d. Hierarchical Reduction
If more than 3 chunks, summarize the concatenated chunk summaries again to get the final output. This prevents the final summary from being a mechanical list of chunk summaries.

```python
def final_summary(chunk_summaries, model, tokenizer):
    combined = " ".join(chunk_summaries)
    return summarize_chunks([combined], model, tokenizer)[0]
```

**Deliverable:** `inference/summarizer.py` — callable function `summarize_pdf(path) -> str`.

---

## Block 5 — Evaluation

**Goal:** Measure ROUGE-1, ROUGE-2, and ROUGE-L on the CNN/DailyMail test split.

**Steps:**

1. Install evaluation library:
   ```bash
   pip install rouge-score
   ```
2. Define compute function:
   ```python
   from rouge_score import rouge_scorer

   def compute_rouge(predictions, references):
       scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                          use_stemmer=True)
       scores = [scorer.score(ref, pred)
                 for ref, pred in zip(references, predictions)]
       return {
           "rouge1": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
           "rouge2": sum(s["rouge2"].fmeasure for s in scores) / len(scores),
           "rougeL": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
       }
   ```
3. Run on 1K test samples (full 11K test set takes ~2 hours):
   ```python
   test_subset = dataset["test"].select(range(1000))
   ```
4. Target benchmarks:
   - ROUGE-1 ≥ 0.42
   - ROUGE-2 ≥ 0.20
   - **ROUGE-L ≥ 0.44** ← primary metric

**Deliverable:** `evaluation/eval.py` + logged results in `evaluation/results.json`.

---

## Block 6 — Streamlit App

**Goal:** Web UI where a user uploads a PDF and gets a summary in under 8 seconds.

**Steps:**

1. App structure (`app/app.py`):
   ```python
   import streamlit as st
   from inference.summarizer import load_model, summarize_pdf
   import tempfile, time

   @st.cache_resource
   def get_model():
       return load_model("models/bart-finetuned/final")

   st.title("Document Summarizer")
   uploaded = st.file_uploader("Upload a PDF", type="pdf")

   if uploaded:
       model, tokenizer = get_model()
       with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
           f.write(uploaded.read())
           tmp_path = f.name

       with st.spinner("Summarizing..."):
           t0 = time.time()
           summary = summarize_pdf(tmp_path, model, tokenizer)
           elapsed = time.time() - t0

       st.subheader("Summary")
       st.write(summary)
       st.caption(f"Generated in {elapsed:.1f}s")
       st.download_button("Download summary", summary, "summary.txt")
   ```

2. Run the app:
   ```bash
   streamlit run app/app.py
   ```

**Key UX details:**
- `@st.cache_resource` loads the model once — not on every upload
- Show word count of summary below the text area
- Add a sidebar with options: summary length (short / medium / long), language

**Deliverable:** Running Streamlit app at `localhost:8501`.

---

## Block 7 — End-to-End Test & Performance Check

**Goal:** Verify the full pipeline hits the 8-second target on a real 50-page PDF.

**Steps:**

1. Download a 50-page test PDF (e.g., an arXiv paper or annual report)
2. Time the full pipeline:
   ```python
   import time
   t0 = time.time()
   summary = summarize_pdf("test_50page.pdf", model, tokenizer)
   print(f"Time: {time.time() - t0:.2f}s | Words: {len(summary.split())}")
   ```
3. If over 8 seconds, optimize:
   - Reduce `num_beams` from 4 → 2 (fastest single change)
   - Batch chunks: pass all chunks to `model.generate()` at once
   - Use `fp16` inference: `model.half().to("cuda")`
4. Confirm summary word count is 250–350 words

**Deliverable:** Benchmark log with time and word count for 3 different PDFs.

---

## Build Order & Dependencies

```
Block 1 (Environment)
    └── Block 2 (Data Pipeline)
            └── Block 3 (Fine-Tuning)
                    └── Block 5 (Evaluation)
Block 1 (Environment)
    └── Block 4 (Inference Pipeline)
            ├── Block 3 (Fine-Tuning) ← needs the checkpoint
            └── Block 6 (Streamlit App)
                        └── Block 7 (End-to-End Test)
```

Blocks 2 and 4 can be developed in parallel once Block 1 is done.

---

## Estimated Timeline

| Block | Effort | Notes |
|---|---|---|
| 1 — Environment | 1–2 hrs | One-time setup |
| 2 — Data Pipeline | 2–3 hrs | Dataset download + tokenization |
| 3 — Fine-Tuning | 4–18 hrs | Mostly GPU wait time |
| 4 — Inference | 3–4 hrs | Chunking logic needs testing |
| 5 — Evaluation | 1–2 hrs | Straightforward once model is ready |
| 6 — Streamlit App | 2–3 hrs | Fast with Streamlit's simplicity |
| 7 — E2E Test | 1 hr | Benchmark + iterate |
| **Total** | **~2–3 days** | Excluding GPU training time |

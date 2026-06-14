# Automated Document Summarizer

Upload a PDF and get a concise, abstractive summary powered by
[BART-large-CNN](https://huggingface.co/facebook/bart-large-cnn), with a
Streamlit UI, adjustable summary length/quality, and optional translation.

## Requirements

- Python 3.10+ (tested on 3.14)
- ~3 GB free disk space for model weights (downloaded on first run)
- CPU-only works fine; an NVIDIA GPU + CUDA will speed up generation

## Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If you have an NVIDIA GPU with CUDA 11.8, install `torch` first with GPU
> support before the line above — see the comment at the top of
> `requirements.txt`.

### (Optional) Authenticate with Hugging Face

Model downloads work fine anonymously, but logging in avoids rate limits and
is required if you later use a private or gated model:

```powershell
.\venv\Scripts\hf.exe auth login
```

Paste an access token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
(read access is sufficient). This is stored in your user profile, not the
repo.

## Running the app

```powershell
.\venv\Scripts\python.exe -m streamlit run app/app.py
```

This opens the UI at [http://localhost:8501](http://localhost:8501). The
first run downloads `facebook/bart-large-cnn` (~1.6 GB), so it may take a
few minutes.

### Using it

1. Upload a PDF using the file uploader.
2. In the sidebar, choose:
   - **Summary length** — short / medium / long
   - **Output language** — English or a translated summary (Spanish, French, German, Hindi)
   - **Beam search width** — higher = better quality but slower
3. Wait for the summary to generate. Time elapsed and word count are shown
   below the result.
4. Use **Download summary** to save the result as a `.txt` file.

## Other scripts

| Script | Purpose |
| --- | --- |
| `python -m inference.summarizer <pdf_path> [short\|medium\|long]` | Summarize a PDF from the command line |
| `python training/prepare_data.py` | Build the CNN/DailyMail training dataset (Block 2) |
| `python training/train.py` | Fine-tune BART on the prepared dataset (Block 3) |
| `python evaluation/eval.py` | Compute ROUGE scores against the test set (Block 5) |
| `python evaluation/benchmark.py <pdf1> <pdf2> ...` | Time the end-to-end pipeline against the 8s / 250-350 word targets (Block 7) |
| `python evaluation/make_test_pdf.py` | Generate synthetic multi-page test PDFs |

See `build-plan.md` for the full project plan, targets, and tuning notes.

## Notes on performance (CPU)

- The app automatically uses all CPU cores (`torch.set_num_threads`) and
  defaults to a narrower beam search (`num_beams=2`) on CPU for speed.
- Long documents are split into overlapping chunks, summarized individually,
  then hierarchically combined into a final summary.
- If a summary takes longer than 8 seconds, try lowering the beam search
  width in the sidebar.

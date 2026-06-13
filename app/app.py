"""
Block 6 — Streamlit App

Web UI: upload a PDF, get a summary back in (ideally) under 8 seconds.

Run with:
    streamlit run app/app.py
"""

import os
import sys
import tempfile
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.summarizer import DEFAULT_NUM_BEAMS, get_device, load_model, summarize_pdf  # noqa: E402

# Output language options. English is native to the model; other languages
# are produced by translating the English summary with a small Helsinki-NLP
# MarianMT model, downloaded on first use.
LANGUAGES = {
    "English": None,
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
}


@st.cache_resource(show_spinner="Loading summarization model...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Loading translation model...")
def get_translator(target_lang_code: str):
    from transformers import pipeline

    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang_code}")


st.set_page_config(page_title="Document Summarizer", page_icon="📄")
st.title("📄 Automated Document Summarizer")
st.caption("Upload a PDF and get a concise summary powered by BART-large.")

with st.sidebar:
    st.header("Options")
    length = st.radio(
        "Summary length",
        options=["short", "medium", "long"],
        index=1,
        help="Controls the target length of the final summary.",
    )
    language_name = st.selectbox("Output language", list(LANGUAGES.keys()))
    num_beams = st.slider(
        "Beam search width",
        min_value=1,
        max_value=6,
        value=DEFAULT_NUM_BEAMS,
        help="Higher = better quality but slower. Lower this if generation is too slow.",
    )
    st.markdown("---")
    st.caption(f"Running on: **{get_device().upper()}**")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    model, tokenizer = get_model()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    try:
        with st.spinner("Summarizing..."):
            t0 = time.time()
            summary = summarize_pdf(tmp_path, model, tokenizer, length=length, num_beams=num_beams)
            elapsed = time.time() - t0
    finally:
        os.remove(tmp_path)

    if not summary:
        st.warning("No extractable text found in this PDF (it may be scanned/image-only).")
    else:
        lang_code = LANGUAGES[language_name]
        if lang_code:
            with st.spinner(f"Translating to {language_name}..."):
                try:
                    translator = get_translator(lang_code)
                    output_text = translator(summary, max_length=512)[0]["translation_text"]
                except Exception as exc:
                    st.warning(f"Translation to {language_name} unavailable ({exc}); showing English summary.")
                    output_text = summary
        else:
            output_text = summary

        st.subheader("Summary")
        st.write(output_text)

        word_count = len(output_text.split())
        st.caption(f"Generated in {elapsed:.1f}s | {word_count} words")
        if elapsed > 8:
            st.info("This took longer than the 8-second target (see build-plan.md Block 7 for tuning tips).")

        st.download_button("Download summary", output_text, file_name="summary.txt")

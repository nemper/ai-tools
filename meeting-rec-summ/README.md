# Meeting Recording Summarizer

Archived Streamlit tool for summarizing meeting transcripts, transcribing audio/video, and describing images with OpenAI models.

## What It Does

- Summarizes uploaded `.txt`, `.pdf`, and `.docx` meeting transcripts in Serbian.
- Supports `Kratak` and `Dugacak` summary modes.
- Provides a temperature slider for summary generation.
- Falls back to a LangChain `map_reduce` summarization chain when the input is longer than 275000 characters. In that fallback mode, long summaries are unavailable.
- Transcribes audio/video files to text with OpenAI Whisper (`whisper-1`), converting to MP3 and chunking longer files before transcript correction.
- Describes images from either a local JPG upload or an image URL using the configured vision-capable OpenAI chat model.

## Dependencies

This app depends on:

- OpenAI API access.
- Python packages listed in `requirements.txt`.

It is **standalone** — the helpers it used to import from the private `myfunc`
package (`myfunc.asistenti`, `myfunc.mojafunkcija`, `myfunc.varvars_dicts`) are
now re-implemented locally in **`app_utils.py`**. `Dunja_Zapisnik.py` is kept as
a thin reference that re-exports the `asistenti` helpers from `app_utils.py`.

> Note: the summary download helper (`sacuvaj_dokument`) exports `.txt` and
> `.docx`; the original PDF export relied on an external `wkhtmltopdf` binary and
> was dropped to keep the tool self-contained.

## Environment

Create environment variables before running:

- `OPENAI_API_KEY`: OpenAI API key used by transcription, summarization, and image description.
- `DEPLOYMENT_ENVIRONMENT`: optional; when set to `Streamlit`, the app uses the configured login flow.

See `.env.example` for a minimal template.

## Run

From this directory:

```bash
streamlit run Openai_Zapisnik.py
```

This tool is archived and not intended for deployment.

# Analysis & archival cleanup — gdrive2pinecone

Date: 2026-06-28. Reviewer: Claude (Opus 4.8).

## What the tool does

A staged CLI pipeline (`step1` → `step1b` → `step2` → `step2b` → `step3`, plus
`export_sorted.py`) that downloads Google Drive documents, extracts per-page text
(with OCR fallback), attaches source URLs, and upserts OpenAI embeddings into
Pinecone. See `README.md` for the full step-by-step table.

## State of the code

The Python is actually in good shape: clear function decomposition, docstrings,
defensive error handling, parallel download/processing, and the Pinecone usage is
already on the modern v3+ client (`from pinecone import Pinecone`). No functional
bugs were found, so **the scripts were left unchanged**.

## Problems found (non-code) & changes made

1. **Empty README.** It contained only `# Add client json`. Replaced with a real
   README documenting the pipeline, the run order, requirements (incl. the
   external Tesseract dependency), and all configuration knobs.
2. **No `.gitignore` — secret-leak risk.** `step1.py` loads a Google
   service-account JSON key (`denty.json`); there was nothing stopping it (or a
   `.env`, or the downloaded corpus) from being committed. Added a `.gitignore`
   that excludes `*.json` keys, `.env`, and all pipeline intermediates/outputs.
3. **No environment documentation.** Added `.env.example` listing
   `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_HOST`, `NAMESPACE`, and a note
   that Drive auth is a JSON key file rather than an env var.
4. **Misleading CI workflow removed.** `.github/workflows/main_gdrive-2-azure.yml`
   deployed this repo as an **Azure Web App** (zipping a `.streamlit/` folder),
   but this tool is a set of batch scripts with no web server — the workflow
   could never produce a working deployment and referenced a stale Azure publish
   profile secret. Removed it.

## Left as-is (deliberately)

- Project-specific placeholders (`FOLDER_IDS = ['...']`, `CREDENTIALS_FILE`,
  `sorting_id`/`date_filter` in `export_sorted.py`, the hardcoded `date`
  `20240930` in `step2.py`) are intentional per-deployment values, now documented
  in the README rather than guessed at.

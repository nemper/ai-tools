# gdrive2pinecone

A small, script-based pipeline that pulls documents out of one or more **Google
Drive** folders, turns them into clean per-page text, and **upserts** the
resulting chunks (with OpenAI embeddings) into a **Pinecone** index. It also
includes a helper to export records back out of Pinecone, sorted, to a text
file.

> Archived tool. The steps are plain CLI scripts meant to be run in order; each
> one reads/writes intermediate files on disk (`gdrive_files/`, a CSV, and
> `gdrive_jsons/`). Configuration lives in the `CONFIG` constants at the top of
> each script and in environment variables — edit those before running.

## Pipeline

| Step | Script | What it does | Reads | Writes |
|------|--------|--------------|-------|--------|
| 1  | `step1.py`    | Recursively list & download every file from the configured Drive folders (parallel) and record name→URL pairs | Drive API | `gdrive_files/`, `gdrive_names_and_urls.csv` |
| 1b | `step1b.py`   | Drop redundant `_en` files when a same-named `_sr`/`_hr` version exists (file + CSV row) | CSV, `gdrive_files/` | updated CSV, pruned files |
| 2  | `step2.py`    | Extract text per PDF page (with OCR fallback via Tesseract), keep only "meaningful" pages | `gdrive_files/` | `gdrive_jsons/*.json` |
| 2b | `step2b.py`   | Fill each JSON page's `url` field from the CSV (matched on `source`) | CSV, `gdrive_jsons/` | updated JSONs |
| 3  | `step3.py`    | Embed each page (`text-embedding-3-large`) and upsert to Pinecone in batches | `gdrive_jsons/` | Pinecone index (+ `err_log.txt` on errors) |
| —  | `export_sorted.py` | Query a namespace by a `date` metadata filter and dump the matching `context` fields, sorted, to a file | Pinecone index | `pinecone_out.txt` |

Typical run:

```bash
python step1.py     # download + CSV
python step1b.py    # optional language de-dup
python step2.py     # PDF -> JSON (OCR fallback)
python step2b.py    # attach URLs to JSON
python step3.py     # embed + upsert to Pinecone
```

## Requirements

- Python 3.12+, dependencies in `requirements.txt` (`pip install -r requirements.txt`).
- **Tesseract OCR** installed and on `PATH` (used by `pytesseract` in `step2.py`
  for scanned/image-only PDF pages).
- A Google **service account** with read access to the target Drive folders; its
  JSON key file referenced by `CREDENTIALS_FILE` in `step1.py`.

## Configuration

Environment variables (see `.env.example`):

- `OPENAI_API_KEY` — embeddings in `step3.py`.
- `PINECONE_API_KEY` — Pinecone auth.
- `PINECONE_HOST` — full index host URL (used by `step3.py` / `export_sorted.py`).
- `NAMESPACE` — Pinecone namespace for `export_sorted.py`.

In-script constants to set before running:

- `step1.py`: `CREDENTIALS_FILE` (service-account JSON), `FOLDER_IDS` (Drive
  folder IDs), `DOWNLOAD_DIR`, `CSV_FILE`.
- `step3.py`: `namespace` (passed to `do_embeddings`).
- `export_sorted.py`: the `sorting_id` metadata key and `date_filter` value.

## Notes

- Do **not** commit the service-account JSON key or any `.env` file.
- This pipeline was extracted from a specific project ("denty"), so some default
  names and IDs are placeholders (`'...'`) — replace them with your own.

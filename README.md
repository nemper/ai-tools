# ai-tools — archived AI tools monorepo

This repository is an **archive** of several older, no-longer-used internal tools.
Each was once its own repository; they were merged into this single monorepo with
their **shared git history preserved**.

**Current status:** these tools are not actively used. They are being kept in the
best possible shape for archival — reviewed and modernized by current LLMs
(Claude + Codex): dependencies updated, dead code and artifacts removed, configs
sanitized, and the formerly-private `myfunc` dependency removed and re-implemented
locally (vendored as `app_utils.py` inside each tool that needed it).

Each tool has its **own README/readme** and an **`ANALYSIS.md`** describing what it
does, problems found, and changes made — start there for details.

## Tools

| Tool | What it is |
|------|------------|
| [`basic-vectorstore-coder`](basic-vectorstore-coder/) | Streamlit + LangChain `RetrievalQA` over a Pinecone index — generates code/answers. |
| [`gdrive2pinecone`](gdrive2pinecone/) | CLI pipeline: Google Drive docs → text/OCR → OpenAI embeddings upserted to Pinecone. |
| [`meeting-rec-summ`](meeting-rec-summ/) | Streamlit meeting-transcript summarizer + Whisper audio/video transcription + image description. |
| [`odoo15-project-bug-tracker`](odoo15-project-bug-tracker/) | Odoo 15 addon (technical name `positive_bugs`) adding a bug tracker to the Project app. |
| [`write-in-style`](write-in-style/) | Streamlit app that writes text in a given style using Pinecone hybrid search + OpenAI. |

## Notes

- Most tools are Streamlit/LangChain apps; each has its own `requirements.txt`
  (and, where a full freeze was preserved, a `requirements-lock.txt`).
- They depend on external services (OpenAI, Pinecone, etc.); see each tool's
  `.env.example` for the expected environment variables. No real secrets are
  committed.

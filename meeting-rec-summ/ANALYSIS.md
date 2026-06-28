# Analysis

## Tool Purpose

`meeting-rec-summ` is an archived Streamlit tool for meeting-related AI workflows:

- Summarizes uploaded `.txt`, `.pdf`, and `.docx` transcripts into Serbian meeting notes.
- Offers `Kratak` and `Dugacak` summary modes with a temperature control.
- Uses a LangChain `map_reduce` fallback for documents over 275000 characters.
- Provides sidebar helpers for OpenAI Whisper transcription of audio/video files with MP3 conversion and chunking.
- Provides sidebar helpers for image description from a local JPG file or image URL.

The main app is `Openai_Zapisnik.py`. It intentionally imports private helpers and configuration from `myfunc.*`.

## Problems Found

- The `ChatOpenAI` construction used by the map-reduce path had `temperature=temp` inside a trailing comment, so the temperature slider was ignored in that path.
- `README.md` documented an unrelated code summarization app instead of this meeting transcript/audio/image tool.
- `config.yaml` contained real employee emails and bcrypt password hashes.
- The project did not include an example environment file.
- `Dunja_Zapisnik.py` had redundant `client = openai` assignments, including one that shadowed a function parameter.

## Changes Made

- Fixed `Openai_Zapisnik.py` so `temperature=temp` is passed as an actual `ChatOpenAI` argument.
- Added small docstrings and spacing cleanup in `Openai_Zapisnik.py`.
- Removed redundant OpenAI client aliases in `Dunja_Zapisnik.py` without changing prompt keys, UI strings, or `myfunc` imports.
- Rewrote `README.md` to accurately describe transcript summarization, Whisper transcription, image description, dependencies, environment variables, and the real Streamlit entry point.
- Sanitized `config.yaml` to placeholder `admin` and `demo` users with fake bcrypt-format hashes and placeholder emails.
- Added `.env.example` with `OPENAI_API_KEY` and `DEPLOYMENT_ENVIRONMENT`.

## Update — removed the private `myfunc` dependency

The tool imported helpers from `myfunc.asistenti`, `myfunc.mojafunkcija`, and
`myfunc.varvars_dicts`, which are no longer available. These are now
re-implemented from scratch in a local, self-contained `app_utils.py`:

- `varvars_dicts`: `work_prompts()` (freshly authored prompt templates matching
  how each key is used) and the `work_vars` dict.
- `mojafunkcija`: `positive_login` (streamlit-authenticator over `config.yaml`),
  `initialize_session_state`, and `sacuvaj_dokument` (now `.txt` + `.docx`; the
  PDF path that needed the external `wkhtmltopdf` binary was dropped).
- `asistenti`: `priprema`, `transkript`, `read_local_image`, `read_url_image`,
  `generate_corrected_transcript`, `delete_mp3_files` — based faithfully on the
  former reference copies.

Other changes:

- `Openai_Zapisnik.py` now imports everything from `app_utils`.
- `Dunja_Zapisnik.py` was reduced to a thin reference that re-exports the
  `asistenti` helpers from `app_utils` (no more ~300 lines of duplication).
- `requirements.txt`: dropped the `myfunc` git line; added the now-direct deps
  `streamlit-authenticator`, `PyYAML`, `Pillow`, `requests`, `python-docx`.
- Added `test_app_utils.py` (a pytest contract test for the prompt keys,
  placeholders and `work_vars` shape; requires the deps to be installed).

### Pre-existing quirk left untouched

The >275000-char `map_reduce` path builds its `PromptTemplate` via
`mprompts["summary_begin"].format(text="text", opis="opis")`, which substitutes
the placeholders before declaring them as `input_variables` — a pre-existing bug
in the app logic, not in the vendored helpers, so it was left as-is.

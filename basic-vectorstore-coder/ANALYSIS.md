# basic-vectorstore-coder analysis

## What this tool does

`Koder.py` is an archived Streamlit app that wraps a LangChain RetrievalQA chain. It embeds user requests with OpenAI embeddings, retrieves context from the Pinecone index `embedings1` in namespace `koder`, and asks a chat model to return code or explanations through the existing Serbian UI.

## Problems found

- The app used the removed Pinecone v2 API (`pinecone.init` and `pinecone.Index`) while `requirements.txt` pinned `pinecone-client==3.0.2`.
- The vector store import came from the deprecated community Pinecone integration instead of `langchain-pinecone`.
- LangSmith tracing was always enabled and still pointed at the old `https://api.langchain.plus` endpoint.
- Missing API keys would fail later with low-level exceptions instead of a clear Streamlit error.
- `requirements.txt` was a full pip freeze rather than a direct dependency list.
- `config.yaml` contained real employee emails and bcrypt password hashes.

## Changes made

- Modernized Pinecone setup to `from pinecone import Pinecone`, `pc = Pinecone(api_key=...)`, and `pc.Index("embedings1")`.
- Replaced `langchain_community.vectorstores.Pinecone` with `langchain_pinecone.PineconeVectorStore`, keeping namespace `koder` and text field `text`.
- Kept `RetrievalQA` for minimal behavior-preserving code, with a comment noting its newer LangChain deprecation.
- Updated LangSmith to `https://api.smith.langchain.com` and only enables tracing when `LANGCHAIN_API_KEY` is present.
- Added an early Streamlit environment-variable check for `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_API_ENV`.
- Added concise module/function docstrings and comments around the Pinecone v3 compatibility choices.
- Moved the original frozen dependency list to `requirements-lock.txt` and slimmed `requirements.txt` to direct dependencies.
- Replaced authentication users and preauthorized emails with placeholder-only values.
- Added `.env.example` with the expected environment variables.

## Update — removed the private `myfunc` dependency

The app imported `st_style`, `positive_login`, `init_cond_llm` and `show_logo`
from the private `myfunc.mojafunkcija` package, which is no longer available.
These are now re-implemented from scratch in a local, self-contained
`app_utils.py`:

- `st_style()` — hides Streamlit's default menu/footer/header chrome.
- `init_cond_llm()` — sidebar model + temperature selectors, returns `(model, temp)`.
- `show_logo()` — optional sidebar logo (env `APP_LOGO_URL`) with a text fallback.
- `positive_login(main, " ")` — streamlit-authenticator login over `config.yaml`
  (lazy import; only needed when `DEPLOYMENT_ENVIRONMENT=Streamlit`).

`Koder.py` now imports from `app_utils`, and the `myfunc` git line was dropped
from `requirements.txt` (the helpers' deps — `streamlit-authenticator`, `pyyaml`
— were already present).


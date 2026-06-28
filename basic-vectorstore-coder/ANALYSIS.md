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

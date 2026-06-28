# Write In Style

Archived Streamlit tool for generating Serbian text in a requested writing style
using OpenAI models with Pinecone-backed context retrieval. The active entry
point for the archived state is:

```bash
streamlit run MultiTool_app.py
```

The private `myfunc` dependency has been removed; the helper functions it used
are vendored locally in `app_utils.py`. The app still needs working OpenAI and
Pinecone credentials. It is kept here as an internal archive of experimental
variants and is not prepared for deployment.

## Files

| File | Role |
| --- | --- |
| `MultiTool_app.py` | Main Streamlit entry point. Uses `custom_llm_agent.our_custom_agent`, hybrid Pinecone retrieval, OpenAI, and export helpers. |
| `custom_llm_agent.py` | Support module for the main app. Defines the custom LangChain agent tools. |
| `multi_ret.py` | Experimental retrieval-router Streamlit script. |
| `Pisi_u_stilu_FT.py` | Older standalone fine-tuned-model style writer. |
| `Pisi_u_stilu_Hybrid.py` | Older standalone hybrid-search style writer. |
| `Pisi_u_stilu_Self.py` | Older standalone self-query retrieval style writer. |
| `Pisi_u_stilu_Test.py` | Older standalone test variant with extra namespace/model controls. |
| `Test_setup.py` | Development multi-tool chatbot experiment. |
| `Test_dva_alata.py` | Development two-tool chatbot experiment for hybrid search and CSV. |
| `csvtest.py` | Small CSV-agent Streamlit experiment. |
| `acs.py` | Small Azure Cognitive Search command-line experiment. |
| `sql.py` | Small SQL-agent Streamlit experiment. |

`PRAVILNIK O ORGANIZACIJI I SISTEMATIZACIJI RADNIH MESTA.txt` is included as a
sample source document for internal testing and should be reviewed before any
public redistribution.

## Configuration

Use `.env.example` as the inventory of environment variables referenced by the
code. The main and variant scripts expect OpenAI and Pinecone credentials; some
experiments also reference Serper/Google search, Azure Cognitive Search, or a
local SQL database.

`config.yaml` contains placeholder Streamlit-authenticator credentials only.
Replace them with real credentials outside the archive workflow.

## Dependencies

`requirements.txt` is intentionally slimmed to direct project dependencies. The
original full pip-freeze has been preserved unchanged in `requirements-lock.txt`
for historical reference.

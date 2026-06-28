"""Archived Streamlit RetrievalQA app backed by a Pinecone vector index.

The app keeps its original UI flow and private ``myfunc`` helpers while using
the Pinecone v3 client and the dedicated LangChain Pinecone integration.
"""

from __future__ import annotations

import os

from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st

from myfunc.mojafunkcija import st_style, positive_login, init_cond_llm, show_logo


REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_API_ENV")
INDEX_NAME = "embedings1"
NAMESPACE = "koder"
TEXT_FIELD = "text"

version = "05.11.23. (Streamlit, Pinecone, LangChain)"


def configure_langsmith() -> None:
    """Enable LangSmith tracing only when credentials are configured."""
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return

    os.environ["LANGCHAIN_PROJECT"] = "Koder"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


def get_required_env() -> dict[str, str]:
    """Return required env vars or stop Streamlit with a clear error."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        st.error(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set them before running Koder."
        )
        st.stop()

    return {name: os.environ[name] for name in REQUIRED_ENV_VARS}


def build_vectorstore(
    embeddings: OpenAIEmbeddings, pinecone_api_key: str
) -> PineconeVectorStore:
    """Connect LangChain to the archived Pinecone index."""
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key=TEXT_FIELD,
        namespace=NAMESPACE,
    )


configure_langsmith()

st.set_page_config(page_title="Koder", page_icon="🖥️", layout="wide")
st_style()


def main():
    """Render the app and answer user requests with Pinecone retrieval."""
    env_vars = get_required_env()
    openai_api_key = env_vars["OPENAI_API_KEY"]
    pinecone_api_key = env_vars["PINECONE_API_KEY"]
    # Pinecone v3 no longer takes an environment in client initialization, but
    # this archived app still validates PINECONE_API_ENV for deployment parity.
    _pinecone_api_env = env_vars["PINECONE_API_ENV"]

    # Create the embeddings model and LangChain vector store wrapper.
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = build_vectorstore(embeddings, pinecone_api_key)
    show_logo()
    # Get user input
    st.markdown(f"<p style='font-size: 10px; color: grey;'>{version}</p>", unsafe_allow_html=True)
    st.subheader("Koristeći LangChain i Streamlit...")
    with st.expander("Pročitajte uputstvo:"):
        st.caption("""
                   Prethodni korak bio je kreiranje pitanja. To smo radili pomocu besplatnog CHATGPT modela. Iz svake oblasti (ili iz dokumenta) zamolimo CHATGPT da kreira relevantna pitanja. Na pitanja mozemo da odgovorimo sami ili se odgovori mogu izvuci iz dokumenta.\n
                   Ukoliko zelite da vam model kreira odgovore, odaberite ulazni fajl sa pitanjma iz prethodnog koraka. Opciono, ako je za odgovore potreban izvor, odaberite i fajl sa izvorom. Unesite sistemsku poruku (opis ponasanja modela) i naziv FT modela. Kliknite na Submit i sacekajte da se obrada zavrsi. Fajl sa odgovorima cete kasnije korisiti za kreiranje FT modela.\n
                   Pre prelaska na sledecu fazu OBAVEZNO pregledajte izlazni dokument sa odgovorima i korigujte ga po potrebi.
                   """)
        st.divider()

    # Initialize ChatOpenAI and the QA chain.
    st.session_state["izlaz"] = ""
    model, temp = init_cond_llm()
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model, temperature=temp)
    # RetrievalQA is deprecated in newer LangChain releases; kept here to
    # preserve the original one-call QA behavior for this archived tool.
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), verbose=False
    )

    # Save the user input in the session state
    placeholder = st.empty()
    st.session_state["task"] = ""

    # Create a form with a text input and a submit button
    with placeholder.form(key="my_form", clear_on_submit=True):
        query = (
            "Using langchain and streamlite, "
            + st.text_area(
                label="Detaljno opišite šta želite da uradim (kod, objašnjenje ili sl): ",
                key="1",
                value=st.session_state["task"],
                help="Npr. Napravi kod koji će da ispiše Hello World!",
            )
            + "."
        )
        submit_button = st.form_submit_button(
            label="Submit", help="Kliknite ovde da pokrenete izvršavanje"
        )

        # If the submit button is clicked, clear the session state and run the query
        if submit_button:
            st.session_state["task"] = ""
            with st.spinner("Sačekajte trenutak..."):
                st.session_state["izlaz"] = qa.run(query)
                st.write(st.session_state["izlaz"])

    if "izlaz" in st.session_state:
        st.download_button(
            "Download as .txt",
            st.session_state["izlaz"],
            file_name="koder.txt",
            help="Kliknite ovde da preuzmete fajl",
        )


# koristi se samo za deployment na streamlit cloudu
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()

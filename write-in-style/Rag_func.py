import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent


def set_namespace():
    # with st.sidebar:
    st.session_state.name_semantic = st.selectbox(
        "Namespace za Semantic Search",
        (
            "koder",
            "positive",
            "pravnikprazan",
            "pravnikprefix",
            "pravnikschema",
            "pravnikfull",
            "bisprazan",
            "bisprefix",
            "bisschema",
            "bisfull",
        ),
        help="Pitanja o Positive uopstena",
    )
    st.session_state.name_self = st.selectbox(
        "Namespace za SelfQuery Search",
        ("sistematizacija3",),
        help="Pitanja o meta poljima",
    )
    st.session_state.name_hybrid = st.selectbox(
        "Namespace za Hybrid Search",
        (
            "pravnikkraciprazan",
            "pravnikkraciprefix",
            "pravnikkracischema",
            "pravnikkracifull",
            "bishybridprazan",
            "bishybridprefix",
            "bishybridschema",
            "bishybridfull",
            "pravnikprazan",
            "pravnikprefix",
            "pravnikschema",
            "pravnikfull",
            "bisprazan",
            "bisprefix",
            "bisschema",
            "bisfull",
        ),
        help="Pitanja o opisu radnih mesta",
    )


def read_csv(upit):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        st.session_state.uploaded_file.name,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    odgovor = agent.run(upit)
    return str(odgovor)


# semantic search - klasini model
def rag(upit):
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_API_ENV"],
    )
    index_name = "embedings1"

    index = pinecone.Index(index_name)
    # vectorstore = Pinecone(
    #     index=index, embedding=OpenAIEmbeddings(), text_key=upit, namespace=namespace
    # )
    text = "text"
    odg = Pinecone(
        index=index,
        embedding=OpenAIEmbeddings(),
        text_key=text,
        namespace=st.session_state.name_semantic,
    ).as_retriever(search_kwargs={"k": st.session_state.broj_k})
    ceo_odgovor = odg.get_relevant_documents(
        upit,
    )
    odgovor = ""
    for member in ceo_odgovor:
        odgovor += member.page_content + "\n\n"

    return odgovor


# selfquery search - pretrazuje po meta poljima
def selfquery(upit):
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_API_ENV"],
    )

    llm = ChatOpenAI(temperature=0)
    # Define metadata fields
    metadata_field_info = [
        AttributeInfo(name="title", description="Tema dokumenta", type="string"),
        AttributeInfo(name="keyword", description="reci za pretragu", type="string"),
        AttributeInfo(
            name="text", description="The Content of the document", type="string"
        ),
        AttributeInfo(
            name="source", description="The Source of the document", type="string"
        ),
    ]

    # Define document content description
    document_content_description = "Sistematizacija radnih mesta"

    index_name = "embedings1"
    text = "text"
    # Izbor stila i teme
    index = pinecone.Index(index_name)
    vector = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=OpenAIEmbeddings(),
        text_key=text,
        namespace=st.session_state.name_self,
    )
    ret = SelfQueryRetriever.from_llm(
        llm,
        vector,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True,
        search_kwargs={"k": st.session_state.broj_k},
    )

    ceo_odgovor = ret.get_relevant_documents(upit)
    odgovor = ""
    for member in ceo_odgovor:
        odgovor += member.page_content + "\n\n"

    return odgovor


# hybrid search - kombinacija semantic i selfquery metoda po kljucnoj reci
def hybrid_query(upit):
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY_POS"],
        environment=os.environ["PINECONE_ENVIRONMENT_POS"],
    )
    # # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    index_name = "bis"

    index = pinecone.Index(index_name)
    bm25_encoder = BM25Encoder().default()

    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
        namespace=st.session_state.name_hybrid,
        top_k=st.session_state.broj_k,
        alpha=0.5,
    )
    ceo_odgovor = vectorstore.get_relevant_documents(upit)
    odgovor = ""
    for member in ceo_odgovor:
        odgovor += member.page_content + "\n\n"

    return odgovor


# pretrqaga iz csv fajla

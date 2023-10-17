import os
import io
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
from myfunc.mojafunkcija import st_style
from mojafunkcija import positive_login


st.set_page_config(page_title="Multi Tool Chatbot", page_icon="👉", layout="wide")

st_style()

version = "17.10.23. - Svi search Agent i memorija"
st.markdown(
    f"<p style='font-size: 10px; color: grey;'>{version}</p>",
    unsafe_allow_html=True,
)


def set_namespace():
    # with st.sidebar:
    st.session_state.name_semantic = st.selectbox(
        "Namespace za Semantic Search",
        (
            "positive",
            "pravnikprazan",
            "pravnikprefix",
            "pravnikschema",
            "pravnikfull",
            "bisprazan",
            "bisprefix",
            "bisschema",
            "bisfull",
            "koder",
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


if "name_semantic" not in st.session_state:
    st.session_state.name_semantic = "positive"
if "name_self" not in st.session_state:
    st.session_state.name_self = "sistematizacija3"
if "name_hybrid" not in st.session_state:
    st.session_state.name_hybrid = "pravnikkraciprazan"
if "broj_k" not in st.session_state:
    st.session_state.broj_k = 3
if "alpha" not in st.session_state:
    st.session_state.alpha = 0.5

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = "x.csv"
if "direct_semantic" not in st.session_state:
    st.session_state.direct_semantic = True
if "direct_hybrid" not in st.session_state:
    st.session_state.direct_hybrid = True
if "direct_self" not in st.session_state:
    st.session_state.direct_self = True
if "direct_csv" not in st.session_state:
    st.session_state.direct_csv = True
if "input_prompt" not in st.session_state:
    st.session_state.input_prompt = True


def main():
    st.subheader("Multi Tool Chatbot - Setup")
    with st.expander("Pročitajte uputstvo 🧜‍♂️"):
        st.caption(
            """
                    Na ovom mestu podesavate parametre sistema za testiranje. Za rad CSV agenta potrebno je da uploadujete csv fajl sa struktuiranim podacima.
                    Za rad ostalih agenata potrebno je da odlucit eda li cete korisiti originalni prompt ili upit koji formira agent. Takodje, odaberite namespace za svaki metod.
                    Izborom izlaza odlucujete da li ce se odgovor vratiti direktno iz alata ili ce se korisiti dodatni LLM za formiranje odgovora.
                    Za hybrid search odredite koeficijent alpha koji odredjuje koliko ce biti zastupljena pretraga po kljucnim recima, a koliko po semantickom znacenju.
                    Mozete odabrati i broj dokumenata koji se vracaju iz indeksa.
                        """
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        set_namespace()
        st.caption("Izbor prompta")
        st.session_state.input_prompt = st.radio(
            "Da li zelite da koristite vas originalni prompt?",
            [True, False],
            help="Ako je odgovor False, onda se koristi upit koji formira Agent",
        )
    with col2:
        st.caption("Da li zelite direktan odgovor?")
        st.session_state.direct_semantic = st.radio(
            "Semantic search",
            [True, False],
            help="Pitanja o Positive uopstena",
        )

        st.session_state.direct_hybrid = st.radio(
            "Hybrid search",
            [True, False],
            help="Pitanja o opisu radnih mesta",
        )

        st.session_state.direct_self = st.radio(
            "Self search",
            [True, False],
            help="Pitanja o meta poljima",
        )

        st.session_state.direct_csv = st.radio(
            "CSV search",
            [True, False],
            help="Pitanja o struktuiranim podacima",
        )

    with col3:
        st.session_state.uploaded_file = st.file_uploader(
            "Choose a CSV file", accept_multiple_files=False, type="csv", key="csv_key"
        )
        if st.session_state.uploaded_file is not None:
            with io.open(st.session_state.uploaded_file.name, "wb") as file:
                file.write(st.session_state.uploaded_file.getbuffer())

        st.session_state.broj_k = st.number_input(
            "Podesi broj dokumenata koje vraca index - radi za sva tri indexa",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key="broj_k_key",
            help="Broj dokumenata koji se vraćaju iz indeksa",
        )
        st.session_state.alpha = st.slider(
            "Podesi odnos izmedju keyword i semantic searcha za hybrid search",
            0.0,
            1.0,
            0.5,
            0.1,
            help="Koeficijent koji određuje koliko će biti zastupljena pretraga po ključnim rečima, a koliko po semantičkom značenju. 0-0.4 pretezno Kljucne reci , 0.5 podjednako, 0.6-1 pretezno semanticko znacenje",
        )


def read_csv(upit):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        st.session_state.uploaded_file.name,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )
    # za prosledjivanje originalnog prompta alatu alternativa je upit
    if st.session_state.input_prompt == True:
        odgovor = agent.run(st.session_state.fix_prompt)
    else:
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

    # verizja sa score-om
    # za prosledjivanje originalnog prompta alatu alternativa je upit
    if st.session_state.input_prompt == True:
        odg = Pinecone(
            index=index,
            embedding=OpenAIEmbeddings(),
            text_key=text,
            namespace=st.session_state.name_semantic,
        ).similarity_search_with_score(
            st.session_state.fix_prompt, k=st.session_state.broj_k
        )
    else:
        odg = Pinecone(
            index=index,
            embedding=OpenAIEmbeddings(),
            text_key=text,
            namespace=st.session_state.name_semantic,
        ).similarity_search_with_score(upit, k=st.session_state.broj_k)

    ceo_odgovor = odg

    # verzija bez score-a
    # odg = Pinecone(
    #    index=index,
    #    embedding=OpenAIEmbeddings(),
    #    text_key=text,
    #    namespace=st.session_state.name_semantic,
    # ).as_retriever(search_kwargs={"k": st.session_state.broj_k})

    # ceo_odgovor = odg.get_relevant_documents(
    #     st.session_state.fix_prompt,
    # )

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
    # za prosledjivanje originalnog prompta alatu alternativa je upit
    if st.session_state.input_prompt == True:
        ceo_odgovor = ret.get_relevant_documents(st.session_state.fix_prompt)
    else:
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
    # za prosledjivanje originalnog prompta alatu alternativa je upit
    if st.session_state.input_prompt == True:
        ceo_odgovor = vectorstore.get_relevant_documents(st.session_state.fix_prompt)
    else:
        ceo_odgovor = vectorstore.get_relevant_documents(upit)

    odgovor = ""
    for member in ceo_odgovor:
        odgovor += member.page_content + "\n\n"

    return odgovor


deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()

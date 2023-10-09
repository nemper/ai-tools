# program za pisanje u stilu neke osobe, uzima stil i temu iz Pinecone indexa

# uvoze se biblioteke
import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import LLMChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from html2docx import html2docx
from myfunc.mojafunkcija import st_style, positive_login, open_file
import markdown
import pdfkit
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

version = "09.10.23. Hybrid"


def main():
    # Retrieving API keys from env
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY_POS"],
        environment=os.environ["PINECONE_ENVIRONMENT_POS"],
    )
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize OpenAI embeddings and LLM and all variables

    if "model" not in st.session_state:
        st.session_state.model = ""
    if "temp" not in st.session_state:
        st.session_state.temp = 1.0
    if "text" not in st.session_state:
        st.session_state.text = "text"
    if "index_name" not in st.session_state:
        st.session_state.index_name = "bis"
    if "namespace" not in st.session_state:
        st.session_state.namespace = "bis01"
    if "odgovor" not in st.session_state:
        st.session_state.odgovor = ""
    if "tematika" not in st.session_state:
        st.session_state.tematika = ""
    if "stil" not in st.session_state:
        st.session_state.stil = ""
    ft_model = "Standard"
    # Izbor stila i teme
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Hybrid Search Test 🏙️")
    with st.expander("Pročitajte uputstvo 🧝"):
        st.caption(
            """
                Hybrid Search se bazira na pretrazi indeksa prema kjucnim recima i prema semantickom znacenju.
                App ce kombinovati odgovre na obe pretrage i predloziti podatke potrebne za odgovor. 
                Odatale ce LLM preuzeti zadatak da odgovori na pitanje. Trenutno nisu podrzani namsepace-ovi.
                   """
        )

    with st.sidebar:
        st.session_state.model = "gpt-3.5-turbo-0613"
        st.session_state.stil = (
            "You are a helpful assistent. You always answer in the Serbian language."
        )

        st.session_state.temp = st.slider(
            "Set temperature (0=strict, 1=creative)", 0.0, 2.0, step=0.1, value=0.0
        )
        st.caption("Temperatura za hybrid search treba de je što bliže 0")

    # define model, vestorstore and retriever
    # vazno ako ne stavimo u session state, jako usporava jer inicijalizacija dugo traje!
    if "index" not in st.session_state:
        st.session_state.index = pinecone.Index(st.session_state.index_name)
    if "bm25_encoder" not in st.session_state:
        st.session_state.bm25_encoder = BM25Encoder().default()
    with st.sidebar:
        st.session_state.namespace = st.selectbox(
            "Odaberite oblast",
            ("bis01",),
        )
    zahtev = ""
    llm = ChatOpenAI(
        model_name=st.session_state.model,
        temperature=st.session_state.temp,
        openai_api_key=openai_api_key,
    )
    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=st.session_state.bm25_encoder,
        index=st.session_state.index,
        namespace=st.session_state.namespace,
    )

    # Prompt template - Loading text from the file
    prompt_file = st.file_uploader(
        "Izaberite početni prompt koji možete editovati ili pišite prompt od početka za definisanje vašeg zahteva",
        key="upload_prompt",
        type="txt",
    )
    prompt_t = ""
    if prompt_file is not None:
        prompt_t = prompt_file.getvalue().decode("utf-8")
    else:
        prompt_t = " "

    # Prompt
    with st.form(key="stilovi", clear_on_submit=False):
        zahtev = st.text_area(
            "Opišite temu, iz oblasti Positive, ili opšte teme. Objasnite i formu željenog teksta: ",
            prompt_t,
            key="prompt_prva",
            height=150,
        )
        submit_button = st.form_submit_button(label="Submit")

    # pocinje obrada, prvo se pronalazi tematika, zatim stil i na kraju se generise odgovor
    if zahtev != " " and zahtev != "":
        with st.spinner("Obrađujem temu..."):
            uk_teme = ""
            st.session_state.tematika = vectorstore.get_relevant_documents(zahtev)
            for document in st.session_state.tematika:
                uk_teme += document.page_content + "\n"

        # Read prompt template from the file
        sve_zajedno = open_file("prompt_FT.txt")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            st.session_state.stil
        )
        system_message = system_message_prompt.format()
        human_message_prompt = HumanMessagePromptTemplate.from_template(sve_zajedno)
        human_message = human_message_prompt.format(
            zahtev=zahtev, uk_teme=uk_teme, ft_model=ft_model
        )
        prompt = ChatPromptTemplate(messages=[system_message, human_message])

        # Create LLM chain with chatbot prompt
        chain = LLMChain(llm=llm, prompt=prompt)

        with st.expander("Model i Prompt", expanded=False):
            st.write(
                f"Korišćen je prompt: {prompt.messages[0].content} ->  {prompt.messages[1].content} - >"
            )
        # Run chain to get chatbot's answer
        with st.spinner("Pišem tekst..."):
            try:
                st.session_state.odgovor = chain.run(prompt=prompt)
            except Exception as e:
                st.warning(
                    f"Nisam u mogućnosti da završim tekst. Ovo je opis greške:\n {e}"
                )

    # Izrada verzija tekstova za fajlove formnata po izboru
    # html to docx
    if st.session_state.odgovor != "":
        with st.expander("FINALNI TEKST", expanded=True):
            st.markdown(st.session_state.odgovor)
        html = markdown.markdown(st.session_state.odgovor)
        buf = html2docx(html, title="Zapisnik")

        options = {
            "encoding": "UTF-8",  # Set the encoding to UTF-8
            "no-outline": None,
            "quiet": "",
        }
        try:
            pdf_data = pdfkit.from_string(html, cover_first=False, options=options)
            st.download_button(
                label="Download TekstuStilu.pdf",
                data=pdf_data,
                file_name="TekstuStilu.pdf",
                mime="application/octet-stream",
            )
        except:
            st.write(
                "Za pdf fajl restartujte app za 5 minuta. Osvezavanje aplikacije je u toku"
            )
        st.download_button(
            "Download TekstuStilu.txt",
            st.session_state.odgovor,
            file_name="TekstuStilu.txt",
        )

        st.download_button(
            label="Download TekstuStilu.docx",
            data=buf.getvalue(),
            file_name="TekstuStilu.docx",
            mime="docx",
        )


# Login
st_style()
# Koristi se samo za deploy na streamlit.io
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()

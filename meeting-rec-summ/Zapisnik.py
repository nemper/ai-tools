# This code does summarization

# Importing necessary modules
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import PromptTemplate
import streamlit as st
import os
from html2docx import html2docx
import markdown
import openai
import pdfkit
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    open_file,
    init_cond_llm,
    greska,
    show_logo,
    def_chunk,
)

# XXX
from expression_chain import get_expression_chain
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from vanilla_chain import get_llm_chain

client = Client()
# XXX

# these are the environment variables that need to be set for LangSmith to work
os.environ["LANGCHAIN_PROJECT"] = "Zapisnik"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ.get("LANGCHAIN_API_KEY")


st.set_page_config(page_title="Zapisnik", page_icon="📜", layout="wide")
st_style()
show_logo()


def main():
    side_zapisnik()
    # Read OpenAI API key from env
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # initial prompt
    prompt_string = open_file("prompt_summarizer.txt")
    prompt_string_pam = open_file("prompt_pam.txt")
    opis = "opis"
    st.subheader("Zapisnik")  # Setting the title for Streamlit application
    with st.expander("Procitajte uputstvo:"):
        st.caption(
            "Po potrebi, sa leve strane,  mozete konvertovati svoj MP3 fajl u tekst za obradu."
        )

        st.caption(
            "U svrhe testiranja mozete birati GPT 4 8K ili GPT 3.5 Turbo 16k modele."
        )
        st.caption(
            "Date su standardne instrukcije koji mozete promeniti po potrebi. Promprove mozete cuvati i uploadovati u .txt formatu"
        )
        st.caption(
            "* dokumenti do velicine 4.000 karaktera ce biti tretirani kao jedan. Dozvoljeni formati su .txt, .docx i .pdf"
        )

    uploaded_file = st.file_uploader(
        "Izaberite tekst za sumarizaciju",
        key="upload_file_sumarizacija",
        type=["txt", "pdf", "docx"],
        help="Odabir dokumenta",
    )

    if "dld" not in st.session_state:
        st.session_state.dld = "Zapisnik"

    # markdown to html
    html = markdown.markdown(st.session_state.dld)
    # html to docx
    buf = html2docx(html, title="Zapisnik")
    # create pdf
    options = {
        "encoding": "UTF-8",  # Set the encoding to UTF-8
        "no-outline": None,
        "quiet": "",
    }

    pdf_data = pdfkit.from_string(html, False, options=options)

    # summarize chosen file
    if uploaded_file is not None:
        model, temp = init_cond_llm()
        # Initializing ChatOpenAI model
        llm = ChatOpenAI(
            model_name=model, temperature=temp, openai_api_key=openai.api_key
        )

        prva_file = st.file_uploader(
            "Izaberite pocetni prompt koji mozete editovati ili pisite prompt od pocetka",
            key="upload_prva",
            type="txt",
            help="Odabir dokumenta",
        )

        if prva_file is not None:
            prva = open_file(prva_file.name)  # Loading text from the file
        else:
            prva = " "
        druga_file = st.file_uploader(
            "Izaberite finalni prompt koji mozete editovati ili pisite prompt od pocetka",
            key="upload_druga",
            type="txt",
            help="Odabir dokumenta",
        )

        if druga_file is not None:
            druga = open_file(druga_file.name)  # Loading text from the file
        else:
            druga = " "

        with open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.getbuffer())

        if ".pdf" in uploaded_file.name:
            loader = UnstructuredPDFLoader(uploaded_file.name, encoding="utf-8")
        else:
            # Creating a file loader object
            loader = UnstructuredFileLoader(uploaded_file.name, encoding="utf-8")

        result = loader.load()  # Loading text from the file
        chunk_size = 5000
        chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )  # Creating a text splitter object
        duzinafajla = len(result[0].page_content)
        # Splitting the loaded text into smaller chunks
        texts = text_splitter.split_documents(result)
        chunkova = len(texts)
        st.success(
            f"Tekst je dugacak {duzinafajla} karaktera i podeljen je u {chunkova} delova."
        )
        if chunkova == 1:
            st.info(
                "Tekst je kratak i bice obradjen u celini koristeci samo drugi prompt"
            )

        # XXX
        out_elements = [
            "Zapisnik -",
            "m=" + model.rsplit("-", 1)[-1],
            "t=" + str(temp),
            "chunk s_o=" + str(chunk_size / 1000) + "k_" + str(chunk_overlap),
        ]
        out_name = " ".join(out_elements)
        # XXX
        with st.form(key="my_form", clear_on_submit=False):
            opis = st.text_area(
                "Unestite instrukcije za pocetnu sumarizaciju (kreiranje vise manjih delova teksta): ",
                prva,
                key="prompt_prva",
                height=150,
                help="Napisite prompt za pocetnu sumarizaciju",
            )

            opis_kraj = st.text_area(
                "Unestite instrukcije za finalnu sumarizaciju (kreiranje finalne verzije teksta): ",
                druga,
                key="prompt_druga",
                height=150,
                help="Napisite prompt za finalnu sumarizaciju",
            )

            PROMPT = PromptTemplate(
                template=prompt_string, input_variables=["text", "opis"]
            )  # Creating a prompt template object
            PROMPT_pam = PromptTemplate(
                template=prompt_string_pam, input_variables=["text", "opis_kraj"]
            )  # Creating a prompt template object
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                with st.spinner("Sacekajte trenutak..."):
                    chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        verbose=True,
                        map_prompt=PROMPT,
                        combine_prompt=PROMPT_pam,
                        token_max=4000,
                    )
                    # Load the summarization chain with verbose mode
                    try:
                        suma = AIMessage(
                            content=chain.run(
                                input_documents=texts, opis=opis, opis_kraj=opis_kraj
                            )
                        )

                        st.session_state.dld = suma.content
                        html = markdown.markdown(st.session_state.dld)
                        buf = html2docx(html, title="Zapisnik")
                        pdf_data = pdfkit.from_string(html, False, options=options)
                    except Exception as e:
                        greska(e)

        if st.session_state.dld != "Zapisnik":
            st.write("Downloadujte vase promptove")
            col4, col5 = st.columns(2)
            with col4:
                st.download_button(
                    "Download prompt 1 as .txt",
                    opis,
                    file_name="prompt1.txt",
                    help="Odabir dokumenta",
                )
            with col5:
                st.download_button(
                    "Download prompt 2 as .txt",
                    opis_kraj,
                    file_name="prompt2.txt",
                    help="Odabir dokumenta",
                )
            st.write("Downloadujte vas zapisnik")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "Download Zapisnik as .txt",
                    st.session_state.dld,
                    file_name=out_name + ".txt",
                    help="Odabir dokumenta",
                )
            with col2:
                st.download_button(
                    label="Download Zapisnik as .pdf",
                    data=pdf_data,
                    file_name=out_name + ".pdf",
                    mime="application/octet-stream",
                    help="Odabir dokumenta",
                )
            with col3:
                st.download_button(
                    label="Download Zapisnik as .docx",
                    data=buf.getvalue(),
                    file_name=out_name + ".docx",
                    mime="docx",
                    help="Odabir dokumenta",
                )

            with st.expander("Summary", True):
                # Generate the summary by running the chain on the input documents and store it in an AIMessage object
                st.write(st.session_state.dld)  # Displaying the summary

        # XXX
        if prompt := st.chat_input(
            placeholder="Unesite sve napomene/komentare koje imate u vezi sa performansama programa."
        ):
            st.chat_message("user", avatar="🥸").write(prompt)
            st.session_state["user_feedback"] = prompt
            st.chat_input(placeholder="Vaš feedback je sačuvan!", disabled=True)
            st.session_state.feedback = None
            st.session_state.feedback_update = None
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Samo sekund!")
                run_collector = RunCollectorCallbackHandler()
                message_placeholder.markdown(
                    "Samo jos ocenite od 1 do 5 dobijene rezultate."
                )

                memory = ConversationBufferMemory(
                    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
                    return_messages=True,
                    memory_key="chat_history",
                )

                chain = get_llm_chain("Hi", memory)

                full_response = chain.invoke(
                    {"input": "Hi."},
                    config=RunnableConfig(
                        callbacks=[run_collector],
                        tags=["Streamlit Chat"],
                    ),
                )["text"]

                message_placeholder.markdown(
                    "Samo jos ocenite od 1 do 5 dobijene rezultate."
                )
                run = run_collector.traced_runs[0]
                run_collector.traced_runs = []
                st.session_state.run_id = run.id
                wait_for_all_tracers()
                client.share_run(run.id)

        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="faces",
                key=f"feedback_{st.session_state.run_id}",
            )
            scores = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}
            if feedback:
                score = scores[feedback["score"]]
                feedback = client.create_feedback(
                    st.session_state.run_id, "user_score", score=score
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback.id),
                    "score": score,
                }

        if st.session_state.get("feedback"):
            feedback = st.session_state.get("feedback")
            feedback_id = feedback["feedback_id"]
            score = feedback["score"]

            st.session_state.feedback_update = {
                "comment": st.session_state["user_feedback"],
                "feedback_id": feedback_id,
            }
            client.update_feedback(feedback_id)
            st.chat_input(placeholder="To je to - hvala puno!", disabled=True)

        if st.session_state.get("feedback_update"):
            feedback_update = st.session_state.get("feedback_update")
            feedback_id = feedback_update.pop("feedback_id")
            client.update_feedback(feedback_id, **feedback_update)
            st.session_state.feedback = None
            st.session_state.feedback_update = None
        # XXX


def fix_names():
    with st.sidebar:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        model, temp = init_cond_llm()
        chat = ChatOpenAI(model=model, temperature=temp)
        template = (
            "You are a helpful assistant that fixes misspelled names in transcript."
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        result_string = ""
        prompt = PromptTemplate(
            template="Please only fix the names of the people mentioned that are misspelled in this text: ### {text} ### The correct names are {ucesnici}. Do not write any comment, just the original text with corrected names. If there are no corrections to be made, just write the original text again ",
            input_variables=["ucesnici", "text"],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        dokum = st.file_uploader(
            "Izaberite .txt",
            key="upload_file_fix_names",
            type=["txt"],
            help="Izaberite .txt fajl koji zelite da obradite",
        )

        if dokum:
            loader = UnstructuredFileLoader(dokum.name, encoding="utf-8")

            data = loader.load()
            chunk_size, chunk_overlap = def_chunk()
            # Split the document into smaller parts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            texts = text_splitter.split_documents(data)
            new_text = []
            for txt in texts:
                new_text.append(txt.page_content)
            with st.form(key="imena"):
                ucesnici = st.text_input(
                    "Unesi imena ucesnika: ",
                    help="Imena ucesnika odvojiti zarezom i razmakom",
                )
                submit = st.form_submit_button(
                    label="Submit", help="Submit dugme pokrece izvrsenje programa"
                )
                if submit:
                    with st.spinner("Obrada teksta u toku..."):
                        # Get a chat completion from the formatted messages
                        for text in new_text:
                            result = chat(
                                chat_prompt.format_prompt(
                                    ucesnici=ucesnici,
                                    text=text,
                                ).to_messages()
                            )
                            result_string += result.content
                            result_string = result_string.replace("\n", " ")

                        with st.expander("Obradjen tekst"):
                            st.write(result_string)
                        with open(f"out_{dokum.name}", "w", encoding="utf-8") as file:
                            file.write(result_string)
                        st.success(
                            f"Texts saved to out_{dokum.name} and are now ready for Embeddings"
                        )


def transkript():
    # Read OpenAI API key from env

    with st.sidebar:  # App start
        st.info("Konvertuje MP3 u TXT")
        audio_file = st.file_uploader(
            "Choose MP3 file MAX SIZE 25Mb",
            type="mp3",
            key="audio_",
            help="Odabir dokumenta",
        )
        # transcript_json= "transcript"
        transcritpt_text = "transcript"
        if audio_file is not None:
            placeholder = st.empty()
            st.session_state["question"] = ""

            with placeholder.form(key="my_jezik", clear_on_submit=False):
                jezik = st.selectbox(
                    "Odaberite jezik izvornog teksta 👉",
                    (
                        "sr",
                        "en",
                        "th",
                        "de",
                        "fr",
                        "hu",
                        "it",
                        "ja",
                        "ko",
                        "pt",
                        "ru",
                        "es",
                        "zh",
                    ),
                    key="jezik",
                    help="Odabir jezika",
                )

                submit_button = st.form_submit_button(label="Submit")

                if submit_button:
                    with st.spinner("Sacekajte trenutak..."):
                        transcript = openai.Audio.transcribe(
                            "whisper-1", audio_file, language=jezik
                        )
                        # transcript_dict = {"text": transcript.text}
                        transcritpt_text = transcript.text
                        with st.expander("Transkript"):
                            # Create an expander in the Streamlit application with label 'Koraci'
                            st.info(transcritpt_text)
                            # Display the intermediate steps inside the expander
            if transcritpt_text is not None:
                st.download_button(
                    "Download transcript",
                    transcritpt_text,
                    file_name="transcript.txt",
                    help="Odabir dokumenta",
                )


def side_zapisnik():
    with st.sidebar:
        izbor_app = st.selectbox(
            "Izaberite pomocnu akciju",
            ("Transkript", "Fix names"),
            help="Odabir akcije za pripremu zapisnika",
        )
        if izbor_app == "Transkript":
            transkript()

        elif izbor_app == "Fix names":
            fix_names()


# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, "16.09.23.")
else:
    if __name__ == "__main__":
        main()

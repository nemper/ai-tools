from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

import streamlit as st
import os
import PyPDF2
import re
import io
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    open_file,)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
# from pydub import AudioSegment

from docx import Document

from myfunc.mojafunkcija import sacuvaj_dokument
from myfunc.asistenti import (audio_izlaz, 
                              priprema, 
                              dugacki_iz_kratkih)
import nltk

from openai import OpenAI

# Setting the title for Streamlit application
st.set_page_config(page_title="Zapisnik", page_icon="👉", layout="wide")
st_style()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"Dugacki Zapisnik"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client=OpenAI()

# The variables `story` and `uputstvo` in the `DocumentAnalyzer` class play distinct roles in the context of processing and analyzing a document:

# ### `story`
# - **Purpose:** This variable accumulates the ongoing analysis of the document. It is used to build the entire narrative or analytical output as the process iterates through the document. Each time the loop runs and generates a new part of the analysis, it is appended to `story`, gradually constructing the complete analysis.
# - **Usage:** Initially, `story` is an empty string. In each iteration of the loop, a part of the analysis (`story_part`) is added to `story`. This incremental approach allows the function to handle and accumulate large amounts of text that might exceed the single response limit of the API.

# ### `uputstvo`
# - **Purpose:** The variable `uputstvo` (Serbian for "instructions") contains the instructions or prompts that guide the AI in analyzing the document. It sets the context and requests for the AI, outlining what is expected from the analysis.
# - **Usage and Changes in Code:**
#     - Initially, `uputstvo` is set to a detailed instruction for starting the analysis, guiding the AI to summarize key points and delve into an in-depth analysis.
#     - As the loop continues (after the first iteration), `uputstvo` changes to prompt the AI to continue its analysis based on what has already been written. This new prompt includes a directive to continue from where the last analysis left off, providing the AI with the context of the `story` so far. This helps in maintaining the continuity and coherence of the analysis, ensuring that the AI's output is a seamless continuation of the previous parts.

# The change in `uputstvo` from initial instructions to a continuation prompt is crucial for managing the flow of the document analysis. It ensures that each new piece of generated content is logically and contextually connected to the existing analysis, creating a coherent and comprehensive final output.


class DocumentAnalyzer:
    """
    A class for analyzing documents using the GPT-4 model from OpenAI.

    Attributes:
        model (str): The model identifier for the OpenAI API.
        temperature (float): Controls randomness in the output generation.
        max_tokens (int): The maximum number of tokens to generate in each request.

    Methods:
        analyze_document(document_text): Analyzes the provided document and returns a detailed analysis.
    """
    
    def __init__(self, model="gpt-4-turbo-preview", temperature=0, max_tokens=1024):
        """
        Initializes the DocumentAnalyzer with the specified model parameters.

        Args:
            model (str): The model identifier for the OpenAI API. Default is "gpt-4-turbo-preview".
            temperature (float): Controls randomness in the output generation. Default is 0.
            max_tokens (int): The maximum number of tokens to generate in each request. Default is 1024.
        """
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_document(self, document_text):
        """
        Analyzes the provided document text, generating a detailed analysis by summarizing key points, themes, 
        and findings, and providing in-depth insights on the implications and context.

        The analysis continues iteratively until all relevant parts of the document are covered, 
        ensuring a comprehensive and self-contained output.

        Args:
            document_text (str): The text of the document to be analyzed.

        Returns:
            str: A detailed analysis of the document.
        """
        
        start=True
        total_prompt_tokens = 0
        total_completion_tokens = 0
        story = ""
        uputstvo = f"""I have a document that I need a detailed analysis of. 
        Please start by summarizing the key points, themes, or findings in each document. 
        After summarizing each document, provide an in-depth analysis focusing on the implications, context, and any critical insights. 
        Begin with the firsttopic in the document and continue until you reach the end of your output limit. 
        I will then instruct you to continue with the analysis, moving on to the next topic of the document or further elaborating on the points already discussed. 
        Ensure that each part of the analysis is comprehensive and self-contained to the extent possible, 
        allowing for a seamless continuation in the follow-up responses. """

        while True:
            if not start:
                uputstvo=f"""Continue the analysis with the next section or provide further details on the last discussed points. 
                    Here is what you wrote so far: {story}"""
        
            completion = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                temperature=0,
                max_tokens=1024,
                stop=None,
                messages=[
                    {"role": "system", "content": f"[Use only the Serbian language.] Here is the document >>> {document_text}."},
                    {"role": "user", "content": uputstvo}
                ]
            )
            start=False
            story_part = completion.choices[0].message.content
            story += story_part + " "
            finish_reason = completion.choices[0].finish_reason
            if finish_reason != "length":
                break
        return story


if "init_prompts" not in st.session_state:
    st.session_state.init_prompts = True
    from myfunc.retrievers import PromptDatabase
    with PromptDatabase() as db:
        prompt_map = db.get_prompts_by_names(["result1", "result2"],["SUM_PAM", "SUM_SUMARIZATOR"])
        st.session_state.result1 = prompt_map.get("result1", "You are helpful assistant that always writes in Sebian.")
        st.session_state.result2 = prompt_map.get("result2", "You are helpful assistant that always writes in Sebian.")


version = "29.12.23."
# this function does summarization of the text 
def main():

    def read_pdf(file):
        pdfReader = PyPDF2.PdfFileReader(file)
        count = pdfReader.numPages
        text = ""
        for i in range(count):
            page = pdfReader.getPage(i)
            text += page.extractText()
        return text

    def read_docx(file):
        doc = Document(file)
        text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    doc_io = ""
    with st.sidebar:
        priprema()
    # Read OpenAI API key from envtekst za
    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    # initial prompt
    opis = "opis"
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Zapisnik OpenAI ✍️")  # Setting the title for Streamlit application
    with st.expander("Pročitajte uputstvo 🧜"):
        st.caption(
            """
                   ### Korisničko Uputstvo za Sažimanje Teksta i Transkribovanje 🧜

Dobrodošli na alat za sažimanje teksta i transkribovanje zvučnih zapisa! Ovaj alat vam pruža mogućnost generisanja sažetaka tekstova sastanaka, kao i transkribovanja zvučnih zapisa. Evo kako možete koristiti ovaj alat:

#### Sažimanje Teksta

1. **Učitavanje Teksta**
   - U prozoru "Izaberite tekst za sumarizaciju" učitajte tekstualni dokument koji želite sažeti. Podržani formati su .txt, .pdf i .docx.

2. **Unos Promptova**
   - Unesite instrukcije za sažimanje u polje "Unesite instrukcije za sumarizaciju". Ovo vam omogućava da precizirate želje za sažimanje.
   - Opciono mozete učitati prethodno sačuvani .txt fajl sa promptovima u opciji "Izaberite prompt koji možete editovati, prihvatite default tekst ili pišite prompt od početka".
 
**Generisanje Sažetka**
   - Mozete odabrati opcije Kratki i Dugacki Summary. Kratki summary kao izlaz daje jednu stranicu A4. 
        - Dugacki summary daje otprilike jednu stranicu A4 po temi, ali traje duze i koristi mnogo vise tokena. Za sada sa Dugacki summary nije moguce kroistiti User prompt. 
   - Pritisnite dugme "Submit" kako biste pokrenuli proces sažimanja. Sažetak će se prikazati u prozoru "Sažetak". Takođe, imate opciju preuzimanja sažetka kao .txt, .docx i .pdf.
   - Ukoliko je dokument duzi od 275000 karaktera, bice primenjen drugi, sporiji nacim rada, zbog trenutog ogranicenja GPT-4 modela na 4000 tokena za izlaz. U ovom slucaju dugacki summary nije dostupan.

#### Transkribovanje Zvučnih Zapisa

1. **Učitavanje Zvučnog Zapisa**
   - U bočnoj traci, kliknite na opciju "Transkribovanje zvučnih zapisa" u padajućem meniju. Učitajte zvučni zapis (.mp3) koji želite transkribovati. \
   Možete poslušati sadržaj fajla po potrebi. **Napomena:** Zvučni zapis ne sme biti veći od 25Mb. 

2. **Odabir Jezika**
   - Izaberite jezik izvornog teksta zvučnog zapisa u padajućem meniju "Odaberite jezik izvornog teksta".

3. **Generisanje Transkripta**
   - Pritisnite dugme "Submit" kako biste pokrenuli proces transkribovanja. Transkript će se prikazati u prozoru "Transkript". Takođe, možete preuzeti transkript kao .txt.

   #### Čitanje slika iz fajla i sa URL-a

1. **Učitavanje slike**
   - U bočnoj traci, kliknite na opciju "Čitanje sa slike iz fajla" ili "Čitanje sa slike sa URL-a" u padajućem meniju. Učitajte sliku (.jpg) koji želite da bude opisana. Prikazaće se preview slike.

2. **Uputstvo**
   - Korigujte uputsvo po potrebi.

3. **Generisanje opisa**
   - Pritisnite dugme "Submit" kako biste pokrenuli proces opisivanja. Opis će se prikazati u prozoru "Opis slike". Takođe, možete preuzeti opis kao .txt.

**Napomena:**
- Za transkribovanje zvučnih zapisa koristi se OpenAI Whisper model. Zvučni zapis mora biti u .MP3 formatu i ne veći od 25Mb.
- Za sažimanje teksta i citanje sa slika koristi se odgovarajući OpenAI GPT-4 model.
- Sve generisane datoteke možete preuzeti pomoću odgovarajućih dugmadi za preuzimanje u bočnoj traci.

Srećno sa korišćenjem alata za sažimanje teksta i transkribovanje! 🚀 
                   """
        )
    uploaded_file = st.file_uploader(
        "Izaberite tekst za sumarizaciju",
        key="upload_file",
        type=["txt", "pdf", "docx"],
        help = "Odabir dokumenta",
    )

    if "dld" not in st.session_state:
        st.session_state.dld = "Zapisnik"

    # summarize chosen file
    if uploaded_file is not None:
        
        # Initializing ChatOpenAI model
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview", temperature=0
        )

        prva_file = st.file_uploader(
            "Izaberite prompt koji možete editovati, prihvatite default tekst ili pišite prompt od početka",
            key="upload_prva",
            type="txt",
            help = "Odabir dokumenta",
        )
        if prva_file is not None:
            prva = prva_file.getvalue().decode("utf-8")  # Loading text from the file
        else:
            prva = """Write a detailed summary. Be sure to describe every topic and the name used in the text. \
Write it as a newspaper article. Write only in Serbian language. Give it a Title and subtitles where appropriate \
and use markdown such is H1, H2, etc."""

        with io.open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.getbuffer())

        if ".pdf" in uploaded_file.name:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(pdf_reader.pages)
            text_content = ""

            for page in range(num_pages):
                page_obj = pdf_reader.pages[page]
                text_content += page_obj.extract_text()
            text_content = text_content.replace("•", "")
            text_content = re.sub(r"(?<=\b\w) (?=\w\b)", "", text_content)
            with io.open("temp.txt", "w", encoding="utf-8") as f:
                f.write(text_content)

            loader = UnstructuredFileLoader("temp.txt", encoding="utf-8")
        else:
            # Creating a file loader object
            loader = UnstructuredFileLoader(file_path=uploaded_file.name, encoding="utf-8")

        result = loader.load()

        out_name = "Zapisnik"

        ye_old_way = False
        if len(result[0].page_content) > 275000:
            ye_old_way = True
            st.warning("Vaš dokument je duži od 275000 karaktera. Koristiće se map reduce document chain (radi sporije, a daje drugačije rezultate) - ovo je temporary rešenje. Za ovu opciju dugacki summary nije dostupan.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=75000, chunk_overlap=5000,)
        texts = text_splitter.split_documents(result)

        with st.form(key="my_form", clear_on_submit=False):
            opis = st.text_area(
                "Unesite instrukcije za sumarizaciju : ",
                prva,
                key="prompt_prva",
                height=150,
                help = "Unos prompta koji opisuje fokus sumiranja, željenu dužinu sažetka, formatiranje, ton, jezik, itd."
            )
            audio_i = st.checkbox("Glasovna naracija")



            koristi_dugacak = st.radio(label="Kakav sazetak:", options=["Kratak", "Dugacak"], horizontal=True, label_visibility="collapsed")

            submit_button = st.form_submit_button(label="Submit")
            
            if submit_button:
                with st.spinner("Sačekajte trenutak..."):
                    if ye_old_way:
                        opis_kraj = opis
                        opis = "Summarize comprehensively the content of the document."
                        chain = load_summarize_chain(
                            llm,
                            chain_type="map_reduce",
                            verbose=True,
                            map_prompt=PromptTemplate(template=st.session_state.result2.format(text="text", opis="opis"), input_variables=["text", "opis"]),
                            combine_prompt=PromptTemplate(template=st.session_state.result1.format(text="text", opis_kraj="opis_kraj"), input_variables=["text", "opis_kraj"]),
                            token_max=4000,)

                        suma = AIMessage(
                            content=chain.invoke(
                                {"input_documents": texts, "opis": opis, "opis_kraj": opis_kraj})["output_text"]
                                ).content
                    elif koristi_dugacak == "Kratak":
                        prompt_template = """ "{additional_variable}"
                        "{text}"
                        SUMMARY:"""
                        prompt = PromptTemplate.from_template(prompt_template)
                        prompt.input_variables = ["text", "additional_variable"] 
                        # Define LLM chain
                        llm_chain = LLMChain(llm=llm, prompt=prompt)
                        # Define StuffDocumentsChain
                        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")       

                        suma = AIMessage(
                            content=stuff_chain.invoke({"input_documents": result, "additional_variable": opis})["output_text"]
                        ).content
                        # st.write(type(suma.content))
                    elif koristi_dugacak == "Dugacak":
                        analyzer = DocumentAnalyzer()
                        suma = analyzer.analyze_document(result)
                    try:
                        st.session_state.dld = suma.contect
                    except:
                        st.session_state.dld = suma
                    
        if st.session_state.dld != "Zapisnik":
            with st.sidebar:
                            
                st.download_button(
                    "Download prompt as .txt", opis, file_name="prompt1.txt", help = "Čuvanje zadatog prompta"
                )
                
                sacuvaj_dokument(st.session_state.dld, out_name)
                
            if audio_i == True:
                            st.write("Glasovna naracija")    
                            audio_izlaz(st.session_state.dld)    
            with st.expander("Sažetak", True):
                # Generate the summary by running the chain on the input documents and store it in an AIMessage object
                st.write(st.session_state.dld)  # Displaying the summary

# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()

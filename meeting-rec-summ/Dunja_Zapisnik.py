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
import openai 
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
                            )
import nltk


# Setting the title for Streamlit application
st.set_page_config(page_title="Zapisnik", page_icon="👉", layout="wide")
st_style()



# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"Dugacki Zapisnik"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


client = openai

version = "29.12.23."
#samo_jednom = False
# download and install punkt only once
#if "samo_jednom" not in st.session_state:
#    st.session_state.samo_jednom = True
#    nltk.download('punkt')
def dugacki_iz_kratkih(uploaded_text, entered_prompt):
    """ Generate a summary of a long text. 
        Parameters: 
            uploaded_text (str): The long text.
            entered_prompt (str): The prompt.
        """    
   
    uploaded_text = uploaded_text[0].page_content

    if uploaded_text is not None:
        all_prompts = {
                 "p_system_0" : (
                    "You are helpful assistant."
                ),
                 "p_user_0" : ( 
                     "[In Serbian, using markdown formatting] At the begining of the text write the Title [formatted as H1] for the whole text, the date (dd.mm.yy), topics that vere discussed in the numbered list. After that list the participants in a numbered list. "
                ),
                "p_system_1": (
                    "You are a helpful assistant that identifies the main topics in a provided text. Please ensure clarity and focus in your identification."
                ),
                "p_user_1": (
                    "Please provide a numerated list of up to 10 main topics described in the text - one topic per line. Avoid including any additional text or commentary."
                ),
                "p_system_2": (
                    "You are a helpful assistant that corrects structural mistakes in a provided text, ensuring the response follows the specified format. Address any deviations from the request."
                ),
                "p_user_2": (
                    "Please check if the previous assistant's response adheres to this request: 'Provide a numerated list of topics - one topic per line, without additional text.' Correct any deviations or structural mistakes. If the response is correct, re-send it as is."
                ),
                "p_system_3": (
                    "You are a helpful assistant that summarizes parts of the provided text related to a specific topic. Ask for clarification if the context or topic is unclear."
                ),
                "p_user_3": (
                    "[In Serbian, using markdown formatting, use H2 as a top level] Please summarize the above text, focusing only on the topic {topic}. Start with a simple title, followed by 2 empty lines before and after the summary. "
                ),
                "p_system_4": (
                    "You are a helpful assistant that creates a conclusion of the provided text. Ensure the conclusion is concise and reflects the main points of the text."
                ),
                "p_user_4": (
                    "[In Serbian, using markdown formatting, use H2 as a top level ] Please create a conclusion of the above text. The conclusion should be succinct and capture the essence of the text."
                )
            }

        

        def get_response(p_system, p_user_ext):
            client = openai
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                temperature=0,
                messages=[
                    {"role": "system", "content": all_prompts[p_system]},
                    {"role": "user", "content": uploaded_text},
                    {"role": "user", "content": p_user_ext}
                ]
            )
            return response.choices[0].message.content.strip()


        response = get_response("p_system_1", all_prompts["p_user_1"])
        
        # ovaj double check je veoma moguce bespotreban, no sto reskirati
        response = get_response("p_system_2", all_prompts["p_user_2"]).split('\n')
        topics = [item for item in response if item != ""]  # just in case - triple check

        # Prvi deo teksta sa naslovom, datumom, temama i ucesnicima
        formatted_pocetak_summary = f"{get_response('p_system_0', all_prompts['p_user_0'])}"
        
        # Start the final summary with the formatted 'pocetak_summary'
        final_summary = formatted_pocetak_summary + "\n\n"
        i = 0
        imax = len(topics)

        for topic in topics:
            summary = get_response("p_system_3", f"{all_prompts['p_user_3'].format(topic=topic)}")
            st.info(f"Summarizing topic: {topic} - {i}/{imax}")
            final_summary += f"{summary}\n\n"
            i += 1

        final_summary += f"{get_response('p_system_4', all_prompts['p_user_4'])}"
        
        return final_summary
    
    else:
        return "Please upload a text file."



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
                            map_prompt=PromptTemplate(template=open_file("prompt_summarizer.txt"), input_variables=["text", "opis"]),
                            combine_prompt=PromptTemplate(template=open_file("prompt_pam.txt"), input_variables=["text", "opis_kraj"]),
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

                        suma = dugacki_iz_kratkih(result, opis)

                        # suma = AIMessage(content=dugacki_iz_kratkih(result, opis))
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

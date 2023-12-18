from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

import streamlit as st
import os
from html2docx import html2docx
import markdown
import pdfkit
import PyPDF2
import re
import io
from openai import OpenAI
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    open_file,)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from pydub import AudioSegment

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

from myfunc.mojafunkcija import (audio_izlaz, 
                                 priprema, 
                                 generate_corrected_transcript, 
                                 dugacki_iz_kratkih)


# Setting the title for Streamlit application
st.set_page_config(page_title="Zapisnik", page_icon="👉", layout="wide")
st_style()
client = OpenAI()
version = "14.12.23. - Opisi"

# this function does summarization of the text 
def main():
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
   - Mozete odabrati opcij eKratki i Dugacki Summary. Kratki summary kao izlaz daje jednu stranicu A4. Dugacki summary daje otprilike jednu stranicu A4 po temi, ali traje duze i koristi mnogo vise tokena. 
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

    # markdown to html
    html = markdown.markdown(st.session_state.dld)
    # html to docx
    buf = html2docx(html, title="Zapisnik")

    options = {
        "encoding": "UTF-8",  # Set the encoding to UTF-8
        "no-outline": None,
        "quiet": "",
    }

    pdf_data = pdfkit.from_string(html, cover_first=False, options=options)

    # summarize chosen file
    if uploaded_file is not None:
        
        # Initializing ChatOpenAI model
        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview", temperature=0
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
                            content=chain.run(
                                input_documents=texts, opis=opis, opis_kraj=opis_kraj))
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
                            content=stuff_chain.run(input_documents=result, additional_variable=opis)
                        )
                    elif koristi_dugacak == "Dugacak":
                        suma = AIMessage(content=dugacki_iz_kratkih(uploaded_file, opis))

                    st.session_state.dld = suma.content
                    html = markdown.markdown(st.session_state.dld)
                    buf = html2docx(html, title="Zapisnik")
                    # Creating a document object
                    doc = Document(io.BytesIO(buf.getvalue()))
                    # Iterate over the paragraphs and set them to justified
                    for paragraph in doc.paragraphs:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    # Creating a byte buffer object
                    doc_io = io.BytesIO()
                    doc.save(doc_io)
                    doc_io.seek(0)  # Rewind the buffer to the beginning


                    pdf_data = pdfkit.from_string(html, False, options=options)

        if st.session_state.dld != "Zapisnik":
            with st.sidebar:
                            
                st.download_button(
                    "Download prompt as .txt", opis, file_name="prompt1.txt", help = "Čuvanje zadatog prompta"
                )
                                       
                st.download_button(
                    "Download Zapisnik as .txt",
                    st.session_state.dld,
                    file_name=out_name + ".txt",
                    help= "Čuvanje sažetka",
                )
            
                st.download_button(
                    label="Download Zapisnik as .docx",
                    data=doc_io,
                    file_name=out_name + ".docx",
                    mime="docx",
                    help= "Čuvanje sažetka",
                )
            
                st.download_button(
                    label="Download Zapisnik as .pdf",
                    data=pdf_data,
                    file_name=out_name + ".pdf",
                    mime="application/octet-stream",
                    help= "Čuvanje sažetka",
                )
            if audio_i == True:
                            st.write("Glasovna naracija")    
                            audio_izlaz(st.session_state.dld)    
            with st.expander("Sažetak", True):
                # Generate the summary by running the chain on the input documents and store it in an AIMessage object
                st.write(st.session_state.dld)  # Displaying the summary


# this function corrects the transcriptmora 3.5 turbo 16 k zbog duzine completition (gpt4 max 4k tokena za sada)
# opcija da se prvo izbroje tokeni pa ili radi segmentacija ili se koristi gpt4 za krace a gpt3.5 turbo za duze
# def generate_corrected_transcript(system_prompt, audio_file, jezik):
        
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo-16k",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": transcribe(audio_file, jezik) # does transcription of the audio file
#             }
#         ]
#     )

#     return response.choices[0].message.content
                


# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()

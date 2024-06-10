import io
import nltk
import os
import PyPDF2
import re
import streamlit as st

from openai import OpenAI

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai.chat_models import ChatOpenAI

from myfunc.asistenti import priprema
from myfunc.mojafunkcija import initialize_session_state, positive_login, sacuvaj_dokument
from myfunc.varvars_dicts import work_prompts, work_vars

mprompts = work_prompts()
client=OpenAI()

default_values = {
    "dld": "Zapisnik",
}
initialize_session_state(default_values)


version = "07.06.24."

# this class does long summarization of the text 
class MeetingTranscriptSummarizer:
    def __init__(self, transcript, temperature, number_of_topics):
        self.transcript = transcript
        self.temperature = temperature
        self.number_of_topics = number_of_topics

    def get_response(self, prompt, text):
        response = client.chat.completions.create(
            model=work_vars["names"]["openai_model"],
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": prompt + "Use only the Serbian Language"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content

    def summarize(self):
        introduction = self.get_response(mprompts["date_participants_summary"], self.transcript)
        topic_identification_prompt = mprompts["topic_list_summary"].format(number_of_topics = self.number_of_topics)
        topics = self.get_response(topic_identification_prompt, self.transcript).split('\n')
        lista_tema=""
        st.success("Identifikovane su teme:")
        for topic in topics:
            lista_tema +=topic + "\n"
        st.success(lista_tema)

        summaries = []
        for topic in topics:
            summary_prompt = mprompts["topic_summary"].format(topic = topic)
            summary = self.get_response(summary_prompt, self.transcript)
            summaries.append(f"## Tema: {topic} \n{summary}")
            st.info(f"Obradjujem temu: {topic}")
        
        conclusion = self.get_response(mprompts["conclusion_summary"], self.transcript)
        full_text = (
            f"## Sastanak koordinacije AI Tima\n\n{introduction}\n\n ## Teme sastanka\n\n" + 
            "\n".join([f"{topic}" for topic in topics]) + "\n\n"
            + "\n\n".join(summaries) 
            + f"\n\n## Zaključak\n\n{conclusion}"
        )
        return full_text
    
#main function
def main():
    if "file_content" not in st.session_state:
        st.session_state.file_content = None
    with st.sidebar:
        priprema()

    # initial prompt
    opis = "opis"
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Zapisnik OpenAI NEW ✍️")  # Setting the title for Streamlit application
    with st.expander("Pročitajte uputstvo 🧜"):
        st.caption(
            """
                   ### Korisničko Uputstvo za Sažimanje Teksta i Transkribovanje 🧜

Dobrodošli na alat za sažimanje teksta i transkribovanje zvučnih zapisa! Ovaj alat vam pruža mogućnost generisanja sažetaka tekstova sastanaka, kao i transkribovanja zvučnih zapisa. Evo kako možete koristiti ovaj alat:

#### Sažimanje Teksta

1. **Učitavanje Teksta**
   - U prozoru "Izaberite tekst za sumarizaciju" učitajte tekstualni dokument koji želite sažeti. Podržani formati su .txt, .pdf i .docx.

**Generisanje Sažetka**
   - Mozete odabrati opcije Kratki i Dugacki Summary. Kratki summary kao izlaz daje jednu stranicu A4. 
   - Mozete odabrati temperaturu. Visa tempratura daje manje detrministicki odgovor.
   - Dugacki summary daje otprilike 2-3 teme po jednoj stranici A4, ali traje duze i koristi mnogo vise tokena. Za dugacki summary mozete odrediti i maksimalni broj glavnih tema. Ostale identifikovane teme bice obradjene pod tackom Razno 
   - Pritisnite dugme "Submit" kako biste pokrenuli proces sažimanja. Sažetak će se prikazati u prozoru "Sažetak". Takođe, imate opciju preuzimanja sažetka kao .txt, .docx i .pdf.
   - Ukoliko je dokument duzi od 275000 karaktera, bice primenjen drugi, sporiji nacim rada, zbog trenutog ogranicenja GPT-4 modela na 4000 tokena za izlaz. U ovom slucaju dugacki summary nije dostupan.

#### Transkribovanje Zvučnih Zapisa

1. **Učitavanje Zvučnog Zapisa**
   - U bočnoj traci, kliknite na opciju "Transkribovanje zvučnih zapisa" u padajućem meniju. Učitajte zvučni zapis (.mp3) koji želite transkribovati. \
   Možete poslušati sadržaj fajla po potrebi. **Napomena:** Zvučni zapis ne sme biti veći od 25Mb. Optimalni zapis je mp3, mono, 16000 Hz, 145-185 kbps, ne duzi od 1 sata

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

    # summarize chosen file
    if uploaded_file is not None:
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
        st.session_state.file_content = result[0].page_content

        out_name = "Zapisnik"
        if st.session_state.file_content:
            ye_old_way = False
            if len(result[0].page_content) > 275000:
                ye_old_way = True
                st.warning("Vaš dokument je duži od 275000 karaktera. Koristiće se map reduce document chain (radi sporije, a daje drugačije rezultate) - ovo je temporary rešenje. Za ovu opciju dugacki summary nije dostupan.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=75000, chunk_overlap=5000,)
            texts = text_splitter.split_documents(result)

            with st.form(key="my_form", clear_on_submit=False):
                opis = mprompts["intro_summary"]
                col1, col2, col3 = st.columns(3)
                with col2:
                    temp = st.slider("Temperatura:", min_value=0.0, max_value=1.0, value=0.0, step=0.1, help="Manja temperatura je precizniji odgovor. Max temperatura modela je 2, ali nije omogucena u ovom slucaju")
                with col3:
                    broj_tema= st.number_input("Broj glavnih tema za duzi sazetak max:", min_value=1, max_value=10, value=5, step=1, help="Max broj glavnih tema. Model moze odabrati i manji broj tema, a ostale ce biti obradjene pod tackom Razno")
                with col1:    
                    koristi_dugacak = st.radio(label="Obim sazetka:", options=["Kratak", "Dugacak"], help='Kratki sazetrak je oko jedne strane A4. Dugacki sazetak zavisi od broja tema, otprilike 2-3 teme po stranici A4')

                submit_button = st.form_submit_button(label="Submit")
                
                if submit_button:
                    # Initializing ChatOpenAI model
                    llm = ChatOpenAI(
                        model_name=work_vars["names"]["openai_model"], temperature=temp
                        )

                    st.info(f"Temperatura je {temp}")
                    with st.spinner("Sačekajte trenutak..."):
                        if ye_old_way:
                            opis_kraj = opis
                            opis = "Summarize comprehensively the content of the document."
                            chain = load_summarize_chain(
                                llm,
                                chain_type="map_reduce",
                                verbose=True,
                                map_prompt=PromptTemplate(template=mprompts["summary_begin"].format(text="text", opis="opis"), input_variables=["text", "opis"]),
                                combine_prompt=PromptTemplate(template=mprompts["summary_end"].format(text="text", opis_kraj="opis_kraj"), input_variables=["text", "opis_kraj"]),
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
                            llm_chain = LLMChain(llm=llm, prompt=prompt)

                            # Define StuffDocumentsChain
                            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")       

                            suma = AIMessage(
                                content=stuff_chain.invoke({"input_documents": result, "additional_variable": opis})["output_text"]
                            ).content
                        
                        elif koristi_dugacak == "Dugacak":
                            ulaz= st.session_state.file_content
                            summarizer = MeetingTranscriptSummarizer(
                                transcript=ulaz, 
                                temperature=temp, 
                                number_of_topics=broj_tema
                                )
                            
                            suma = summarizer.summarize()
                        st.session_state.dld = suma

                        
                        directory = os.getcwd()
                        for filename in os.listdir(directory):
                            if filename.endswith('.txt') and filename not in ['requirements.txt', "prompt1.txt", out_name]:
                                file_path = os.path.join(directory, filename)
                                os.remove(file_path)
                            elif filename.endswith('.docx') or filename.endswith('.pdf'):
                                file_path = os.path.join(directory, filename)
                                os.remove(file_path)
                    
            if st.session_state.dld != "Zapisnik":
                with st.sidebar:
                                
                    st.download_button(
                        "Download prompt as .txt", opis, file_name="prompt1.txt", help = "Čuvanje zadatog prompta"
                    )
                    
                    sacuvaj_dokument(st.session_state.dld, out_name)
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

from langchain_community.document_loaders import UnstructuredFileLoader
import streamlit as st
import os
import PyPDF2
import re
import io
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    )

from myfunc.mojafunkcija import sacuvaj_dokument
from myfunc.asistenti import (audio_izlaz, 
                              priprema, 
                            )
import nltk
from openai import OpenAI

# Setting the title for Streamlit application
st.set_page_config(page_title="Zapisnik", page_icon="👉", layout="wide")
st_style()

temp=0
tokens=0
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"Dugacki Zapisnik"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

client=OpenAI()

############
# The variables `story` and `uputstvo` in the `DocumentAnalyzer` class play distinct roles in the context of processing and analyzing a document:
#
# ### `story`
# - **Purpose:** This variable accumulates the ongoing analysis of the document. It is used to build the entire narrative or analytical output as the process iterates through the document. Each time the loop runs and generates a new part of the analysis, it is appended to `story`, gradually constructing the complete analysis.
# - **Usage:** Initially, `story` is an empty string. In each iteration of the loop, a part of the analysis (`story_part`) is added to `story`. This incremental approach allows the function to handle and accumulate large amounts of text that might exceed the single response limit of the API.
#
# ### `uputstvo`
# - **Purpose:** The variable `uputstvo` (Serbian for "instructions") contains the instructions or prompts that guide the AI in analyzing the document. It sets the context and requests for the AI, outlining what is expected from the analysis.
# - **Usage and Changes in Code:**
#     - Initially, `uputstvo` is set to a detailed instruction for starting the analysis, guiding the AI to summarize key points and delve into an in-depth analysis.
#     - As the loop continues (after the first iteration), `uputstvo` changes to prompt the AI to continue its analysis based on what has already been written. This new prompt includes a directive to continue from where the last analysis left off, providing the AI with the context of the `story` so far. This helps in maintaining the continuity and coherence of the analysis, ensuring that the AI's output is a seamless continuation of the previous parts.
#
# The change in `uputstvo` from initial instructions to a continuation prompt is crucial for managing the flow of the document analysis. It ensures that each new piece of generated content is logically and contextually connected to the existing analysis, creating a coherent and comprehensive final output.
############


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
    
    def __init__(self, model="gpt-4-turbo-preview", temperature=temp, max_tokens=tokens):
        """
        Initializes the DocumentAnalyzer with the specified model parameters.

        Args:
            model (str): The model identifier for the OpenAI API. Default is "gpt-4-turbo-preview".
            temperature (float): Controls randomness in the output generation. Default is 0.
            max_tokens (int): The maximum number of tokens to generate in each request. Default is 4096.
        """
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        

    def analyze_document(self, document_text, prompt, opis_sistem, opis_nastavak):
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
        st.info(f"Temperatura je {self.temperature}")
        st.info(f"Broj tokena je {self.max_tokens}")    
        dolara = 0
        total_prompt = 0
        total_completion = 0
        start=True
        story = ""
        uputstvo = f"""Based on this instructions: >> {prompt} << process this document >>> {document_text} """
        while True:
            if not start:
                uputstvo=f"""{opis_nastavak} {prompt}  
                Here is what you wrote so far: {story}"""
        
            completion = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=None,
                messages=[
                    {"role": "system", "content": opis_sistem},
                    {"role": "user", "content": uputstvo}
                ]
            )
            start=False
            total_prompt +=  completion.usage.prompt_tokens
            total_completion+= completion.usage.completion_tokens
            dolara += (total_prompt/1000*0.01)+(total_completion/1000*0.03)
            story_part = completion.choices[0].message.content
            story += story_part + " "
            finish_reason = completion.choices[0].finish_reason
            st.info(f" Razlog zavrsetka je {completion.choices[0].finish_reason}")
            if finish_reason != "length":
                dolara = round(dolara,3)
                st.info(f"Utroseno je {total_prompt} tokena za prompt i {total_completion} tokena za odgovor. ukupno je utroseno {dolara} dolara")
                break
        return story


version = "22.03.24."


# this function does summarization of the text 
def main():
    with st.sidebar:
        priprema()
    # initial prompt
    opis = "opis"
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Zapisnik OpenAI - DUNJA ✍️")  # Setting the title for Streamlit application
    with st.expander("Pročitajte uputstvo 🧜"):
        st.caption(
            """
                   ### Korisničko Uputstvo za Sažimanje Teksta i Transkribovanje 🧜
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
        
        prva = """I have a document that I need a detailed analysis of. 
        Please start by summarizing the key points, themes, or findings in each document. 
        After summarizing each document, provide an in-depth analysis focusing on the implications, context, and any critical insights. 
        Begin with the firsttopic in the document and continue until you reach the end of your output limit. 
        I will then instruct you to continue with the analysis, moving on to the next topic of the document or further elaborating on the points already discussed. 
        Ensure that each part of the analysis is comprehensive and self-contained to the extent possible, 
        allowing for a seamless continuation in the follow-up responses. """
        
        sistem = "[Use only the Serbian language.] Use markdown and create the Title and Subtitles"
        
        nastavak = "Continue rewriting from where you stopped the last time. Do not repeat yourself." 
        
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
       
        with st.form(key="my_form", clear_on_submit=False):
            opis_sistem = st.text_area(
                "Unesite sistemske instrukcije : ",
                sistem,
                key="prompt_sistem",
                height=150,
                help = "Unos prompta koji opisuje fokus sumiranja, željenu dužinu sažetka, formatiranje, ton, jezik, itd."
            )
            opis = st.text_area(
                "Unesite instrukcije za sumarizaciju : ",
                prva,
                key="prompt_prva",
                height=150,
                help = "Unos prompta koji opisuje fokus sumiranja, željenu dužinu sažetka, formatiranje, ton, jezik, itd."
            )
            opis_nastavak = st.text_area(
                "Unesite instrukcije za nastavak sumarizacije : ",
                nastavak,
                key="prompt_nastavak",
                height=150,
                help = "Unos prompta koji opisuje fokus sumiranja, željenu dužinu sažetka, formatiranje, ton, jezik, itd."
            )
            audio_i = st.checkbox("Glasovna naracija")
            temp=st.slider("Temperatura", min_value = 0.0, max_value = 2.0, value = 0.0, step = 0.1)
            tokens=st.slider("Tokena", min_value = 256, max_value = 4096, value = 1024, step = 128)

            submit_button = st.form_submit_button(label="Submit")
            
            if submit_button:
          
                with st.spinner("Sačekajte trenutak..."):
                    analyzer = DocumentAnalyzer(temperature = temp, max_tokens = tokens)
                    try:
                        suma = analyzer.analyze_document(result, opis, opis_sistem, opis_nastavak)
                        st.session_state.dld = suma
                    except Exception as e:
                        st.error(f"Nisam u mogucnosati da ispunim zahtev {e}")
                            
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

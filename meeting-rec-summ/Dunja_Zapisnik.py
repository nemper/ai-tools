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
        uputstvo = f"""Based on this instructions: >> 
                        {prompt} << process this document >>> 
                        {document_text} """
        while True:
            if not start:
                uputstvo=f"""{opis_nastavak} Based on this instructions: >> 
                            {prompt} << process this document >>> 
                            {document_text}  << Here is what you wrote so far: >> 
                            {story}"""
        
            completion = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=1,
                stop=None,
                messages=[
                    {"role": "system", "content": opis_sistem},
                    {"role": "user", "content": uputstvo}
                ]
            )
            start=False
            total_prompt +=  completion.usage.prompt_tokens
            total_completion+= completion.usage.completion_tokens
            dolara_za_sada = round((total_prompt/1000*0.01)+(total_completion/1000*0.03),3)
            dolara += (total_prompt/1000*0.01)+(total_completion/1000*0.03)
            story_part = completion.choices[0].message.content
            st.success(f"Prompt this part: ")
            st.write(uputstvo)
            st.success(f"Story this part: ")
            st.write(story_part)
            st.info(f"Tokens this part:{completion.usage.prompt_tokens} tokena za prompt i {completion.usage.completion_tokens} tokena za odgovor. ukupno je utroseno {dolara_za_sada} dolara")
            story += story_part 
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
        
        prva = """To perform a detailed analysis of your document, follow a structured and systematic approach as outlined below. This will ensure a thorough examination of the content, capturing all pertinent information and insights. Here’s how to proceed:

1. **"Datum"**: Begin by identifying the date specified in the document. This is crucial for contextualizing the information. Format this date in the day-month-year (dd.mm.yyyy) format. If the document does not clearly state the date, record it as 00.00.0000. This notation acknowledges that the date is either unspecified or not a focal point in the document. The correct identification of the date will set the temporal context for the analysis, providing a timeline for the events or information presented.

2. **"Akteri i spomenute persone"**: Compile a comprehensive list of all individuals, organizations, and entities mentioned within the document. Number each entry to keep track of the various participants and mentioned persons. This list will serve as a reference point for understanding the key players involved and their roles or significance within the context of the document. It is important to capture every name mentioned to ensure a full understanding of the interactions, relationships, or dynamics presented.

3. **"Ključne tačke"**: Identify and list the key points, themes, or findings detailed in the document. Present these in a summarised form, akin to headlines or topic titles, which you will later elaborate on. This section should act as a scaffold for the detailed analysis, highlighting the primary subjects and arguments that the document addresses. Each key point serves as a gateway to deeper exploration and understanding of the document's content.

4. **"Izjave učesnika"**: Carefully extract and quote the relevant statements made by the participants in relation to each theme or key point. These quotes are vital for supporting the analysis, providing direct insights or perspectives from the involved parties. The accuracy and relevance of these quotations are paramount, as they will be used to bolster the interpretation and understanding of the document's themes.

5. **"Detaljna analiza"**: Delve into a detailed examination of each previously listed key point or theme. Discuss each in a separate, well-structured paragraph. This section is the core of your analysis, where each topic is to be explored comprehensively. Leave no stone unturned; every aspect of the topic should be covered, ensuring a complete and thorough understanding. This in-depth analysis requires a critical approach, assessing the implications, nuances, and subtleties of each point.

6. **"Zaključak"**: Conclude the analysis with a concise yet comprehensive summary that encapsulates the main findings, themes, and insights derived from the document. This conclusion should tie together all the threads of analysis, providing a coherent and integrated overview of the document's content and significance.

7. **"Dalji koraci i preporuke"**: Finally, identify any further steps and recommendations that emerge from the analysis. This section should go beyond the content already discussed, proposing actions, considerations, or strategies based on the insights gained. Provide a deep analysis of these steps and recommendations, focusing on their implications, the context surrounding them, and any critical insights that they offer.

Ensure that each segment of the analysis, from the initial identification of the date and participants to the detailed exploration of key points and the concluding recommendations, is comprehensive and self-contained. This approach allows for a seamless continuation of the analysis, whether in further discussions, subsequent analyses, or practical applications of the findings.
"""
        
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

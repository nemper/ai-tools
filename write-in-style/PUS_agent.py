import os
import io
import sys
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import Tool, AgentType, initialize_agent

import streamlit as st
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    StreamHandler,
    StreamlitRedirect,
    init_cond_llm,
    open_file,
)
from Rag_func import selfquery, hybrid_query, rag, read_csv, set_namespace

version = "16.10.23. - Svi search Agent i memorija"

st.set_page_config(page_title="Multi Tool Chatbot", page_icon="👉", layout="wide")


def new_chat():
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.memory.clear()
    st.session_state["messages"] = []


def main():
    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
    )

    st.subheader(
        """
                 AI Asistent testira rad AI asistenta za BIS i Pravnik
                 """
    )

    with st.expander("Pročitajte uputstvo 🧜‍♂️"):
        st.caption(
            """
                   Testiramo rad BIS i Pravnik sa upotrebom agenta
                    """
        )

    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "cot" not in st.session_state:
        st.session_state["cot"] = ""
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    search = GoogleSerperAPIWrapper()

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

    with st.sidebar:
        st.button("New Chat", on_click=new_chat)
        model, temp = init_cond_llm()

    col1, col2, col3 = st.columns(3)
    with col1:
        set_namespace()
    with col2:
        st.caption("Da li zelite direktan odgovor?")
        direct_semantic = st.radio(
            "Semantic search",
            [True, False],
            key="direct_semantic",
            help="Pitanja o Positive uopstena",
        )

        direct_hybrid = st.radio(
            "Hybrid search",
            [True, False],
            key="direct_hybrid",
            help="Pitanja o opisu radnih mesta",
        )

        direct_self = st.radio(
            "Self search",
            [True, False],
            key="direct_self",
            help="Pitanja o meta poljima",
        )

        direct_csv = st.radio(
            "CSV search",
            [True, False],
            key="direct_csv",
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
            "Set number of returned documents - radi za sva tri indexa",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key="broj_k_key",
            help="Broj dokumenata koji se vraćaju iz indeksa",
        )
        st.session_state.alpha = st.slider(
            "Set alpha - radi samo za hybrid search",
            0.0,
            1.0,
            0.5,
            0.1,
            help="Koeficijent koji određuje koliko će biti zastupljena pretraga po ključnim rečima, a koliko po semantičkom značenju. 0-0.4 pretezno Kljucne reci , 0.5 podjednako, 0.6-1 pretezno semanticko znacenje",
        )

    st.session_state.tools = [
        Tool(
            name="search",
            func=search.run,
            description="Google search tool. Useful when you need to answer questions about recent events or if someone asks for the current time or date.",
        ),
        Tool(
            name="Semantic search",
            func=rag,
            verbose=False,
            description="Useful for when you are asked about topics including Positive doo and their portfolio. Input should contain Positive.",
            return_direct=direct_semantic,
        ),
        Tool(
            name="Hybrid search",
            func=hybrid_query,
            verbose=False,
            description="Useful for when you are asked about topics that will list items about opis radnih mesta.",
            return_direct=direct_hybrid,
        ),
        Tool(
            name="Self search",
            func=selfquery,
            verbose=False,
            description="Useful for when you are asked about topics that will look for keyword.",
            return_direct=direct_self,
        ),
        Tool(
            name="CSV search",
            func=read_csv,
            verbose=True,
            description="Useful for when you are asked about structured data like numbers, counts or sums",
            return_direct=direct_csv,
        ),
    ]

    download_str = []
    if "open_api_key" not in st.session_state:
        # Retrieving API keys from env
        st.session_state.open_api_key = os.environ.get("OPENAI_API_KEY")
    # Read OpenAI API key from env

    if "SERPER_API_KEY" not in st.session_state:
        # Retrieving API keys from env
        st.session_state.SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=4
        )
    if "sistem" not in st.session_state:
        st.session_state.sistem = open_file("prompt_turbo.txt")
    if "odgovor" not in st.session_state:
        st.session_state.odgovor = open_file("odgovor_turbo.txt")
    if "system_message_prompt" not in st.session_state:
        st.session_state.system_message_prompt = (
            SystemMessagePromptTemplate.from_template(st.session_state.sistem)
        )
    if "human_message_prompt" not in st.session_state:
        st.session_state.human_message_prompt = (
            HumanMessagePromptTemplate.from_template("{text}")
        )
    if "chat_prompt" not in st.session_state:
        st.session_state.chat_prompt = ChatPromptTemplate.from_messages(
            [
                st.session_state.system_message_prompt,
                st.session_state.human_message_prompt,
            ]
        )

    name = st.session_state.get("name")

    placeholder = st.empty()

    pholder = st.empty()
    with pholder.container():
        if "stream_handler" not in st.session_state:
            st.session_state.stream_handler = StreamHandler(pholder)
    st.session_state.stream_handler.reset_text()

    chat = ChatOpenAI(
        openai_api_key=st.session_state.open_api_key,
        temperature=temp,
        model=model,
        streaming=True,
        callbacks=[st.session_state.stream_handler],
    )
    upit = []

    if upit := st.chat_input("Postavite pitanje"):
        formatted_prompt = st.session_state.chat_prompt.format_prompt(
            text=upit
        ).to_messages()
        # prompt[0] je system message, prompt[1] je tekuce pitanje
        pitanje = formatted_prompt[0].content + formatted_prompt[1].content

        with placeholder.container():
            st_redirect = StreamlitRedirect()
            sys.stdout = st_redirect

            fix_prompt = formatted_prompt[1].content
            agent_chain = initialize_agent(
                tools=st.session_state.tools,
                llm=chat,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                messages=st.session_state.chat_prompt,
                verbose=True,
                memory=st.session_state.memory,
                handle_parsing_errors=True,
                prompt=fix_prompt,
                max_iterations=4,
            )
            output = agent_chain.invoke(input=pitanje)
            output_text = output.get("output", "")

            #            output_text = chat.predict(pitanje)
            st.session_state.stream_handler.clear_text()
            st.session_state.past.append(f"{name}: {upit}")
            st.session_state.generated.append(f"AI Asistent: {output_text}")
            # Calculate the length of the list
            num_messages = len(st.session_state["generated"])

            # Loop through the range in reverse order
            for i in range(num_messages - 1, -1, -1):
                # Get the index for the reversed order
                reversed_index = num_messages - i - 1
                # Display the messages in the reversed order
                st.info(st.session_state["past"][reversed_index], icon="🤔")

                st.success(st.session_state["generated"][reversed_index], icon="👩‍🎓")

                # Append the messages to the download_str in the reversed order
                download_str.append(st.session_state["past"][reversed_index])
                download_str.append(st.session_state["generated"][reversed_index])
            download_str = "\n".join(download_str)

            with st.sidebar:
                st.download_button("Download", download_str)


st_style()
# Koristi se samo za deploy na streamlit.io
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()

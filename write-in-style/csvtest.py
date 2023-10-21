import streamlit as st
from langchain.agents import AgentType
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from myfunc.mojafunkcija import init_cond_llm

st.subheader("Testiranje modela na osnovu csv fajla")
st.caption("Ver. 21.10.23")
st.divider()
model, temp = init_cond_llm()
with st.form("my_form"):
    upit = st.text_area("Sistem: ", value="Pisi iskljucivo na srpskom jeziku. ")
    posalji = st.form_submit_button("Posalji")
if posalji:
    agent = create_csv_agent(
        ChatOpenAI(temperature=temp, model=model),
        "pravilnik.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    odgovor = agent.run(upit)
    st.write(odgovor)

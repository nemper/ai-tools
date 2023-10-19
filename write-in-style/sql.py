from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import streamlit as st

db = SQLDatabase.from_uri(f"mysql+pymysql://root:Present1!@localhost:3310/sys")
llm = ChatOpenAI(temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

st.subheader("Upit u SQL bazu")
pitanje = st.text_input("Unesi upit u SQL bazu")
if pitanje:
    odgovor = agent_executor.run(pitanje)
    st.write(odgovor)

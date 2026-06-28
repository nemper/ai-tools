"""Small experimental Streamlit SQL-agent utility.

Tests a LangChain SQL agent against a local database; it is not the main
style-writing app. Run locally with: streamlit run sql.py
"""

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os

llm = ChatOpenAI(temperature=0)

# Connection string comes from the DATABASE_URL env var; the default is a
# non-secret placeholder (never hardcode real credentials).
db = SQLDatabase.from_uri(
    os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost:3306/dbname")
)

# cini se da je ovo najbolje resenje za sada. deluje da chat modeli kao sto je klasican turbo ne rade sa ovim alatom.
# to je sasvim moguce i za CSV tool. Treba i tamo u samom toolu definisati OpenAI a ne chatOpenAI, slicno ovome
toolkit = SQLDatabaseToolkit(
    db=db, llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
)


# ovde moze Chat model, ali treba dodati i handle_parsing_errors=True
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

st.subheader("Upit u SQL bazu")
st.caption("Ver. 24.10.23")
pitanje = st.text_input("Unesi upit u SQL bazu")
if pitanje:
    pitanje = (
        "Show only top 5 results for the query. If you can not find the answer, say I don.t know. When using LIKE allways add N in fornt of '%  "
        + pitanje
    )

    odgovor = agent_executor.run(pitanje)
    st.write(odgovor)

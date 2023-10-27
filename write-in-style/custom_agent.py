from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish, OutputParserException

from typing import List, Union
from re import search, DOTALL

search = SerpAPIWrapper()

import openai
import pinecone
from pinecone_text.sparse import BM25Encoder
import streamlit as st
import os

from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI


# citanje csv fajla i pretraga po njemu
def read_csv(upit):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        st.session_state.uploaded_file.name,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )
    # za prosledjivanje originalnog prompta alatu alternativa je upit
    if st.session_state.input_prompt == True:
        odgovor = agent.run(st.session_state.fix_prompt)
    else:
        odgovor = agent.run(upit)
    return str(odgovor)


# hybrid search - kombinacija semantic i selfquery metoda po kljucnoj reci
def hybrid_query(upit):
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY_POS"],
        environment=os.environ["PINECONE_ENVIRONMENT_POS"],
    )
    # # Initialize OpenAI embeddings
    # embeddings = OpenAIEmbeddings()
    index_name = "bis"
    index = pinecone.Index(index_name)
    # za prosledjivanje originalnog prompta alatu alternativa je upit
    if st.session_state.input_prompt == True:
        ceo_odgovor = st.session_state.fix_prompt
    else:
        ceo_odgovor = upit
    odgovor = ""

    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)["data"][0][
            "embedding"
        ]

    def hybrid_score_norm(dense, sparse, alpha: float):
        """Hybrid score using a convex combination

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: scale between 0 and 1
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hs = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }
        return [v * alpha for v in dense], hs

    def hybrid_query(question, top_k, alpha):
        bm25 = BM25Encoder().default()
        sparse_vector = bm25.encode_queries(question)
        dense_vector = get_embedding(question)
        hdense, hsparse = hybrid_score_norm(
            dense_vector, sparse_vector, alpha=st.session_state.alpha
        )

        result = index.query(
            top_k=top_k,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
            namespace=st.session_state.name_hybrid,
        )
        # return search results as dict
        return result.to_dict()

    # st.session_state.tematika = vectorstore.get_relevant_documents(zahtev)
    st.session_state.tematika = hybrid_query(
        ceo_odgovor, top_k=st.session_state.broj_k, alpha=st.session_state.alpha
    )
    for ind, item in enumerate(st.session_state.tematika["matches"]):
        if item["score"] > st.session_state.score:
            st.info(f'Za odgovor broj {ind + 1} score je {item["score"]}')
            odgovor += item["metadata"]["context"] + "\n\n"
    return odgovor


tools = [
        Tool(
            name="Web search",
            func=search.run,
            description="""
            This tool uses Google Search to find the most relevant and up-to-date information on the web. \
            It accepts a query string as input and returns a list of search results, including web pages, news articles, images, and more. \
            This tool is particularly useful when you need comprehensive information on a specific topic (that isn't related to the company Positive doo), \
            want to explore different viewpoints, or are looking for the latest news.
            Please note that the quality and relevance of results may depend on the specificity of your query. The input must not include the word 'Positive'.
            """,
        ),
        Tool(
            name="Hybrid search",
            func=hybrid_query,
            verbose=True,
            description="""
            This tool combines the capabilities of Pinecone's semantic and keyword search to provide a comprehensive search solution. \
            It uses machine learning models for semantic understanding and also matches specific keywords in the database. \
            This tool is ideal when you need the flexibility of semantic understanding and the precision of keyword matching. 
            The input must include the word 'Positive', i.e. it should be about the company Positive doo.
            """,
            return_direct=st.session_state.direct_hybrid,
        ),
        Tool(
            name="CSV search",
            func=read_csv,
            verbose=True,
            description="""
            This tool should be use when you are asked about structured data, e.g: numbers, counts or sums. 
            The input must include the word 'Positive', i.e. it should be about the company Positive doo.
            """,
            return_direct=st.session_state.direct_csv,
        ),
	]


template = """
Answer the following questions using one of the provided tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, needs to be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation cycle can repeat up to 3 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""


"""
CustomPromptTemplate includes additional information in the prompt, such as the agent's intermediate steps and a list of available tools. 
The intermediate_steps are extracted from the kwargs argument and formatted into a string.
"""
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        kwargs["agent_scratchpad"] = thoughts

        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # `agent_scratchpad`, `tools`, and `tool_names` variables are omitted because they are generated dynamically - `intermediate_steps` is included because it's needed.
    input_variables=["input", "intermediate_steps"]
)


"""
CustomOutputParser extracts the action and action input from the LLM's output. 
If the output contains “Final Answer:”, it returns an AgentFinish object with the final answer. 
Otherwise, it uses a regular expression to extract the action and action input, and returns an AgentAction object.
"""
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        match = search(r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", llm_output, DOTALL)

        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")

        return AgentAction(tool=match.group(1).strip(), tool_input=match.group(2).strip(" ").strip('"'), log=llm_output)


llm_chain = LLMChain(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), prompt=prompt)
tool_names = [tool.name for tool in tools]
output_parser = CustomOutputParser()

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent_executor.run("Who won Wimbledon in 2023?")

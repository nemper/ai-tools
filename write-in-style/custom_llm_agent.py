"""
                   _|¯|_            
                ¯\__(ツ)__/¯   
                    [  ]
                     []
                    / /
                  _/ /_           
"""

def our_custom_agent(question: str, session_state: dict):
    from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        StringPromptTemplate,
        )
    from langchain.schema import AgentAction, AgentFinish, OutputParserException
    from langchain.utilities import GoogleSerperAPIWrapper

    from os import environ
    from re import search, DOTALL
    from typing import List, Union

    from openai import Embedding
    import pinecone
    from pinecone_text.sparse import BM25Encoder
    from myfunc.mojafunkcija import open_file


    environ.get("OPENAI_API_KEY")

    # Tool #1 Web search
    web_search = GoogleSerperAPIWrapper(environment=environ["SERPER_API_KEY"])

    # Tool #2 Pinecone Hybrid search
    def hybrid_search_process(upit):
        pinecone.init(
            api_key=environ["PINECONE_API_KEY_POS"],
            environment=environ["PINECONE_ENVIRONMENT_POS"],
            )
        index = pinecone.Index("bis")

        alpha=session_state["alpha"]
        def hybrid_query():
            get_embedding = lambda text, model="text-embedding-ada-002": Embedding.create(
                input=[text.replace("\n", " ")], 
                model=model,
                )["data"][0]["embedding"]

            # alpha * dense + (1 - alpha) * sparse
            hybrid_score_norm = lambda dense, sparse, alpha: (
                [v * alpha for v in dense], 
                {"indices": sparse["indices"], 
                "values": [v * (1 - alpha) for v in sparse["values"]],
                }
                ) if 0 <= alpha <= 1 else ValueError("Alpha must be between 0 and 1")

            hdense, hsparse = hybrid_score_norm(
                dense=get_embedding(upit), 
                sparse=BM25Encoder().default().encode_queries(upit), 
                alpha=alpha
                )
            
            return index.query(top_k=session_state["broj_k"], 
                            vector=hdense,
                            sparse_vector=hsparse,
                            include_metadata=True,
                            namespace=session_state["namespace"],
                            ).to_dict()

        session_state["tematika"] = hybrid_query()
        
        uk_teme = ""
        for _, item in enumerate(session_state["tematika"]["matches"]):
            if item["score"] > session_state["score"]:
                uk_teme += item["metadata"]["context"] + "\n\n"


        system_message = SystemMessagePromptTemplate.from_template(
            template=session_state["stil"]
            ).format()
        
        human_message = HumanMessagePromptTemplate.from_template(
            template=open_file("prompt_FT.txt")
            ).format(zahtev=question, 
                    uk_teme=uk_teme, 
                    ft_model=session_state["model"],
                    )
        
        return ChatPromptTemplate(messages=[system_message, human_message])


    tools = [
        Tool(
            name="Web search",
            func=web_search.run,
            description="Google search tool. Use this when you need to answer questions about recent events or if someone asks for the current time or date.",
            ),
        Tool(
            name="Pinecone Hybrid search",
            func=hybrid_search_process,
            verbose=True,
            description="Use this when you are asked about topics including Positive doo and their portfolio. Question must include 'radno mesto'.",
            ),
        ]


    template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:
    {tools}

    Only answer questions using the tools above. If you can't use a tool to answer a question, say "I don't know".
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat up to 3 times, if necessary)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

    Question: {input}
    {agent_scratchpad}"""


    class CustomPromptTemplate(StringPromptTemplate):
        template: str
        tools: List[Tool]

        def format(self, **kwargs) -> str:
            intermediate_steps = kwargs.pop("intermediate_steps")   # Get the intermediate steps (AgentAction, Observation tuples)

            kwargs["agent_scratchpad"] = "".join([f"{action.log}\nObservation: {observation}\nThought: " for action, observation in intermediate_steps])
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

            return self.template.format(**kwargs)


    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is a dictionary with a single `output` key
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                    )
            match = search(
                pattern=r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", 
                string=llm_output, 
                flags=DOTALL,
                )
            if not match:
                raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
            
            # action, action input and logs
            return AgentAction(
                tool=match.group(1).strip(), 
                tool_input=match.group(2).strip(" ").strip('"'), 
                log=llm_output,
                )


    llm_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4", verbose=True), 
        prompt=CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "intermediate_steps"],
            )
        )
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=CustomOutputParser(),
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools],
        )
    
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True).run(question)

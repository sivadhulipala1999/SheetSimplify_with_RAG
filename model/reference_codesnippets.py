"""File with a bunch a code snippets I found online. Not a good way to organize such content. To be refactored"""


# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents.agent_types import AgentType
# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import AgentExecutor

# pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
# db = SQLDatabase.from_uri(pg_uri)

# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
# )


# agent_executor = create_sql_agent(
#     llm=llm,
#     db=db,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )

# agent_executor.run(
#     "what is the id of host spencer ?"
# )


# import os
# from langchain import PromptTemplate, HuggingFaceHub, LLMChain, OpenAI, SQLDatabase, HuggingFacePipeline
# from langchain.agents import create_csv_agent
# from langchain.chains.sql_database.base import SQLDatabaseChain
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
# import transformers

# model_id = 'google/flan-t5-xxl'
# config = AutoConfig.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config)
# pipe = pipeline('text2text-generation',
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_length=1024
#                 )
# local_llm = HuggingFacePipeline(pipeline=pipe)

# agent = create_csv_agent(llm=local_llm, path="dummy_data.csv", verbose=True)
# agent.run('how many unique status are there?')

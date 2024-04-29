from transformers import pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate

import torch
import pandas as pd


"""The file contains all the functions which would be good for reference"""


def prep_local_llm_chain(model_name, prompt, tokenizer, data):
    """Add the prompt to the Local LLM via a chain to make it work for our case"""

    # Configure quantization in the LLM to make it more efficient
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    #
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config)

    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    agent_executor = create_pandas_dataframe_agent(
        reader_llm, data, agent_type="tool-calling", verbose=True
    )

    return agent_executor


def prep_prompt():
    """Build a standard chat prompt with RAG as the idea -- did not work for my case"""
    messages = [
        ("system", "Answer the query using the context provided. Be succinct. Your name is The Data Master"),
        ("human", "Hello! I come to you for understanding the data."),
        ("ai", "Hey! All good. Just ask me the exact questions and I will be glad to help."),
        ("human", "{user_input}")
    ]
    chat_template = ChatPromptTemplate.from_messages(messages)
    messages = chat_template.format_messages(
        user_input="What are the columns in the data?")
    return messages

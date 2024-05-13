"""Prepares the RAG LLM for a chat with the user"""

from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import json
import os


def prep_llm_chain(repo_id, prompt, retriever):
    """Prepare the LLM Agent which answers based on the data given and follows the provided prompt"""
    llm = HuggingFaceHub(
        repo_id=repo_id, huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_kwargs={"temperature": 0.5, "max_length": 1024})

    agent_executor = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return agent_executor


def chat(llm):
    """Initiates a chat with the LLM via the command. Run this script directly via command line for this."""
    print("> Chat with the Sheet_Simplify. Please enter your queries here. Press 'quit' to stop the chat")
    while True:
        user_input = input("> ")
        if user_input == "quit":
            break
        else:
            print("Sheet Simplify: ", llm.invoke(
                user_input).split("assistant")[1])


def prep_rag_prompt():
    """Prepare the prompt to make the LLM a RAG based model that answers queries like a chatbot"""

    # Note that Mistral does not take system prompts directly and hence a bit of formatting is needed
    # Different models have different prompt structures which is absolutely painful to navigate through
    # source of this fix: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/41
    sys_prompt = """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer."""

    prompt = """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}"""

    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + sys_prompt + suffix
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    input_text = sys_format + user_format + assistant_format

    prompt_in_chat_format = [
        {
            "role": "user",
            "content": input_text,
        },
    ]

    return PromptTemplate.from_template(input_text)


def prep_vectorstore_csv(filepath, embedding_model_name):
    """Takes the CSV data and loads it into the FAISS vector store using HuggingFace embeddings"""
    loader = CSVLoader(file_path=filepath, encoding="utf-8", csv_args={
        'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},)
    vectorstore = FAISS.from_documents(
        data, embeddings, distance_strategy=DistanceStrategy.COSINE)
    return vectorstore.as_retriever()


def setup(environment_name, user_api_key=None):
    """Setup the LLM Chain for further usage from API or Streamlit app"""
    with open(f"model/{environment_name}/properties.json") as f:
        json_contents = f.read()
    env_data = json.loads(json_contents)

    if user_api_key is None:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = env_data["api_key"]
    else:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = user_api_key
    model_name = env_data["huggingface_model_name"]
    embedding_model_name = env_data["embedding_model_name"]

    # Prep the CSV data
    retriever = prep_vectorstore_csv("data/Train.csv", embedding_model_name)

    # Prepare the prompt for the LLM and chain it to the model
    # tokenizer, prompt_template = prep_rag_prompt(model_name)
    prompt_template = prep_rag_prompt()

    # prep the LLM Agent
    llm_chain = prep_llm_chain(model_name, prompt_template, retriever)

    return llm_chain


def main(environment_name, user_api_key=None, API_CALL=False, APP_INVOKE=False):
    llm_chain = setup(environment_name, user_api_key)
    chat(llm_chain)


if __name__ == "__main__":
    main("dev")

import streamlit as st
from model import llm

st.title("Sheet Simplify")


user_api_key = st.sidebar.text_input('API Key', type='password')


def invoke_llm(user_api_key, text):
    llm_chain = llm.setup("dev", user_api_key)
    st.info(llm_chain.invoke(text).split(
        "assistant")[1].split("<|im_end|>")[0])


with st.form("my_form"):
    text = st.text_area("Enter text:", "Give a summary of the data provided")
    submitted = st.form_submit_button("Submit")
    if (user_api_key is None) or (user_api_key == ""):
        st.warning("Please enter your API key. It is mandatory", icon="⚠")
    else:
        invoke_llm(user_api_key, text)

from langchain_ollama import ChatOllama
import streamlit as st

llm =  ChatOllama(model= "qwen2.5:3b")
st.title("Chat-Ollama")
user_input = st.chat_input("Ask something")

if user_input:
    response= llm.invoke(user_input)
    st.write(response.content)


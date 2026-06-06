from langchain_google_genai import ChatGoogleGenerativeAI   # pip install langchain-google-genai
import streamlit as st

llm = ChatGoogleGenerativeAI(model= "gemini-2.5-flash",
                             google_api_key = "google_api_key")

# response = llm.invoke("What is langchain")
# print(" Response ::  ", response.content)

st.title("AI Assistant")

Question = st.chat_input("Ask Anything")
if Question:

    with st.chat_message("User"):
        st.write(Question)
    response = llm.invoke(Question)
    with st.chat_message("assistant"):
        st.write(response.content)


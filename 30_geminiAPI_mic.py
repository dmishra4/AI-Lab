from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from langchain_ollama import ChatOllama

# # Load Gemini Model
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key="google_api_key"
# )

# Load Gemini Model
llm = ChatOllama(
    model="qwen2.5:3b",
)

st.title("AI Assistant")

# Mic input
voice_text = speech_to_text(
    language="en",
    use_container_width=True,
    just_once=True,
    key="voice_input"
)

# Create conversation history once
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Debug Button
if st.button("Show History"):
    st.write("Conversation History:")
    st.write(st.session_state.conversation_history)

# Display previous messages
for message in st.session_state.conversation_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
question = st.chat_input("Ask Anything")

# Prefer voice if available
if voice_text:
    question = voice_text

if question:

    # Show user question
    with st.chat_message("user"):
        st.write(question)

    # Store user question
    st.session_state.conversation_history.append(
        HumanMessage(content=question)
    )

    # Send COMPLETE history to Gemini
    response = llm.invoke(
        st.session_state.conversation_history
    )

    # Show AI response
    with st.chat_message("assistant"):
        st.write(response.content)

    # Store AI response
    st.session_state.conversation_history.append(
        AIMessage(content=response.content)
    )


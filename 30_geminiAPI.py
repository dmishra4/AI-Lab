from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

# Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="google_api_key"
)

st.title("AI Assistant")

# Create conversation history once
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

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


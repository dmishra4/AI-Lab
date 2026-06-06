from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

# # Load model
# llm = ChatOllama(model="qwen2.5:3b")

# ---------- Load Model (only once) ----------
@st.cache_resource
def load_model():
    return ChatOllama(model="qwen2.5:3b")

llm = load_model()

# Title
st.title("My AI Assistant")

# Session memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Show previous chat messages
for message in st.session_state.conversation_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Input box
user_input = st.chat_input("Type your message")

# When user enters message
if user_input:

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Store user message
    st.session_state.conversation_history.append(
        HumanMessage(content=user_input)
    )

    # # Get AI response
    # response = llm.invoke(
    #     st.session_state.conversation_history
    # )
    # shows history before sending to LLM
    st.write("conversation_history :: ",st.session_state.conversation_history)
    # ✅ CHANGE 1: Use stream() instead of invoke()
    stream = llm.stream(
        st.session_state.conversation_history
    )
    print(type(stream))
    # <generator object>   ← NOT a list, NOT history

    # Words come out one by one like:
    # for chunk in stream:
    #     print(chunk.content)
    #
    #     # bot_reply = stream.content

    # # Show AI response
    # with st.chat_message("assistant"):
    #     st.write(bot_reply)
    # ✅ CHANGE 2: Use write_stream() — displays word by word
    with st.chat_message("assistant"):
        bot_reply = st.write_stream(stream)  # returns full text when done

    # Store AI response
    st.session_state.conversation_history.append(
        AIMessage(content=bot_reply)


    )

 # Clear chat button
if st.button("🔄 Clear Chat"):
        st.session_state.conversation_history = []
        st.rerun()
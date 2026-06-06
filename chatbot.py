from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from datetime import datetime

@st.cache_resource
def load_model():
    return ChatOllama(model="qwen2.5:3b")

llm = load_model()

# ✅ Initialize session states
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []         # list of old chats

if "viewing_old_chat" not in st.session_state:
    st.session_state.viewing_old_chat = None  # which old chat is open

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 💬 Chat History")
    st.markdown("---")

    if len(st.session_state.saved_chats) == 0:
        st.write("No saved chats yet.")
    else:
        # Show each saved chat as a button
        for i, saved in enumerate(st.session_state.saved_chats):
            if st.button(
                f"🗂️ {saved['title']}",
                key=f"chat_{i}",
                use_container_width=True
            ):
                # Open this old chat
                st.session_state.viewing_old_chat = i

    st.markdown("---")

    # Show delete all option
    if len(st.session_state.saved_chats) > 0:
        if st.button("🗑️ Delete All History", use_container_width=True):
            st.session_state.saved_chats = []
            st.session_state.viewing_old_chat = None
            st.rerun()

# ==================== MAIN AREA ====================

# ✅ If user clicked an old chat — show it
if st.session_state.viewing_old_chat is not None:

    index = st.session_state.viewing_old_chat
    old_chat = st.session_state.saved_chats[index]

    st.title(f"📖 {old_chat['title']}")
    st.caption(f"Saved on: {old_chat['time']}")
    st.markdown("---")

    # Show old messages
    for message in old_chat["messages"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    st.markdown("---")

    # Button to go back to current chat
    if st.button("⬅️ Back to Current Chat"):
        st.session_state.viewing_old_chat = None
        st.rerun()

# ✅ Normal current chat view
else:

    st.title("🙏 My AI Assistant")

    # Show current chat messages
    for message in st.session_state.conversation_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Input box
    user_input = st.chat_input("Type your message...")

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.conversation_history.append(
            HumanMessage(content=user_input)
        )

        stream = llm.stream(
            st.session_state.conversation_history
        )

        with st.chat_message("assistant"):
            bot_reply = st.write_stream(stream)

        st.session_state.conversation_history.append(
            AIMessage(content=bot_reply)
        )

    # ✅ Clear Chat button
    if st.button("🔄 Clear Chat"):
        if len(st.session_state.conversation_history) > 0:

            # Save current chat before clearing
            now = datetime.now().strftime("%d %b %Y %I:%M %p")
            first_question = st.session_state.conversation_history[0].content
            title = first_question[:30] + "..." if len(first_question) > 30 else first_question

            st.session_state.saved_chats.append({
                "title": title,        # first question as title
                "time": now,           # date and time
                "messages": st.session_state.conversation_history.copy()
            })

            # Clear current chat
            st.session_state.conversation_history = []
            st.rerun()
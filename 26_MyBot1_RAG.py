from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile
import os

@st.cache_resource
def load_model():
    return ChatOllama(model="qwen2.5:3b")

llm = load_model()

st.title("My AI Assistant")

# ✅ Session memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

# ==================== SIDEBAR — File Upload ====================
with st.sidebar:
    st.markdown("### 📎 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT or DOCX",
        type=["pdf", "txt", "docx"]
    )

    if uploaded_file is not None:

        # ✅ Save uploaded file temporarily on disk
        # document_loaders need a file PATH not file object
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name         # get temp file path

        # ✅ Use correct loader based on file type
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)

        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)

        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)

        # ✅ Load and split document
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        # ✅ Combine all chunks into one text
        st.session_state.document_text = "\n".join(
            [chunk.page_content for chunk in chunks]
        )

        # Clean up temp file
        os.unlink(tmp_path)

        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.caption(f"Total chunks: {len(chunks)}")
        st.caption(f"Total characters: {len(st.session_state.document_text)}")

    # Show preview
    if st.session_state.document_text:
        st.markdown("### 👁️ Preview")
        st.text_area(
            "Document Content",
            st.session_state.document_text[:500] + "...",
            height=200
        )

        if st.button("🗑️ Remove Document"):
            st.session_state.document_text = ""
            st.rerun()

# ==================== MAIN CHAT ====================

for message in st.session_state.conversation_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

user_input = st.chat_input("Type your message...")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    # ✅ Add document context if uploaded
    if st.session_state.document_text:
        full_message = f"""
        Answer based on the document below.

        Document:
        {st.session_state.document_text}

        Question: {user_input}
        """
    else:
        full_message = user_input

    st.session_state.conversation_history.append(
        HumanMessage(content=full_message)
    )

    stream = llm.stream(
        st.session_state.conversation_history
    )

    with st.chat_message("assistant"):
        bot_reply = st.write_stream(stream)

    st.session_state.conversation_history.append(
        AIMessage(content=bot_reply)
    )

# Clear chat button
if st.button("🔄 Clear Chat"):
    st.session_state.conversation_history = []
    st.rerun()
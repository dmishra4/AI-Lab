from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage

# =========================================================
# 1. LOAD LLM
# =========================================================

llm = ChatOllama(
    model="qwen2.5:3b"
)

# =========================================================
# 2. LOAD DOCUMENT
# =========================================================

loader = TextLoader("sample.txt")
documents = loader.load()

# =========================================================
# 3. SPLIT DOCUMENTS
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = splitter.split_documents(documents)

# =========================================================
# 4. EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# 5. VECTOR DATABASE
# =========================================================

vectordb = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./chroma_db"
)

retriever = vectordb.as_retriever()

# =========================================================
# 6. RAG AGENT
# =========================================================

def rag_agent(query):

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(query)

    # Combine chunks
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Final prompt
    prompt = f"""
    You are a strict QA Assistant. 
    Rules:
    1.Answer ONLY from provided context and ignore unrelated context.
    2. If answer is not present in context,
        say:
        "I do not know based on provided documents"
    3. Do not use your own knowledge.
    

    Context:
    {context}

    Question:
    {query}
    """

    # Ask LLM
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content


# =========================================================
# 7. CHAT LOOP
# =========================================================

print("\n===== RAG AGENT STARTED =====")

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    answer = rag_agent(query)

    print("\n===== ANSWER =====")
    print(answer)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage

# =========================================================
# 1. LOAD LLMs
# =========================================================

# Main LLM
general_llm = ChatOllama(
    model="qwen2.5:3b"
)

# Router LLM
router_llm = ChatOllama(
    model="phi3:mini"
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

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
    You are a strict QA Assistant. 
    Rules:
    1.Answer ONLY from provided context and ignore unrelated context.

    Context:
    {context}

    Question:
    {query}
    """

    response = general_llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content


# =========================================================
# 7. GENERAL AGENT
# =========================================================

def general_agent(query):

    response = general_llm.invoke(
        [HumanMessage(content=query)]
    )

    return response.content


# =========================================================
# 8. ROUTER AGENT
# =========================================================

def router_agent(query):

    routing_prompt = f"""
    You are an intelligent router.

    Decide route:
    - rag -> document/internal/company questions
    - general -> generic questions

    Query:
    {query}

    Return ONLY:
    rag
    or
    general
    """

    response = router_llm.invoke(
        [HumanMessage(content=routing_prompt)]
    )

    return response.content.strip().lower()


# =========================================================
# 9. MAIN SYSTEM
# =========================================================

def ai_system(query):

    route = router_agent(query)

    print(f"\n[ROUTER DECISION]: {route}")

    if "rag" in route:
        return rag_agent(query)

    else:
        return general_agent(query)


# =========================================================
# 10. CHAT LOOP
# =========================================================

print("\n===== MULTI AGENT SYSTEM STARTED =====")

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    answer = ai_system(query)

    print("\n===== ANSWER =====")
    print(answer)
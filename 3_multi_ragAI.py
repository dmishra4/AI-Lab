
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
 # pip install sentence-transformers
 # pip install chromadb


#Load Multiple Documents
loader = DirectoryLoader(
    "logs",
    glob="*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(documents)
print("\n\n")
#Splitt text into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)
for text in texts:
    print(text)
    print("---------------")
#create embedding

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

for index, text in enumerate(texts):

    print(f"\nChunk {index + 1}:")
    print(text.page_content)

    vector = embeddings.embed_query(
        text.page_content
    )

    print("\nEmbedding Vector (first 10 values):")
    print(vector[:10])

    print("\nVector Length:")
    print(len(vector))

    print("\n---------------------------")

# store vectors in chromeDB

db = Chroma.from_documents(texts,embeddings)

print(db)
print("\n\n")

# load local qwen model

llm = OllamaLLM(model="qwen2.5:3b")

#Ask Question
query = "why did deployment failed"


# create retriver
retriever = db.as_retriever(
    search_kwargs={
        "k": 2,
    }
)

# Retrieve relevant document
docs = retriever.invoke(query)

## Combine retrieved text
#context = "\n".join([doc.page_content for doc in docs])
context = ""

for doc in docs:
    context = context + doc.page_content + "\n"

print("Context for LLM :: ",context)

# Build prompt
prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

# Get answer from Qwen
response = llm.invoke(prompt)

print("\nAnswer:")
print(response)
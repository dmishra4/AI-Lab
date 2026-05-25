from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# Load local Ollama model
llm = ChatOllama(model="qwen2.5:3b")

# Define graph state
class GraphState(TypedDict):
    question: str
    answer: str

# Node function
def chatbot(state: GraphState):
    response = llm.invoke(state["question"])

    return {
        "answer": response.content
    }

# Build graph
builder = StateGraph(GraphState)

builder.add_node("chatbot", chatbot)

builder.set_entry_point("chatbot")

builder.add_edge("chatbot", END)

graph = builder.compile()

# Run graph
result = graph.invoke({
    "question": "Explain LangGraph in simple words"
})

print("\nANSWER:\n")
print(result["answer"])
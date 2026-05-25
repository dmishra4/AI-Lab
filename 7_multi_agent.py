from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# ---------------- MODELS ----------------

# Agent 1 model
phi3_llm = ChatOllama(model="phi3:mini")

# Agent 2 model
qwen_llm = ChatOllama(model="qwen2.5:3b")

# ---------------- SHARED STATE ----------------

class GraphState(TypedDict):
    logs: str
    root_cause: str
    fix: str

# ---------------- AGENT 1 ----------------
# Error Analysis Agent

def error_analysis_agent(state: GraphState):

    print("\n[Agent 1] Analyzing logs...")

    prompt = f"""
    Analyze these logs and identify root cause:

    {state['logs']}
    """

    response = phi3_llm.invoke(prompt)

    return {
        "root_cause": response.content
    }

# ---------------- AGENT 2 ----------------
# Fix Suggestion Agent

def fix_suggestion_agent(state: GraphState):

    print("\n[Agent 2] Suggesting fix...")

    prompt = f"""
    Suggest fix for this root cause:

    {state['root_cause']}
    """

    response = qwen_llm.invoke(prompt)

    return {
        "fix": response.content
    }

# ---------------- BUILD GRAPH ----------------

builder = StateGraph(GraphState)

builder.add_node("error_analysis_agent", error_analysis_agent)
builder.add_node("fix_suggestion_agent", fix_suggestion_agent)

builder.set_entry_point("error_analysis_agent")

builder.add_edge("error_analysis_agent", "fix_suggestion_agent")
builder.add_edge("fix_suggestion_agent", END)

graph = builder.compile()

# ---------------- RUN GRAPH ----------------

result = graph.invoke({

    "logs": """
    TEST FAILED
    NullPointerException at LoginPage.java:45
    Browser session disconnected
    """
})

# ---------------- OUTPUT ----------------

print("\n=========== ROOT CAUSE ===========")
print(result["root_cause"])

print("\n=========== FIX ===========")
print(result["fix"])
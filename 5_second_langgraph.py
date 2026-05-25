from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# Load local Ollama model
llm = ChatOllama(model="qwen2.5:3b")

# ---------------- STATE ----------------

class GraphState(TypedDict):
    logs: str
    errors: str
    root_cause: str
    fix: str

# ---------------- NODE 1 ----------------
# Get Logs

def get_logs(state: GraphState):

    sample_logs = """
    TEST FAILED
    NullPointerException at LoginPage.java:45
    Browser session disconnected
    """

    return {
        "logs": sample_logs
    }

# ---------------- NODE 2 ----------------
# Extract Errors

def extract_errors(state: GraphState):

    logs = state["logs"]

    extracted = []

    for line in logs.splitlines():
        if "FAILED" in line or "Exception" in line:
            extracted.append(line)

    return {
        "errors": "\n".join(extracted)
    }

# ---------------- NODE 3 ----------------
# Analyze Root Cause

def analyze_root_cause(state: GraphState):

    prompt = f"""
    Analyze this error and identify root cause:

    {state["errors"]}
    """

    response = llm.invoke(prompt)

    return {
        "root_cause": response.content
    }

# ---------------- NODE 4 ----------------
# Suggest Fix

def suggest_fix(state: GraphState):

    prompt = f"""
    Suggest fix for this issue:

    {state["root_cause"]}
    """

    response = llm.invoke(prompt)

    return {
        "fix": response.content
    }

# ---------------- BUILD GRAPH ----------------

builder = StateGraph(GraphState)

builder.add_node("get_logs", get_logs)
builder.add_node("extract_errors", extract_errors)
builder.add_node("analyze_root_cause", analyze_root_cause)
builder.add_node("suggest_fix", suggest_fix)

builder.set_entry_point("get_logs")

builder.add_edge("get_logs", "extract_errors")
builder.add_edge("extract_errors", "analyze_root_cause")
builder.add_edge("analyze_root_cause", "suggest_fix")
builder.add_edge("suggest_fix", END)

graph = builder.compile()

# ---------------- RUN GRAPH ----------------

result = graph.invoke({})

# ---------------- OUTPUT ----------------

print("\n=========== ERRORS ===========")
print(result["errors"])

print("\n=========== ROOT CAUSE ===========")
print(result["root_cause"])

print("\n=========== FIX ===========")
print(result["fix"])

###############################
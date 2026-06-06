from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# Tools — no changes needed
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together. Use this when user wants to add numbers."""
    return a + b

@tool
def sub_numbers(a: int, b: int) -> int:
    """Subtract two numbers. Use this when user wants to subtract numbers."""
    return a - b

@tool
def get_temple_timing(temple_name: str) -> str:
    """Get opening timing of a temple."""
    timings = {
        "shirdi": "Darshan opens at 4:00 AM and closes at 11:00 PM",
        "tirupati": "Darshan opens at 3:00 AM and closes at 10:00 PM",
        "default": "Temple opens at 6:00 AM and closes at 9:00 PM"
    }
    return timings.get(temple_name.lower(), timings["default"])

# LLM + Agent — 3 lines
llm = ChatOllama(model="qwen2.5:3b")
tools = [add_numbers, sub_numbers, get_temple_timing]
agent = create_react_agent(llm, tools)

def run_agent(user_question):
    print(f"\nUser: {user_question}")
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_question}]
    })
    print(f"\nFinal Answer: {result['messages'][-1].content}")

# Test
run_agent("What is 25 + 37?")
run_agent("What is 80 - 37?")
run_agent("What are the timings of Shirdi temple?")
run_agent("Who are you?")
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import json

# ✅ STEP 1 — Define a Tool (this is your MCP Server in simple form)
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together. Use this when user wants to add numbers."""
    return a + b

@tool
def sub_numbers (a : int, b : int) -> int:
    """Subtract two numbers together. Use this when user wants to subtract numbers."""
    return a-b

@tool
def get_temple_timing(temple_name: str) -> str:
    """Get opening timing of a temple."""
    timings = {
        "shirdi": "Darshan opens at 4:00 AM and closes at 11:00 PM",
        "tirupati": "Darshan opens at 3:00 AM and closes at 10:00 PM",
        "default": "Temple opens at 6:00 AM and closes at 9:00 PM"
    }
    return timings.get(temple_name.lower(), timings["default"])

# ✅ STEP 2 — Give tools to LLM
llm = ChatOllama(model="qwen2.5:3b")
tools = [add_numbers, get_temple_timing,sub_numbers]
llm_with_tools = llm.bind_tools(tools)   # LLM now knows about tools

# ✅ STEP 3 — Create tool lookup dictionary
tool_map = {
    "add_numbers": add_numbers,
    "get_temple_timing": get_temple_timing,
    "sub_numbers": sub_numbers,
}

def run_agent(user_question):

    print(f"\nUser: {user_question}")

    messages = [HumanMessage(content=user_question)]

    # ✅ STEP 4 — LLM decides which tool to call
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    print(f"LLM thinking: {response.content}")
    print(f"response.tool_calls: {response.tool_calls}")

    # ✅ STEP 5 — Check if LLM wants to use a tool
    if response.tool_calls:

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nLLM decided to use tool: {tool_name}")
            print(f"With arguments: {tool_args}")

            # ✅ STEP 6 — Actually run the tool
            tool_result = tool_map[tool_name].invoke(tool_args)

            print(f"Tool result: {tool_result}")

            # ✅ STEP 7 — Give tool result back to LLM
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # ✅ STEP 8 — LLM gives final answer using tool result
        final_response = llm_with_tools.invoke(messages)
        print(f"\nFinal Answer: {final_response.content}")

    else:
        # LLM answered directly without tool
        print(f"\nFinal Answer: {response.content}")


# ✅ Test it
run_agent("What is 25 + 37?")
run_agent("What is 80 - 37?")
run_agent("What are the timings of Shirdi temple?")
run_agent("Who are you?")   # no tool needed
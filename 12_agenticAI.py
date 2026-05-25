from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
#from langchain.agents import create_
from langchain_ollama import ChatOllama

# =====================================================
# LLM
# =====================================================

llm = ChatOllama(
    model="qwen2.5:3b"
)

# =====================================================
# TOOL 1
# =====================================================

@tool
def calculator(expression: str) -> str:
    """
    Useful for math calculations.
    """

    result = eval(expression)

    return str(result)

# =====================================================
# TOOL 2
# =====================================================

@tool
def log_tool(query: str) -> str:
    """
    ALWAYS use this tool for:
    - Selenium issues
    - Selenium failures
    - timeout errors
    - automation problems
    - browser automation failures
    - test execution failures

    Never answer Selenium questions directly.
    Always call this tool first.
    """

    print("\n[LOG TOOL EXECUTED]")

    return "Selenium failed due to timeout."

# =====================================================
# TOOL LIST
# =====================================================

tools = [
    calculator,
    log_tool
]

# =====================================================
# AGENT
# =====================================================

agent = create_react_agent(
    tools=tools,
    model=llm,

)

# =====================================================
# CHAT LOOP
# =====================================================

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [ ("user", query)]
        }
    )

    print("\nFINAL ANSWER:")
    print(response["messages"][-1].content)
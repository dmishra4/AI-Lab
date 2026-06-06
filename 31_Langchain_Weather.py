from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# STEP 1 — Same Tools (no change needed)
@tool
def get_weather(city: str) -> str:
    """Get the current temperature of a city."""
    weather_data = {
        "pune": 32,
        "mumbai": 34,
        "delhi": 40,
        "bangalore": 28
    }
    temp = weather_data.get(city.lower(), 30)
    return str(temp)

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="google_api_key"
)

# STEP 2 — LLM + Agent (3 lines only!)
#llm = ChatOllama(model="qwen2.5:3b")
tools = [get_weather, add_numbers]
agent = create_react_agent(llm, tools)
# agent = create_react_agent(
#     llm,
#     tools,
#     prompt="You are a helpful assistant. You MUST use the add_numbers tool for any arithmetic. Never calculate in your head."
# )

# STEP 3 — Run
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Find the temperature of Pune and then add 10 to it. Give me the final result."
    }]
})

print(f"Complete Result  ::  {result}")

# ADD HERE ↓
for i, message in enumerate(result['messages']):
    print(f"Step {i} → {type(message).__name__}")

# Tool calls
for message in result['messages']:
    msg_type = type(message).__name__
    if msg_type == "ToolMessage":
        print(f"Tool Called: {message.name} → Result: {message.content}")

# Final Answer
print(f"\nFinal Answer: {result['messages'][-1].content}")



### Outcome

#
# Complete Result  ::  {'messages': [HumanMessage(content='Find the temperature of Pune and then add 10 to it. Give me the final result.', additional_kwargs={}, response_metadata={}, id='7e47e6eb-6cd8-41b6-97e4-b3e448d6add0'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_weather', 'arguments': '{"city": "Pune"}'}, '__gemini_function_call_thought_signatures__': {'a5d3cead-273e-4180-92cb-7b345bb757be': 'CvsCAQw51seCaNq9T2IpRRcIxO5GWmV8AKJJj/vYslu3FAkH//PM1PO9nJ7ZPJbDRmvj9ttOBPXzZsj+QEqYOmS7bC1hDJhSM2G7iMxq/XqeaBCFwGJRie8ELAxUW1s1+ueTqmBePVPDl6FKoQ5ZkEehzzsO06K23Z0xEUrgt7oISrRjs/6v+ayHn50JHs6U+vn6wSkHM2QY2MbQ7u9VtBxalzikhHO/bDy3/z2HWr1Wj67TY0liG1jy7xT+PY/cGvYfPEYXpTHum6zSR/EKBbPw3My8qR5X6fxL7XL85UPV++fezIcg55h333DXe9Sfy/z/z+he72/mOUGa26KiY2Ob7z4CpyeJcme51QigxDhm5RlErMOqgvoP+d4oddvzF63JFM4S1NZxxhr6RU93F3XEZfe8XeIJYh/yZhf5U5wRcH7YUW7OY6eF3dwVDddtXlbZXx8wO0IgRyqR9Mnm8Ip7j8HCvwKj914X8v2+YAfYxV4743kEVRJxpIaUCQ=='}}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019e8dc5-1e21-7b80-81db-de52e92b6ddd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'Pune'}, 'id': 'a5d3cead-273e-4180-92cb-7b345bb757be', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 109, 'output_tokens': 115, 'total_tokens': 224, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 100}}), ToolMessage(content='32', name='get_weather', id='c094d146-e5bd-4ba9-991b-47e3b3fc0321', tool_call_id='a5d3cead-273e-4180-92cb-7b345bb757be'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'add_numbers', 'arguments': '{"a": 32, "b": 10}'}, '__gemini_function_call_thought_signatures__': {'12b6c3ba-134e-46f1-81a8-e3189ffb4f15': 'CosCAQw51seZLvYwwtjt2CIgAlXkEIuPj+saopcQGTpHwDJ7Vh+2j8PR+oNx0X8UCTMr1wM2upjIeF5CxTlZUFMhJ0b3fO/GB6yCBDrkie4obX3z8S8IgsJsu+NazLJ/7eFXm78La3DCQGGyscCug0YlXGUClmlgfSO5Cs9t4rrZMKz0C0XIROuHghuEFIkAp/uUfuRyyUhwYqnX2tDzY89FWeIgM54YwU/Gk7dCI27FyjYa3nqzwHjJlBCpScg/J/Wrs/zJT6tqli58SYKXnwUhbY7iP/yQQ3+FilZgnpAYR0JYw/Sznc6ogOgIZzRIAIacotV4YdDbUSAf0obzgtTefPtCihEU10SdW2a6'}}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019e8dc5-2513-7841-bce9-124bef0e7aa9-0', tool_calls=[{'name': 'add_numbers', 'args': {'a': 32, 'b': 10}, 'id': '12b6c3ba-134e-46f1-81a8-e3189ffb4f15', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 140, 'output_tokens': 102, 'total_tokens': 242, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 80}}), ToolMessage(content='42', name='add_numbers', id='e3592599-5646-4a61-9390-e6bce9900b43', tool_call_id='12b6c3ba-134e-46f1-81a8-e3189ffb4f15'), AIMessage(content=[{'type': 'text', 'text': 'The final result is 42.', 'extras': {'signature': 'CvoBAQw51sfNcd3WhsSRENJtF4ckbCMWW8pISkJZPJD8jK0C9B8kTK7GGIJC+fabu35awNkvgULww2TfHYbxAr42OEYy3XbeEX8RQrq8EBKDGfNroCa7nluBLEnbSDSLalEJd8E6Bd7ZknHHQH4uCoMU8aWEfnvcux7xFT8Qs5V2uwKFR4OebIHi5Y7ZNdV/c8uJefAht+nc/NurVD4taG+02/cUyah2muxdLvDYXwpmiW9L97jqhbSTfT1FuLhSDortwL4oLxSmFFuqQSdxBg5NUocyMpnS+R1zOgXRqviTx3aIlGQPjAD5MBiquCRqCEsDzZAF0bk+Q4Tg/Q=='}}], additional_kwargs={}, response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}, id='lc_run--019e8dc5-2ad9-7793-8bdd-e8aa1fd40225-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 178, 'output_tokens': 77, 'total_tokens': 255, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 69}})]}
# Step 0 → HumanMessage
# Step 1 → AIMessage
# Step 2 → ToolMessage
# Step 3 → AIMessage
# Step 4 → ToolMessage
# Step 5 → AIMessage
# Tool Called: get_weather → Result: 32
# Tool Called: add_numbers → Result: 42
#
# Final Answer: [{'type': 'text', 'text': 'The final result is 42.', 'extras': {'signature': 'CvoBAQw51sfNcd3WhsSRENJtF4ckbCMWW8pISkJZPJD8jK0C9B8kTK7GGIJC+fabu35awNkvgULww2TfHYbxAr42OEYy3XbeEX8RQrq8EBKDGfNroCa7nluBLEnbSDSLalEJd8E6Bd7ZknHHQH4uCoMU8aWEfnvcux7xFT8Qs5V2uwKFR4OebIHi5Y7ZNdV/c8uJefAht+nc/NurVD4taG+02/cUyah2muxdLvDYXwpmiW9L97jqhbSTfT1FuLhSDortwL4oLxSmFFuqQSdxBg5NUocyMpnS+R1zOgXRqviTx3aIlGQPjAD5MBiquCRqCEsDzZAF0bk+Q4Tg/Q=='}}]



#from langchain_community.llms import Ollama      Ollama class is depricated
#
# from langchain_ollama import OllamaLLM
#
#
# llm = OllamaLLM(model ="qwen2.5:3b")
#
# response = llm.invoke("Who are you")
# print(response)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# Load ChatOllama model
llm = ChatOllama(model="qwen2.5:3b")

# This list stores the entire conversation
conversation_history = []
print("Chatbot is ready! Type 'exit' to quit.\n")
while True:
    # Take user input
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to history
    conversation_history.append(HumanMessage(content=user_input))

    # Send full history to model
    response = llm.invoke(conversation_history)

    # Extract reply
    bot_reply = response.content

    # Add bot reply to history
    conversation_history.append(AIMessage(content=bot_reply))

    print(f"\nBot: {bot_reply}\n")

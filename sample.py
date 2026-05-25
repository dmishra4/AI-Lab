from  langchain_community.chat_models import ChatOllama
import os

token = os.getenv("GITHUB_TOKEN")
print("The value of token is :: " , token)

# llm = ChatOllama(
#     model = "qwen2.5:3b"
# )
#
# run = llm.invoke("Capital of India")
# print(run.content)
#
#
# print(ChatOllama(model="qwen2.5:3b").invoke("Write python to send telegram pdf").content)
#
#
#
#
#
#
#
#
# def sub(a : int, b: int) -> int:
#     return a-b
#
# c= sub(5,2)
# print(c)
# print(sub(9,3))
#
# def sum(m: list, n: list):
#     return m+n
#
# print(sum([1,2],[3,4]))

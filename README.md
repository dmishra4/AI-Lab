# AI-Lab

This repository contains my AI, NLP, BERT, LangChain, and Agentic AI learning projects.

git remote add origin https://github.com/dmishra4/AI-Lab.git
git branch -M main
git push -u origin main


(.venv) PS C:\Users\Deepak Kumar\PycharmProjects\FirstPythonProject> 

(.venv) PS C:\Users\Deepak Kumar\PycharmProjects\FirstPythonProject>

#########
cd..
deactivate
 .\.venv\Scripts\activate
Go to settings --> chnage python interpreter too then run the scripts.
#########################
python -m venv rag_env
 rag_env\Scripts\activate


#########


from langchain_ollama import OllamaLLM


llm = OllamaLLM(model ="qwen2.5:3b")
response = llm.invoke("Who are you")
print(response)
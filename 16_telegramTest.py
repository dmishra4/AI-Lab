import requests

BOT_TOKEN = "8754994532:AAH0K5XAatlKH0hLR-sTF_rSRjPJBuuxj8Y"
CHAT_ID = "7905977685"

message = """
AI ALERT

pipelines failed
5 selenium test case failed
Login module issue detected

"""
url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

response = requests.post(
    url,
    data= {
    "chat_id" : CHAT_ID,
     "text" : message
    }
)

print(response.json())

print(response.json()["result"]["text"])



import requests

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Generate PDF
doc = SimpleDocTemplate("AI_Report.pdf")

styles = getSampleStyleSheet()

content = """
Pipeline Failed

5 Selenium test cases failed.
Login module issue detected.
"""

story = []

story.append(
    Paragraph(content, styles['BodyText'])
)

doc.build(story)

print("PDF Generated")

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


# -----------------------------------------
# SEND PDF REPORT
# -----------------------------------------

document_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

with open("AI_Report.pdf", "rb") as file:

    response = requests.post(
        document_url,
        data={
            "chat_id": CHAT_ID
        },
        files={
            "document": file
        }
    )

print(response.json())



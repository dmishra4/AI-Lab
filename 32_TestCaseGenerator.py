import requests
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="google api key",
    temperature=0
)

OWNER = "dmishra4"
REPO = "AI-Lab"

url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"

response = requests.get(url)
response.raise_for_status()

latest_commit = response.json()[0]

latest_sha = latest_commit["sha"]
latest_message = latest_commit["commit"]["message"]

print(f"Latest SHA: {latest_sha}")
print(f"Commit Message: {latest_message}")

# Step 2: Get details of latest commit
details_url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits/{latest_sha}"

response = requests.get(details_url)
response.raise_for_status()

commit_details = response.json()

diff_text = ""

for file in commit_details["files"]:

    filename = file["filename"]

    patch = file.get("patch", "")

    diff_text += f"\nFILE: {filename}\n"
    diff_text += patch
    diff_text += "\n\n"

print(diff_text)
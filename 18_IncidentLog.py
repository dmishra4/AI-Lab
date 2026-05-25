import ollama

# -------------------------
# Read incident logs
# -------------------------
def read_log_file():

    with open("incident.log", "r") as file:
        return file.read()

log_data = read_log_file()

# -------------------------
# Prompt for AI
# -------------------------
prompt = f"""
Analyze this production incident log.

Provide:
1. Root cause
2. Severity
3. Suggested fix
4. Responsible team

Logs:
{log_data}
"""

# -------------------------
# Send to Ollama
# -------------------------
response = ollama.chat(
    model='qwen2.5:3b',
    messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ]
)

# -------------------------
# Print AI Response
# -------------------------
print("\n===== INCIDENT ANALYSIS =====\n")

print(response['message']['content'])
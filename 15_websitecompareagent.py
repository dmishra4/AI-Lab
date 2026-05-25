import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama

# =====================================================
# LLM
# =====================================================

llm = ChatOllama(
    model="qwen2.5:3b"
)

# =====================================================
# TOOL 1 - WEBSITE CONTENT EXTRACTOR
# =====================================================

def extract_visible_text(url):

    print(f"\n[EXTRACTING CONTENT]: {url}")

    headers = {
        "user-Agent" : "Mozilla/5.0"
    }

    response = requests.get(url,
                            headers=headers)

    print("Response is :: ", response.text)

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )
    print("Soup is :: ", soup)
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text(separator="\n")

    # Clean lines
    lines = []

    for line in text.splitlines():

        if line.strip():
            cleaned_line = line.strip()

            lines.append(cleaned_line)

    final_text = "\n".join(lines)

    return final_text

# =====================================================
# TOOL 2 - CONTENT COMPARISON
# =====================================================

def compare_content(uat_text, prod_text):

    uat_lines = set(uat_text.splitlines())
    prod_lines = set(prod_text.splitlines())

    missing_in_uat = prod_lines - uat_lines

    missing_in_prod = uat_lines - prod_lines

    return {
        "missing_in_uat": list(missing_in_uat),
        "missing_in_prod": list(missing_in_prod)
    }

# =====================================================
# TOOL 3 - AI ANALYZER
# =====================================================

def analyze_differences(diff):

    prompt = f"""
    Analyze website content differences.

    Missing in UAT:
    {diff['missing_in_uat']}

    Missing in PROD:
    {diff['missing_in_prod']}

    Summarize:
    - major content gaps
    - possible impact
    - important missing sections
    """

    response = llm.invoke(prompt)

    return response.content

# =====================================================
# MAIN FLOW
# =====================================================

uat_url = input("Enter UAT URL: ")
prod_url = input("Enter PROD URL: ")

# Extract content
uat_text = extract_visible_text(uat_url)

prod_text = extract_visible_text(prod_url)

# Compare
diff = compare_content(
    uat_text,
    prod_text
)

# Show raw differences without using AI
print("\n===== MISSING IN UAT =====")

for item in diff["missing_in_uat"][:20]:
    print("-", item)

print("\n===== MISSING IN PROD =====")

for item in diff["missing_in_prod"][:20]:
    print("-", item)

# AI Summary
print("\n===== AI ANALYSIS =====")

analysis = analyze_differences(diff)

print(analysis)
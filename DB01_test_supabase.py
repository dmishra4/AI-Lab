from supabase import create_client     # pip install supabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="google_api_key"
)

# ==========================
# Supabase Configuration
# ==========================
url = "https://lrkmzuwqjmdjemfxsykf.supabase.co"
key = "Paster key here"

supabase = create_client(url, key)

# ==========================
# Load Embedding Model
# ==========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# Read Incidents
# ==========================
response = supabase.table("incidents").select("*").execute()

print("\nIncidents from Supabase:")
print(response.data)

# ==========================
# Generate & Store Embeddings
# ==========================
for incident in response.data:

    incident_id = incident["id"]
    text = incident["incident_text"]

    print(f"\nProcessing Incident {incident_id}")
    print(f"Text: {text}")

    # Generate embedding
    vector = model.encode(text).tolist()

    print(f"Embedding Length = {len(vector)}")

    # Update Supabase
    supabase.table("incidents").update(
        {
            "embedding": vector
        }
    ).eq("id", incident_id).execute()

    print("Embedding Stored")

print("\nEmbeddings stored successfully.")

# ==========================
# Search Similar Incident
# ==========================

new_incident = "Oracle database connection failure"

print("\nSearching for:", new_incident)

query_vector = model.encode(new_incident).tolist()

result = supabase.rpc(
    "match_incidents",
    {
        "query_embedding": query_vector,
        "match_count": 3
    }
).execute()

print("\nSimilar Incidents Found:")

for row in result.data:
    print(
        f"Incident: {row['incident_number']} | "
        f"Assigned To: {row['assigned_to']} | "
        f"Similarity: {round(row['similarity'], 3)}"
    )

# # Recommendation
# best_match = result.data[0]
#
# print("\n====================")
# print("RECOMMENDATION")
# print("====================")
#
# print(
#         f"Assign Incident To : {best_match['assigned_to']}"
#     )
#
# print(
#         f"Confidence         : "
#         f"{round(best_match['similarity'] * 100, 2)}%"
#     )

response = llm.invoke(
    f"""
    New Incident:
    {new_incident}

    Similar Incidents Retrieved From Vector Database:

    {result.data}

    Based on the above incidents:
    1. Recommend the best engineer
    2. Explain why
    3. Mention confidence level
    """
)

print(response.content)
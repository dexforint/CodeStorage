import requests
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("APERTIS_TOKEN")


headers = {"Authorization": f"Bearer {token}"}

# Step 1: Search for the library
search_response = requests.get(
    "https://context7.com/api/v2/libs/search",
    headers=headers,
    params={"libraryName": "react", "query": "I need to manage state"},
)
libraries = search_response.json()["results"]
# print(libraries.keys(), f"{libraries=}")
best_match = libraries[0]
print(f"Found: {best_match['title']} ({best_match['id']})")

# Step 2: Get documentation context
context_response = requests.get(
    "https://context7.com/api/v2/context",
    headers=headers,
    params={"libraryId": best_match["id"], "query": "How do I use useState?"},
)
print(context_response.status_code)
print(context_response.text)
docs = context_response.json()


# for doc in docs:
#     print(f"Title: {doc['title']}")
#     print(f"Content: {doc['content'][:200]}...")

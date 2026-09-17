import requests

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = '{"action": "deckNames", "version": 6}'

response = requests.post("http://127.0.0.1:8765", headers=headers, data=data)
print(response.text)
print(response.json())

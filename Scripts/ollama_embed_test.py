import requests

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = '{\n    "model": "manutic/nomic-embed-code",\n    "messages": [{"role": "user", "content": "Hello!"}]\n  }'

response = requests.post("http://localhost:11434/api/chat", headers=headers, data=data)
print(response)

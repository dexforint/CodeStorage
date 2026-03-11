import requests

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = '{\n  "model": "deepseek-r1:1.5b",\n  "prompt": "Why is the sky blue?"\n}'

response = requests.post("https://mlvoca.com/api/generate", headers=headers, data=data)
print(response.text)

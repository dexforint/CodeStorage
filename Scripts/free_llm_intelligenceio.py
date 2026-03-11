import requests
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("INTELLIGENCEIO_TOKEN")

url = "https://api.intelligence.io.solutions/api/v1/models?page=1&page_size=50"

headers = {"Authorization": f"Bearer {token}"}

response = requests.get(url, headers=headers)
data = response.json()

for model_info in data["data"]:
    print(model_info["id"])

# moonshotai/Kimi-K2-Thinking
# zai-org/GLM-4.7-Flash
# zai-org/GLM-4.7
# deepseek-ai/DeepSeek-V3.2
# moonshotai/Kimi-K2-Instruct-0905
# meta-llama/Llama-3.2-90B-Vision-Instruct
# openai/gpt-oss-120b
# Qwen/Qwen2.5-VL-32B-Instruct
# deepseek-ai/DeepSeek-R1-0528
# zai-org/GLM-4.6
# Qwen/Qwen3-Next-80B-A3B-Instruct
# Intel/Qwen3-Coder-480B-A35B-Instruct-int4-mixed-ar
# meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
# mistralai/Mistral-Nemo-Instruct-2407
# openai/gpt-oss-20b
# meta-llama/Llama-3.3-70B-Instruct
# mistralai/Mistral-Large-Instruct-2411

import requests

url = "https://api.intelligence.io.solutions/api/v1/chat/completions"

payload = {
    "messages": [{"content": "Привет! Кто ты?", "role": "developer"}],
    "model": "moonshotai/Kimi-K2-Thinking",
}
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.text)

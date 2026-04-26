import ollama

client = ollama.Client(host="http://localhost:11434")

response = client.chat(
    model="gemma4:latest",
    messages=[
        {
            "role": "system",
            "content": "Ты полезный ИИ-ассистент. Отвечай кратко и по делу.",
        },
        {
            "role": "user",
            "content": "Почему небо голубое? Ответь в двух предложениях.",
        },
    ],
    options={
        "temperature": 0.7,
        "num_predict": 150,
    },
)

print("Ответ модели:")
print(response["message"]["content"])

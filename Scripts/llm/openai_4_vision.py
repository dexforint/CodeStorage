from openai import OpenAI
import base64

client = OpenAI()

# ─── Анализ по URL ────────────────────────────────────────────────────────────
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Что изображено на картинке? Опиши подробно."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
                        "detail": "high",  # "low" / "high" / "auto"
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)


# ─── Анализ локального файла (base64) ─────────────────────────────────────────
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_local_image(image_path: str, question: str) -> str:
    base64_image = encode_image(image_path)
    ext = image_path.split(".")[-1]  # jpg, png, etc.

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# result = analyze_local_image("screenshot.png", "Найди все ошибки в коде на скриншоте")


# ─── Несколько изображений сразу ─────────────────────────────────────────────
def compare_images(url1: str, url2: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Сравни эти два изображения. Чем они похожи и чем отличаются?",
                    },
                    {"type": "image_url", "image_url": {"url": url1}},
                    {"type": "image_url", "image_url": {"url": url2}},
                ],
            }
        ],
    )
    return response.choices[0].message.content

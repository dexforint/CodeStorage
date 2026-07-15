from openai import OpenAI
import requests
from pathlib import Path
import base64

client = OpenAI()

# ─── Генерация изображения (URL) ──────────────────────────────────────────────
response = client.images.generate(
    model="dall-e-3",  # "dall-e-2" или "dall-e-3"
    prompt="Футуристический город на Марсе, закат, фиолетовое небо, кинематографический стиль",
    n=1,  # dall-e-3 поддерживает только n=1
    size="1024x1024",  # "256x256" / "512x512" / "1024x1024" / "1792x1024" / "1024x1792"
    quality="hd",  # "standard" / "hd" (только dall-e-3)
    style="vivid",  # "vivid" / "natural" (только dall-e-3)
    response_format="url",  # "url" или "b64_json"
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt  # Как DALL-E интерпретировал промпт

print(f"🖼️  URL: {image_url}")
print(f"📝 Revised prompt: {revised_prompt}")


# ─── Сохранить изображение локально ──────────────────────────────────────────
def save_image(url: str, filename: str = "image.png"):
    img_data = requests.get(url).content
    Path(filename).write_bytes(img_data)
    print(f"✅ Сохранено: {filename}")


save_image(image_url)


# ─── Вариации изображения (dall-e-2) ─────────────────────────────────────────
def create_variation(image_path: str) -> str:
    with open(image_path, "rb") as img:
        response = client.images.create_variation(
            image=img,
            n=1,
            size="1024x1024",
        )
    return response.data[0].url


# ─── Редактирование изображения (inpainting, dall-e-2) ───────────────────────
def edit_image(image_path: str, mask_path: str, prompt: str) -> str:
    """Редактирует часть изображения по маске."""
    with open(image_path, "rb") as img, open(mask_path, "rb") as mask:
        response = client.images.edit(
            image=img,
            mask=mask,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
    return response.data[0].url

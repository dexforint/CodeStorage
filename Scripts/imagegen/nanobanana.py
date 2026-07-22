import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import base64

load_dotenv()

api_key = os.getenv("GEMINI_TOKEN")

client = genai.Client(api_key=api_key)

interaction = client.interactions.create(
    model="gemini-3.1-flash-image",
    input="Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
)

with open("./data/images/generated_image.png", "wb") as f:

    f.write(base64.b64decode(interaction.output_image.data))

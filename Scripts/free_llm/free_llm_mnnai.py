import openai
from prettytable import PrettyTable

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MNNAI_TOKEN")


client = openai.OpenAI(base_url="https://api.mnnai.ru/v1", api_key=api_key)

# models = client.models.list()

# mytable = PrettyTable()
# mytable.field_names = ["Model", "Free tier", "Price"]


# for model in models.data:
#     if not ("Free" in model.tiers):
#         continue

#     mytable.add_row(
#         [
#             model.id,
#             "Free" in model.tiers,
#             str(
#                 model.pricing["in_cost_per_million"]
#                 if "in_cost_per_million" in model.pricing
#                 else "None"
#             )
#             + " + "
#             + str(
#                 model.pricing["out_cost_per_million"]
#                 if "out_cost_per_million" in model.pricing
#                 else "None"
#             ),
#         ]
#     )

# print(mytable)


resp = client.chat.completions.create(
    model="qwen-3.5-397b-a17b",
    messages=[
        {
            "role": "user",
            "content": "Напиши Python функцию для суммирования двух чисел",
        }
    ],
)

print(resp)
print("#####")
print(resp.choices[0].message.content)


# +------------------------------+-----------+---------------+
# |            Model             | Free tier |     Price     |
# +------------------------------+-----------+---------------+
# |           gpt-5.2            |    True   |   1.75 + 14   |
# |         gpt-5.2-chat         |    True   |   1.75 + 14   |
# |           gpt-5.1            |    True   |   1.25 + 10   |
# |         gpt-5.1-chat         |    True   |   1.25 + 10   |
# |            gpt-5             |    True   |   1.25 + 10   |
# |          gpt-5-mini          |    True   |    0.25 + 2   |
# |          gpt-5-nano          |    True   |   0.05 + 0.4  |
# |          gpt-5-chat          |    True   |   1.25 + 10   |
# |           gpt-4.1            |    True   |     1 + 4     |
# |         gpt-4.1-mini         |    True   |   0.2 + 0.8   |
# |         gpt-4.1-nano         |    True   |   0.05 + 0.2  |
# |         gpt-oss-120b         |    True   |   0.15 + 0.6  |
# |         gpt-oss-20b          |    True   |   0.05 + 0.2  |
# |           o4-mini            |    True   |   1.1 + 4.4   |
# |            gpt-4o            |    True   |     1 + 5     |
# |     gpt-4o-audio-preview     |    True   |    2.5 + 10   |
# |         gpt-4o-mini          |    True   |   0.07 + 0.3  |
# |           o3-mini            |    True   |   0.55 + 2.2  |
# |         grok-3-mini          |    True   |   0.3 + 0.5   |
# |         glm-4.7-free         |    True   |   0.6 + 2.2   |
# |         glm-4.5-air          |    True   |   0.2 + 1.1   |
# |      deepseek-v3.2-exp       |    True   |   0.2 + 0.5   |
# |    deepseek-v3.1-terminus    |    True   |    0.27 + 1   |
# |        deepseek-v3.1         |    True   |   0.2 + 0.8   |
# |       deepseek-v3-0324       |    True   |  0.035 + 0.55 |
# |         deepseek-v3          |    True   |  0.035 + 0.55 |
# |       deepseek-r1-0528       |    True   |    0.07 + 1   |
# |     mistral-large-latest     |    True   |     1 + 3     |
# |    mistral-medium-latest     |    True   |    0.2 + 1    |
# |   magistral-medium-latest    |    True   |    1 + 2.5    |
# |     mistral-small-latest     |    True   |  0.05 + 0.15  |
# |       pixtral-12b-2409       |    True   | 0.075 + 0.075 |
# |    gemini-3-flash-preview    |    True   |    0.5 + 3    |
# |       gemini-2.5-flash       |    True   |  0.15 + 1.25  |
# |       gemini-2.0-flash       |    True   |  0.075 + 0.3  |
# |        gemma-3-27b-it        |    True   |  0.09 + 0.17  |
# |       gemma-3n-e4b-it        |    True   |  0.02 + 0.04  |
# |        gemma-3-4b-it         |    True   |  0.02 + 0.04  |
# |        gemma-3-12b-it        |    True   |  0.03 + 0.03  |
# |      qwen-3.5-397b-a17b      |    True   |   0.6 + 3.6   |
# |      qwen-3-coder-plus       |    True   |   1.1 + 2.2   |
# | qwen-3-next-80b-a3b-thinking |    True   |   0.3 + 0.3   |
# |       qwen-3-235b-a22b       |    True   |   0.1 + 0.3   |
# |    qwen-3-235b-a22b-2507     |    True   |   0.1 + 0.3   |
# |         qwen-3-coder         |    True   |     2 + 2     |
# |       llama-4-maverick       |    True   | 0.135 + 0.425 |
# |        llama-4-scout         |    True   |  0.09 + 0.295 |
# |        llama-3.3-70b         |    True   |  0.27 + 0.45  |
# |         llama-3.2-3b         |    True   |  0.03 + 0.03  |
# |        llama-3.1-70b         |    True   |  0.27 + 0.45  |
# |            sonar             |    True   |     1 + 1     |
# |     kimi-k2-0711-preview     |    True   |   0.6 + 2.5   |
# |     kimi-k2-0905-preview     |    True   |   0.6 + 2.5   |
# |       kimi-k2-thinking       |    True   |   0.6 + 2.5   |
# |          kimi-k2.5           |    True   |    0.6 + 3    |
# |         sd-3.5-large         |    True   |  None + None  |
# |        sd-3.5-medium         |    True   |  None + None  |
# |           flux-dev           |    True   |  None + None  |
# |         flux-schnell         |    True   |  None + None  |
# |           dall-e-3           |    True   |  None + None  |
# |    text-embedding-3-small    |    True   |  0.01 + 0.01  |
# |    text-embedding-3-large    |    True   | 0.065 + 0.065 |
# |    text-embedding-ada-002    |    True   |  0.05 + 0.05  |
# |    omni-moderation-latest    |    True   |  None + None  |
# |          whisper-1           |    True   |  None + None  |
# |      gpt-4o-transcribe       |    True   |    2.5 + 10   |
# |    gpt-4o-mini-transcribe    |    True   |    1.25 + 5   |
# +------------------------------+-----------+---------------+

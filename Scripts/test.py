from openai import OpenAI

client = OpenAI(
    base_url="https://aihubmix.com/v1",
    api_key="sk-ykEuMBKUQLwr3xdDEa44Db2c33054bCc81C1E1AbDe1513Fb",
)

completion = client.chat.completions.create(
    model="claude-opus-4-7-think",
    messages=[
        {
            "role": "assistant",
            "content": "Provide helpful, friendly, and well-structured answers.",
        },
        {
            "role": "user",
            "content": "I'm visiting US for 7 days. Can you help me plan a simple travel itinerary?",
        },
    ],
    #   max_tokens=10000, # Defaults to 4096 for OpenAI compatible interface; enable this parameter for longer text generation
)

print(completion.choices[0].message.content)

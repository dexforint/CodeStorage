from openai import OpenAI

# g4f api

# Initialize the OpenAI client
client = OpenAI(
    api_key="secret",  # Set an API key (use "secret" if your provider doesn't require one)
    base_url="http://localhost:1337/v1",  # Point to your local or custom API endpoint
)

# Create a chat completion request
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Specify the model to use
    messages=[
        {"role": "user", "content": "Write a poem about a tree"}
    ],  # Define the input message
    stream=True,  # Enable streaming for real-time responses
)

# Handle the response
if isinstance(response, dict):
    # Non-streaming response
    print(response.choices[0].message.content)
else:
    # Streaming response
    for token in response:
        content = token.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)

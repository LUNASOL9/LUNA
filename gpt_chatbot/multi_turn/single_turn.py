from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    )

    print("Assistant:", response.choices[0].message.content)

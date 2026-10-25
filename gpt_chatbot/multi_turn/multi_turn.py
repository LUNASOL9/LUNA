from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def get_ai_response(messages):
    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=messages
    )
    return response.choices[0].message.content

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})
    
    ai_response = get_ai_response(messages)
    messages.append({"role": "assistant", "content": ai_response})
    
    print("Assistant:", ai_response)
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]

for model in models_to_test:

    print("\n================================")
    print("Testing:", model)
    print("================================")

    try:

        response = client.models.generate_content(
            model=model,
            contents="Say hello in one short sentence."
        )

        if response and response.text:
            print("SUCCESS:")
            print(response.text)

        else:
            print("EMPTY RESPONSE")

    except Exception as e:

        print("ERROR:")
        print(e)
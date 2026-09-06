from google import genai

from tools import calculator


client = genai.Client()


tools = [
    calculator
]


print("🤖 Mini AI Agent Tool Test")
print("---------------------------")


question = input("\nAsk something: ")


try:

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question,
        config={
            "tools": tools
        }
    )

    print("\n========== AI ANSWER ==========")
    print(response.text)


except Exception as e:

    print("\nAI Error:")
    print(e)
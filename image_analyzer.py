import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_bytes,
    mime_type,
    question="Describe this image."
):

    if not image_bytes:

        return (
            "❌ Image data is empty."
        )


    try:

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )


        prompt = f"""
You are Mini AI Agent's image analysis system.

Analyze the provided image carefully.

User question:
{question}

Instructions:

1. Describe only what you can actually see.
2. Do not invent details.
3. If there is text, explain or read it when possible.
4. If the image contains an error or screenshot,
   explain the visible issue clearly.
5. Keep the answer simple and useful.
"""


        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=[
                image_part,
                prompt
            ]

        )


        if response is None:

            return (
                "❌ Gemini returned no response."
            )


        answer = getattr(
            response,
            "text",
            None
        )


        if answer:

            return answer.strip()


        return (
            "❌ Gemini returned an empty answer."
        )


    except Exception as e:

        return (
            "❌ Image analysis failed:\n"
            + str(e)
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "       IMAGE ANALYZER"
    )

    print(
        "========================================"
    )

    print(
        "\nImage analyzer is ready."
    )
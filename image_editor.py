import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()

# ==============================
# SETTINGS
# ==============================

MODEL_NAME = "gemini-3.1-flash-image"

IMAGE_FOLDER = Path("images")
OUTPUT_FOLDER = Path("edited_images")


# ==============================
# FIND IMAGE
# ==============================

def find_image():

    supported_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]

    if not IMAGE_FOLDER.exists():
        IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        return None

    images = []

    for file in IMAGE_FOLDER.iterdir():

        if file.is_file() and file.suffix.lower() in supported_extensions:
            images.append(file)

    if not images:
        return None

    return images[0]


# ==============================
# EDIT IMAGE
# ==============================

def edit_image(image_path, instruction):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY .env file mein nahi mila."
        )

    client = genai.Client(
        api_key=api_key
    )

    # Original image read karo
    with open(image_path, "rb") as file:
        image_bytes = file.read()

    # MIME type
    extension = Path(image_path).suffix.lower()

    if extension in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"

    elif extension == ".png":
        mime_type = "image/png"

    elif extension == ".webp":
        mime_type = "image/webp"

    else:
        raise ValueError(
            "Unsupported image format."
        )

    # AI editing prompt
    prompt = f"""
Edit the provided image according to the user's instruction.

USER INSTRUCTION:
{instruction}

IMPORTANT:
- Preserve the person's identity.
- Keep the face natural.
- Do not unnecessarily change facial features.
- Make realistic and high-quality edits.
- Change only what the user requested.
- Keep the original composition where possible.
"""

    # Gemini image editing
    interaction = client.interactions.create(

        model=MODEL_NAME,

        input=[
            {
                "type": "image",
                "data": base64.b64encode(
                    image_bytes
                ).decode("utf-8"),
                "mime_type": mime_type
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    )

    # Output folder
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_FOLDER /
        f"{Path(image_path).stem}_edited.png"
    )

    # Generated image save karo
    for step in interaction.steps:

        if step.type == "model_output":

            for content_block in step.content:

                if content_block.type == "image":

                    with open(
                        output_path,
                        "wb"
                    ) as file:

                        file.write(
                            base64.b64decode(
                                content_block.data
                            )
                        )

                    return output_path

    raise RuntimeError(
        "Gemini ne edited image return nahi ki."
    )


# ==============================
# MAIN
# ==============================

def main():

    print("\n================================")
    print("       AI IMAGE EDITOR")
    print("================================")

    image_path = find_image()

    if image_path is None:

        print("\n❌ images folder mein photo nahi mili.")

        print("\nPhoto yahan rakho:")
        print(
            IMAGE_FOLDER.resolve()
        )

        print(
            "\nSupported formats: "
            "JPG, JPEG, PNG, WEBP"
        )

        return

    print("\n📸 Photo found:")
    print(image_path)

    print("\nExample instructions:")
    print("1. Face ko naturally clear aur sharp karo.")
    print("2. Background ko Chandigarh University campus karo.")
    print("3. Lighting improve karo.")
    print("4. Photo ko professional portrait jaisa karo.")

    instruction = input(
        "\n✏️ Photo mein kya edit karna hai: "
    ).strip()

    if not instruction:

        print(
            "\n❌ Editing instruction empty hai."
        )

        return

    print("\n🤖 AI image editing started...")
    print("Please wait...\n")

    try:

        result = edit_image(
            image_path,
            instruction
        )

        print(
            "================================"
        )

        print(
            "✅ IMAGE EDITED SUCCESSFULLY"
        )

        print(
            "================================"
        )

        print(
            "\nSaved at:"
        )

        print(
            result
        )

    except Exception as e:

        print(
            "\n❌ Image editing failed:"
        )

        print(
            str(e)
        )


if __name__ == "__main__":
    main()
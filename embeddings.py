import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class EmbeddingModel:

    def __init__(
        self,
        model_name="gemini-embedding-001"
    ):
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables."
            )

        print("Loading Gemini embedding model...")

        self.client = genai.Client(
            api_key=api_key
        )

        print("Gemini embedding model loaded successfully.")

    def create_embedding(self, text):

        if not text or not text.strip():
            return []

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768
            )
        )

        if not result.embeddings:
            raise ValueError(
                "No embedding returned from Gemini API."
            )

        return result.embeddings[0].values
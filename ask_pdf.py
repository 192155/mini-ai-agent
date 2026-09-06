from google import genai

from embeddings import EmbeddingModel
from vector_store import VectorStore


VECTOR_STORE_PATH = "data/java_vectors.pkl"


# ==============================
# LOAD AI MODEL
# ==============================

print("Loading Gemini AI...")

client = genai.Client()

print("Gemini AI loaded successfully.")


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("Loading embedding model...")

embedding_model = EmbeddingModel()

print("Loading PDF knowledge...")

vector_store = VectorStore()
vector_store.load(VECTOR_STORE_PATH)

print("PDF knowledge loaded successfully.")


# ==============================
# ASK QUESTION
# ==============================

while True:

    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        print("Program closed.")
        break

    if not question.strip():
        print("Please type a question.")
        continue


    # ==============================
    # SEARCH PDF
    # ==============================

    question_embedding = embedding_model.create_embedding(
        question
    )

    results = vector_store.search(
        question_embedding,
        top_k=3
    )


    if not results:
        print("\nNo relevant information found in PDF.")
        continue


    # ==============================
    # CREATE CONTEXT
    # ==============================

    context = ""

    for i, result in enumerate(results):

        context += f"""
SOURCE {i + 1}:
{result["text"]}

"""


    # ==============================
    # SEND TO GEMINI
    # ==============================

    prompt = f"""
You are a helpful Java study assistant.

Answer the user's question using ONLY the information
provided from the Java PDF.

If the information is not available in the PDF,
say that the answer was not found in the PDF.

Explain the answer clearly and simply.

User Question:
{question}

PDF Information:
{context}
"""


    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        print("\n========== AI ANSWER ==========\n")

        print(response.text)

        print("\n===============================\n")


    except Exception as e:

        print("\nAI Error:")
        print(e)
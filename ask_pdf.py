import os
from dotenv import load_dotenv
from google import genai

from embeddings import EmbeddingModel
from vector_store import VectorStore
from tools import web_search

load_dotenv()

VECTOR_STORE_PATH = "data/java_vectors.pkl"

print("Loading AI embedding model...")
embedding_model = EmbeddingModel()

print("Loading PDF knowledge...")
vector_store = VectorStore()
vector_store.load(VECTOR_STORE_PATH)

print("Loading Gemini AI...")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.6-flash"


def ask_gemini(question, context):
    prompt = f"""
You are a helpful AI assistant.

The user asked:
{question}

Use the following information to answer the question:

{context}

Instructions:
1. If the provided information contains the answer, answer using it.
2. If the information is insufficient, clearly say that the PDF information is insufficient.
3. Do not invent information.
4. Give a clear and easy-to-understand answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


def answer_question(question):

    # --------------------------------
    # STEP 1: Search PDF
    # --------------------------------
    question_embedding = embedding_model.create_embedding(question)

    pdf_results = vector_store.search(
        question_embedding,
        top_k=3
    )

    pdf_context = ""

    for result in pdf_results:
        pdf_context += "\n" + result["text"]

    # --------------------------------
    # STEP 2: Ask Gemini about PDF
    # --------------------------------
    pdf_answer = ask_gemini(
        question,
        pdf_context
    )

    # --------------------------------
    # STEP 3: Decide whether web search
    # is required
    # --------------------------------

    check_prompt = f"""
Question:
{question}

PDF-based answer:
{pdf_answer}

Answer only YES or NO.

Return YES if the PDF-based answer is insufficient
or the question requires information outside the PDF.

Return NO if the PDF contains enough information.
"""

    check_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=check_prompt
    )

    decision = check_response.text.strip().upper()

    # --------------------------------
    # STEP 4: Web Search if needed
    # --------------------------------

    if "YES" in decision:

        print("\n🌐 Searching the web...")

        web_results = web_search(question)

        final_context = f"""
PDF Information:
{pdf_context}

Web Search Information:
{web_results}
"""

        final_answer = ask_gemini(
            question,
            final_context
        )

        return final_answer

    # --------------------------------
    # STEP 5: Return PDF answer
    # --------------------------------

    return pdf_answer


# ====================================
# MAIN PROGRAM
# ====================================

print("\n================================")
print("       MINI AI AGENT")
print("================================")

while True:

    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        print("Program closed.")
        break

    if not question.strip():
        print("Please type a question.")
        continue

    try:

        answer = answer_question(question)

        print("\n========== AI ANSWER ==========\n")
        print(answer)

    except Exception as e:

        print("\nAI Error:")
        print(e)
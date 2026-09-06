import os
import time

from dotenv import load_dotenv
from google import genai

from chat_manager import ChatManager
from embeddings import EmbeddingModel
from vector_store import VectorStore
from web_search import web_search
from calculator import calculator
from ai_router import route_question

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
VECTOR_STORE_PATH = "data/all_vectors.pkl"

client = genai.Client(api_key=API_KEY)

chat_manager = ChatManager()

_embedding_model = None
_vector_store = None


# =========================================================
# PDF SYSTEM
# =========================================================

def load_pdf_system():
    global _embedding_model, _vector_store

    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = EmbeddingModel()
        print("Embedding model loaded.")

    if _vector_store is None:
        print("Loading vector store...")

        _vector_store = VectorStore()

        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found: {VECTOR_STORE_PATH}"
            )

        _vector_store.load(VECTOR_STORE_PATH)

        print("Vector store loaded.")

    return _embedding_model, _vector_store


# =========================================================
# GEMINI ERROR HANDLER
# =========================================================

def get_gemini_error_message(error):

    error_text = str(error)

    if (
        "429" in error_text
        or "RESOURCE_EXHAUSTED" in error_text
        or "quota" in error_text.lower()
    ):
        return (
            "⚠️ Gemini API quota exhausted.\n\n"
            "Your current Gemini free-tier "
            "request limit has been reached.\n\n"
            "Please wait for the quota to reset "
            "or use a Gemini API plan with higher limits."
        )

    if (
        "503" in error_text
        or "UNAVAILABLE" in error_text
    ):
        return (
            "⚠️ Gemini is temporarily busy.\n\n"
            "Please try again after a short while."
        )

    if (
        "API key" in error_text
        or "API_KEY" in error_text
    ):
        return (
            "❌ Gemini API key problem.\n\n"
            "Check GEMINI_API_KEY in your .env file."
        )

    if (
        "404" in error_text
        or "NOT_FOUND" in error_text
    ):
        return (
            "❌ Gemini model was not found.\n\n"
            f"Current model: {MODEL_NAME}"
        )

    return "❌ Gemini API error:\n" + error_text


# =========================================================
# GEMINI ANSWER GENERATOR
# =========================================================

def generate_answer(
    question,
    context="",
    conversation=""
):

    prompt = f"""
You are Mini AI Agent.

Answer the user's question clearly,
accurately and helpfully.

Previous conversation:

{conversation}

Relevant knowledge:

{context}

User question:

{question}

Instructions:

1. Give a direct answer.
2. If relevant knowledge is provided,
   use it.
3. Do not invent information.
4. Keep the answer easy to understand.
5. Use bullet points when useful.
6. If the user asks for code,
   provide complete working code.
"""

    for attempt in range(2):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            answer = getattr(
                response,
                "text",
                None
            )

            if answer:
                return answer.strip()

            return "❌ Gemini returned an empty response."

        except Exception as error:

            error_text = str(error)

            print(
                f"Gemini error "
                f"(attempt {attempt + 1}/2): "
                f"{error_text}"
            )

            # -----------------------------------------
            # QUOTA ERROR
            # -----------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):
                return get_gemini_error_message(error)

            # -----------------------------------------
            # TEMPORARY SERVER ERROR
            # -----------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                if attempt == 0:

                    time.sleep(5)

                    continue

                return get_gemini_error_message(error)

            # -----------------------------------------
            # OTHER ERROR
            # -----------------------------------------

            return get_gemini_error_message(error)

    return "❌ Unable to generate answer."


# =========================================================
# CHECK GEMINI QUOTA ERROR
# =========================================================

def is_gemini_quota_error(answer):

    if not answer:
        return False

    text = answer.lower()

    return (
        "gemini api quota exhausted" in text
        or "quota exhausted" in text
        or "resource_exhausted" in text
    )


# =========================================================
# OFFLINE PDF FALLBACK
# =========================================================

def build_pdf_offline_answer(
    question,
    results
):

    if not results:

        return (
            "❌ I could not find relevant information "
            "in your uploaded PDFs."
        )

    output = []

    output.append(
        "⚠️ Gemini is currently unavailable because "
        "the API quota has been exhausted."
    )

    output.append("")

    output.append(
        "📚 Offline PDF Mode"
    )

    output.append("")

    output.append(
        "I found the following relevant excerpts "
        "from your notes:"
    )

    output.append("")

    for index, result in enumerate(
        results[:3],
        start=1
    ):

        text = result.get(
            "text",
            ""
        )

        score = result.get(
            "score",
            0
        )

        if isinstance(text, dict):

            content = text.get(
                "text",
                ""
            )

            source = text.get(
                "source",
                "Uploaded PDF"
            )

        else:

            content = str(text)

            source = "Uploaded PDF"

        content = content.strip()

        if not content:
            continue

        # Keep fallback readable.
        if len(content) > 900:
            content = content[:900] + "..."

        output.append(
            f"### 📄 Excerpt {index}"
        )

        output.append(
            f"**Source:** {source}"
        )

        output.append(
            f"**Relevance:** {score:.3f}"
        )

        output.append("")

        output.append(content)

        output.append("")

    output.append(
        "💡 Gemini answer generation will work again "
        "when your API quota becomes available."
    )

    return "\n".join(output)


# =========================================================
# CALCULATOR
# =========================================================

def handle_calculator(question):

    result = calculator(question)

    return {
        "answer": result,
        "tool": "calculator",
        "tool_name": "🧮 Calculator",
        "source": "Calculator",
        "sources": []
    }


# =========================================================
# PDF SEARCH
# =========================================================

def search_pdf(question):

    try:

        embedding_model, vector_store = (
            load_pdf_system()
        )

        query_embedding = (
            embedding_model.create_embedding(
                question
            )
        )

        results = vector_store.search(
            query_embedding,
            top_k=5,
            threshold=0.30
        )

        if not results:

            return {
                "answer": (
                    "❌ I could not find relevant "
                    "information in your uploaded PDFs."
                ),
                "tool": "pdf",
                "tool_name": "📚 PDF Knowledge",
                "source": "No matching PDF source found.",
                "sources": []
            }

        context_parts = []

        sources = []

        # -----------------------------------------
        # PREPARE PDF CONTEXT
        # -----------------------------------------

        for result in results:

            text = result.get(
                "text",
                ""
            )

            if isinstance(text, dict):

                content = text.get(
                    "text",
                    ""
                )

                source_name = text.get(
                    "source",
                    "Uploaded PDF"
                )

            else:

                content = str(text)

                source_name = "Uploaded PDF"

            context_parts.append(
                content
            )

            if source_name not in sources:

                sources.append(
                    source_name
                )

        context = "\n\n".join(
            context_parts
        )

        # -----------------------------------------
        # ASK GEMINI
        # -----------------------------------------

        answer = generate_answer(
            question,
            context=context
        )

        # -----------------------------------------
        # OFFLINE FALLBACK
        # -----------------------------------------

        if is_gemini_quota_error(answer):

            offline_answer = (
                build_pdf_offline_answer(
                    question,
                    results
                )
            )

            return {
                "answer": offline_answer,
                "tool": "pdf_offline",
                "tool_name": "📚 PDF Knowledge • Offline Mode",
                "source": (
                    ", ".join(sources)
                    if sources
                    else "Uploaded PDF"
                ),
                "sources": sources
            }

        # -----------------------------------------
        # NORMAL PDF RESPONSE
        # -----------------------------------------

        return {
            "answer": answer,
            "tool": "pdf",
            "tool_name": "📚 PDF Knowledge",
            "source": (
                ", ".join(sources)
                if sources
                else "Uploaded PDF"
            ),
            "sources": sources
        }

    except Exception as error:

        return {
            "answer": (
                "❌ PDF search failed:\n"
                + str(error)
            ),
            "tool": "pdf",
            "tool_name": "📚 PDF Knowledge",
            "source": "PDF System Error",
            "sources": []
        }


# =========================================================
# WEB SEARCH
# =========================================================

def handle_web_search(question):

    try:

        search_result = web_search(
            question
        )

        if not search_result:

            return {
                "answer": "No web results found.",
                "tool": "web",
                "tool_name": "🌐 Web Search",
                "source": "Web Search",
                "sources": []
            }

        answer = generate_answer(
            question,
            context=search_result
        )

        sources = []

        for line in search_result.splitlines():

            line = line.strip()

            if line.startswith("URL:"):

                url = line.replace(
                    "URL:",
                    ""
                ).strip()

                if (
                    url
                    and url not in sources
                ):

                    sources.append(
                        url
                    )

        return {
            "answer": answer,
            "tool": "web",
            "tool_name": "🌐 Web Search",
            "source": (
                f"{len(sources)} web source(s)"
                if sources
                else "Web Search"
            ),
            "sources": sources
        }

    except Exception as error:

        return {
            "answer": (
                "❌ Web search failed:\n"
                + str(error)
            ),
            "tool": "web",
            "tool_name": "🌐 Web Search",
            "source": "Web Search Error",
            "sources": []
        }


# =========================================================
# GENERAL AI
# =========================================================

def handle_general(
    question,
    conversation=""
):

    answer = generate_answer(
        question,
        context="",
        conversation=conversation
    )

    return {
        "answer": answer,
        "tool": "general",
        "tool_name": "🤖 Gemini AI",
        "source": "Gemini AI",
        "sources": []
    }


# =========================================================
# MAIN AGENT
# =========================================================

def ask_agent(
    question,
    chat_id=None
):

    question = (
        question or ""
    ).strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "tool": "general",
            "tool_name": "🤖 Gemini AI",
            "source": "",
            "sources": [],
            "chat_id": chat_id
        }

    # -----------------------------------------
    # CREATE CHAT
    # -----------------------------------------

    if not chat_id:

        chat_id = (
            chat_manager.create_chat(
                "New Chat"
            )
        )

    elif not chat_manager.chat_exists(
        chat_id
    ):

        chat_id = (
            chat_manager.create_chat(
                "New Chat"
            )
        )

    # -----------------------------------------
    # CONVERSATION
    # -----------------------------------------

    conversation = (
        chat_manager.get_context(
            chat_id,
            max_messages=10
        )
    )

    # -----------------------------------------
    # ROUTER
    # -----------------------------------------

    tool = route_question(
        question
    )

    print(
        "Selected tool:",
        tool
    )

    # -----------------------------------------
    # TOOL EXECUTION
    # -----------------------------------------

    if tool == "calculator":

        result = handle_calculator(
            question
        )

    elif tool == "pdf":

        result = search_pdf(
            question
        )

    elif tool == "web":

        result = handle_web_search(
            question
        )

    else:

        result = handle_general(
            question,
            conversation
        )

    # -----------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------

    chat_manager.add_message(
        chat_id,
        "user",
        question
    )

    # -----------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------------------

    chat_manager.add_message(
        chat_id,
        "assistant",
        result.get(
            "answer",
            ""
        )
    )

    # -----------------------------------------
    # AUTO TITLE
    # -----------------------------------------

    messages = (
        chat_manager.get_messages(
            chat_id
        )
    )

    if len(messages) == 2:

        chat_manager.generate_title(
            chat_id,
            question
        )

    result["chat_id"] = chat_id

    return result


# =========================================================
# TERMINAL MODE
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "======================================"
    )

    print(
        "          MINI AI AGENT"
    )

    print(
        "======================================"
    )

    print(
        "Agent is running..."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    current_chat_id = None

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nGoodbye 👋"
            )

            break

        result = ask_agent(
            question,
            current_chat_id
        )

        current_chat_id = (
            result.get(
                "chat_id"
            )
        )

        print()

        print(
            "🤖 Answer:"
        )

        print(
            result.get(
                "answer",
                ""
            )
        )

        print()

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print(
            "🧠 Tool Used:",
            result.get(
                "tool_name",
                "🤖 Gemini AI"
            )
        )

        source = result.get(
            "source",
            ""
        )

        if source:

            print(
                "📄 Source:",
                source
            )

        sources = result.get(
            "sources",
            []
        )

        if sources:

            print(
                "\n🔗 Sources:"
            )

            for item in sources:

                print(
                    "-",
                    item
                )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        )
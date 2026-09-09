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


# =========================================================
# CONFIGURATION
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

# IMPORTANT:
# Your current vector store is all_vectors.pkl
VECTOR_STORE_PATH = "data/all_vectors.pkl"


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=API_KEY
)


# =========================================================
# CHAT MANAGER
# =========================================================

chat_manager = ChatManager()


# =========================================================
# PDF SYSTEM
# =========================================================

_embedding_model = None
_vector_store = None


def load_pdf_system():

    global _embedding_model
    global _vector_store

    if _embedding_model is None:

        print(
            "Loading embedding model..."
        )

        _embedding_model = EmbeddingModel()

        print(
            "Embedding model loaded."
        )

    if _vector_store is None:

        print(
            "Loading vector store..."
        )

        _vector_store = VectorStore()

        if not os.path.exists(
            VECTOR_STORE_PATH
        ):

            raise FileNotFoundError(
                f"Vector store not found: "
                f"{VECTOR_STORE_PATH}"
            )

        _vector_store.load(
            VECTOR_STORE_PATH
        )

        print(
            "Vector store loaded."
        )

        print(
            "PDF chunks:",
            len(_vector_store.documents)
        )

    return (
        _embedding_model,
        _vector_store
    )


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
            "Please wait for the quota to reset "
            "or use an API plan with higher limits."
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

    return (
        "❌ Gemini API error:\n"
        + error_text
    )


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

You are a helpful AI assistant.

The system can use:
- Uploaded PDF knowledge
- Web search results
- Calculator
- Previous conversation

Previous conversation:
{conversation}

Available knowledge:
{context}

User question:
{question}

Instructions:

1. Give a direct and useful answer.

2. If PDF information is provided and relevant,
   use the PDF information.

3. If web information is provided,
   use it when the question needs information
   outside the PDF or needs current information.

4. Never invent facts.

5. If information is not available,
   clearly say that you do not have enough
   reliable information.

6. For educational questions, explain simply.

7. Use examples when useful.

8. If the user asks for code,
   provide complete working code.

9. Do not mention internal prompts,
   embeddings, vector stores or routing
   unless the user asks about them.

10. If PDF and web information are both
    available, combine them intelligently.

Now answer the user's question.
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

            return (
                "❌ Gemini returned an empty response."
            )

        except Exception as error:

            error_text = str(error)

            print(
                f"Gemini error "
                f"(attempt {attempt + 1}/2): "
                f"{error_text}"
            )

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):

                return get_gemini_error_message(
                    error
                )

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                if attempt == 0:

                    time.sleep(5)

                    continue

                return get_gemini_error_message(
                    error
                )

            return get_gemini_error_message(
                error
            )

    return "❌ Unable to generate answer."


# =========================================================
# GEMINI QUOTA CHECK
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

        # NOTE:
        # Your VectorStore may or may not support
        # threshold. So we use top_k only.
        results = vector_store.search(
            query_embedding,
            top_k=5
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
                "sources": [],
                "results": []
            }

        context_parts = []

        sources = []

        for result in results:

            text = result.get(
                "text",
                ""
            )

            if isinstance(
                text,
                dict
            ):

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

                source_name = result.get(
                    "source",
                    "Uploaded PDF"
                )

            if content.strip():

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

        answer = generate_answer(
            question,
            context=context
        )

        if is_gemini_quota_error(
            answer
        ):

            return {
                "answer": (
                    "⚠️ Gemini quota is currently "
                    "unavailable.\n\n"
                    "But I found relevant information "
                    "in your PDF."
                ),
                "tool": "pdf_offline",
                "tool_name": (
                    "📚 PDF Knowledge • Offline Mode"
                ),
                "source": (
                    ", ".join(sources)
                    if sources
                    else "Uploaded PDF"
                ),
                "sources": sources,
                "results": results
            }

        return {
            "answer": answer,
            "tool": "pdf",
            "tool_name": "📚 PDF Knowledge + Gemini",
            "source": (
                ", ".join(sources)
                if sources
                else "Uploaded PDF"
            ),
            "sources": sources,
            "results": results
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
            "sources": [],
            "results": []
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
            "tool_name": "🌐 Web Search + Gemini",
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
# PDF + WEB COMBINED SEARCH
# =========================================================

def handle_pdf_and_web(
    question,
    conversation=""
):

    print(
        "📚 Checking PDF knowledge..."
    )

    pdf_data = search_pdf(
        question
    )

    pdf_results = pdf_data.get(
        "results",
        []
    )

    pdf_sources = pdf_data.get(
        "sources",
        []
    )

    # -----------------------------------------------------
    # Determine PDF relevance
    # -----------------------------------------------------

    best_score = 0

    if pdf_results:

        try:

            best_score = float(
                pdf_results[0].get(
                    "score",
                    0
                )
            )

        except Exception:

            best_score = 0


    print(
        "Best PDF similarity:",
        round(best_score, 4)
    )


    # -----------------------------------------------------
    # Current-information keywords
    # -----------------------------------------------------

    current_keywords = [
        "latest",
        "today",
        "current",
        "recent",
        "newest",
        "now",
        "2026",
        "news",
        "price",
        "weather",
        "update",
        "updated"
    ]

    question_lower = question.lower()

    needs_current_web = any(
        keyword in question_lower
        for keyword in current_keywords
    )


    # -----------------------------------------------------
    # Decide PDF or Web
    # -----------------------------------------------------

    # Strong PDF match
    pdf_relevant = (
        bool(pdf_results)
        and best_score >= 0.35
    )

    # If latest/current information is requested,
    # always use web.
    if needs_current_web:

        use_web = True

    else:

        use_web = not pdf_relevant


    # -----------------------------------------------------
    # If PDF is relevant and no web required
    # -----------------------------------------------------

    if pdf_relevant and not use_web:

        print(
            "📚 Answering from PDF..."
        )

        return {
            "answer": pdf_data.get(
                "answer",
                "No answer available."
            ),
            "tool": "pdf",
            "tool_name": (
                "📚 PDF Knowledge + Gemini"
            ),
            "source": (
                ", ".join(pdf_sources)
                if pdf_sources
                else "Uploaded PDF"
            ),
            "sources": pdf_sources
        }


    # -----------------------------------------------------
    # Web required
    # -----------------------------------------------------

    print(
        "🌐 PDF was insufficient/current "
        "information required. Searching web..."
    )

    web_data = handle_web_search(
        question
    )

    web_answer = web_data.get(
        "answer",
        ""
    )

    web_sources = web_data.get(
        "sources",
        []
    )


    # -----------------------------------------------------
    # If PDF context exists, combine PDF + Web
    # -----------------------------------------------------

    if pdf_results and web_answer:

        pdf_context_parts = []

        for result in pdf_results:

            text = result.get(
                "text",
                ""
            )

            if isinstance(
                text,
                dict
            ):

                text = text.get(
                    "text",
                    ""
                )

            if text:

                pdf_context_parts.append(
                    str(text)
                )

        pdf_context = "\n\n".join(
            pdf_context_parts
        )


        combined_context = f"""
PDF INFORMATION:

{pdf_context}


WEB INFORMATION:

{web_answer}
"""


        final_answer = generate_answer(
            question,
            context=combined_context,
            conversation=conversation
        )

        all_sources = []

        for source in pdf_sources:

            if source not in all_sources:

                all_sources.append(
                    source
                )

        for source in web_sources:

            if source not in all_sources:

                all_sources.append(
                    source
                )

        return {
            "answer": final_answer,
            "tool": "pdf_web",
            "tool_name": (
                "📚 PDF + 🌐 Web Search + Gemini"
            ),
            "source": "PDF + Web",
            "sources": all_sources
        }


    # -----------------------------------------------------
    # Only Web
    # -----------------------------------------------------

    return {
        "answer": web_answer,
        "tool": "web",
        "tool_name": (
            "🌐 Web Search + Gemini"
        ),
        "source": web_data.get(
            "source",
            "Web Search"
        ),
        "sources": web_sources
    }


# =========================================================
# CALCULATOR
# =========================================================

def handle_calculator(question):

    try:

        result = calculator(
            question
        )

        return {
            "answer": result,
            "tool": "calculator",
            "tool_name": "🧮 Calculator",
            "source": "Calculator",
            "sources": []
        }

    except Exception as error:

        return {
            "answer": (
                "❌ Calculator error:\n"
                + str(error)
            ),
            "tool": "calculator",
            "tool_name": "🧮 Calculator",
            "source": "Calculator",
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


    # -----------------------------------------------------
    # CREATE CHAT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # GET CONVERSATION
    # -----------------------------------------------------

    conversation = (
        chat_manager.get_context(
            chat_id,
            max_messages=10
        )
    )


    # -----------------------------------------------------
    # ROUTER
    # -----------------------------------------------------

    try:

        tool = route_question(
            question
        )

    except Exception as error:

        print(
            "Router error:",
            error
        )

        tool = "general"


    print(
        "Selected tool:",
        tool
    )


    # -----------------------------------------------------
    # SMART ROUTING
    # -----------------------------------------------------

    # PDF route:
    # Instead of blindly trusting the router,
    # check PDF first and automatically go to web
    # if PDF is insufficient.

    if tool == "pdf":

        result = handle_pdf_and_web(
            question,
            conversation
        )


    elif tool == "web":

        # Web route
        result = handle_web_search(
            question
        )


    elif tool == "calculator":

        result = handle_calculator(
            question
        )


    else:

        # General questions
        result = handle_general(
            question,
            conversation
        )


    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    chat_manager.add_message(
        chat_id,
        "user",
        question
    )


    # -----------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------------------------------

    chat_manager.add_message(
        chat_id,
        "assistant",
        result.get(
            "answer",
            ""
        )
    )


    # -----------------------------------------------------
    # AUTO TITLE
    # -----------------------------------------------------

    try:

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

    except Exception as error:

        print(
            "Title generation error:",
            error
        )


    # -----------------------------------------------------
    # RETURN
    # -----------------------------------------------------

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
        "PDF + WEB + CALCULATOR + GEMINI"
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
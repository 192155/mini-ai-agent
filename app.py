import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# =========================================================
# CONFIGURATION
# =========================================================

VERCEL_API_URL = "https://mini-ai-agent-pi.vercel.app/api/chat"

APP_TITLE = "Mini AI Agent"

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    .feature-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }

    .small-text {
        font-size: 13px;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 Mini AI Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your personal AI assistant powered by Gemini AI</div>',
    unsafe_allow_html=True
)


# =========================================================
# FUNCTIONS
# =========================================================

def clear_chat():
    """
    Clear the current local chat display
    and create a new chat ID.
    """
    st.session_state.messages = []
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.processing = False


def ask_vercel_api(question):
    """
    Send the user's question to the deployed
    Mini AI Agent API on Vercel.
    """

    try:

        response = requests.post(
            VERCEL_API_URL,
            json={
                "question": question,
                "chat_id": st.session_state.chat_id
            },
            timeout=60
        )

        # -------------------------------------------------
        # Successful HTTP response
        # -------------------------------------------------

        if response.status_code == 200:

            try:
                result = response.json()
            except Exception:
                return {
                    "success": False,
                    "answer": "❌ API returned an invalid response.",
                    "tool": "error",
                    "tool_name": "Vercel API",
                    "source": "Vercel",
                    "sources": []
                }

            return result

        # -------------------------------------------------
        # API error
        # -------------------------------------------------

        return {
            "success": False,
            "answer": (
                f"❌ Vercel API Error: {response.status_code}\n\n"
                f"{response.text}"
            ),
            "tool": "error",
            "tool_name": "Vercel API",
            "source": "Vercel",
            "sources": []
        }

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "answer": (
                "⏱️ The AI server took too long to respond.\n\n"
                "Please try again."
            ),
            "tool": "error",
            "tool_name": "Vercel API",
            "source": "Vercel",
            "sources": []
        }

    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "answer": (
                "🌐 Could not connect to the AI server.\n\n"
                "Please check your internet connection and try again."
            ),
            "tool": "error",
            "tool_name": "Vercel API",
            "source": "Vercel",
            "sources": []
        }

    except Exception as e:

        return {
            "success": False,
            "answer": f"❌ Unexpected error:\n\n{str(e)}",
            "tool": "error",
            "tool_name": "Vercel API",
            "source": "Vercel",
            "sources": []
        }


def process_question(question):

    question = question.strip()

    if not question:
        return

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    st.session_state.processing = True

    # -----------------------------------------------------
    # Call Vercel API
    # -----------------------------------------------------

    result = ask_vercel_api(question)

    # -----------------------------------------------------
    # Extract answer
    # -----------------------------------------------------

    answer = result.get(
        "answer",
        "❌ No answer received from the AI."
    )

    # Update chat ID if API returns one
    returned_chat_id = result.get("chat_id")

    if returned_chat_id:
        st.session_state.chat_id = returned_chat_id

    # -----------------------------------------------------
    # Add assistant message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "tool": result.get("tool", "general"),
            "tool_name": result.get(
                "tool_name",
                "🤖 Gemini AI"
            ),
            "source": result.get("source", ""),
            "sources": result.get("sources", [])
        }
    )

    st.session_state.processing = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Mini AI Agent")

    st.markdown("---")

    # -----------------------------------------------------
    # New Chat
    # -----------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        clear_chat()
        st.rerun()

    st.markdown("---")

    # -----------------------------------------------------
    # Quick Prompts
    # -----------------------------------------------------

    st.subheader("⚡ Quick Questions")

    quick_prompts = [
        "Explain Java OOP in simple language",
        "What is DBMS?",
        "Explain Artificial Intelligence",
        "What is Machine Learning?",
        "Calculate 125 × 48",
        "What is Agile methodology?"
    ]

    for prompt in quick_prompts:

        if st.button(
            prompt,
            use_container_width=True
        ):

            process_question(prompt)
            st.rerun()

    st.markdown("---")

    # -----------------------------------------------------
    # System Status
    # -----------------------------------------------------

    st.subheader("📡 System Status")

    st.markdown(
        """
        <div class="status-box">
        🟢 <b>Vercel API</b><br>
        <span class="small-text">Connected</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="status-box">
        🤖 <b>Gemini AI</b><br>
        <span class="small-text">Online</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.getenv("TAVILY_API_KEY"):

        st.markdown(
            """
            <div class="status-box">
            🌐 <b>Web Search</b><br>
            <span class="small-text">Configured</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-box">
            🌐 <b>Web Search</b><br>
            <span class="small-text">Local key not detected</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Project Information
    # -----------------------------------------------------

    st.subheader("📚 Features")

    st.markdown(
        """
        <div class="feature-box">
        🤖 <b>Gemini AI</b><br>
        <span class="small-text">
        Intelligent AI responses
        </span>
        </div>

        <div class="feature-box">
        🌐 <b>Web Search</b><br>
        <span class="small-text">
        Search current information
        </span>
        </div>

        <div class="feature-box">
        📄 <b>PDF RAG</b><br>
        <span class="small-text">
        Answer questions from documents
        </span>
        </div>

        <div class="feature-box">
        🧮 <b>Calculator</b><br>
        <span class="small-text">
        Perform mathematical calculations
        </span>
        </div>

        <div class="feature-box">
        💬 <b>Chat Memory</b><br>
        <span class="small-text">
        Maintain conversation context
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "Mini AI Agent • CSE-AIML Project"
    )


# =========================================================
# MAIN CHAT AREA
# =========================================================

if not st.session_state.messages:

    st.markdown("## 👋 Hello!")

    st.write(
        "I'm your Mini AI Agent. Ask me anything!"
    )

    st.markdown("---")

    st.subheader("💡 Try asking")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "☕ Explain Java",
            use_container_width=True
        ):

            process_question(
                "Explain Java in simple language"
            )

            st.rerun()

    with col2:

        if st.button(
            "📚 What is DBMS?",
            use_container_width=True
        ):

            process_question(
                "Explain DBMS in simple language"
            )

            st.rerun()

    with col3:

        if st.button(
            "🤖 What is AI?",
            use_container_width=True
        ):

            process_question(
                "What is Artificial Intelligence?"
            )

            st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    role = message.get("role", "assistant")

    if role == "user":

        with st.chat_message("user"):

            st.markdown(
                message.get("content", "")
            )

    else:

        with st.chat_message("assistant"):

            st.markdown(
                message.get("content", "")
            )

            # ---------------------------------------------
            # Tool information
            # ---------------------------------------------

            tool_name = message.get("tool_name")

            if tool_name:

                st.caption(
                    f"🔧 {tool_name}"
                )

            # ---------------------------------------------
            # Source information
            # ---------------------------------------------

            source = message.get("source")

            if source:

                st.caption(
                    f"📌 Source: {source}"
                )

            # ---------------------------------------------
            # Sources
            # ---------------------------------------------

            sources = message.get("sources")

            if sources:

                with st.expander("🔗 Sources"):

                    if isinstance(sources, list):

                        for item in sources:

                            st.write(
                                str(item)
                            )

                    elif isinstance(sources, dict):

                        for key, value in sources.items():

                            st.write(
                                f"**{key}:** {value}"
                            )

                    else:

                        st.write(
                            str(sources)
                        )


# =========================================================
# CHAT INPUT
# =========================================================

chat_input = st.chat_input(
    "Ask Mini AI Agent anything..."
)

if chat_input:

    process_question(chat_input)

    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <span class="small-text">
        🤖 Mini AI Agent | Powered by Gemini AI | 
        Built for CSE-AIML Project
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
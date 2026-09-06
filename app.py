import os
import streamlit as st

from chat_manager import ChatManager
from agent import ask_agent
from voice_transcriber import transcribe_audio
from voice_output import text_to_speech, get_available_voices
from image_analyzer import analyze_image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mini AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #020617 100%
        );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border-radius: 24px;
        background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.95),
            rgba(15,23,42,0.95)
        );
        border: 1px solid
        rgba(255,255,255,0.08);
        box-shadow:
        0 20px 50px
        rgba(0,0,0,0.25);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 10px;
    }

    .tool-card {
        padding: 14px 18px;
        border-radius: 15px;
        background:
        rgba(255,255,255,0.045);
        border:
        1px solid
        rgba(255,255,255,0.08);
        margin-top: 12px;
        margin-bottom: 10px;
    }

    .source-card {
        padding: 14px 18px;
        border-radius: 15px;
        background:
        rgba(255,255,255,0.035);
        border:
        1px solid
        rgba(255,255,255,0.06);
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .status-card {
        padding: 15px;
        border-radius: 15px;
        background:
        rgba(255,255,255,0.04);
        border:
        1px solid
        rgba(255,255,255,0.07);
        text-align: center;
        min-height: 90px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 45px;
        padding: 25px;
        border-top:
        1px solid
        rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #0f172a,
            #020617
        );
    }

    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        transition: 0.2s;
    }

    textarea {
        border-radius: 15px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CHAT MANAGER
# =========================================================

@st.cache_resource
def load_chat_manager():
    return ChatManager()


chat_manager = load_chat_manager()


# =========================================================
# SESSION STATE
# =========================================================

if "chat_id" not in st.session_state:

    st.session_state.chat_id = (
        chat_manager.create_chat(
            "New Chat"
        )
    )


if "voice_enabled" not in st.session_state:

    st.session_state.voice_enabled = True


if "voice_rate" not in st.session_state:

    st.session_state.voice_rate = 0


if "voice_volume" not in st.session_state:

    st.session_state.voice_volume = 100


if "selected_voice" not in st.session_state:

    st.session_state.selected_voice = None


if "quick_prompt" not in st.session_state:

    st.session_state.quick_prompt = ""


if "gemini_status" not in st.session_state:

    st.session_state.gemini_status = "Configured"


# =========================================================
# CURRENT CHAT
# =========================================================

def load_current_chat():

    return chat_manager.get_messages(
        st.session_state.chat_id
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <h1>🤖 Mini AI Agent</h1>

        <p>
        Intelligent AI assistant with
        PDF Knowledge • Web Search • Calculator
        • Voice • Image Analysis • Memory
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Control Panel")


    # -----------------------------------------------------
    # NEW CHAT
    # -----------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        new_chat_id = (
            chat_manager.create_chat(
                "New Chat"
            )
        )

        st.session_state.chat_id = (
            new_chat_id
        )

        st.rerun()


    st.divider()


    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    st.subheader("💬 Chat History")

    chats = chat_manager.get_all_chats()

    if chats:

        for chat in reversed(chats):

            chat_id = chat.get("id")

            title = chat.get(
                "title",
                "New Chat"
            )

            if st.button(
                title,
                key=f"history_{chat_id}",
                use_container_width=True
            ):

                st.session_state.chat_id = (
                    chat_id
                )

                st.rerun()

    else:

        st.caption(
            "No chats yet."
        )


    st.divider()


    # -----------------------------------------------------
    # VOICE SETTINGS
    # -----------------------------------------------------

    st.subheader("🎙️ Voice Settings")


    st.session_state.voice_enabled = (
        st.toggle(
            "🔊 Voice Output",
            value=st.session_state.voice_enabled
        )
    )


    st.session_state.voice_rate = (
        st.slider(
            "Voice Speed",
            min_value=-5,
            max_value=5,
            value=st.session_state.voice_rate
        )
    )


    st.session_state.voice_volume = (
        st.slider(
            "Voice Volume",
            min_value=0,
            max_value=100,
            value=st.session_state.voice_volume
        )
    )


    voices = get_available_voices()


    if voices:

        voice_options = [
            "Default"
        ] + voices

        selected_voice = st.selectbox(
            "🗣️ Windows Voice",
            voice_options
        )

        if selected_voice == "Default":

            st.session_state.selected_voice = (
                None
            )

        else:

            st.session_state.selected_voice = (
                selected_voice
            )

    else:

        st.info(
            "No Windows speech voices detected."
        )


    st.divider()


    # -----------------------------------------------------
    # QUICK PROMPTS
    # -----------------------------------------------------

    st.subheader("⚡ Quick Prompts")


    quick_prompts = [

        "Explain machine learning",

        "Explain inheritance in Java",

        "What is normalization in DBMS?",

        "What is the latest Python version?"

    ]


    for index, prompt in enumerate(
        quick_prompts
    ):

        if st.button(
            prompt,
            key=f"quick_prompt_{index}",
            use_container_width=True
        ):

            st.session_state.quick_prompt = (
                prompt
            )

            st.rerun()


    st.divider()


    # -----------------------------------------------------
    # CHAT MANAGEMENT
    # -----------------------------------------------------

    st.subheader("🗑️ Chat Management")


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "Clear",
            use_container_width=True
        ):

            chat_manager.clear_chat(
                st.session_state.chat_id
            )

            st.rerun()


    with col2:

        if st.button(
            "Delete",
            use_container_width=True
        ):

            chat_manager.delete_chat(
                st.session_state.chat_id
            )

            new_chat_id = (
                chat_manager.create_chat(
                    "New Chat"
                )
            )

            st.session_state.chat_id = (
                new_chat_id
            )

            st.rerun()


# =========================================================
# DISPLAY EXISTING CHAT
# =========================================================

messages = load_current_chat()


for message in messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )


    if role == "user":

        with st.chat_message("user"):

            st.markdown(
                content
            )

    else:

        with st.chat_message("assistant"):

            st.markdown(
                content
            )


# =========================================================
# RESULT DETAILS
# =========================================================

def display_result_details(result):

    tool_name = result.get(
        "tool_name",
        "🤖 Gemini AI"
    )


    source = result.get(
        "source",
        ""
    )


    sources = result.get(
        "sources",
        []
    )


    tool = result.get(
        "tool",
        "general"
    )


    # -----------------------------------------------------
    # TOOL
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="tool-card">

        🧠 <b>Tool Used:</b>
        {tool_name}

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # SOURCE
    # -----------------------------------------------------

    if source:

        st.markdown(
            f"""
            <div class="source-card">

            📄 <b>Source:</b>
            {source}

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # PDF SOURCES
    # -----------------------------------------------------

    if sources and tool in [
        "pdf",
        "pdf_offline"
    ]:

        st.markdown(
            """
            <div class="source-card">

            📚 <b>PDF Sources</b>

            </div>
            """,
            unsafe_allow_html=True
        )


        for pdf_source in sources:

            st.markdown(
                f"📄 `{pdf_source}`"
            )


    # -----------------------------------------------------
    # WEB SOURCES
    # -----------------------------------------------------

    elif sources and tool == "web":

        st.markdown(
            """
            <div class="source-card">

            🌐 <b>Web Sources</b>

            </div>
            """,
            unsafe_allow_html=True
        )


        for url in sources:

            st.markdown(
                f"- {url}"
            )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(question):

    question = (
        question or ""
    ).strip()


    if not question:

        return


    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # ASSISTANT
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Thinking..."
        ):

            result = ask_agent(
                question,
                st.session_state.chat_id
            )


        answer = result.get(
            "answer",
            "No answer returned."
        )


        st.markdown(
            answer
        )


        display_result_details(
            result
        )


        # -------------------------------------------------
        # UPDATE GEMINI STATUS
        # -------------------------------------------------

        answer_lower = answer.lower()


        if (
            "quota exhausted" in answer_lower
            or "resource_exhausted" in answer_lower
        ):

            st.session_state.gemini_status = (
                "Quota Limited"
            )

        elif (
            "gemini api error" in answer_lower
            or "gemini api key problem" in answer_lower
        ):

            st.session_state.gemini_status = (
                "Error"
            )

        else:

            st.session_state.gemini_status = (
                "Online"
            )


        # -------------------------------------------------
        # VOICE OUTPUT
        # -------------------------------------------------

        if st.session_state.voice_enabled:

            with st.spinner(
                "🔊 Generating voice..."
            ):

                success, audio_result = (
                    text_to_speech(
                        answer,
                        rate=st.session_state.voice_rate,
                        volume=st.session_state.voice_volume,
                        voice_name=st.session_state.selected_voice
                    )
                )


            if success:

                st.audio(
                    audio_result,
                    format="audio/wav"
                )

            else:

                st.warning(
                    f"Voice output failed: {audio_result}"
                )


    # -----------------------------------------------------
    # UPDATE CHAT ID
    # -----------------------------------------------------

    new_chat_id = result.get(
        "chat_id"
    )


    if new_chat_id:

        st.session_state.chat_id = (
            new_chat_id
        )


# =========================================================
# TEXT CHAT INPUT
# =========================================================

quick_prompt = (
    st.session_state.quick_prompt
)

st.session_state.quick_prompt = ""


chat_input = st.chat_input(
    "Ask Mini AI Agent anything..."
)


if chat_input:

    process_question(
        chat_input
    )

elif quick_prompt:

    process_question(
        quick_prompt
    )


# =========================================================
# VOICE INPUT
# =========================================================

st.divider()

st.subheader(
    "🎙️ Voice Input"
)


audio_value = st.audio_input(
    "Speak to Mini AI Agent"
)


if audio_value:

    with st.spinner(
        "🎧 Understanding your voice..."
    ):

        try:

            audio_bytes = (
                audio_value.getvalue()
            )

            transcript = transcribe_audio(
                audio_bytes,
                "wav"
            )

        except Exception as error:

            transcript = (
                "❌ Voice processing failed: "
                + str(error)
            )


    if transcript.startswith("❌"):

        st.error(
            transcript
        )

    else:

        st.success(
            f"🗣️ You said: {transcript}"
        )

        process_question(
            transcript
        )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

st.divider()

st.subheader(
    "🖼️ Image Analysis"
)


uploaded_image = st.file_uploader(
    "Upload an image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    key="image_uploader"
)


image_question = st.text_input(
    "What do you want to know about the image?",
    value="Describe this image.",
    key="image_question"
)


if uploaded_image:

    st.image(
        uploaded_image,
        caption="Uploaded Image",
        use_container_width=True
    )


    if st.button(
        "🔍 Analyze Image",
        use_container_width=True
    ):

        with st.spinner(
            "🔎 Analyzing image..."
        ):

            try:

                image_bytes = (
                    uploaded_image.getvalue()
                )


                image_result = analyze_image(
                    image_bytes,
                    uploaded_image.type,
                    image_question
                )


            except Exception as error:

                image_result = (
                    "❌ Image analysis failed:\n"
                    + str(error)
                )


        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                image_result
            )


            st.markdown(
                """
                <div class="tool-card">

                🧠 <b>Tool Used:</b>
                🖼️ Image Analysis

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# SYSTEM STATUS
# =========================================================

st.divider()

st.subheader(
    "📊 System Status"
)


col1, col2, col3, col4 = (
    st.columns(4)
)


# ---------------------------------------------------------
# GEMINI STATUS
# ---------------------------------------------------------

with col1:

    gemini_status = (
        st.session_state.gemini_status
    )

    st.markdown(
        f"""
        <div class="status-card">

        🤖<br>

        <b>Gemini AI</b><br>

        {gemini_status}

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PDF STATUS
# ---------------------------------------------------------

with col2:

    pdf_status = (
        "Ready"
        if os.path.exists(
            "data/all_vectors.pkl"
        )
        else "Not Indexed"
    )


    st.markdown(
        f"""
        <div class="status-card">

        📚<br>

        <b>PDF Knowledge</b><br>

        {pdf_status}

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# WEB STATUS
# ---------------------------------------------------------

with col3:

    web_status = (
        "Ready"
        if os.getenv(
            "TAVILY_API_KEY"
        )
        else "Not Configured"
    )


    st.markdown(
        f"""
        <div class="status-card">

        🌐<br>

        <b>Web Search</b><br>

        {web_status}

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# VOICE STATUS
# ---------------------------------------------------------

with col4:

    voice_status = (
        "Ready"
        if voices
        else "Unavailable"
    )


    st.markdown(
        f"""
        <div class="status-card">

        🎙️<br>

        <b>Voice System</b><br>

        {voice_status}

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🤖 <b>Mini AI Agent</b>

        <br><br>

        PDF Knowledge • Smart Routing •
        Web Search • Calculator •
        Voice • Image Analysis • Memory

        <br><br>

        Mini AI Agent v7.0

    </div>
    """,
    unsafe_allow_html=True
)
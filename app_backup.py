import os
import time

import streamlit as st

import agent
from agent import ask_agent

from chat_manager import ChatManager

from pdf_manager import (
    index_single_pdf,
    list_pdfs,
    delete_pdf,
    rebuild_knowledge_base
)

from image_analyzer import analyze_image

from voice_transcriber import (
    transcribe_audio
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mini AI Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Mini AI Agent")

st.caption(
    "Gemini AI • PDF RAG • Web Search • Calculator • "
    "Memory • Voice • Image"
)


# ============================================================
# CHAT MANAGER
# ============================================================

@st.cache_resource
def get_chat_manager():

    return ChatManager()


chat_manager = get_chat_manager()


# ============================================================
# SESSION STATE
# ============================================================

if "current_chat_id" not in st.session_state:

    chats = chat_manager.get_all_chats()

    if chats:

        st.session_state.current_chat_id = (
            chats[-1]["id"]
        )

    else:

        st.session_state.current_chat_id = (
            chat_manager.create_chat(
                "New Chat"
            )
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("💬 Chats")


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        chat_id = chat_manager.create_chat(
            "New Chat"
        )

        st.session_state.current_chat_id = (
            chat_id
        )

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # CHAT LIST
    # --------------------------------------------------------

    chats = chat_manager.get_all_chats()

    for chat in reversed(chats):

        chat_id = chat["id"]

        title = chat.get(
            "title",
            "New Chat"
        )

        if len(title) > 25:

            title = title[:25] + "..."

        is_current = (
            chat_id
            ==
            st.session_state.current_chat_id
        )

        prefix = "👉 " if is_current else ""

        if st.button(
            prefix + "💬 " + title,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state.current_chat_id = (
                chat_id
            )

            st.rerun()


    st.divider()


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.header("📚 Knowledge Base")

    pdfs = list_pdfs()

    if not pdfs:

        st.info(
            "No PDFs uploaded."
        )

    else:

        for pdf in pdfs:

            filename = pdf["filename"]

            status = pdf["status"]

            chunks = pdf.get(
                "chunks",
                0
            )


            if status == "indexed":

                icon = "🟢"

            elif status == "changed":

                icon = "🟡"

            elif status == "not_indexed":

                icon = "⚪"

            else:

                icon = "🔴"


            st.write(
                f"{icon} **{filename}**"
            )

            st.caption(
                f"Status: {status} | Chunks: {chunks}"
            )


            if st.button(
                "🗑️ Delete",
                key=f"delete_{filename}",
                use_container_width=True
            ):

                with st.spinner(
                    "Deleting PDF..."
                ):

                    success = delete_pdf(
                        filename
                    )


                if success:

                    agent.embedding_model = None

                    agent.vector_store = None

                    st.success(
                        "PDF deleted successfully."
                    )

                    time.sleep(0.5)

                    st.rerun()

                else:

                    st.error(
                        "Could not delete PDF."
                    )


    st.divider()


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    st.subheader("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="pdf_uploader"
    )


    if uploaded_file is not None:

        if st.button(
            "➕ Add PDF",
            use_container_width=True
        ):

            os.makedirs(
                "documents",
                exist_ok=True
            )


            file_path = os.path.join(
                "documents",
                uploaded_file.name
            )


            try:

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )


                with st.spinner(
                    "Indexing PDF..."
                ):

                    success = index_single_pdf(
                        file_path
                    )


                if success:

                    agent.embedding_model = None

                    agent.vector_store = None

                    st.success(
                        "PDF added successfully!"
                    )

                    time.sleep(1)

                    st.rerun()

                else:

                    st.error(
                        "PDF indexing failed."
                    )


            except Exception as e:

                st.error(
                    f"PDF upload error: {e}"
                )


    # ========================================================
    # REBUILD
    # ========================================================

    st.divider()


    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True
    ):

        with st.spinner(
            "Rebuilding Knowledge Base..."
        ):

            success = (
                rebuild_knowledge_base()
            )


        if success:

            agent.embedding_model = None

            agent.vector_store = None

            st.success(
                "Knowledge Base rebuilt!"
            )

            time.sleep(1)

            st.rerun()

        else:

            st.error(
                "Knowledge Base rebuild failed."
            )


    st.divider()


    st.caption(
        "Mini AI Agent v2.0"
    )


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat_id = (
    st.session_state.current_chat_id
)


current_chat = chat_manager.get_chat(
    current_chat_id
)


# ============================================================
# CHAT HEADER
# ============================================================

if current_chat:

    st.subheader(
        f"💬 {current_chat.get('title', 'New Chat')}"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

messages = chat_manager.get_messages(
    current_chat_id
)


for message in messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )


    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )


# ============================================================
# INPUT TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "💬 Text",
        "🎤 Voice",
        "📷 Image"
    ]
)


# ============================================================
# TEXT TAB
# ============================================================

with tab1:

    question = st.chat_input(
        "Ask anything..."
    )


    if question:

        question = question.strip()


        if question:

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )


            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    answer = ask_agent(
                        question,
                        current_chat_id
                    )


                st.markdown(
                    answer
                )


            st.rerun()


# ============================================================
# VOICE TAB
# ============================================================

with tab2:

    st.subheader(
        "🎤 Ask using your voice"
    )


    audio = st.audio_input(
        "Record your question"
    )


    if audio is not None:

        st.audio(
            audio
        )


        if st.button(
            "🧠 Transcribe & Ask",
            key="voice_button",
            use_container_width=True
        ):

            with st.spinner(
                "Understanding your voice..."
            ):

                audio_bytes = audio.getvalue()

                question = transcribe_audio(
                    audio_bytes,
                    "wav"
                )


            if question.startswith(
                "❌"
            ):

                st.error(
                    question
                )

            else:

                st.success(
                    "Voice understood!"
                )


                st.write(
                    "🗣️ You said:"
                )


                st.info(
                    question
                )


                with st.spinner(
                    "Thinking..."
                ):

                    answer = ask_agent(
                        question,
                        current_chat_id
                    )


                st.markdown(
                    "### 🤖 AI Answer"
                )


                st.markdown(
                    answer
                )


                st.rerun()


# ============================================================
# IMAGE TAB
# ============================================================

with tab3:

    st.subheader(
        "📷 Ask about an image"
    )


    image_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_uploader"
    )


    image_question = st.text_input(
        "What do you want to know about this image?",
        value="Describe this image."
    )


    if image_file is not None:

        st.image(
            image_file,
            caption="Uploaded Image",
            use_container_width=True
        )


        if st.button(
            "🔍 Analyze Image",
            key="image_button",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                image_bytes = (
                    image_file.getvalue()
                )

                mime_type = (
                    image_file.type
                )


                answer = analyze_image(
                    image_bytes,
                    mime_type,
                    image_question
                )


            st.markdown(
                "### 🤖 AI Analysis"
            )


            st.markdown(
                answer
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🤖 Mini AI Agent • "
    "Gemini + RAG + Web + Calculator + "
    "Memory + Voice + Vision"
)
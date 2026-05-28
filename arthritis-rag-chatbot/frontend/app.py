import streamlit as st
import sys
import os

# ==========================
# PATH FIX
# ==========================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from backend.rag_engine import (
    ask_question
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title=
    "Medical Research Assistant",

    page_icon="🩺",

    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.title(
        "🩺 Arthritis RAG"
    )

    st.write(
        "AI-powered medical "
        "research assistant "
        "for arthritis and "
        "bone disorders."
    )

    st.markdown(
        "### Supported Topics"
    )

    st.markdown("""
    - Osteoarthritis
    - Rheumatoid Arthritis
    - Osteoporosis
    - Biomarkers
    - Treatments
    - Bone Health
    """)

    st.divider()

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()

# ==========================
# TITLE
# ==========================

st.title(
    "🩺 Arthritis Medical Research Assistant"
)

st.caption(
    "Research-based answers "
    "for arthritis, osteoporosis, "
    "rheumatoid arthritis and more."
)

# ==========================
# SESSION MEMORY
# ==========================

if "messages" not in (
    st.session_state
):

    st.session_state.messages = []

# ==========================
# WELCOME SCREEN
# ==========================

if len(
    st.session_state.messages
) == 0:

    st.info("""
👋 Welcome!

Ask research-based medical questions such as:

• What are biomarkers for osteoporosis?  
• Tell me about rheumatoid arthritis  
• Can treatment slow cartilage degeneration?  
• What causes osteoarthritis?  
""")

# ==========================
# DISPLAY CHAT HISTORY
# ==========================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ==========================
# USER INPUT
# ==========================

user_prompt = st.chat_input(
    "Ask a medical question..."
)

if user_prompt:

    # ----------------------
    # Save user message
    # ----------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    # ----------------------
    # Show user message
    # ----------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_prompt
        )

    # ----------------------
    # Assistant Response
    # ----------------------

    response = None

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Analyzing medical research..."
        ):

            try:

                response = ask_question(
                    user_prompt
                )

                st.markdown(
                    response
                )

            except Exception:

                st.error(
                    "Medical assistant is temporarily unavailable."
                )

    # ----------------------
    # Save Assistant Response
    # ----------------------

    if response:

        st.session_state.messages.append({
            "role":
                "assistant",

            "content":
                response
        })

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "Built by Chirag Kaushik • "
)
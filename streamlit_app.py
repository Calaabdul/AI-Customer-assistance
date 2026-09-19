import os
import html

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="SupportAI",
    page_icon="✦",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    .stApp {
        background: #f7f8fc;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 1.5rem;
    }

    .navbar {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 55px;
    }

    .brand {
        font-size: 21px;
        font-weight: 700;
        color: #111827;
    }

    .brand span {
        color: #6366f1;
    }

    .status {
        font-size: 13px;
        color: #6b7280;
    }

    .hero {
        text-align: center;
        margin-bottom: 35px;
    }

    .hero h1 {
        font-size: 45px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 12px;
    }

    .hero p {
        color: #6b7280;
        font-size: 16px;
    }

    .panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 28px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
    }

    .panel-title {
        font-size: 18px;
        font-weight: 650;
        color: #111827;
    }

    .panel-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .response {
        background: #111827;
        color: white;
        border-radius: 18px;
        padding: 26px;
        margin-top: 25px;
    }

    .response-label {
        color: #a5b4fc;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .response-body {
        font-size: 16px;
        line-height: 1.7;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        font-size: 15px;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVBAR
# ============================================================

st.markdown(
    """
    <div class="navbar">
        <div class="brand">
            Support<span>AI</span>
        </div>

        <div class="status">
            AI Customer Support
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Resolve customer issues faster.</h1>

        <p>
            Send a customer message and let the AI
            analyze and respond to it.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    """
    <div class="panel">
        <div class="panel-title">
            Customer message
        </div>

        <div class="panel-subtitle">
            Describe the customer's issue.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


message = st.text_area(
    "Customer message",
    placeholder=(
        "Example: I was charged twice for my subscription."
    ),
    height=170,
    label_visibility="collapsed",
)


if st.button(
    "Generate response",
    type="primary",
):

    if not message.strip():

        st.warning(
            "Please enter a customer message."
        )

    else:

        with st.spinner(
            "Analyzing message..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": message
                    },
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                final_response = html.escape(
                    data["response"]
                )

                st.markdown(
                    f"""
                    <div class="response">

                        <div class="response-label">
                            AI Response
                        </div>

                        <div class="response-body">
                            {final_response}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except requests.RequestException as error:

                st.error(
                    f"API connection failed: {error}"
                )


st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:55px;
        color:#9ca3af;
        font-size:13px;
    ">
        LangChain · LangGraph · Groq
    </div>
    """,
    unsafe_allow_html=True,
)
import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DEFENCE CHATBOT | TACTICAL C4ISR UPLINK",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# ==================  BACKEND  (DO NOT MODIFY)  ==============
# ============================================================
# Everything in this block is your original chatbot logic:
# env/API setup, embeddings, vector DB, retrieval, prompt,
# and the NVIDIA completion call. Nothing here has changed.
# ============================================================

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("NVIDIA_API_KEY not found in .env")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


@st.cache_resource
def load_database():
    return Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )


db = load_database()


def ask_question(query: str):
    """
    Unchanged backend pipeline: retrieve -> build context ->
    build prompt -> call NVIDIA model -> return (answer, results).
    This is exactly the logic that lived inline in your original
    'if query:' block — only wrapped in a function so the new
    frontend can call it cleanly. No retrieval/prompt/model
    parameters were altered.
    """

    # Retrieve
    results = db.similarity_search(
        query,
        k=3
    )

    # Context
    context_parts = []

    for i, document in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")
        context_parts.append(
            f"""
SOURCE {i}: {source}

{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    # Prompt
    prompt = f"""
You are Defense-RAG, an AI assistant for defense and
security information.

Answer the user's question using ONLY the retrieved
knowledge base.

RULES:

1. Use only the provided context.
2. Do not invent facts.
3. If the answer is not available, say:
   "I could not find this information in the knowledge base."
4. Give a concise and clear answer.
5. Do not provide harmful operational instructions.
6. Mention relevant sources when possible.

KNOWLEDGE BASE:

{context}

USER QUESTION:

{query}
"""

    # Generate
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual defense "
                    "knowledge assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    answer = completion.choices[0].message.content

    return answer, results


# ============================================================
# ==================  FRONTEND  (REDESIGNED)  ================
# ============================================================


# ------------------------------------------------------------
# SESSION STATE  (names unchanged: messages, page, last_sources)
# ------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page" not in st.session_state:
    st.session_state.page = "landing"

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


def go_to_chat():
    st.session_state.page = "chat"


def go_to_landing():
    st.session_state.page = "landing"


def clear_chat():
    st.session_state.messages = []
    st.session_state.last_sources = []


def extract_source_briefs(results):
    """
    Frontend-only helper: turns raw LangChain Document results
    into small serializable dicts so they can be attached to a
    chat message and rendered later. Does not touch ask_question()
    or any retrieval/prompt logic.
    """
    briefs = []
    for i, document in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")
        snippet = document.page_content.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."
        briefs.append({
            "index": i,
            "source": source,
            "snippet": snippet
        })
    return briefs


# ------------------------------------------------------------
# GLOBAL CSS — TACTICAL C4ISR HUD THEME
# ------------------------------------------------------------

def inject_css(page: str):

    if page == "landing":
        app_background = """
            background:
                radial-gradient(circle at 50% 35%, rgba(57,255,20,0.06), transparent 55%),
                radial-gradient(circle at 15% 85%, rgba(255,176,0,0.05), transparent 45%),
                linear-gradient(180deg, #050706 0%, #0b1410 55%, #06100b 100%);
        """
    else:
        app_background = """
            background:
                radial-gradient(circle at 12% 8%, rgba(57,255,20,0.05), transparent 32%),
                radial-gradient(circle at 88% 18%, rgba(255,176,0,0.04), transparent 30%),
                linear-gradient(180deg, #050706 0%, #0a0f0c 100%);
        """

    st.markdown(
        f"""
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&family=Inter:wght@400;500;600;700&display=swap');

        /* ---------- HIDE DEFAULT STREAMLIT CHROME ---------- */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
        [data-testid="stDecoration"] {{ display: none; }}
        [data-testid="stHeader"] {{ background: transparent; }}

        :root {{
            --phosphor: #39ff14;
            --amber: #ffb000;
            --gunmetal: #1b211d;
            --obsidian: #050706;
            --olive: #0b1410;
        }}

        * {{
            font-family: 'Inter', sans-serif;
        }}

        .mono, .mono * {{
            font-family: 'Share Tech Mono', monospace !important;
        }}

        html, body, .stApp {{
            {app_background}
            color: #cfe8d4;
        }}

        /* ---------- SCANLINES + VIGNETTE OVERLAY ---------- */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            z-index: 5;
            background-image: repeating-linear-gradient(
                0deg,
                rgba(57,255,20,0.028) 0px,
                rgba(57,255,20,0.028) 1px,
                transparent 2px,
                transparent 4px
            );
            mix-blend-mode: overlay;
            animation: scanshift 9s linear infinite;
        }}
        @keyframes scanshift {{
            0% {{ background-position-y: 0px; }}
            100% {{ background-position-y: 400px; }}
        }}

        .stApp::after {{
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            z-index: 4;
            box-shadow: inset 0 0 220px rgba(0,0,0,0.85);
        }}

        .grid-bg {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            z-index: 0;
            background-image:
                linear-gradient(rgba(57,255,20,0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(57,255,20,0.05) 1px, transparent 1px);
            background-size: 46px 46px;
        }}

        .block-container {{
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 6rem;
            position: relative;
            z-index: 1;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #060a08 0%, #0c1712 100%);
            border-right: 1px solid rgba(57,255,20,0.20);
        }}
        section[data-testid="stSidebar"] * {{ color: #b9d9bd; }}

        /* ---------- BUTTONS ---------- */
        .stButton > button {{
            border-radius: 3px;
            border: 1px solid rgba(57,255,20,0.45);
            background: rgba(11, 20, 16, 0.9);
            color: var(--phosphor);
            font-family: 'Share Tech Mono', monospace;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            font-size: 13px;
            transition: 0.18s;
            padding: 0.6rem 1rem;
            box-shadow: 0 0 0 rgba(57,255,20,0);
        }}
        .stButton > button:hover {{
            border-color: var(--phosphor);
            background: rgba(57,255,20,0.10);
            box-shadow: 0 0 18px rgba(57,255,20,0.45), inset 0 0 12px rgba(57,255,20,0.15);
            color: #eaffef;
        }}
        .stButton > button:active {{
            transform: translateY(1px);
        }}

        /* ---------- LANDING PAGE ---------- */
        .landing-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 78vh;
            text-align: center;
            position: relative;
        }}

        .crosshair {{
            position: absolute;
            top: 50%; left: 50%;
            width: 460px; height: 460px;
            margin-left: -230px; margin-top: -290px;
            border: 1px solid rgba(57,255,20,0.22);
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
            animation: spin 14s linear infinite;
        }}
        .crosshair::before, .crosshair::after {{
            content: "";
            position: absolute;
            background: rgba(57,255,20,0.28);
        }}
        .crosshair::before {{ top: 50%; left: -14px; right: -14px; height: 1px; }}
        .crosshair::after {{ left: 50%; top: -14px; bottom: -14px; width: 1px; }}
        @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}

        .radar-sweep {{
            position: absolute;
            top: 50%; left: 50%;
            width: 460px; height: 460px;
            margin-left: -230px; margin-top: -290px;
            border-radius: 50%;
            background: conic-gradient(from 0deg, rgba(57,255,20,0.30), transparent 22%);
            animation: sweep 3.4s linear infinite;
            pointer-events: none;
            z-index: 0;
        }}
        @keyframes sweep {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}

        .ring {{
            position: absolute;
            top: 50%; left: 50%;
            border-radius: 50%;
            border: 1px solid rgba(57,255,20,0.14);
            pointer-events: none;
            z-index: 0;
        }}
        .ring.r1 {{ width: 320px; height: 320px; margin-left: -160px; margin-top: -250px; }}
        .ring.r2 {{ width: 200px; height: 200px; margin-left: -100px; margin-top: -190px; }}

        .dossier {{
            position: relative;
            z-index: 1;
            padding: 44px 54px;
            border-radius: 4px;
            background: rgba(6, 12, 9, 0.72);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border: 1px solid rgba(57,255,20,0.35);
            box-shadow:
                0 0 40px rgba(57,255,20,0.10),
                0 25px 70px rgba(0,0,0,0.6);
            max-width: 680px;
        }}
        .dossier-corner {{
            position: absolute;
            width: 22px; height: 22px;
            border-color: var(--phosphor);
        }}
        .dossier-corner.tl {{ top: -1px; left: -1px; border-top: 2px solid; border-left: 2px solid; }}
        .dossier-corner.tr {{ top: -1px; right: -1px; border-top: 2px solid; border-right: 2px solid; }}
        .dossier-corner.bl {{ bottom: -1px; left: -1px; border-bottom: 2px solid; border-left: 2px solid; }}
        .dossier-corner.br {{ bottom: -1px; right: -1px; border-bottom: 2px solid; border-right: 2px solid; }}

        .clearance-tag {{
            display: inline-block;
            font-family: 'Share Tech Mono', monospace;
            font-size: 11px;
            letter-spacing: 3px;
            color: var(--amber);
            border: 1px solid rgba(255,176,0,0.5);
            padding: 4px 12px;
            border-radius: 2px;
            margin-bottom: 18px;
            background: rgba(255,176,0,0.06);
        }}

        .landing-icon {{ font-size: 46px; margin-bottom: 8px; filter: drop-shadow(0 0 12px rgba(57,255,20,0.5)); }}

        .landing-title {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 44px;
            font-weight: 700;
            letter-spacing: 6px;
            margin: 4px 0 12px 0;
            color: var(--phosphor);
            text-shadow: 0 0 18px rgba(57,255,20,0.55), 0 0 42px rgba(57,255,20,0.25);
        }}

        .landing-subtitle {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 14px;
            color: #b9d9bd;
            letter-spacing: 1.5px;
            margin-bottom: 22px;
            text-transform: uppercase;
        }}

        .uplink-status {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 12.5px;
            color: var(--phosphor);
            letter-spacing: 1px;
            margin-bottom: 26px;
            opacity: 0.9;
        }}
        .uplink-status .blink-cursor {{
            display: inline-block;
            width: 8px;
            background: var(--phosphor);
            margin-left: 3px;
            animation: blink 1s steps(1) infinite;
        }}
        @keyframes blink {{ 50% {{ opacity: 0; }} }}

        .landing-badges {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 24px;
            flex-wrap: wrap;
        }}
        .mini-badge {{
            font-family: 'Share Tech Mono', monospace;
            padding: 5px 12px;
            border-radius: 2px;
            background: rgba(57,255,20,0.06);
            border: 1px solid rgba(57,255,20,0.28);
            font-size: 10.5px;
            color: #bdf2c4;
            letter-spacing: 1px;
        }}
        .mini-badge.amber {{
            border-color: rgba(255,176,0,0.35);
            color: #ffd487;
            background: rgba(255,176,0,0.05);
        }}

        /* ---------- CHAT PAGE ---------- */

        .chat-topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 22px;
            border-radius: 3px;
            background: rgba(9, 16, 13, 0.9);
            border: 1px solid rgba(57,255,20,0.30);
            box-shadow: 0 0 24px rgba(57,255,20,0.06), 0 16px 40px rgba(0,0,0,0.4);
            margin-bottom: 18px;
        }}
        .chat-topbar-title {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 19px;
            font-weight: 700;
            letter-spacing: 2px;
            color: var(--phosphor);
            text-shadow: 0 0 10px rgba(57,255,20,0.4);
        }}
        .chat-topbar-sub {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 11.5px;
            color: #9fc7a5;
            margin-top: 4px;
            letter-spacing: 0.5px;
        }}
        .status-dot {{
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--phosphor);
            margin-right: 6px;
            box-shadow: 0 0 8px var(--phosphor);
            animation: blink 1.4s infinite;
        }}

        .telemetry-box {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 11px;
            color: #9fc7a5;
            background: rgba(57,255,20,0.04);
            border: 1px solid rgba(57,255,20,0.18);
            border-radius: 3px;
            padding: 10px 12px;
            margin: 6px 0;
            line-height: 1.9;
        }}
        .telemetry-box b {{ color: var(--phosphor); }}
        .sidebar-status {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 12px;
            letter-spacing: 1px;
            color: var(--phosphor);
            margin: 10px 0;
        }}

        .chat-scroll {{
            padding: 6px 4px 10px 4px;
        }}

        .msg-row {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin: 16px 0;
        }}
        .msg-row.user {{ flex-direction: row-reverse; }}

        .avatar {{
            flex-shrink: 0;
            width: 38px; height: 38px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 17px;
            border: 1px solid rgba(255,255,255,0.16);
        }}
        .avatar.ai {{
            background: rgba(57,255,20,0.10);
            border-color: rgba(57,255,20,0.4);
        }}
        .avatar.user {{
            background: rgba(80,120,160,0.18);
            border-color: rgba(120,160,200,0.4);
        }}

        .bubble-wrap {{
            max-width: 74%;
            position: relative;
        }}

        .bubble {{
            padding: 15px 18px;
            border-radius: 2px;
            line-height: 1.6;
            font-size: 14.5px;
            font-family: 'Inter', sans-serif;
            color: #e7f3ea;
            position: relative;
        }}
        .bubble.ai {{
            background: rgba(9, 22, 15, 0.85);
            border: 1px solid rgba(57,255,20,0.32);
            box-shadow: 0 0 18px rgba(57,255,20,0.06);
        }}
        .bubble.user {{
            background: rgba(16, 24, 33, 0.9);
            border: 1px solid rgba(110,150,190,0.35);
        }}

        /* HUD targeting corner accents on AI bubble */
        .bubble.ai::before, .bubble.ai::after,
        .hud-corner-tr, .hud-corner-bl {{
            content: "";
            position: absolute;
            width: 12px; height: 12px;
            border-color: var(--phosphor);
        }}
        .bubble.ai::before {{ top: -1px; left: -1px; border-top: 2px solid; border-left: 2px solid; }}
        .bubble.ai::after {{ bottom: -1px; right: -1px; border-bottom: 2px solid; border-right: 2px solid; }}

        .bubble-name {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .bubble-name.ai {{ color: var(--phosphor); }}
        .bubble-name.user {{ color: #9dc0e0; }}

        .typing-line {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 13px;
            color: var(--phosphor);
            letter-spacing: 1px;
        }}
        .typing-line .cursor {{
            display: inline-block;
            width: 8px;
            background: var(--phosphor);
            margin-left: 4px;
            animation: blink 0.9s steps(1) infinite;
        }}

        .empty-state {{
            text-align: center;
            padding: 46px 20px;
            color: #6f8a75;
            font-family: 'Share Tech Mono', monospace;
            letter-spacing: 1px;
        }}
        .empty-state-icon {{ font-size: 40px; margin-bottom: 12px; filter: drop-shadow(0 0 10px rgba(57,255,20,0.4)); }}

        /* ---------- SOURCES / CLASSIFIED REFERENCES ---------- */
        div[data-testid="stExpander"] {{
            border: 1px dashed rgba(255,176,0,0.35) !important;
            background: rgba(255,176,0,0.03) !important;
            border-radius: 2px !important;
            margin-top: 6px;
        }}
        div[data-testid="stExpander"] summary {{
            font-family: 'Share Tech Mono', monospace !important;
            font-size: 11.5px !important;
            letter-spacing: 1.5px !important;
            color: var(--amber) !important;
            text-transform: uppercase;
        }}
        .ref-item {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 11.5px;
            color: #d9c58f;
            border-left: 2px solid rgba(255,176,0,0.4);
            padding: 6px 10px;
            margin: 6px 0;
            background: rgba(255,176,0,0.03);
        }}
        .ref-tag {{ color: var(--amber); font-weight: 700; }}

        textarea {{
            background: #060b08 !important;
            color: #d9f5dd !important;
            border: 1px solid rgba(57,255,20,0.35) !important;
            font-family: 'Share Tech Mono', monospace !important;
        }}

        </style>
        <div class="grid-bg"></div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# LANDING PAGE — "SECURE BOOT SEQUENCE"
# ------------------------------------------------------------

def render_landing():

    inject_css("landing")

    st.markdown(
        """
        <div class="landing-wrap">
            <div class="ring r1"></div>
            <div class="ring r2"></div>
            <div class="radar-sweep"></div>
            <div class="crosshair"></div>

            <div class="dossier">
                <div class="dossier-corner tl"></div>
                <div class="dossier-corner tr"></div>
                <div class="dossier-corner bl"></div>
                <div class="dossier-corner br"></div>

                <div class="clearance-tag">TOP SECRET // NOFORN // C4ISR</div>
                <div class="landing-icon">🛡️</div>
                <div class="landing-title">DEFENCE CHATBOT</div>
                <div class="landing-subtitle">AI-Powered Defence Knowledge Assistant</div>
                <div class="uplink-status">
                    &gt; ENCRYPTED CONNECTION ESTABLISHED<span class="blink-cursor">&nbsp;</span>
                </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("🎯 INITIALIZE UPLINK", use_container_width=True, type="primary"):
            go_to_chat()
            st.rerun()

    st.markdown(
        """
                <div class="landing-badges">
                    <div class="mini-badge">🟢 SYSTEM ONLINE</div>
                    <div class="mini-badge amber">🔒 SECURE CHANNEL</div>
                    <div class="mini-badge">📡 KNOWLEDGE-GROUNDED</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; color:#6f8a75; font-family:'Share Tech Mono',monospace; font-size:11px; margin-top:26px; letter-spacing:1.5px; position:relative; z-index:1;">
            BUILT FOR AI HACKATHON &nbsp;//&nbsp; KNOWLEDGE-GROUNDED RESPONSES ONLY
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# CHAT PAGE — "COMMS LOG"
# ------------------------------------------------------------

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📁 CLASSIFIED REFERENCES ({len(sources)})", expanded=False):
        for item in sources:
            st.markdown(
                f"""
                <div class="ref-item">
                    <span class="ref-tag">REF-{item['index']:02d}</span> &nbsp;|&nbsp; {item['source']}<br>
                    <span style="opacity:0.75;">{item['snippet']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_message(message):

    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(
            f"""
            <div class="msg-row user">
                <div class="avatar user">🧑‍✈️</div>
                <div class="bubble-wrap">
                    <div class="bubble user">
                        <div class="bubble-name user">OPERATOR</div>
                        {content}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="msg-row ai">
                <div class="avatar ai">🛡️</div>
                <div class="bubble-wrap">
                    <div class="bubble ai">
                        <div class="bubble-name ai">DEFENCE INTEL SYSTEM</div>
                        {content}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        render_sources(message.get("sources"))


def render_chat():

    inject_css("chat")

    # ---- Sidebar: Mission Control Panel ----
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:15px 0 16px 0;">
                <div style="font-size:42px; filter: drop-shadow(0 0 10px rgba(57,255,20,0.5));">🛡️</div>
                <div class="mono" style="font-size:16px; font-weight:700; letter-spacing:2px; color:#39ff14;">DEFENCE CHATBOT</div>
                <div class="mono" style="color:#6f8a75; font-size:10.5px; margin-top:5px; letter-spacing:1px;">MISSION CONTROL PANEL</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="sidebar-status"><span class="status-dot"></span>STATUS: SECURE</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="telemetry-box">
                NODE ID: <b>DFC-ALPHA-07</b><br>
                UPLINK: <b>STABLE</b><br>
                ENCRYPTION: <b>AES-256</b><br>
                LATENCY: <b>~ 120ms</b><br>
                THREAT LEVEL: <b>NOMINAL</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        if st.button("← Back to Home", use_container_width=True):
            go_to_landing()
            st.rerun()

        if st.button("🗑️ ABORT / CLEAR CHAT", use_container_width=True):
            clear_chat()
            st.rerun()

        st.divider()
        st.markdown(
            """
            <div class="mono" style="font-size:11px; color:#6f8a75; line-height:1.7; letter-spacing:0.5px;">
                ⚠ ADVISORY: This system provides information
                from its configured knowledge base only. Not a
                substitute for official defence or government
                sources.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---- Top bar: Secure Transmission Header ----
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.markdown(
            """
            <div class="chat-topbar">
                <div>
                    <div class="chat-topbar-title">🛡️ DEFENCE CHATBOT // COMMS LOG</div>
                    <div class="chat-topbar-sub"><span class="status-dot"></span>NODE ACTIVE &nbsp;//&nbsp; AES-256 ENCRYPTED &nbsp;//&nbsp; AI-POWERED DEFENCE KNOWLEDGE ASSISTANT</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with top_r:
        st.write("")
        if st.button("← Home", use_container_width=True):
            go_to_landing()
            st.rerun()

    # ---- Conversation ----
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

        if len(st.session_state.messages) == 0:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">🎖️</div>
                    <div>&gt; AWAITING OPERATOR INPUT...<br>QUERY THE KNOWLEDGE BASE TO BEGIN TRANSMISSION.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for message in st.session_state.messages:
                render_message(message)

        typing_placeholder = st.empty()

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Input ----
    query = st.chat_input("Transmit query to Defence Intel System...")

    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        # show terminal-style decrypting indicator while backend works
        with chat_container:
            typing_placeholder.markdown(
                """
                <div class="msg-row ai">
                    <div class="avatar ai">🛡️</div>
                    <div class="bubble-wrap">
                        <div class="bubble ai">
                            <div class="bubble-name ai">DEFENCE INTEL SYSTEM</div>
                            <div class="typing-line">&gt; DECRYPTING TRANSMISSION<span class="cursor">&nbsp;</span></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        try:
            answer, results = ask_question(query)
            sources_list = extract_source_briefs(results)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources_list}
            )
            st.session_state.last_sources = results

        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠ TRANSMISSION ERROR — Unable to generate a response right now. ({e})",
                    "sources": []
                }
            )

        typing_placeholder.empty()
        st.rerun()


# ============================================================
# ROUTING
# ============================================================

if st.session_state.page == "landing":
    render_landing()
else:
    render_chat()
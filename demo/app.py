"""
Entry point — run with:
  GUARD_MODEL=sg .venv/bin/streamlit run demo/app.py
"""
import pathlib
import streamlit as st
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")

# ── Navigation ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Singlish Hate Speech Guardrail",
    page_icon="🛡️",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/chat.py",    title="Chat",    icon=":material/chat:"),
    st.Page("pages/samples.py", title="Samples", icon=":material/table:"),
])
pg.run()

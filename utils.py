"""
utils.py
========
Shared helpers: path management, CSS injection, Streamlit UI cards,
and miscellaneous formatting utilities.
"""

import os
import base64
from datetime import datetime
import streamlit as st


# ── Directory layout ─────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
ENCODINGS_DIR = os.path.join(BASE_DIR, "encodings")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
DB_DIR        = os.path.join(BASE_DIR, "database")
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")


def ensure_directories() -> None:
    """Create all required application directories."""
    for d in [DATASET_DIR, ENCODINGS_DIR, REPORTS_DIR, DB_DIR]:
        os.makedirs(d, exist_ok=True)


def student_image_dir(roll_no: str, name: str) -> str:
    """Return (and create) the per-student image directory."""
    folder = os.path.join(DATASET_DIR, f"{roll_no}_{name.replace(' ', '_')}")
    os.makedirs(folder, exist_ok=True)
    return folder


# ── Custom CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root palette ──────────────────────────────────────────────── */
:root {
    --bg-primary:   #0a0f1e;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --accent-cyan:  #00e5ff;
    --accent-green: #00ff9d;
    --accent-purple:#9d6bff;
    --accent-amber: #ffb830;
    --text-primary: #e8edf5;
    --text-muted:   #8899aa;
    --border:       rgba(0,229,255,0.15);
    --glow-cyan:    0 0 20px rgba(0,229,255,0.25);
    --glow-green:   0 0 20px rgba(0,255,157,0.25);
    --radius:       14px;
}

/* ── Base overrides ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem !important;
    padding: 6px 0 !important;
}

/* ── Header banner ─────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2a4a 50%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;  left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(0,229,255,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.main-header p {
    color: var(--text-muted);
    margin: 0.4rem 0 0 !important;
    font-size: 0.95rem;
}

/* ── Stat cards ────────────────────────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--glow-cyan);
}
.stat-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--accent-cyan);
    line-height: 1;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.stat-number.green  { color: var(--accent-green); }
.stat-number.purple { color: var(--accent-purple); }
.stat-number.amber  { color: var(--accent-amber); }
.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

/* ── Section card ──────────────────────────────────────────────── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-cyan);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Badge ─────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 30px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-green  { background: rgba(0,255,157,0.15); color: var(--accent-green);  border: 1px solid rgba(0,255,157,0.3); }
.badge-cyan   { background: rgba(0,229,255,0.15); color: var(--accent-cyan);   border: 1px solid rgba(0,229,255,0.3); }
.badge-amber  { background: rgba(255,184,48,0.15); color: var(--accent-amber); border: 1px solid rgba(255,184,48,0.3); }
.badge-red    { background: rgba(255,75,75,0.15);  color: #ff6b6b;             border: 1px solid rgba(255,75,75,0.3); }

/* ── Alert boxes ───────────────────────────────────────────────── */
.alert-success {
    background: rgba(0,255,157,0.1);
    border: 1px solid rgba(0,255,157,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: var(--accent-green);
    font-weight: 600;
}
.alert-error {
    background: rgba(255,75,75,0.1);
    border: 1px solid rgba(255,75,75,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #ff6b6b;
    font-weight: 600;
}
.alert-info {
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: var(--accent-cyan);
    font-weight: 600;
}

/* ── Streamlit widget tweaks ───────────────────────────────────── */
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), #0080ff) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
h2, h3 { color: var(--text-primary) !important; }
hr { border-color: var(--border) !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Reusable UI components ───────────────────────────────────────────────────

def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div class="main-header">
        <h1>🎓 {title}</h1>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def stat_card(number: int | str, label: str, color: str = "") -> str:
    cls = f"stat-number {color}".strip()
    return f"""
    <div class="stat-card">
        <div class="{cls}">{number}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def alert(message: str, kind: str = "info") -> None:
    """kind: 'success' | 'error' | 'info'"""
    st.markdown(f'<div class="alert-{kind}">{message}</div>', unsafe_allow_html=True)


# ── Misc helpers ─────────────────────────────────────────────────────────────

def csv_download_button(df, filename: str, label: str = "⬇️ Download CSV") -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def format_date(d: str) -> str:
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return d


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

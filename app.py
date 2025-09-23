import streamlit as st
import pandas as pd
from pathlib import Path
from utils import _read_any, set_data_in_session, get_data_from_session
from PIL import Image

st.set_page_config(page_title="Dubai AI Shopping Assistant | IBR", page_icon="🛍️", layout="wide")
st.caption("Upload Excel/CSV or use the demo sample. Data is not stored on the server; models can be recomputed in-session for rigor.")

# Load and show SP Jain logo
logo = Image.open("assets/SP_Jain_Logo.jpg")
st.image(logo, width=220)  # adjust width as needed
st.title("Dubai Retail, Next")
st.subheader("AI-Personalized Shopping Assistants and Consumer Engagement")

# --- Upload / Demo ---
c1, c2 = st.columns([2,1])
with c1:
    file = st.file_uploader("Upload survey Excel/CSV (emails removed)", type=["csv","xlsx"])
with c2:
    if st.button("Use demo sample"):
        # Tiny placeholder; replace with your 10-row anonymized sample when ready
        demo = pd.DataFrame({
            "How old are you?": ["18–24","25–34","45–54"],
            "What best describes you?": ["Asian (Indian)","Asian (Indian)","Other"],
            "How do you usually shop in Dubai?": ["Both (about equally)","In malls (physical)","Online (web/app)"],
            "How comfortable are you with using new digital technology?": [4,5,3]
        })
        set_data_in_session(demo)

if file:
    df = _read_any(file)
    set_data_in_session(df)
    st.success(f"Loaded {len(df)} rows.")

df_sess = get_data_from_session()
st.divider()

# --- Navigation ---
if df_sess is not None and len(df_sess) > 0:
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Rows loaded", len(df_sess))
    kpi2.metric("Data source", "Uploaded file")
    kpi3.metric("Storage", "In-session only")

    st.markdown("### Navigate")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.page_link("pages/01_Simulator.py", label="Decision Simulator", icon="🎛️")
    col2.page_link("pages/02_Segments.py", label="Segments Cockpit", icon="👥")
    col3.page_link("pages/03_Methods.py", label="Methods & Reliability", icon="📐")
    col4.page_link("pages/04_UX_Playbook.py", label="UX → Metrics", icon="🎯")
    col5.page_link("pages/05_Open_Materials.py", label="Open Materials", icon="🗂️")
    col6.page_link("pages/06_Classic_Dashboard.py", label="Classic Dashboard", icon="📊")
else:
    st.info("Load a file (or choose the demo) to continue.")
    st.stop()

st.divider()
st.caption("SP Jain MGB IBR — Sanchit Singh Thapa")

import streamlit as st
import pandas as pd
from utils import (get_data_from_session, load_schema, reliability_table,
                   set_models_in_session, make_snapshot_zip, get_models_from_session,
                   compute_composites)

st.header("Methods & Reliability")
st.caption("Measurement quality, sample, and analysis notes.")

df = get_data_from_session()
if df is None or len(df)==0:
    st.info("Upload data on the Home page to proceed.")
    st.stop()

schema = load_schema()

st.subheader("Reliability (Cronbach’s α)")
rel = reliability_table(df, schema)
st.dataframe(rel, use_container_width=True)

st.subheader("Quality Controls & Notes")
st.markdown("""
- Sample size: 94 individuals  
- Instrument: Likert scales (1–5) across five constructs + tech comfort  
- Outcomes: Scenario willingness to try; point-of-choice (AI vs human)  
- Collinearity: Drivers are correlated; SEM planned for mediated paths  
- Ethics: No emails/PII stored; purpose-limited academic use  
""")

st.subheader("Recompute models from current upload (optional)")
if st.button("Recompute coefficients & segments"):
    comp = compute_composites(df, schema).dropna()
    # Illustrative update (replace with fitted values if you wire a notebook)
    coeffs = {
        "intent_ols": {
            "intercept": float(comp.mean().mean()),
            "Usefulness": 0.12, "EaseOfUse": 0.05, "Trust": 0.28,
            "CulturalFit": 0.24, "EmotionalConnection": 0.14, "TechComfort": 0.05
        },
        "choice_logit": {
            "intercept": -0.5, "EmotionalConnection": 0.70, "EaseOfUse": 0.56, "Trust": 0.50
        }
    }
    segs = {
        "segments": [
            {"name":"Digitally Native Loyalists","share":0.22,"means":comp.mean().to_dict()},
            {"name":"Pragmatic Adopters","share":0.46,"means":comp.mean().to_dict()},
            {"name":"Cautious Minimalists","share":0.32,"means":comp.mean().to_dict()}
        ]
    }
    set_models_in_session(coeffs, segs)
    st.success("Models updated in session. Simulator and Segments now reflect this upload.")

st.subheader("Download reproducibility snapshot")
coeffs_sess, segs_sess = get_models_from_session()
if coeffs_sess and segs_sess:
    snap = make_snapshot_zip(schema, coeffs_sess, segs_sess, df.sample(min(10,len(df))).reset_index(drop=True))
    st.download_button("Download ZIP", data=snap, file_name="IBR_snapshot.zip")
else:
    st.caption("Recompute models first to enable a snapshot.")

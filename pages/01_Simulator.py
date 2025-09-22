import streamlit as st
from utils import load_coeffs, get_models_from_session, predict_intent, predict_choice_probability

st.header("Decision Simulator")
st.caption("Change perceptions → see predicted willingness and AI-offer choice probability.")

# Prefer session-updated models; fallback to files
coeffs_sess, _ = get_models_from_session()
coeffs = coeffs_sess or load_coeffs()

with st.sidebar:
    st.subheader("Inputs (1–5)")
    use = st.slider("Usefulness", 1.0, 5.0, 3.5, 0.1)
    ease = st.slider("Ease of Use", 1.0, 5.0, 3.5, 0.1)
    trust = st.slider("Trust", 1.0, 5.0, 3.0, 0.1)
    culture = st.slider("Cultural Fit", 1.0, 5.0, 3.5, 0.1)
    emotion = st.slider("Emotional Connection", 1.0, 5.0, 3.2, 0.1)
    tech = st.slider("Tech Comfort", 1.0, 5.0, 4.0, 0.1)

comp = {
    "Usefulness": use,
    "EaseOfUse": ease,
    "Trust": trust,
    "CulturalFit": culture,
    "EmotionalConnection": emotion,
    "TechComfort": tech
}

c1, c2 = st.columns(2)
with c1:
    st.metric("Predicted Willingness (1–5)", f"{predict_intent(comp, coeffs):.2f}")
with c2:
    st.metric("Probability of Choosing AI Offer", f"{predict_choice_probability(comp, coeffs)*100:.1f}%")

st.info("Coefficients are evidence-inspired for transparency; final paper advances to mediated SEM and robust inference.")

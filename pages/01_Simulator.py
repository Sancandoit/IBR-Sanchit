import streamlit as st
import pandas as pd
from utils import (
    load_coeffs, load_segments,
    get_models_from_session, get_data_from_session,
    compute_composites, predict_intent, predict_choice_probability,
    get_query_params, set_query_params
)

st.header("What-If Simulator")
st.caption("Compare a baseline against a proposed scenario. Use presets, share a link, or export the summary.")

# ---- Models (prefer session then fallback) ----
coeffs_sess, segs_sess = get_models_from_session()
coeffs = coeffs_sess or load_coeffs()
segments_data = (segs_sess or load_segments())["segments"]

# ---- Baseline from uploaded data (construct means) or defaults ----
df = get_data_from_session()
if df is not None and len(df) > 0:
    comp = compute_composites(df)
    baseline_means = comp.mean(numeric_only=True).to_dict() if comp is not None and not comp.empty else {}
else:
    baseline_means = {}

# Safe defaults if any construct missing
def _d(name, fallback):
    return float(round(baseline_means.get(name, fallback), 2))

baseline_defaults = {
    "Usefulness": _d("Usefulness", 3.5),
    "EaseOfUse": _d("EaseOfUse", 3.5),
    "Trust": _d("Trust", 3.2),
    "CulturalFit": _d("CulturalFit", 3.4),
    "EmotionalConnection": _d("EmotionalConnection", 3.2),
    "TechComfort": _d("TechComfort", 4.0),
}

# ---- Read presets from URL (so we can share scenarios) ----
qp = get_query_params()
prefill = {k: float(qp.get(k, [baseline_defaults[k]])[0]) for k in baseline_defaults.keys()}
mode = qp.get("mode", ["baseline"])[0]

# ---- Layout ----
colA, colB = st.columns(2)

with colA:
    st.subheader("Baseline")
    st.caption("Use data-derived averages or tweak to match today’s reality.")
    b_use = st.slider("Usefulness (baseline)", 1.0, 5.0, float(prefill["Usefulness"]), 0.1, key="b_use")
    b_ease = st.slider("Ease of Use (baseline)", 1.0, 5.0, float(prefill["EaseOfUse"]), 0.1, key="b_ease")
    b_trust = st.slider("Trust (baseline)", 1.0, 5.0, float(prefill["Trust"]), 0.1, key="b_trust")
    b_cult = st.slider("Cultural Fit (baseline)", 1.0, 5.0, float(prefill["CulturalFit"]), 0.1, key="b_cult")
    b_emot = st.slider("Emotional Connection (baseline)", 1.0, 5.0, float(prefill["EmotionalConnection"]), 0.1, key="b_emot")
    b_tech = st.slider("Tech Comfort (baseline)", 1.0, 5.0, float(prefill["TechComfort"]), 0.1, key="b_tech")

with colB:
    st.subheader("Scenario")
    st.caption("Dial the levers you plan to change with UX, copy, or ops.")
    s_use = st.slider("Usefulness (scenario)", 1.0, 5.0, float(prefill["Usefulness"]), 0.1, key="s_use")
    s_ease = st.slider("Ease of Use (scenario)", 1.0, 5.0, float(prefill["EaseOfUse"]), 0.1, key="s_ease")
    s_trust = st.slider("Trust (scenario)", 1.0, 5.0, float(prefill["Trust"]), 0.1, key="s_trust")
    s_cult = st.slider("Cultural Fit (scenario)", 1.0, 5.0, float(prefill["CulturalFit"]), 0.1, key="s_cult")
    s_emot = st.slider("Emotional Connection (scenario)", 1.0, 5.0, float(prefill["EmotionalConnection"]), 0.1, key="s_emot")
    s_tech = st.slider("Tech Comfort (scenario)", 1.0, 5.0, float(prefill["TechComfort"]), 0.1, key="s_tech")

# ---- Presets (Segments) ----
with st.expander("Presets by segment", expanded=False):
    preset_cols = st.columns(len(segments_data))
    for i, seg in enumerate(segments_data):
        means = seg["means"]
        if preset_cols[i].button(seg["name"]):
            # apply segment means to SCENARIO only (baseline stays as-is)
            s_use = means.get("Usefulness", s_use)
            s_ease = means.get("EaseOfUse", s_ease)
            s_trust = means.get("Trust", s_trust)
            s_cult = means.get("CulturalFit", s_cult)
            s_emot = means.get("EmotionalConnection", s_emot)
            s_tech = means.get("TechComfort", s_tech)
            # update the sliders instantly by writing to session state
            st.session_state["s_use"] = s_use
            st.session_state["s_ease"] = s_ease
            st.session_state["s_trust"] = s_trust
            st.session_state["s_cult"] = s_cult
            st.session_state["s_emot"] = s_emot
            st.session_state["s_tech"] = s_tech

# ---- Compute predictions ----
baseline_comp = {
    "Usefulness": b_use, "EaseOfUse": b_ease, "Trust": b_trust,
    "CulturalFit": b_cult, "EmotionalConnection": b_emot, "TechComfort": b_tech
}
scenario_comp = {
    "Usefulness": s_use, "EaseOfUse": s_ease, "Trust": s_trust,
    "CulturalFit": s_cult, "EmotionalConnection": s_emot, "TechComfort": s_tech
}

b_intent = predict_intent(baseline_comp, coeffs)
s_intent = predict_intent(scenario_comp, coeffs)
b_prob = predict_choice_probability(baseline_comp, coeffs)
s_prob = predict_choice_probability(scenario_comp, coeffs)

# ---- KPIs with deltas ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Willingness (baseline)", f"{b_intent:.2f}")
k2.metric("Willingness (scenario)", f"{s_intent:.2f}", f"{(s_intent-b_intent):+.2f}")
k3.metric("AI-Offer Probability (baseline)", f"{b_prob*100:.1f}%")
k4.metric("AI-Offer Probability (scenario)", f"{s_prob*100:.1f}%", f"{(s_prob-b_prob)*100:+.1f} pp")

st.info("Evidence mode coefficients are used for clarity. For the paper, mediated SEM will quantify indirect effects.")

# ---- Share link + export ----
left, right = st.columns([3,2])
with left:
    if st.button("Copy sharable link for this scenario"):
        set_query_params({
            "Usefulness": s_use, "EaseOfUse": s_ease, "Trust": s_trust,
            "CulturalFit": s_cult, "EmotionalConnection": s_emot, "TechComfort": s_tech,
            "mode": "scenario"
        })
        st.success("URL updated. Copy from the browser address bar and share.")
with right:
    summary = pd.DataFrame([
        {"Metric":"Willingness", "Baseline": round(b_intent,2), "Scenario": round(s_intent,2), "Delta": round(s_intent-b_intent,2)},
        {"Metric":"AI-Offer Probability (%)", "Baseline": round(b_prob*100,1), "Scenario": round(s_prob*100,1), "Delta": round((s_prob-b_prob)*100,1)}
    ])
    st.download_button("Download scenario summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="what_if_summary.csv")

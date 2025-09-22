import streamlit as st
import pandas as pd
from utils import load_segments, get_models_from_session

st.header("Segments Cockpit")
st.caption("Profiles, shares, and activation playbook by latent class.")

# Prefer session segments; fallback to file
_, seg_sess = get_models_from_session()
seg_data = (seg_sess or load_segments())["segments"]

df = pd.DataFrame([{
    "Segment": s["name"],
    "Share": f'{int(s["share"]*100)}%',
    **s["means"]
} for s in seg_data])

st.dataframe(df, use_container_width=True)

st.markdown("### Activation playbook")
for seg in seg_data:
    with st.expander(seg["name"], expanded=False):
        st.write("**Profile (construct means)**")
        st.write(", ".join([f"{k} {v}" for k, v in seg["means"].items()]))
        st.write("**Pain points**")
        if seg["name"].startswith("Digitally"):
            st.write("- Seeks advanced features; keep trust visible to avoid complacency.")
        elif seg["name"].startswith("Pragmatic"):
            st.write("- Needs transparent privacy controls and obvious value (time saved, smart deals).")
        else:
            st.write("- Low trust & emotion; prefers human fallback and gentle onboarding.")
        st.write("**Experiments**")
        st.write("- A/B: Trust UI vs baseline • A/B: Warm micro-copy vs neutral • A/B: One-tap vs multi-step")

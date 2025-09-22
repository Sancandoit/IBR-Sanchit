import streamlit as st

st.header("UX → Metrics Playbook")
st.caption("Design moves mapped to target metrics and supporting evidence.")

tiles = [
    ("Privacy mini-ledger", "Target: Trust → Willingness", "Evidence: Trust has the strongest relationship with willingness. Test ledger vs baseline."),
    ("Consent center (granular)", "Target: Trust", "Evidence: Trust co-occurs with Ease; clarity increases perceived safety."),
    ("Bilingual by default", "Target: Cultural Fit → Willingness & Emotion", "Evidence: Cultural cues co-occur with feeling understood."),
    ("Warm, culturally familiar micro-copy", "Target: Emotional Connection → Choice", "Evidence: Higher odds of choosing the AI offer when emotion is high."),
    ("One-tap accept/decline + progress cue", "Target: Ease → Choice & Trust", "Evidence: Ease supports choice and appears with Trust.")
]

for title, target, note in tiles:
    with st.container():
        st.subheader(title)
        st.write(target)
        st.write(note)
        st.divider()

import streamlit as st
import pandas as pd

st.header("UX → Metrics Playbook")
st.caption("Design moves mapped to target metrics and supporting evidence. Export as CSV below.")

# ---- Playbook data (edit freely) ----
PLAYBOOK = [
    {
        "title": "Privacy mini-ledger",
        "icon": "🛡️",
        "target": "Trust → Willingness",
        "evidence": "Trust shows the strongest link with willingness; test ledger vs baseline.",
        "experiment": "A/B: ledger on vs off; measure intent lift and opt-in rate",
        "expected": "↑ intent, ↑ opt-in"
    },
    {
        "title": "Consent center (granular)",
        "icon": "✅",
        "target": "Trust",
        "evidence": "Trust co-occurs with Ease; clarity boosts perceived safety.",
        "experiment": "A/B: simple vs granular controls; measure trust and drop-off",
        "expected": "↑ trust, ↓ drop-off"
    },
    {
        "title": "Bilingual by default",
        "icon": "🌐",
        "target": "Cultural Fit → Willingness & Emotion",
        "evidence": "Cultural cues co-occur with feeling understood.",
        "experiment": "A/B: EN only vs EN+AR default; measure intent and CSAT",
        "expected": "↑ intent, ↑ CSAT"
    },
    {
        "title": "Warm, familiar micro-copy",
        "icon": "💬",
        "target": "Emotional Connection → Choice",
        "evidence": "Higher odds of choosing AI when emotion is high.",
        "experiment": "A/B: warm vs neutral tone; measure AI-offer take rate",
        "expected": "↑ AI selection"
    },
    {
        "title": "One-tap accept/decline + progress cue",
        "icon": "⚡",
        "target": "Ease → Choice & Trust",
        "evidence": "Ease supports choice and appears with Trust.",
        "experiment": "A/B: 1-tap vs multi-step; measure take rate and time-to-complete",
        "expected": "↑ take rate, ↓ time"
    },
]

# ---- Card styling (inline HTML/CSS) ----
CARD_CSS = """
<style>
.play-card {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 16px 16px 12px 16px;
  margin-bottom: 14px;
}
.tag {
  display:inline-block;
  padding:2px 8px;
  border-radius: 999px;
  font-size: 12px;
  margin-right:6px;
  background: rgba(13,166,166,0.18);
}
.small {
  opacity: 0.85;
  font-size: 14px;
}
.title {
  font-size: 18px;
  font-weight: 700;
}
.row {display: flex; gap: 14px;}
.grow {flex: 1;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ---- Render 2-column grid of cards ----
cols = st.columns(2)
for i, item in enumerate(PLAYBOOK):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="play-card">
          <div class="row">
            <div class="title">{item['icon']} {item['title']}</div>
            <div class="grow"></div>
          </div>
          <div style="margin:8px 0 10px 0;">
            <span class="tag">Target: {item['target']}</span>
          </div>
          <div class="small"><b>Evidence</b> — {item['evidence']}</div>
          <div class="small"><b>Experiment</b> — {item['experiment']}</div>
          <div class="small"><b>Expected</b> — {item['expected']}</div>
        </div>
        """, unsafe_allow_html=True)

# ---- Optional: export the playbook as CSV for the viva handout ----
df = pd.DataFrame(PLAYBOOK)
csv = df[["title","target","evidence","experiment","expected"]].to_csv(index=False).encode("utf-8")
st.download_button("Download Playbook (CSV)", data=csv, file_name="UX_Playbook.csv", type="primary")
st.caption("Tip: add small PNG mocks from Canva to /assets and display them inside cards if you want visuals.")

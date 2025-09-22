import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_schema, normalize_likert, get_data_from_session

st.header("Classic Dashboard (Descriptives)")
st.caption("Legacy charts for demographics, means, and correlations.")

schema = load_schema()

# Prefer the already-uploaded data; fallback to per-page upload
df = get_data_from_session()
if df is None or len(df)==0:
    file = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx"])
    if not file:
        st.stop()
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

st.success(f"Loaded {len(df)} rows.")

age_col = schema["demographics"]["age"]
nat_col = schema["demographics"]["nationality"]
shop_col = schema["demographics"]["shopping"]

c1, c2, c3 = st.columns(3)
c1.metric("Total Responses", len(df))
if schema["tech_comfort"] in df.columns:
    c2.metric("Avg Tech Comfort", round(pd.to_numeric(df[schema["tech_comfort"]], errors="coerce").mean(),2))
online_pct = df[shop_col].value_counts(normalize=True).get('Online (web/app)',0)*100 if shop_col in df else 0
c3.metric("Online Shoppers (%)", round(online_pct,1))

st.subheader("Respondent Profile")
if age_col in df:
    st.plotly_chart(px.bar(df[age_col].value_counts().sort_index(), title="Age Distribution"), use_container_width=True)
if nat_col in df:
    st.plotly_chart(px.bar(df[nat_col].value_counts(), title="Nationality / Identity"), use_container_width=True)

st.subheader("Key Drivers: Means (1–5)")
df_num = normalize_likert(df, schema)
item_list = sum(schema["constructs"].values(), [])
present = [c for c in df_num.columns if any(cc[:15].lower() in c.lower() for cc in item_list)]
means = df_num[present].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
st.plotly_chart(px.bar(means, title="Means"), use_container_width=True)

st.subheader("Correlation Heatmap")
corr = df_num[present].apply(pd.to_numeric, errors="coerce").corr()
st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation"), use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import statsmodels.api as sm

st.title("Dubai AI Shopping Assistant Survey Dashboard")
st.markdown("_Upload your latest Google Forms Excel/CSV to update the dashboard_")

# File upload
uploaded_file = st.file_uploader(
    "Upload your survey data file here (.csv or .xlsx)", type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        # Force openpyxl engine to avoid ImportError
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.success(f"Loaded {len(df)} responses! Ready for analysis.")
else:
    st.info("Please upload your exported Google Forms file to begin.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filter Responses")
    age_filter = st.multiselect("Age", options=df['How old are you?'].unique() if 'How old are you?' in df.columns else [])
    nationality_filter = st.multiselect("Nationality", options=df['What best describes you?'].unique() if 'What best describes you?' in df.columns else [])
    shopping_style_filter = st.multiselect("Shopping Style", options=df['How do you usually shop in Dubai?'].unique() if 'How do you usually shop in Dubai?' in df.columns else [])

    filtered_df = df.copy()
    if age_filter and 'How old are you?' in df.columns:
        filtered_df = filtered_df[filtered_df['How old are you?'].isin(age_filter)]
    if nationality_filter and 'What best describes you?' in df.columns:
        filtered_df = filtered_df[filtered_df['What best describes you?'].isin(nationality_filter)]
    if shopping_style_filter and 'How do you usually shop in Dubai?' in df.columns:
        filtered_df = filtered_df[filtered_df['How do you usually shop in Dubai?'].isin(shopping_style_filter)]

# Quick metrics
st.subheader("Quick Demographics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Responses", len(filtered_df))
if 'How comfortable are you with using new digital technology?' in filtered_df.columns:
    col2.metric("Avg Digital Comfort", round(filtered_df['How comfortable are you with using new digital technology?'].mean(), 2))
if 'How do you usually shop in Dubai?' in filtered_df.columns:
    col3.metric("Online Shoppers (%)", round((filtered_df['How do you usually shop in Dubai?'].value_counts(normalize=True).get('Online (web/app)', 0)) * 100, 1))

st.markdown("---")

# Respondent Profile
if 'How old are you?' in filtered_df.columns and 'What best describes you?' in filtered_df.columns:
    st.subheader("Respondent Profile")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    filtered_df['How old are you?'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='royalblue', title="Age")
    filtered_df['What best describes you?'].value_counts().plot(kind='pie', autopct='%1.0f%%', ax=ax[1], title="Nationality")
    st.pyplot(fig)

# Likert Analysis
likert_map = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5
}

likert_cols = [
    'AI assistants help me make better shopping decisions.',
    'AI assistants save me time in-store or online.',
    'I trust recommendations made by AI shopping assistants.',
    'I feel confident that my data is safe when using AI features.',
    'I find AI shopping assistants easy to use and understand.',
    'It doesn’t take much effort to learn how to use these assistants.',
    'AI shopping assistants give me recommendations that match my taste.',
    'I feel like AI shopping assistants understand what I want.',
    'I appreciate when an AI shopping assistant speaks my language or uses familiar cultural references.',
    'The tone and style of AI assistants in Dubai suit me.',
    'I enjoy chatting with an AI shopping assistant.',
    'Sometimes, AI assistants “get me” better than human staff.'
]

available_cols = [col for col in likert_cols if col in filtered_df.columns]

if available_cols:
    st.subheader("Key Drivers: Means (1=Strongly Disagree, 5=Strongly Agree)")
    means = filtered_df[available_cols].replace(likert_map).mean()
    st.bar_chart(means)
else:
    st.warning("No Likert-scale survey questions found in your data file.")

# Correlation Heatmap
if available_cols:
    st.subheader("Correlation Heatmap of Attitudes")
    likert_num = filtered_df[available_cols].replace(likert_map)
    corr = likert_num.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Regression Analysis
st.subheader("Regression: What drives willingness to recommend AI shopping assistants?")

# Dynamically detect recommendation column
rec_cols = [c for c in filtered_df.columns if "recommend" in c.lower()]
available_cols = [col for col in likert_cols if col in filtered_df.columns]

if rec_cols and available_cols:
    rec_col = rec_cols[0]  # pick first matching column
    st.write(f"Using column: **{rec_col}**")

    # Encode target (Yes=1, No=0, Maybe=0.5, else NaN)
    y = filtered_df[rec_col].map({'Yes': 1, 'No': 0, 'Maybe': 0.5})

    # Prepare predictors
    X = filtered_df[available_cols].replace(likert_map)
    X = sm.add_constant(X)

    # Run regression
    model = sm.OLS(y, X, missing='drop').fit()

    # Show summary
    st.markdown("**Statistical Summary**")
    st.text(model.summary())

    # Show coefficients visually
    coef = model.params.drop("const").sort_values()
    st.subheader("Top Drivers of Recommendation")
    st.bar_chart(coef)

else:
    st.warning("Could not find a recommendation column or Likert predictors in your uploaded file. Please check column names.")

# Word Cloud
st.subheader("Open-Ended Feedback (Word Cloud)")
if 'Any ideas or suggestions for how Dubai retailers can make AI shopping assistants better for you?' in filtered_df.columns:
    text = ' '.join(filtered_df['Any ideas or suggestions for how Dubai retailers can make AI shopping assistants better for you?'].dropna())
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.info("No open-ended feedback available.")
else:
    st.info("Open-ended feedback column not found in uploaded file.")

st.markdown("---")
st.caption("Dashboard by Sanchit Singh Thapa | MBA Research | SP Jain")

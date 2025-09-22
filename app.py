import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import statsmodels.api as sm

st.title("Dubai AI Shopping Assistant Survey Dashboard")
st.markdown("_Upload your latest Google Forms Excel/CSV to update the dashboard_")

uploaded_file = st.file_uploader("Upload your survey data file here (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"Loaded {len(df)} responses! Ready for analysis.")
else:
    st.info("Please upload your exported Google Forms file to begin.")
    st.stop()

# 2. Data Cleaning (example: rename columns if needed)
# st.write(df.head())
# Uncomment and adjust to your column names if necessary

# 3. Sidebar Filters
with st.sidebar:
    st.header("Filter Responses")
    age_filter = st.multiselect("Age", options=df['How old are you?'].unique())
    nationality_filter = st.multiselect("Nationality", options=df['What best describes you?'].unique())
    shopping_style_filter = st.multiselect("Shopping Style", options=df['How do you usually shop in Dubai?'].unique())

    filtered_df = df.copy()
    if age_filter:
        filtered_df = filtered_df[filtered_df['How old are you?'].isin(age_filter)]
    if nationality_filter:
        filtered_df = filtered_df[filtered_df['What best describes you?'].isin(nationality_filter)]
    if shopping_style_filter:
        filtered_df = filtered_df[filtered_df['How do you usually shop in Dubai?'].isin(shopping_style_filter)]

st.subheader("Quick Demographics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Responses", len(filtered_df))
col2.metric("Avg Digital Comfort", round(filtered_df['How comfortable are you with using new digital technology?'].mean(),2))
col3.metric("Online Shoppers (%)", round((filtered_df['How do you usually shop in Dubai?'].value_counts(normalize=True).get('Online (web/app)',0))*100,1))

st.markdown("---")

# 4. Visualizations

# a. Age/Nationality Distribution
st.subheader("Respondent Profile")
fig, ax = plt.subplots(1,2, figsize=(12,4))
filtered_df['How old are you?'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='royalblue', title="Age")
filtered_df['What best describes you?'].value_counts().plot(kind='pie', autopct='%1.0f%%', ax=ax[1], title="Nationality")
st.pyplot(fig)

# b. Likert Averages for Key Constructs
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

st.subheader("Key Drivers: Means (1=Strongly Disagree, 5=Strongly Agree)")
means = filtered_df[likert_cols].replace(likert_map).mean()
st.bar_chart(means)

# c. Correlation Matrix
st.subheader("Correlation Heatmap of Attitudes")
likert_num = filtered_df[likert_cols].replace(likert_map)
corr = likert_num.corr()
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# d. Regression: Which factors predict likelihood to recommend?
st.subheader("Regression: What drives willingness to recommend AI shopping assistants?")
# Example: Predict "Would you recommend using AI-powered shopping assistants to others?" (encode Yes=1, No/Maybe=0)
if 'Would you recommend using AI-powered shopping assistants to others?' in filtered_df.columns:
    y = filtered_df['Would you recommend using AI-powered shopping assistants to others?'].map({'Yes':1, 'No':0, 'Maybe':0.5})
    X = likert_num
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    st.write(model.summary())

# e. Word Cloud: Open-ended feedback
st.subheader("Open-Ended Feedback (Word Cloud)")
if 'Any ideas or suggestions for how Dubai retailers can make AI shopping assistants better for you?' in filtered_df.columns:
    text = ' '.join(filtered_df['Any ideas or suggestions for how Dubai retailers can make AI shopping assistants better for you?'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

st.markdown("---")
st.caption("Dashboard by Sanchit Singh Thapa | MBA Research | SP Jain")

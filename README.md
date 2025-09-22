# 
# IBR- Dubai AI Shopping Assistant Survey Dashboard

This is an interactive Streamlit dashboard that analyzes responses from the "Smart Shopping in Dubai" MBA research survey. It connects to Google Sheets or a local CSV to visualize trends, perform statistical analysis, and reveal insights about consumer engagement with AI shopping assistants in Dubai.

## Features

- Real-time demographic breakdowns (age, nationality, shopping style, digital comfort)
- Distribution and correlation of trust, usefulness, personalization, and engagement scores
- Regression analysis for predictors of adoption/satisfaction
- Visualizations: Bar charts, pie charts, heatmaps, word clouds
- Filter and segment by demographics
- Thematic summary of open-ended responses
- (Optional) Downloadable reports

## Getting Started

1. Clone this repo.
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Place your exported Google Forms CSV as `data.csv` (or connect Google Sheets, see code).
4. Run the dashboard:  
   `streamlit run app.py`

*To use Google Sheets live data, set up service account credentials and update the code accordingly.*

## Data Privacy

All survey data is anonymous and used only for academic research.

## Author

Sanchit Singh Thapa, SP Jain School of Global Management (MBA 2025)
<img width="468" height="642" alt="image" src="https://github.com/user-attachments/assets/18dbaea4-2dc4-42de-9eae-ca499a9ed0a0" />

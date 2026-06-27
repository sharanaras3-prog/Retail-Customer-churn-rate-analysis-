# CX Churn Intelligence Platform
### AI Resolution Bias & Silent Churn Prediction Dashboard

A Streamlit analytics platform investigating why customers marked as **"Successfully Resolved"** by AI support systems churn 30–60 days later — and which customers are most at risk.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cx-churn-intelligence.git
cd cx-churn-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📊 Platform Pages

| Page | Description |
|------|-------------|
| **🏠 Overview** | KPI dashboard, churn summary, silent churner spotlight |
| **1 · Descriptive Analysis** | Cross-tabs, value counts, distributions, score comparisons |
| **2 · Bias Detection** | Chi-square tests, group-wise churn rates, intersectional heatmaps |
| **3 · ML Classification** | Feature engineering, SMOTE, KNN / DT / RF / GB training |
| **4 · Model Evaluation** | ROC-AUC, confusion matrices, Precision/Recall/F1 |
| **5 · Findings** | Bias summary, model recommendations, high-risk watchlist |

---

## 🗂️ Project Structure

```
cx_churn_app/
├── app.py                          # Main entry point (Overview page)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── assets/
│   └── cx_survey_500_responses.xlsx  # Dataset (500 post-AI interaction surveys)
├── pages/
│   ├── 1_Descriptive_Analysis.py
│   ├── 2_Bias_Detection.py
│   ├── 3_ML_Classification.py
│   ├── 4_Model_Evaluation.py
│   └── 5_Findings.py
└── utils/
    ├── data_loader.py              # Cached data loading, feature engineering
    └── plot_style.py               # Shared Plotly theme and colors
```

---

## 📦 Dataset

`assets/cx_survey_500_responses.xlsx` — 500 simulated post-AI-interaction survey responses containing:

- **Customer profile:** tenure, purchase frequency, spend tier
- **Interaction details:** issue type, channel, AI actions offered
- **Satisfaction scores:** AI understanding, true resolution, CES, NPS, overall satisfaction
- **Behavioral intent:** 30-day purchase intent, churn intent
- **Derived variables:** churn binary, NPS segment, resolution binary, high-effort flag

Synthetic demographic fields (age group, income band, support team) are generated with intentional bias patterns for audit purposes.

---

## 🤖 Models

| Model | Use Case |
|-------|----------|
| KNN | Baseline / local similarity |
| Decision Tree | Explainable rules |
| Random Forest | Stable feature importance |
| **Gradient Boosting** | ⭐ Best for Silent Churn detection |

SMOTE oversampling is applied by default to handle class imbalance (~38% churn rate).

---

## 🔑 Key Findings

1. **Silent Churners** (resolved but churned) represent ~35–40% of all churned customers
2. **Older customers (55+)** and **lower-income customers (<$30k)** face systematically higher churn rates
3. **Team Delta** shows significantly worse outcomes across all issue types
4. **Customer Effort Score** and **True Resolution score** are the top churn predictors
5. **Gradient Boosting** achieves the highest ROC-AUC and is most stable for minority-class (churn) detection

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path:** `app.py`
5. Click **Deploy**

---

## 📋 Requirements

- Python 3.9+
- See `requirements.txt` for full dependencies

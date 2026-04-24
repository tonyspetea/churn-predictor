# Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.842-green)

> Deploy to [Streamlit Cloud](https://share.streamlit.io) to add a live demo link here.

A machine learning web app that predicts whether a telecom customer will churn, and **explains why** using SHAP values. Built end-to-end in Python with a Streamlit interface.

---

## What it does

- Predicts churn probability for any customer profile using an XGBoost classifier
- Shows a SHAP waterfall chart explaining the top reasons driving each individual prediction
- Displays overall feature importance so business teams can act on the insights
- Flags customers as low / medium / high risk with clear visual indicators

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.842** |
| Recall (churners) | 80% |
| Precision (non-churners) | 91% |
| Training samples | 5,634 |

> ROC-AUC of 0.842 means the model correctly ranks a churning customer above a non-churning one 84% of the time.

---

## Key findings

The SHAP analysis revealed the top drivers of churn:

1. **Contract type** — two-year contracts reduce churn risk the most
2. **Tenure** — new customers (low tenure) are highest risk
3. **Internet service** — fiber optic customers churn more than DSL
4. **Payment method** — electronic check users show higher churn rates
5. **Monthly charges** — higher charges increase churn probability

---

## Tech stack

- **Data:** IBM Telco Customer Churn dataset (7,043 customers, 21 features)
- **Model:** XGBoost with `scale_pos_weight` to handle class imbalance
- **Explainability:** SHAP TreeExplainer for global and per-customer explanations
- **App:** Streamlit with Plotly visualizations
- **Feature engineering:** Created `AvgMonthlySpend` (TotalCharges / tenure) as an additional predictor

---

## Run locally

```bash
git clone https://github.com/tonyspetea/churn-predictor.git
cd churn-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Project structure

```
churn-predictor/
├── data/                  # Raw dataset
├── model/                 # Saved XGBoost model (.pkl)
├── outputs/               # SHAP importance chart
├── notebooks/             # EDA and modelling notebook
├── app.py                 # Streamlit application
└── requirements.txt
```

---

## Author

**tonyspetea** · [GitHub](https://github.com/tonyspetea)

> Open to freelance data science projects and remote roles. Reach out via GitHub.

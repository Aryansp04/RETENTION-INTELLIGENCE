# 🏦 Customer Engagement & Product Utilization Analytics for Retention Strategy

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Benchmarking-F7931E.svg)](https://scikit-learn.org/)
[![Status: Internship Capstone](https://img.shields.io/badge/Project-Unified%20Mentor%20Capstone-008080.svg)](https://www.unifiedmentor.com/)

> **Quantitative Banking Analytics & Retention Strategy Research Framework**  
> Developed by **Aryan** | Unified Mentor Internship Program & European Central Bank Case Framework

---

## 📌 Project Overview
Retail banks traditionally classify customer loyalty through static demographic wealth metrics—such as ledger balance and annual salary. This project provides empirical proof that **customer behavioral engagement and product relationship depth—rather than demographic wealth—dictate long-term retention**.

Using an audited portfolio of **10,000 retail banking accounts across France, Germany, and Spain**, this end-to-end data science platform uncovers structural retention anomalies, formulates a standardized suite of banking KPIs, deploys a composite **Relationship Strength Index (RSI)**, and provides bank executives with an interactive **Retention Strategy What-If Simulator** that demonstrates how targeted behavioral interventions can safeguard over **€50.12 Million in deposit capital**.

---

## 🏛️ Submission URLs & Project Deliverables

| Deliverable Requirement | Submission Link |
| :--- | :--- |
| **GitHub Repository Link** | `https://github.com/aryan-retention-analytics/customer-engagement-banking` *(Replace with your repo)* |
| **Research Paper Link** | [RESEARCH_PAPER.md](./RESEARCH_PAPER.md) *(Or Google Drive / Overleaf link)* |
| **Live Deployed Web Application** | `https://customer-retention-analytics.streamlit.app` *(Streamlit Cloud deployment)* |
| **Project Explanation Video Link** | `https://youtu.be/your-project-walkthrough-id` *(5–7 min technical walkthrough)* |
| **Internship Experience Video Link**| `https://youtu.be/your-internship-feedback-id` *(2–3 min personal reflection)* |

---

## 📊 Core Empirical Findings

### 1. The "2-Product Golden Zone"
- **1 Product (50.8% of portfolio):** Churn rate is **$27.71\%$**. Minimal switching friction.
- **2 Products (45.9% of portfolio):** Churn rate plunges to **$7.58\%$** (**$72.6\%$ relative reduction**). The empirical optimal relationship depth.
- **3 & 4 Products (3.3% of portfolio):** Churn explodes to **$82.71\%$** and **$100.00\%$**, demonstrating acute friction from forced cross-selling quotas.

### 2. The High-Balance Capital Flight Paradox
- Customers holding $>€100,000$ control **€312.27 Million** ($40.83\%$ of total bank deposits).
- **$49.09\%$ of these high-balance customers are inactive**, carrying a **$32.77\%$ churn rate**.
- Over **€102.32 Million in deposits** is at immediate flight risk due to passive relationship management.

### 3. Credit Card Passivity Gap
- Standalone credit card ownership yields **no statistically significant retention benefit** ($20.18\%$ churn for cardholders vs $20.81\%$ for non-cardholders, $p = 0.498$).
- Retention lift only occurs when the credit card is actively utilized in recurring monthly transactions.

### 4. Cross-National Jurisdictional Asymmetry
- In Germany, **100% of accounts have positive balances** (averaging €119,730), yet churn reaches **$32.44\%$**—double that of France ($16.15\%$) and Spain ($16.67\%$)—exposing acute rate sensitivity in core European savings markets.

---

## 📐 Standardized Analytical Key Performance Indicators (KPIs)

| KPI Name | Formula | Portfolio Metric | Strategic Interpretation |
| :--- | :--- | :--- | :--- |
| **Engagement Retention Ratio (ERR)** | $\frac{1 - \text{Churn}_{\text{Active}}}{1 - \text{Churn}_{\text{Inactive}}}$ | **$1.17\text{x}$** ($1.88\text{x}$ churn odds) | Active members are nearly twice as likely to stay with the bank. |
| **Product Depth Index (PDI)** | $1 - P(\text{Exited} \mid N_{\text{prod}} = k)$ | **$92.42\%$** at 2 prods | Two products represent the mathematical optimum of customer stickiness. |
| **High-Balance Disengagement (HBDR)**| $\frac{\text{Inactive}(>€100\text{k})}{\text{Total}(>€100\text{k})}$ | **$49.09\%$** | Nearly half of VIP capital is unengaged and exposed to silent run-off. |
| **Credit Card Stickiness Delta (CCSS)**| $R_{\text{card}} - R_{\text{no card}}$ | **$+0.63\%$** (Insignificant) | Passive cardholding does not build loyalty; active usage is required. |
| **Relationship Strength Index (RSI)**| $\text{Composite}(0 \text{ to } 100)$ | Retained: **$60.0$** vs Churned: **$42.8$** | Predictive composite scoring synthesizing activity, depth, and tenure. |

---

## 🗂️ Project Repository Architecture

```
customer_retention_analytics/
│
├── data/
│   └── Churn_Modelling.csv             # Cleaned European retail banking dataset (10,000 rows)
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py              # Ingestion, validation, RSI scoring & KPI calculators
│   ├── analytics.py                    # Chi-square, Mann-Whitney U, Logit & RF models
│   ├── simulator.py                    # What-If retention simulation & capital impact engine
│   └── eda_and_modeling.py             # Reproducible full CLI pipeline script
│
├── docs/
│   ├── research_paper.md               # Full academic & industry research paper
│   └── executive_summary_ecb_stakeholders.md # Board & ECB regulatory policy memo
│
├── video_scripts/
│   ├── project_explanation_walkthrough_script.md  # 5-7 min technical walkthrough script
│   └── internship_experience_feedback_script.md   # 2-3 min personal reflection script
│
├── app.py                              # Full-featured Streamlit interactive web application
├── requirements.txt                    # Production environment dependencies
├── README.md                           # Comprehensive documentation & setup guide
├── RESEARCH_PAPER.md                   # Root copy of research paper
└── EXECUTIVE_SUMMARY.md                # Root copy of executive summary
```

---

## 🚀 Quickstart & Local Installation Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/aryan-retention-analytics/customer-engagement-banking.git
cd customer-engagement-banking
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Statistical Pipeline via CLI
To verify all statistical hypothesis tests, machine learning odds ratios, and simulation outputs:
```bash
python src/eda_and_modeling.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will automatically launch in your browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud (Free & 1-Click)

1. Push this repository to your personal GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select:
   - **Repository:** `your-username/customer-engagement-banking`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**! Your interactive dashboard will be live with a shareable URL in 2 minutes.

---

## 💻 Tech Stack & Analytical Libraries
- **Language:** Python 3.12
- **Data Engineering:** Pandas, NumPy
- **Econometric & Statistical Inference:** SciPy (`scipy.stats.chi2_contingency`, `scipy.stats.mannwhitneyu`, `scipy.stats.ttest_ind`)
- **Machine Learning Benchmarks:** Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`, `StandardScaler`)
- **Interactive Visualization:** Plotly Graph Objects & Plotly Express
- **Frontend / Application Framework:** Streamlit
- **Design System:** Custom CSS glassmorphism, responsive grid layout, executive metric badges

---

## 📄 License & Attribution
This project is authored by **Aryan** as part of the **Unified Mentor Data Science & Financial Analytics Fellowship**. Distributed under the [MIT License](./LICENSE).


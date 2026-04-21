# 🛍️ JCPenney Customer Targeting — Advanced Data Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-SQL_Engine-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C?style=for-the-badge)

**A production-style data analytics pipeline for JCPenney retail intelligence**  
*Combining SQL, unsupervised ML, predictive modelling, and a simulated multi-agent AI system*

[📊 View Notebook](#-project-structure) · [🤖 Multi-Agent Pipeline](#-multi-agent-ai-pipeline) · [📈 Key Results](#-key-results) · [🚀 Quick Start](#-quick-start)

</div>

---

## 📌 Project Overview

This project delivers a **consultancy-grade data analytics report** for JCPenney, one of America's largest retail chains. It analyses over **27,000 customer reviews**, **7,900+ products**, and **5,000 registered users** across six datasets to answer one core business question:

> *What drives customer satisfaction at JCPenney — and how can data science be used to retain at-risk customers, segment the customer base, and guide product strategy?*

The pipeline goes far beyond descriptive statistics. It implements:

- **SQL-based analysis** via an in-memory SQLite engine with multi-table JOIN queries
- **RFM (Recency–Frequency–Monetary) segmentation** to classify every customer
- **K-Means clustering** with PCA visualisation to identify distinct product tiers
- **Logistic Regression churn prediction** achieving **94.4% accuracy**
- **Keyword-based sentiment analysis** across price tiers
- A **simulated multi-agent AI pipeline** (SQL Agent → EDA Agent → Modelling Agent → Critic Agent → Synthesis Agent)

---

## 📂 Project Structure

```
jcpenney-customer-targeting/
│
├── 📓 3457775_BD2_Advanced.ipynb          # Full Jupyter notebook (run cell by cell)
├── 🐍 jcpenney_advanced_analysis.py       # Standalone Python script
├── 📄 README.md                           # This file
│
├── 📁 data/
│   ├── products.csv                       # 7,982 products — name, SKU, price, score
│   ├── reviews.csv                        # 39,063 reviews — username, score, text
│   ├── users.csv                          # 5,000 users — DOB, state
│   ├── jcpenney_products.json             # Enriched product data (brand, category, list/sale price)
│   └── jcpenney_reviewers.json            # Enriched reviewer data with purchase history
│
└── 📁 figures/
    ├── fig1_clustering.png                # K-Means elbow curve + PCA scatter
    ├── fig2_rfm_segments.png              # RFM customer segment distribution
    ├── fig3_churn_confusion.png           # Confusion matrix — 94.4% accuracy
    ├── fig4_discount_by_category.png      # SQL: avg discount % by category
    ├── fig5_state_scores.png              # SQL: avg score by state (JOIN query)
    ├── fig6_sentiment_by_tier.png         # Sentiment polarity by price tier
    ├── fig7_cluster_profiles.png          # Normalised cluster feature comparison
    └── fig8_price_vs_score.png            # Pearson correlation: price vs satisfaction
```

---

## 📊 Datasets

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `products.csv` | 7,982 | 6 | SKU, name, description, price, avg score |
| `reviews.csv` | 39,063 | 4 | Product ID, username, score (1–5), review text |
| `users.csv` | 5,000 | 3 | Username, date of birth, US state |
| `jcpenney_products.json` | 7,982 | 15 | Brand, category tree, list/sale price, ratings |
| `jcpenney_reviewers.json` | 5,000 | 4 | Username, DOB, state, products reviewed |

---

## 🛠️ Techniques Used

| # | Technique | Library | Business Purpose |
|---|-----------|---------|-----------------|
| 1 | **Data Cleaning** | `pandas`, `numpy` | Handle nulls, outliers, type coercion |
| 2 | **SQL Queries** | `sqlite3` | Multi-table JOINs, CASE tiers, aggregations |
| 3 | **Statistical Testing** | `scipy.stats` | Pearson r: price vs customer satisfaction |
| 4 | **RFM Segmentation** | `pandas` | Classify 4,983 customers into 5 segments |
| 5 | **K-Means Clustering** | `sklearn` | Unsupervised product grouping (K=4) |
| 6 | **PCA** | `sklearn` | 2D visualisation of cluster structure |
| 7 | **Logistic Regression** | `sklearn` | Predict churned customers (94.4% accuracy) |
| 8 | **Sentiment Analysis** | Custom NLP | Keyword polarity scoring across price tiers |
| 9 | **Multi-Agent Pipeline** | Python | Modular AI commentary chain |
| 10 | **Visualisation** | `matplotlib` | 8 publication-quality figures |

---

## 📈 Key Results

### 🔍 SQL Analysis
| Query | Finding |
|-------|---------|
| Price vs satisfaction (Pearson r) | **r = −0.009** (p = 0.131) — price has **zero** impact on scores |
| Dominant price tier | **Mid ($50–99)** — 4,497 products (largest segment) |
| Highest discounted category | **Hipster** — 83.7% average discount |
| Top state by review volume | **Massachusetts** — 600+ reviewers |
| Average customer age | **50.8 years** — JCPenney skews significantly older |

### 🎯 RFM Customer Segmentation
```
Champions       1,318  ██████████████████████  — VIP loyalty targets
Loyal           1,212  ████████████████████    — Retention priority
Potential Loyal 1,101  ██████████████████      — Upsell opportunity
Lost              750  ████████████            — Reactivation campaign
At Risk           602  ██████████              — Urgent win-back needed
```

### 🔬 K-Means Product Clusters (K=4)
| Cluster | Avg List Price | Avg Sale Price | Avg Discount | Avg Rating | Interpretation |
|---------|---------------|----------------|-------------|------------|----------------|
| 0 | $53.96 | $32.89 | 39.2% | 3.00 | Mid-range mainstream |
| 1 | $50.30 | $29.24 | 41.7% | 2.96 | Value everyday items |
| 2 | $58.01 | $122.48 | — | 3.05 | Sale/clearance outliers |
| 3 | $121.73 | $74.36 | 37.6% | 3.01 | Premium segment |

> PCA captures **60.1%** of total variance in 2 components, confirming cluster separability.

### 🚨 Churn Prediction (Logistic Regression)
```
              precision    recall  f1-score   support
  Retained       0.88      0.45      0.59       114
   Churned       0.95      0.99      0.97     1,132
  Accuracy                           94.4%    1,246
```
> The model correctly identifies **1,125 of 1,132 churned customers** — enabling proactive outreach before customers are fully lost.

### 💬 Sentiment Analysis
| Price Tier | Avg Sentiment |
|------------|--------------|
| Budget (<$20) | 0.586 |
| Value ($20–49) | 0.609 |
| Mid ($50–99) | **0.633** |
| Premium ($100+) | 0.622 |

> Sentiment is broadly positive (all > 0.5) but **Budget tier shows the lowest satisfaction** — suggesting quality-for-price expectations are unmet at the low end.

---

## 🤖 Multi-Agent AI Pipeline

This project simulates a production-style LLM agent architecture where specialised agents handle different analytical tasks, feed outputs to each other, and pass through a validation layer before synthesis.

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  products.csv · reviews.csv · users.csv · JSON files    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               ORCHESTRATOR AGENT                         │
│  Decomposes the business question into sub-tasks         │
│  Routes each task to the appropriate specialist          │
└──────┬──────────────────┬───────────────────────┬───────┘
       │                  │                       │
       ▼                  ▼                       ▼
┌──────────────┐  ┌───────────────┐  ┌───────────────────┐
│  SQL AGENT   │  │   EDA AGENT   │  │  MODELLING AGENT  │
│              │  │               │  │                   │
│ • JOIN queries│  │ • RFM scoring │  │ • K-Means (K=4)  │
│ • Price tiers │  │ • Age/geo     │  │ • Logistic Reg.  │
│ • Discounts  │  │   analysis    │  │ • PCA projection  │
│ • Correlation │  │ • Sentiment   │  │ • Confusion matrix│
└──────┬───────┘  └───────┬───────┘  └─────────┬─────────┘
       │                  │                     │
       └──────────────────┼─────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  CRITIC AGENT                            │
│  ✅ Validates correlation assumptions                    │
│  ✅ Checks RFM Champion count for business viability    │
│  ✅ Flags class imbalance in churn model                │
│  ⚠️  Triggers reruns if outputs fail quality checks     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               SYNTHESIS AGENT                            │
│  Translates validated findings into business language   │
│  Produces actionable recommendations for stakeholders   │
└─────────────────────────────────────────────────────────┘
```

In a production deployment, each agent would be a separate **LLM API call** (e.g. Claude or GPT-4) with a specialised system prompt, passing structured outputs downstream. The architecture demonstrates how real data science teams can use agentic AI to automate the full analytics lifecycle.

---

## 💡 Business Recommendations

Based on the complete analysis, the following actions are recommended for JCPenney leadership:

| Priority | Action | Evidence |
|----------|--------|---------|
| 🔴 Urgent | **Win-back campaign** for 602 At-Risk customers | RFM model — targeted personalised discount within 30 days |
| 🔴 Urgent | **Youth product line** — under-25 segment severely under-served | <25 cohort has ~160 reviewers vs 1,000+ in older groups |
| 🟡 High | **VIP loyalty programme** for 1,318 Champions | RFM — highest value customers, high retention ROI |
| 🟡 High | **Product quality investment** — not pricing | Pearson r = −0.009: price irrelevant to satisfaction |
| 🟢 Medium | **Dynamic discount strategy** for Budget tier | Lowest sentiment score despite cheapest products |
| 🟢 Medium | **Regional targeting** for Massachusetts, Delaware, Vermont | Highest user concentrations in data |
| 🟢 Medium | **Re-engagement of 750 Lost customers** via email campaigns | RFM Lost segment still reachable |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
pip install pandas numpy matplotlib scikit-learn scipy
```

### Option 1 — Jupyter Notebook (recommended)
```bash
git clone https://github.com/Imran3285/jcpenney-customer-targeting.git
cd jcpenney-customer-targeting

# Place data files in a 'data/' subfolder, or adjust paths in the notebook
jupyter notebook 3457775_BD2_Advanced.ipynb
```

### Option 2 — Python Script
```bash
python jcpenney_advanced_analysis.py
```

> **Note:** The script expects data files in the same directory by default. If your data is in a `data/` subfolder, update the file paths at the top of the script.

---

## 📦 Dependencies

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.1.0
scipy>=1.9.0
```

Install all at once:
```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

---

## 📁 Output Figures

All figures are saved automatically to the `figures/` directory when the script runs.

| Figure | Description |
|--------|-------------|
| `fig1_clustering.png` | K-Means elbow curve (K=2–8) + PCA 2D cluster scatter |
| `fig2_rfm_segments.png` | Customer count per RFM segment (Champions → Lost) |
| `fig3_churn_confusion.png` | Logistic Regression confusion matrix with class labels |
| `fig4_discount_by_category.png` | SQL result: top 12 categories by average discount % |
| `fig5_state_scores.png` | SQL JOIN result: average review score by US state |
| `fig6_sentiment_by_tier.png` | Keyword sentiment polarity across 4 price tiers |
| `fig7_cluster_profiles.png` | Normalised parallel-coordinate cluster feature comparison |
| `fig8_price_vs_score.png` | Scatter + regression line: price vs score (Pearson r shown) |

---

## 🗄️ SQL Queries Included

The notebook runs **4 SQL queries** against an in-memory SQLite engine loaded with all datasets:

```sql
-- 1. Average score per state (JOIN across reviews + users)
SELECT u.State, ROUND(AVG(r.Score),3) AS avg_score, COUNT(*) AS review_count
FROM reviews r JOIN users u ON r.Username = u.Username
GROUP BY u.State HAVING review_count >= 10 ORDER BY avg_score DESC LIMIT 15;

-- 2. Discount analysis by product category
SELECT category, ROUND(AVG(discount_pct),2) AS avg_discount_pct
FROM jpp GROUP BY category ORDER BY avg_discount_pct DESC LIMIT 12;

-- 3. Price tier breakdown using CASE logic
SELECT CASE WHEN Price < 20 THEN 'Budget' WHEN Price < 50 THEN 'Value'
            WHEN Price < 100 THEN 'Mid' ELSE 'Premium' END AS tier,
       COUNT(*) AS count FROM products GROUP BY tier;

-- 4. Price vs satisfaction correlation (JOIN products + reviews)
SELECT p.Price, r.Score FROM products p JOIN reviews r ON p.Uniq_id = r.Uniq_id;
```

---

## 🎓 Academic Context

| Field | Detail |
|-------|--------|
| Module | ITNPBD2 — Representing and Manipulating Data |
| University | University of Stirling |
| Student ID | 3457775 |
| Semester | Autumn 2025 |
| AIAS Level | 2 (AI used for drafting assistance; all analysis and code is original) |

---

## 📜 Licence

This project is submitted as academic coursework. The code and analysis are original work by the student. Data files are provided by the module convenor for educational use only.

---

<div align="center">

*Built with Python · Analysed with SQL · Deployed with Git*  
**University of Stirling — MSc Data Science**

</div>

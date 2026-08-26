# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-yellow?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

A machine learning system for detecting fraudulent credit card transactions across 1.29M+ transaction records, achieving a ROC-AUC of 0.983. The project also includes a cost-benefit analysis estimating potential financial impact under defined assumptions.

---

## 📁 Repository Structure

```
CC-Fraud-Detection/
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb   # Main analysis notebook
│
├── report/
│   └── CreditCard_FraudDetection_Presentation.pptx  # Final presentation
│
└── README.md
```

---

## 📌 Problem Statement

Finex Bank processes hundreds of thousands of transactions monthly. Only **0.52%** are fraudulent — but each fraud case costs an average of **$530.66**. The challenge:

- Extreme class imbalance (99.48% legit vs 0.52% fraud)
- Standard accuracy metrics are misleading
- Need to minimize both missed fraud (False Negatives) and false alerts (False Positives)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Records | 1,296,675 |
| Features | 23 |
| Fraud Rate | 0.52% |
| Missing Values | 0 |
| Time Period | 12 months |

---

## 🔍 Key EDA Findings

| Finding | Insight |
|---|---|
| **Transaction Amount** | Fraud avg = $530.66 vs Legit avg = $67.65 (7.8× higher) |
| **Top Fraud Category** | shopping_net (1.6%), misc_net (1.3%), grocery_pos (1.26%) |
| **Time Pattern** | Night (10pm–5am) has 1.65% fraud rate vs 0.1% daytime |
| **Age Group** | Elderly (80–105) most targeted at 0.76% fraud rate |
| **City Hotspots** | Clearwater (3.2%) and Aurora (3.1%) highest fraud cities |

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| **Temporal Features** | Year, Month, Weekday, Hour from transaction timestamp |
| **Customer Age** | Derived from date of birth |
| **Haversine Distance** | Geographic distance between customer and merchant |
| **Columns Removed** | PII (first, last), redundant (unix_time), non-informative (zip, street) |

---

## 🤖 Model Building Journey

| # | Model | Imbalance Handling | F1 Score | ROC-AUC |
|---|---|---|---|---|
| 1 | Logistic Regression | Class Weight | 0.05 | 0.72 |
| 2 | Random Forest | Class Weight | 0.08 | 0.81 |
| 3 | Random Forest | SMOTE | 0.10 | 0.83 |
| 4 | XGBoost | scale_pos_weight | 0.192 | 0.989 |
| 5 | XGBoost | SMOTE | 0.299 | 0.975 |
| 6 | XGBoost + SelectFromModel | SMOTE | 0.436 | 0.977 |
| 7 ⭐ | **XGBoost + SFM + RandomSearchCV** | **SMOTE** | **0.538** | **0.983** |

---

## 🏆 Best Model Results

**XGBoost + SMOTE + SelectFromModel + RandomizedSearchCV**

| Metric | Score |
|---|---|
| ROC-AUC | **0.983** |
| F1 Score | **0.538** |
| Precision | 40.2% |
| Recall | ~85% |
| Feature Reduction | 2,193 → 184 (91.6% reduction) |

---

## 💰 Cost-Benefit Analysis

Using the best model on the test set (per month basis):

| | Before Model | After Model |
|---|---|---|
| Avg fraud transactions/month | 804 | — |
| Fraud caught/month (TF) | — | 200 |
| Fraud missed/month (FN) | — | 45 |
| Cost | **$426,651/month** | **$24,180/month** |
| **Monthly Savings** | | **✅ ~$402,471** |
| **Annual Savings** | | **✅ ~$4.83 Million** |

> Assumptions: avg fraud amount = $530.66, investigation cost per alert = $1.50

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Feature Selection | SelectFromModel |
| Hyperparameter Tuning | RandomizedSearchCV |

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/DurgaPrasad-237/CC-Fruad-Detection.git
cd CC-Fruad-Detection
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

3. Open the notebook
```bash
jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb
```

---

## 📈 Key Takeaways

- **SMOTE consistently outperformed** class-weight approach across all model architectures
- **Feature selection** reduced 91.6% of features with no AUC loss — leaner and faster model
- **XGBoost** was the best architecture due to its ability to handle tabular data with class imbalance
- **Accuracy is misleading** for imbalanced datasets — F1 and ROC-AUC are the right metrics
- Estimated potential annual savings: **~$4.83M under the project's defined cost-benefit assumptions.**

---

## 👤 Author

**Durga Prasad**
- GitHub: [@DurgaPrasad-237](https://github.com/DurgaPrasad-237)

---

⭐ If you found this project useful, please consider giving it a star!

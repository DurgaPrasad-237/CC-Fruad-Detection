# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-yellow?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

A machine learning system for detecting fraudulent credit card transactions across **1.29M+ transaction records**, achieving a **ROC-AUC of 0.983**. The project addresses severe class imbalance through SMOTE, feature engineering, feature selection, and hyperparameter tuning, and includes a cost-benefit analysis estimating potential financial impact under defined project assumptions.

---

## ⭐ Project Highlights

* **1.29M+** transaction records analyzed
* **0.52%** fraud rate
* **0.983 ROC-AUC** achieved by the final XGBoost model
* **0.538 F1-score**
* **40.2% precision**
* **~85% recall**
* **2,193 → 184 features** after feature selection
* **7 model configurations** evaluated
* SMOTE used to address severe class imbalance
* RandomizedSearchCV used for hyperparameter optimization

---

## 📁 Repository Structure

```text
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

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small fraction of total transactions.

This project focuses on detecting fraudulent transactions while addressing the challenges of:

* Extreme class imbalance (**99.48% legitimate vs 0.52% fraudulent**)
* Misleading accuracy in highly imbalanced datasets
* False negatives, where fraudulent transactions are missed
* False positives, where legitimate transactions are incorrectly flagged
* Identifying meaningful temporal, behavioral, and geographic patterns in transactions

The objective was to build and evaluate machine learning models that provide a better balance between **fraud detection and false-positive control**.

---

## 📊 Dataset

| Property       |     Value |
| -------------- | --------: |
| Total Records  | 1,296,675 |
| Features       |        23 |
| Fraud Rate     |     0.52% |
| Missing Values |         0 |
| Time Period    | 12 months |

The dataset contains transaction, customer, merchant, category, geographic, and demographic information used to identify patterns associated with fraudulent activity.

---

## 🔍 Exploratory Data Analysis

### Key Findings

| Finding                | Insight                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| **Transaction Amount** | Fraud average = $530.66 vs Legitimate average = $67.65             |
| **Top Fraud Category** | `shopping_net` (1.6%), `misc_net` (1.3%), `grocery_pos` (1.26%)    |
| **Time Pattern**       | Night (10 PM–5 AM) showed a 1.65% fraud rate                       |
| **Age Group**          | Customers aged 80–105 showed the highest fraud rate at 0.76%       |
| **City Hotspots**      | Clearwater (3.2%) and Aurora (3.1%) showed the highest fraud rates |

### Important Observation

Fraudulent transactions had a substantially higher average transaction amount than legitimate transactions, making transaction amount an important feature for downstream modeling.

---

## ⚙️ Feature Engineering

Several features were engineered to capture temporal, behavioral, and geographic fraud patterns.

| Feature Group           | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| **Temporal Features**   | Year, month, weekday, hour, and time period derived from transaction timestamp |
| **Customer Age**        | Age calculated from date of birth                                              |
| **Geographic Distance** | Haversine distance between customer and merchant locations                     |
| **Transaction Amount**  | Log transformation applied to reduce extreme right skew                        |
| **Columns Removed**     | PII and redundant/non-informative columns removed                              |

### Removed Columns

* `first`
* `last`
* `Unnamed: 0`
* `unix_time`
* `zip`
* `trans_num`
* `street`

The transaction amount showed extreme right skewness, so a logarithmic transformation was applied before modeling.

---

## 🤖 Model Building Journey

Seven model configurations were evaluated to progressively improve fraud detection performance.

|   # | Model                                              | Imbalance Handling |  F1 Score |   ROC-AUC |
| --: | -------------------------------------------------- | ------------------ | --------: | --------: |
|   1 | Logistic Regression                                | Class Weight       |     0.050 |     0.720 |
|   2 | Random Forest                                      | Class Weight       |     0.080 |     0.810 |
|   3 | Random Forest                                      | SMOTE              |     0.100 |     0.830 |
|   4 | XGBoost                                            | `scale_pos_weight` |     0.192 |     0.989 |
|   5 | XGBoost                                            | SMOTE              |     0.299 |     0.975 |
|   6 | XGBoost + SelectFromModel                          | SMOTE              |     0.436 |     0.977 |
| 7 ⭐ | **XGBoost + SelectFromModel + RandomizedSearchCV** | **SMOTE**          | **0.538** | **0.983** |

The modeling process progressed from a baseline Logistic Regression model to an optimized XGBoost pipeline incorporating imbalance handling, feature selection, and hyperparameter tuning.

---

## ⚖️ Handling Class Imbalance

The dataset contains only **0.52% fraudulent transactions**, making class imbalance one of the primary modeling challenges.

Two approaches were evaluated:

### Class Weights

Class weights increase the penalty for incorrectly classifying minority-class observations.

Used with:

* Logistic Regression
* XGBoost

### SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** was used to generate synthetic examples of the minority fraud class.

SMOTE improved the F1-score across the evaluated model configurations and was therefore used in the final XGBoost pipeline.

---

## 🏆 Best Model Results

### Final Model

**XGBoost + SMOTE + SelectFromModel + RandomizedSearchCV**

| Metric                       |           Score |
| ---------------------------- | --------------: |
| ROC-AUC                      |       **0.983** |
| F1 Score                     |       **0.538** |
| Precision                    |       **40.2%** |
| Recall                       |        **~85%** |
| Feature Reduction            | **2,193 → 184** |
| Feature Reduction Percentage |       **91.6%** |

### Feature Selection

SelectFromModel reduced the feature space from **2,193 features to 184 features**, representing a **91.6% reduction**, while maintaining comparable ROC-AUC performance.

### Hyperparameter Tuning

RandomizedSearchCV was used to tune the XGBoost model and improve the final F1-score from **0.436 to 0.538**.

---

## 📈 Model Evaluation

Because of the extreme class imbalance, accuracy alone was not considered an appropriate primary metric.

The models were evaluated using:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Classification Report
* ROC Curve

### Why F1 and ROC-AUC?

* **Precision** measures how many transactions flagged as fraud were actually fraudulent.
* **Recall** measures how many fraudulent transactions were successfully detected.
* **F1-score** balances precision and recall.
* **ROC-AUC** measures the model's ability to distinguish fraudulent transactions from legitimate transactions across classification thresholds.

---

## 💰 Cost-Benefit Analysis

A cost-benefit analysis was performed to estimate the potential financial impact of the final model under defined project assumptions.

### Assumptions

| Metric                                     |   Value |
| ------------------------------------------ | ------: |
| Average transactions/month                 | 154,366 |
| Average fraudulent transactions/month      |     804 |
| Average fraud transaction amount           | $530.66 |
| Investigation cost per flagged transaction |   $1.50 |
| Fraud transactions detected/month          |     200 |
| Fraud transactions missed/month            |      45 |

### Estimated Financial Impact

|                           |   Before Model |   After Model |
| ------------------------- | -------------: | ------------: |
| Fraud-related cost        | $426,651/month | $24,180/month |
| Estimated monthly savings |              — | **~$402,471** |
| Estimated annual savings  |              — |   **~$4.83M** |

> **Note:** These figures represent an estimated financial impact based on the project's defined assumptions and cost-benefit analysis. They should not be interpreted as verified production savings or savings generated from a deployed banking system.

---

## 🛠️ Tech Stack

| Category                | Tools                   |
| ----------------------- | ----------------------- |
| Programming Language    | Python 3.9+             |
| Data Processing         | Pandas, NumPy           |
| Visualization           | Matplotlib, Seaborn     |
| Machine Learning        | Scikit-learn, XGBoost   |
| Imbalance Handling      | imbalanced-learn, SMOTE |
| Feature Selection       | SelectFromModel         |
| Hyperparameter Tuning   | RandomizedSearchCV      |
| Development Environment | Jupyter Notebook        |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/DurgaPrasad-237/CC-Fruad-Detection.git
cd CC-Fruad-Detection
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### 3. Open the Notebook

```bash
jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb
```

---

## 📌 Key Takeaways

* **SMOTE improved minority-class detection** compared with the evaluated class-weight approaches.
* **XGBoost** provided the strongest overall performance among the evaluated model configurations.
* **SelectFromModel reduced the feature space by 91.6%**, from 2,193 to 184 features, while maintaining comparable ROC-AUC performance.
* **RandomizedSearchCV improved the final F1-score** from 0.436 to 0.538.
* **Accuracy alone is misleading** for highly imbalanced fraud datasets; precision, recall, F1-score, and ROC-AUC provide more useful evaluation.
* The final model achieved **0.983 ROC-AUC and ~85% recall** for fraudulent transactions.
* The cost-benefit analysis estimated approximately **$402K/month in potential savings under the project's defined assumptions**.

---

## 🎓 Project Context

This project was completed as a **Capstone Project for an Executive Diploma in Data Science & AI**.

The project demonstrates an end-to-end machine learning workflow:

```text
Raw Transaction Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Class Imbalance Handling
        ↓
Model Training
        ↓
Model Comparison
        ↓
Feature Selection
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Cost-Benefit Analysis
```

---

## 👤 Author

**Durga Prasad**

GitHub: [@DurgaPrasad-237](https://github.com/DurgaPrasad-237)

---

⭐ If you found this project useful, consider giving the repository a star.

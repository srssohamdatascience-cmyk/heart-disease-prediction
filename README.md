
# ❤️ Heart Disease Prediction System

## 📌 Project Overview
This project focuses on building an end-to-end machine learning system to predict the likelihood of heart disease based on patient clinical data. The goal is to assist in early risk identification using data-driven insights.

---

## 🎯 Objective
To develop a classification model that accurately predicts the presence of heart disease while prioritizing **recall**, given the critical nature of minimizing false negatives in healthcare.

---

## 📊 Dataset Description
- Total Records: 918
- Features: 11 input features + 1 target variable
- Target:
  - 0 → No Heart Disease
  - 1 → Heart Disease

---

## 🔍 Data Analysis & Key Findings

### 1. Data Quality
- No missing values or duplicates initially
- Identified **~18.7% invalid cholesterol values (0)** → treated as missing
- Applied **group-wise median imputation** to preserve class distribution

---

### 2. Feature Insights

#### Strong Predictors
- **Max Heart Rate (maxhr):** Lower values associated with disease
- **Oldpeak:** Higher values indicate higher risk
- **ST Slope (Flat):** Strong indicator of heart disease
- **Exercise-induced Angina:** Positively correlated with disease

#### Moderate / Weak Predictors
- **Cholesterol:** Initially misleading due to invalid values
- **Fasting Blood Sugar:** Limited variability but still contributes

---

### 3. Class Distribution
- Slightly imbalanced (55% positive, 45% negative)
- No resampling required

---

## ⚙️ Model Development

### Models Used:
- Logistic Regression (Baseline)
- Random Forest (Non-linear model)

---

## 📈 Model Performance

### Logistic Regression (Final Model)
- Accuracy: 89.1%
- Precision: 89.4%
- Recall: 91.1% ✅
- F1 Score: 90.3%
- ROC-AUC: 0.93

### Random Forest
- Slightly lower performance compared to Logistic Regression

---

## 🧠 Key Insight

> Logistic Regression outperformed Random Forest, indicating that the dataset exhibits **strong linear separability** and does not require complex non-linear modeling.

---

## 🔧 Pipeline Design

- Data Preprocessing:
  - Standard Scaling (Numerical Features)
  - One-Hot Encoding (Categorical Features)
- Implemented using **ColumnTransformer + Pipeline**
- Ensures reproducibility and prevents data leakage

---

## 🌐 Deployment

- Built an interactive web application using **Streamlit**
- Allows real-time prediction based on user input
- Displays:
  - Risk prediction
  - Probability score
  - Risk category (Low / Moderate / High)

---

## 🚀 Features of the App

- User-friendly interface
- Real-time predictions
- Interpretable output
- Clean and modular architecture

---

## 📚 What I Learned

- Importance of **data quality validation** (e.g., detecting invalid values like cholesterol = 0)
- Difference between **statistical vs practical feature importance**
- When simpler models outperform complex ones
- Building **production-ready ML pipelines**
- End-to-end deployment using Streamlit

---

## ⚠️ Disclaimer
This model is for educational purposes only and should not be used as a substitute for professional medical advice.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
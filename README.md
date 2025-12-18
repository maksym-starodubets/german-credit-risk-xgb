# German Credit Risk Analysis using XGBoost

## Project Overview
This project is an introductory machine learning case study developed as part of my academic learning. The objective is to apply a full end-to-end machine learning workflow to a real-world financial dataset by predicting customer credit risk using the **German Credit dataset**.

The project focuses on data preprocessing, model development, hyperparameter tuning, and model interpretation using **XGBoost**, a widely used gradient boosting algorithm.

## Problem Statement
The task is to classify customers as **good** or **bad** credit risks based on demographic and financial attributes. Accurate credit risk assessment is a key problem in banking and lending, and this project demonstrates how machine learning techniques can support such decisions.

## Dataset
- Source: German Credit Risk dataset  
- Target variable: `Risk` (good / bad)  
- Features include:
  - Personal attributes (age, sex, housing, job)
  - Financial information (credit duration, savings, checking account status, loan purpose)

## Methodology

### 1. Data Preprocessing
- Handling missing values using imputation
- One-hot encoding for categorical variables
- Train/validation split (80/20)

### 2. Modeling
- XGBoost classifier (`XGBClassifier`)
- Integrated preprocessing and modeling using `sklearn` pipelines

### 3. Hyperparameter Tuning
- GridSearchCV with 3-fold cross-validation
- Recursive refinement around best-performing parameters

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

### 5. Model Interpretability
- Feature importance analysis
- SHAP values for explainability

## Results
Validation set performance:
- Accuracy: **0.71**
- Precision: **0.72**
- Recall: **0.98**
- F1-score: **0.83**
- ROC AUC: **0.65**

The model achieves high recall, indicating strong performance in identifying high-risk customers, which is often a priority in credit risk applications.

## Key Learnings
- Practical use of machine learning pipelines
- Hyperparameter optimization strategies
- Handling mixed numerical and categorical data
- Interpreting model predictions using feature importance and SHAP
- Understanding trade-offs between evaluation metrics in imbalanced classification problems

## Future Improvements
- Improved feature engineering
- Class imbalance handling with alternative loss functions or sampling
- Probability calibration
- Comparison with simpler baseline models (e.g., logistic regression)


# Employee-Attrition-Performance-model
🚀 Employee Attrition Prediction – End-to-End ML Pipeline (XGBoost + Scikit-Learn)
📌 Project Overview:
Employee attrition (employees leaving a company) is a major challenge for HR teams.
This project builds a complete end-to-end machine learning pipeline that predicts whether an employee is likely to leave the company or not.

The model uses:

1.scikit-learn Pipelines for preprocessing
2.ColumnTransformer for numerical & categorical handling
3.XGBoost as the final classifier
4.EDA to extract meaningful HR insights
5.joblib to save the entire pipeline
6.Streamlit app for deployment 

This project follows industry-level ML workflow used in real companies.

🎯 Problem Statement:
HR teams want to predict which employees are at risk of leaving, so they can take proactive steps (salary adjustments, training, engagement programs).

You will build an ML system that predicts attrition using employee demographic, work environment, and performance data.

📊 Dataset
Dataset: IBM HR Analytics Employee Attrition & Performance (Kaggle)

Attrition (Yes/No)
EDA Summary:
From Exploratory Data Analysis:
🚨 Employees with Overtime = "Yes" have much higher attrition
🚨 Young employees (< 30 yrs) leave more
🚨 Low MonthlyIncome contributes to attrition
🚨 Low JobSatisfaction increases attrition
🚨 Employees with < 2 years at the company leave more
These insights help HR teams take action.


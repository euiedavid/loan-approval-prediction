# Loan Approval Prediction

Exploratory data analysis and machine learning for predicting bank loan approval outcomes.

## Business Problems Addressed

- **Improve loan approval assessment:** Develop a data-driven approach to classify loan applications as likely approved or rejected.
- **Identify approval patterns:** Analyze applicant demographics, financial characteristics, credit history, and property information to identify patterns associated with loan approval.
- **Address data quality issues:** Identify missing values and potential outliers that may affect analysis and predictive modeling.
- **Understand applicant financial profiles:** Examine income, loan amount, and loan-term distributions to understand variability and unusual observations in loan applications.
- **Support consistent decision-making:** Use a machine learning model to provide consistent, data-driven loan approval predictions.

## Project Overview

This project contains two Jupyter notebooks, each serving a different purpose:

### 1. `bank_loan_eda.ipynb`

This notebook focuses on **data understanding and exploratory data analysis (EDA)** of the loan dataset.

It includes:

- Data understanding
- Data quality assessment
- Data visualizations
- Statistical analysis
- Data interpretations
- Key EDA findings
- Conclusion

### 2. `bank_loan_pred.ipynb`

This notebook focuses on building and evaluating a machine learning model to predict bank loan approval outcomes.

The machine learning approach was based on a YouTube tutorial by **Lera Andronova**, which was used as a reference for the overall workflow and implementation. Parts of the code were rewritten and modified to practice writing Python independently and strengthen my understanding of the machine learning workflow.

## Data Analysis & Machine Learning Experience

- **Pandas:** Loaded and inspected the loan dataset, examined its dimensions, data types, non-null counts, and overall structure, identified missing values and duplicate records, and generated descriptive statistics.
- **Data Cleaning & EDA:** Analyzed numerical and categorical variables using mean, median, mode, standard deviation, skewness, frequency distributions, and proportions to understand the dataset's composition and distributions.
- **Data Visualization:** Used **Matplotlib and Seaborn** to visualize numerical distributions, categorical frequencies, potential outliers, and relationships between applicant characteristics and loan approval outcomes.
- **Statistical Analysis:** Applied the **IQR method** to identify potential outliers and used correlation analysis to examine relationships between numerical features.
- **Business Analysis:** Compared applicant characteristics against loan approval status to identify patterns, including differences in approval rates across categorical groups and the strong association between credit history and loan approval.
- **Feature Engineering:** Created additional predictive features such as **Total Income** and **Loan-to-Income ratio**, and converted `Dependents` into a numerical format for modeling.
- **Data Preprocessing:** Built separate preprocessing pipelines for numerical and categorical variables using median/mode imputation, standardization, and one-hot encoding.
- **Machine Learning:** Developed a **Logistic Regression** model to predict loan approval outcomes using a stratified train-test split.
- **Model Evaluation:** Evaluated model performance using **Accuracy, Precision, Recall, F1-score, ROC-AUC, and a Confusion Matrix**.
- **Cross-Validation & Tuning:** Applied **5-fold stratified cross-validation** and performed light hyperparameter tuning on Logistic Regression's `C` parameter using ROC-AUC as the selection metric.
- **Prediction:** Generated loan approval probabilities and Y/N predictions for the unseen test dataset and exported the results to CSV.

## Model Performance

The Logistic Regression model achieved the following results on the hold-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 83.7% |
| Precision | 89.2% |
| Recall | 87.1% |
| ROC-AUC | 87.2% |

The model correctly classified **103 out of 123 applications** in the hold-out set. It correctly identified 74 approved applications and 29 rejected applications.

## About the Dataset

**Source:** Analytics Vidhya — *Loan Prediction* by Anmol Kumar

The dataset contains information about loan applicants and their eligibility for a loan from **Dream Housing Finance Company**.

The target variable is `Loan_Status`, which indicates whether a loan application was approved (`Y`) or rejected (`N`).
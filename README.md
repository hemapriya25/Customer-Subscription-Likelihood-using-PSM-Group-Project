Predicting Customer Subscription Likelihood with Propensity Score Matching
This repository contains the end-to-end data science project analyzing the Bank Marketing Dataset. The project goes beyond standard predictive modelling to apply causal inference techniques, specifically Propensity Score Matching (PSM), to understand the true impact of customer attributes on their likelihood to subscribe to a term deposit.

🎯 Project Goal
The primary goal of this project was twofold:

To build a high-performing classification model to predict which customers are most likely to subscribe to a bank's term deposit.

To use Propensity Score Matching (PSM) to move beyond correlation and estimate the causal effect of specific customer attributes (like having a high bank balance or a housing loan) on their subscription decisions, while controlling for other confounding factors.

📂 Dataset
The analysis is based on the Bank Marketing Data Set from the UCI Machine Learning Repository.

Size: 45,211 instances

Features: 17 attributes, including customer demographics (age, job, marital status), financial status (balance, loans), and campaign interaction details.

Initial data processing involved handling "unknown" values and preparing the data for analysis.

📈 Exploratory Data Analysis (EDA)
The initial EDA revealed several key insights that guided the modelling process:

Imbalanced Target: The dataset is highly imbalanced, with a significantly lower number of 'yes' subscriptions compared to 'no'.

Financial Indicators: Customers with higher bank balances tended to subscribe more often.

Existing Loans: Customers without existing housing or personal loans had a higher propensity to subscribe.

<p align="center">
<img src="https://www.google.com/search?q=https://i.imgur.com/Gj9Xq7v.png" alt="EDA Visualizations" width="800">
</p>
<p align="center"><i>Sample visualizations showing the relationship between financial attributes and subscription outcomes.</i></p>

🛠️ Methodology
The project was structured in two main phases: Predictive Modelling and Causal Inference.

1. Predictive Modelling & Feature Importance
To first identify the most influential predictors, several machine learning models were built and evaluated.

Models Used: XGBoost, Random Forest, and Logistic Regression.

Feature Engineering: Categorical variables were one-hot encoded, and binary variables were label-encoded.

Key Findings: The XGBoost model provided the highest accuracy. SHAP (SHapley Additive exPlanations) analysis revealed that balance, housing, and loan were among the most significant features influencing the subscription outcome.

2. Causal Inference with Propensity Score Matching (PSM)
This was the core of the analysis, designed to isolate the causal impact of specific "treatments." The process was repeated for three separate treatments: having a high bank balance, having a housing loan, and having a personal loan.

The methodology for each treatment followed these steps:

Treatment Definition: A binary treatment variable was created (e.g., treatment = 1 if balance >= median_balance, 0 otherwise).

Propensity Score Estimation: A logistic regression model was trained on a rich set of covariates to predict the probability (propensity score) of a customer receiving the "treatment."

Matching: Treated and control units were matched 1:1 based on their propensity scores, using a caliper of 0.05 to ensure high-quality matches. This step creates a balanced, comparable dataset.

Balance Check: The distribution of propensity scores was visualized before and after matching to confirm that the matching process successfully created balanced treatment and control groups.

Effect Estimation: The Average Treatment Effect on the Treated (ATT) was calculated by comparing the mean subscription outcome of the matched treated and control groups. Statistical significance was confirmed using an independent t-test.

<p align="center">
<img src="https://www.google.com/search?q=https://i.imgur.com/8w3cE4r.png" alt="PSM Balance Check" width="800">
</p>
<p align="center"><i>An example of checking the propensity score balance before and after matching for the housing loan treatment.</i></p>

📊 Results & Conclusion
The analysis successfully identified the causal impact of several key factors:

High Bank Balance: The ATT was +0.0412, indicating that having an above-median bank balance has a positive and significant causal effect, increasing a customer's likelihood of subscribing by approximately 4.12 percentage points, all else being equal.

Housing Loan: The ATT was -0.0718, indicating that having a housing loan has a strong, negative causal effect, decreasing a customer's likelihood of subscribing by approximately 7.18 percentage points.

Personal Loan: The ATT was -0.0526, indicating that having a personal loan also has a significant negative causal effect, decreasing subscription likelihood by approximately 5.26 percentage points.

This project demonstrates the power of moving beyond predictive accuracy to understand the true causal drivers of customer behaviour. The insights generated provide a robust, evidence-based foundation for a more effective and targeted marketing strategy.

💻 Technologies Used
Python

Pandas & NumPy for data manipulation

Scikit-learn for machine learning models and preprocessing

**XGBoost

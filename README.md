# Customer Lifetime Value Prediction

An end-to-end machine learning project for predicting each customer's spending over the next 90 days using historical transaction behavior. The project combines leakage-safe temporal feature engineering, regression modeling, customer ranking, and a GenAI extension that converts predictions into plain-English explanations.

## Project Overview

Customer Lifetime Value (CLV) helps identify customers who are likely to generate higher future value. Transaction-level data is transformed into customer-level behavioral features and used to estimate future 90-day spending.

## Dataset

The project uses `LTV_Transactions.csv`, an e-commerce transaction dataset containing:

- **108,728 transactions**
- **500 unique customers**
- **5 product categories**
- **3 purchase channels**
- Columns: `CUSTOMER_ID`, `TRANSACTION_TIME`, `AMOUNT`, `PRODUCT_CATEGORY`, and `CHANNEL`

Each row represents an individual customer transaction. The data is transformed into customer-level behavioral features before modeling.

## Project Workflow

1. Data loading and exploratory data analysis
2. Leakage-safe cutoff and target-window definition
3. Customer-level feature engineering
4. Train/test split
5. XGBoost and Random Forest regression
6. Model evaluation and comparison
7. Feature importance analysis
8. Customer scoring and ranking
9. GenAI-based prediction explanations
10. Production limitations and deployment considerations

## Feature Engineering

Features are created only from transactions before the cutoff date to reduce target leakage. The prediction target is total customer spending during the future 90-day window.

Key engineered features include total historical spend, transaction count, average transaction amount, days since last purchase, 30/90/180-day spending, recent transaction frequency, average time between purchases, category shares, channel shares, and recent spending trend.

## Models and Evaluation

Two regression models were trained and compared: **XGBoost Regressor** and **Random Forest Regressor**. Performance was evaluated using RMSE, MAE, and R².

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| XGBoost | $3,575.91 | $2,885.34 | 0.71 |
| Random Forest | $3,588.83 | $2,847.27 | 0.71 |

XGBoost achieved the strongest overall performance and was selected for final customer scoring.

## Key Findings

Feature importance analysis showed that historical and recent spending behavior were the strongest predictors of future customer value. Important features included `spend_180d`, `total_spend`, `txn_count`, `txn_count_90d`, and `avg_days_between_txn`.

Overall, spending history and purchase frequency were more influential than product-category or channel preferences.

## Customer Ranking

The best-performing model was used to predict 90-day spending for the full customer set. Customers were then ranked by predicted future spend to identify high-value customers for potential retention, engagement, or account-management strategies.

## GenAI Extension

A GenAI component using **Qwen3-1.7B** translates model outputs into concise, plain-English explanations for non-technical users. The explanations use observed customer behavior to make predictions easier for business stakeholders to interpret.

## Production Considerations

Before production deployment, the pipeline should address temporal data leakage, cold-start customers, distribution shift, validation design, model monitoring, periodic retraining, and safeguards that keep GenAI explanations grounded in observed data.

## Conclusion

This project demonstrates a complete Customer Lifetime Value prediction workflow, from transaction-level data to interpretable customer-level insights. XGBoost and Random Forest were used to estimate future 90-day spending, with XGBoost selected as the final model based on overall performance.

The project also extends traditional machine learning with GenAI-generated explanations, helping bridge the gap between model predictions and non-technical decision makers.

## Repository Structure

```text
Customer-Lifetime-Value-Prediction/
├── Customer_Lifetime_Value_Prediction_GitHub.ipynb
├── LTV_Transactions.csv
└── README.md
```

## Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Random Forest · Transformers · Qwen3-1.7B


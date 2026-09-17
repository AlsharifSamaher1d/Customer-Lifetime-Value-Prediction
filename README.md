# Customer Lifetime Value Prediction

An end-to-end machine learning and GenAI project for predicting customer spending over the next 90 days from historical transaction behavior. The pipeline covers exploratory data analysis, leakage-safe temporal feature engineering, regression modeling, model comparison, feature interpretation, customer ranking, and plain-English GenAI explanations for business stakeholders.

## Project Overview

Customer Lifetime Value (CLV) analysis helps organizations understand which customers are likely to generate greater future value. Rather than relying only on past total spending, this project models future customer behavior using a combination of recency, frequency, monetary value, recent spending windows, purchase timing, category preferences, and purchase-channel behavior.

The main objective is to estimate **future 90-day customer spend**. The project therefore converts transaction-level records into a customer-level modeling dataset while preserving the temporal boundary between historical information and the future target period.

A second objective is interpretability. After identifying high-value customers, the pipeline uses a small language model to convert structured prediction context into concise explanations that can be read by non-technical stakeholders.

## Business Problem

A raw transaction history tells us what customers have already purchased, but businesses often need to know what is likely to happen next. A useful predictive system can support questions such as:

- Which customers are expected to generate the highest value during the next 90 days?
- Which aspects of historical purchasing behavior are most informative for future spending?
- Can model predictions be presented in language that account managers and other non-technical users can understand?

The output of this project is therefore not only a regression prediction, but also a ranked customer view and an interpretable explanation layer.

## Dataset

The project uses `LTV_Transactions.csv`, an e-commerce transaction dataset containing **108,728 transactions** from **500 unique customers**.

### Dataset Summary

| Attribute | Description |
|---|---|
| Transactions | 108,728 |
| Unique customers | 500 |
| Product categories | 5 |
| Purchase channels | 3 |
| Modeling level | Customer |
| Prediction target | Future 90-day spending |

### Original Variables

| Column | Description |
|---|---|
| `CUSTOMER_ID` | Unique customer identifier |
| `TRANSACTION_TIME` | Date/time associated with the transaction |
| `AMOUNT` | Monetary value of the transaction |
| `PRODUCT_CATEGORY` | Product category associated with the purchase |
| `CHANNEL` | Channel through which the transaction was made |

Each row represents an individual transaction. Because the prediction is made at customer level, these records are aggregated and transformed into behavioral features before model training.

## Project Workflow

The project follows the sequence below:

1. Load and inspect transaction data.
2. Perform exploratory data analysis to understand spending, categories, channels, and customer behavior.
3. Define a temporal cutoff separating historical information from the future prediction window.
4. Build customer-level features using only information available before the cutoff.
5. Construct `target_90d_spend` from transactions in the future window.
6. Prepare the modeling dataset and split customers into training and test sets.
7. Train XGBoost and Random Forest regression models.
8. Evaluate both models using RMSE, MAE, and R².
9. Inspect feature importance and compare model behavior.
10. Select the best-performing model and score the complete customer set.
11. Rank customers according to predicted 90-day spending.
12. Use GenAI to produce concise explanations for selected customer predictions.
13. Review limitations that would need to be addressed before production deployment.

## Exploratory Data Analysis

The EDA stage examines both transaction-level and customer-level behavior. The analysis focuses on the distribution of transaction amounts, activity across product categories and purchase channels, transaction frequency per customer, and historical customer spending.

This stage is also used to understand whether a small number of customers dominate spending and whether recent purchase behavior may contain useful predictive information. These observations guide the later feature-engineering strategy.

## Leakage-Safe Temporal Design

Time is a critical part of this project. Features and targets must represent different periods.

All predictive features are calculated from transactions that occurred **before the cutoff date**, while the target is calculated only from transactions in the **future 90-day window**. This prevents the model from using information that would not have been available at prediction time.

Conceptually:

```text
Historical transactions                 Future transactions
<------------------------------------->|<-------------------->
          Feature window               Cutoff    90-day target
```

For each customer:

```text
Historical behavior -> engineered features -> ML model -> predicted 90-day spend
```

This temporal separation is essential because leakage across the cutoff could make offline model performance appear stronger than it would be in a real deployment.

## Feature Engineering

Transaction-level history is converted into customer-level behavioral features. The final modeling dataset contains **23 predictive features**.

### Monetary Behavior

- `total_spend` — total historical customer spending.
- `avg_txn_amount` — average transaction value.
- `spend_30d` — spending during the most recent 30-day window.
- `spend_90d` — spending during the most recent 90-day window.
- `spend_180d` — spending during the most recent 180-day window.

### Purchase Frequency and Recency

- `txn_count` — total number of historical transactions.
- `txn_count_30d` — transaction count during the recent 30-day period.
- `txn_count_90d` — transaction count during the recent 90-day period.
- `days_since_last_purchase` — recency of the customer's most recent historical transaction.
- `avg_days_between_txn` — average time between customer purchases.

### Behavioral Trend

- `spend_trend_90d` — summarizes the direction of recent customer spending behavior.

### Category and Channel Behavior

The pipeline also creates customer-level shares representing how purchasing activity is distributed across product categories and channels. These variables capture behavioral preferences rather than spending magnitude alone.

Examples include category-share features such as `category_share_Apparel`, `category_share_Electronics`, and `category_share_Health & Beauty`, together with channel-share features such as `channel_share_mobile`.

## Prediction Target

The supervised-learning target is:

`target_90d_spend`

It represents the total amount spent by each customer during the future 90-day window. Customers with no qualifying future transactions are assigned a target value of zero.

In the resulting customer-level dataset, the target has a mean of approximately **$13,650.78** and a median of approximately **$11,558.83**, with observed values extending up to approximately **$55,835.58**.

## Modeling Setup

The customer-level dataset contains **500 customers** and **23 predictive features**. An 80/20 split produces:

- **400 training customers**
- **100 test customers**

Two tree-based regression algorithms are compared because they can model nonlinear relationships and interactions among customer-behavior features without requiring a strictly linear relationship between historical activity and future spending.

### XGBoost Regressor

The XGBoost configuration uses up to 500 trees, a learning rate of 0.05, and a maximum tree depth of 6. Early stopping is used to stop training when additional boosting rounds no longer improve evaluation performance. In the notebook run, training stopped after approximately 51 trees.

### Random Forest Regressor

The Random Forest model uses 500 trees with a maximum depth of 15. It provides a strong ensemble-tree baseline and an additional feature-importance perspective for comparison with XGBoost.

## Evaluation Metrics

Three regression metrics are used:

**RMSE (Root Mean Squared Error)** measures the typical prediction error while penalizing larger errors more strongly.

**MAE (Mean Absolute Error)** reports the average absolute difference between predicted and observed spending in the original monetary scale.

**R² (Coefficient of Determination)** describes how much of the variation in future customer spending is explained by the model on the test set.

## Model Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **XGBoost** | **$3,575.91** | $2,885.34 | **0.71** |
| Random Forest | $3,588.83 | **$2,847.27** | **0.71** |

Both models explain approximately **71% of the variation** in future 90-day customer spending on the test set. Their performance is very close: Random Forest has a slightly lower MAE, while XGBoost has a slightly lower RMSE. The notebook selects XGBoost as the final model using the highest R² criterion.

Predicted-versus-actual visualizations are also used to inspect how closely predictions follow observed spending and where larger prediction errors occur.

## Feature Importance

The two models provide complementary views of the strongest predictive signals.

### XGBoost

`spend_180d` is the dominant feature, followed by variables including `txn_count`, `txn_count_90d`, and `total_spend`. This indicates that recent monetary behavior and purchase frequency carry substantial predictive information.

### Random Forest

The strongest features include `total_spend`, `spend_180d`, and `avg_days_between_txn`, followed by transaction-frequency and recent-spending variables.

Across both models, historical spending and purchase frequency are substantially more influential than individual category or channel preferences. This is consistent with the broader EDA observation that customers with stronger historical purchasing behavior tend to be associated with higher future value.

## Customer Scoring and Ranking

After model comparison, the selected XGBoost model is applied to the complete customer feature set to generate `predicted_90d_spend`.

Customers are sorted from highest to lowest predicted future spend, producing a ranked list that could support use cases such as high-value customer identification, account prioritization, retention analysis, and targeted customer-management strategies.

The highest predictions in the current run are approximately **$48K** in expected 90-day spending, illustrating how the regression output can be converted into a practical customer-ranking layer.

## GenAI Extension

The project extends the predictive pipeline with a GenAI explanation layer using **Qwen3-1.7B** through the Hugging Face Transformers ecosystem.

For selected high-value customers, structured information such as predicted 90-day spend, historical spending, transaction count, and recent spending direction is passed to the language model. The system prompt requires JSON-only output containing an `explanation` field with exactly two plain-English sentences.

The generated response is parsed with `json.loads`, demonstrating a structured-output pattern rather than relying on unconstrained free-form generation.

The goal is not for the language model to create new predictions. The regression model remains responsible for the numeric prediction; GenAI is used only to communicate existing model and customer information in a form that is easier for a non-technical account manager to understand.

A templated fallback is included so that the notebook can still run end-to-end if the language model cannot be loaded in the execution environment.

## Production Considerations

This notebook demonstrates the complete analytical pipeline, but several issues require additional work before production deployment.

**Temporal leakage:** Any feature accidentally calculated using transactions after the cutoff can inflate offline performance and fail when genuinely future information is unavailable.

**Cold-start customers:** Customers with little or no transaction history before the cutoff provide weaker behavioral signals, making their predictions less reliable.

**Distribution shift:** Customer behavior, pricing, category mix, and purchasing patterns can change over time. Production systems therefore require monitoring and periodic retraining.

**Validation design:** Early stopping in this notebook checks performance against the test set. This can introduce optimistic bias into final reported performance. A production-quality evaluation should use separate training, validation, and final test sets.

**GenAI grounding:** Generated explanations should only restate information supported by model inputs and observed customer data. They should not invent numerical values, offers, causal claims, or unsupported customer characteristics.

## Potential Future Improvements

Possible extensions include using a dedicated train/validation/test temporal split, evaluating the model across multiple rolling cutoff dates, adding model-drift monitoring, developing a strategy for cold-start customers, tuning model hyperparameters with validation data, adding SHAP-based local explanations, and comparing the current regression approach with alternative CLV modeling techniques.

The GenAI layer could also be extended into a controlled natural-language interface over predictions, provided that generated responses remain grounded in validated model outputs and customer data.

## Repository Structure

```text
Customer-Lifetime-Value-Prediction/
├── Customer_Lifetime_Value_Prediction_GitHub.ipynb
├── LTV_Transactions.csv
└── README.md
```

## Technologies

| Area | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost, Random Forest |
| GenAI | Hugging Face Transformers, Qwen3-1.7B |
| Environment | Jupyter Notebook / Google Colab |

## Conclusion

This project demonstrates an end-to-end approach for transforming raw e-commerce transactions into forward-looking customer insights. Leakage-safe temporal feature engineering converts transaction history into customer-level predictors, while XGBoost and Random Forest provide strong nonlinear regression models for estimating future 90-day spending.

Both models achieved an R² of approximately **0.71**, with XGBoost selected for final customer scoring. Feature-importance analysis indicates that recent and historical spending, transaction frequency, and purchase timing provide the strongest signals for future customer value.

The final GenAI layer demonstrates how predictive analytics can be made more accessible to non-technical stakeholders without replacing the underlying predictive model. Together, the modeling, ranking, interpretation, and production-consideration stages provide a complete foundation for a practical customer-value analytics workflow.

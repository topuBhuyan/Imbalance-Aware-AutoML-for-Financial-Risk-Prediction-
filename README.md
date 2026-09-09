# Imbalance-Aware-AutoML-for-Financial-Risk-Prediction-
An experimental framework for evaluating imbalance-aware strategies in
Automated Machine Learning (AutoML) for financial risk prediction.

The project investigates how different class-imbalance handling strategies
affect minority-class recall, PR-AUC, F1-score, G-mean, and
cost-sensitive decision making.
---
## 📌 Project Overview
Financial risk prediction datasets often contain highly imbalanced classes.

For example, in fraud detection, the number of fraudulent transactions may
be much smaller than the number of legitimate transactions.

Traditional machine learning models can achieve high overall accuracy while
performing poorly on the minority class.

This project investigates whether imbalance-aware strategies can improve
minority-class detection while reducing financial decision cost.

The study compares multiple imbalance-handling strategies across different
datasets, random seeds, and AutoML engines.
---
## 🎯 Research Objectives
The main objectives of this project are:

1. Evaluate the effect of class imbalance on financial risk prediction.
2. Compare different imbalance-handling strategies.
3. Investigate the effectiveness of threshold tuning.
4. Compare XGBoost-GPU and FLAML AutoML engines.
5. Evaluate models using cost-sensitive performance metrics.
6. Determine whether the best strategies remain consistent across datasets
   and AutoML engines.
---
## 🔬 Experimental Design

The experiment consists of:
- 8 datasets
- 7 imbalance-handling strategies
- 10 random seeds
- 2 AutoML engines
- 1,120 total experimental runs

### Experimental Formula
8 Datasets × 7 Strategies × 10 Seeds × 2 Engines

= **1,120 Runs**
---

# 🧩 Imbalance Handling Strategies

Seven strategies are evaluated in this project.

| Strategy | Name | Description |
|----------|------|-------------|
| S0 | Default | No imbalance handling |
| S1 | SMOTE | Synthetic Minority Over-sampling Technique |
| S2 | Objective | Optimize PR-AUC / Average Precision |
| S3 | Class Weight | Assign higher importance to minority samples |
| S4 | Threshold | Tune the classification decision threshold |
| S5 | Combined | Objective optimization + threshold tuning |
| S6 | Full | SMOTE + objective optimization + threshold tuning |
---
# 📊 Datasets

The experiment uses eight datasets with different levels of class imbalance.

| Dataset | Approx. Minority Rate |
|---------|-----------------------|
| Credit Card Fraud | 0.17% |
| Bank Marketing | 11.7% |
| Churn | 14.1% |
| Default Taiwan | 22.1% |
| Adult Income | 23.9% |
| Credit G | 30.0% |
| Australian Credit | 44.5% |
| Credit Approval | 44.5% |

The datasets were selected to provide different levels of class imbalance,
allowing the strategies to be evaluated under both moderate and extreme
imbalance conditions.
---
# 🤖 AutoML Engines

Two AutoML/modeling engines are evaluated.

## 1. XGBoost-GPU

XGBoost is used as the main gradient-boosting model with GPU acceleration.

## 2. FLAML

FLAML is used as the second AutoML engine to determine whether the observed
strategy behavior remains consistent across different AutoML systems.
---
# ⚙️ Experimental Pipeline

The overall workflow is:

Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Train / Validation / Test Split
   ↓
Imbalance Handling Strategy
   ↓
AutoML Model Training
   ↓
Validation-based Threshold Optimization
   ↓
Test Set Prediction
   ↓
Metric Calculation
   ↓
Result Storage
   ↓
Statistical Analysis
   ↓
Visualization
   ↓
Final Comparison
---
# 💰 Cost-Sensitive Decision Making

This project uses an asymmetric financial cost.

False Negative (FN) = 5

False Positive (FP) = 1

Therefore:

Total Cost = 5 × FN + 1 × FP

A false negative is considered more expensive because missing a risky or
fraudulent case can be more harmful than incorrectly flagging a normal case.
---
# 📏 Evaluation Metrics

The models are evaluated using multiple metrics:

- PR-AUC
- ROC-AUC
- Recall
- Precision
- F1-score
- G-mean
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Total Cost

Higher values are preferred for most performance metrics, while lower
Total Cost is preferred.
---
# 🎚️ Threshold Optimization

A major component of this research is decision-threshold optimization.

Instead of always using the default:

threshold = 0.50

the project searches for a validation-set threshold that minimizes the
cost-sensitive objective.

This allows the model to make decisions according to the financial cost
of false negatives and false positives.
---
# 📈 Main Results

## XGBoost-GPU

The average strategy ranks are:

| Strategy | Average Rank |
|----------|--------------|
| S4 Threshold | 2.997 |
| S3 Class Weight | 3.175 |
| S5 Combined | 3.344 |
| S1 SMOTE | 4.173 |
| S6 Full | 4.186 |
| S0 Default | 4.767 |
| S2 Objective | 5.358 |

The Friedman test indicates statistically significant differences between
the strategies.

χ² = 97.285

p < 0.0001
---
## FLAML

The average strategy ranks are:

| Strategy | Average Rank |
|----------|--------------|
| S5 Combined | 2.958 |
| S4 Threshold | 3.523 |
| S3 Class Weight | 3.597 |
| S6 Full | 4.219 |
| S1 SMOTE | 4.333 |
| S2 Objective | 4.370 |
| S0 Default | 5.000 |

Friedman test:

χ² = 105.254

p < 0.0001
---
# 🏆 Overall Finding

The results show that:

**S3 (Class Weight), S4 (Threshold), and S5 (Combined) form the strongest
overall strategy group.**

The exact first-place strategy depends on the AutoML engine:

- XGBoost-GPU → S4 Threshold
- FLAML → S5 Combined

However, both engines identify S3, S4, and S5 as the strongest group.
---
# 📉 Key Insight

One of the most important findings is that threshold-aware strategies can
substantially improve minority-class recall and reduce cost.

The tuned threshold is substantially lower than the default 0.50 threshold,
allowing the model to identify more minority-class cases.
---
# 📊 Visualizations

The project generates multiple types of visualizations.

### Critical Difference Diagrams

Used to compare strategy rankings statistically.

### Bar Charts

Used to compare:

- PR-AUC
- Recall
- F1
- G-mean
- Total Cost

### Box Plots

Used to show the distribution and consistency of strategy performance.

### Dataset-wise Plots

Used to compare strategy performance across individual datasets.

### Heatmaps

Used to visualize:

Dataset × Strategy

relationships for different metrics.

### Efficiency Frontier

Shows the relationship between:

Mean Recall ↔ Mean Cost

The ideal region is:

**High Recall + Low Cost**

### PR and ROC Curves

Used to visualize classification performance on representative datasets.

### Cost Sensitivity Analysis

Tests whether strategy performance remains stable under different
False Negative : False Positive cost ratios.

---

# 📁 Project Structure

Suggested project organization:
```text
Imbalance-Aware-AutoML/
│
├── README.md
│
├── notebook/
│   └── experiment.ipynb
│
├── data/
│   └── README.md
│
├── results/
│   ├── results.csv
│   └── summary_tables/
│
├── figures/
│   ├── xgboost_gpu/
│   ├── flaml/
│   └── comparison/
│
├── src/
│   ├── preprocessing/
│   ├── strategies/
│   ├── models/
│   ├── evaluation/
│   └── visualization/
│
├── requirements.txt
│
└── thesis/
    └── thesis.pdfm

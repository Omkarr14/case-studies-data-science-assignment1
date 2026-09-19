# case-studies-data-science-assignment1

## Individual Task 1 — Data Analysis Code
This repository contains the machine learning analysis code and datasets used in 
Individual Task 1 and Individual Task 2 (COSC2669/COSC2816, RMIT University).

- `analysis.py` — Decision Tree and kNN models applied to two healthcare datasets (Task 1)
- `heart_disease_uci.csv` / `diabetes.csv` — source datasets (see report for citations)
- `summary_results_v2.csv` — Task 1 output evaluation metrics

## Individual Task 2 — Deliberation Analysis
- `task2_part2_analysis.py` — cross-validation comparison, learning curves, and Fairlearn 
  bias analysis extending the Task 1 models
- `cv_vs_single_split.csv` — single-split vs 5-fold cross-validation comparison results
- `fairness_by_sex_heart_dt.csv` — Fairlearn fairness metrics by sex for the Heart Disease 
  Decision Tree model
- `learning_curve_*.png` — learning curve plots for all four model/dataset combinations

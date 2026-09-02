# Merge Conflict Predictor

This repository contains a complete machine learning project built in Python using `scikit-learn`. The project tackles the software engineering problem of predicting merge conflicts in concurrent code branches based on code metrics, developer activity, and commit message keywords.
<br><br>

## Project Overview

Merge conflicts frequently disrupt software development workflows. This project implements an automated classification pipeline using the **Merge Conflicts Dataset** (`MergeConflictsDataset.csv`) to predict whether a merge operation will result in a conflict (`conflict = 1`) or succeed peacefully (`conflict = 0`).

### Key Steps Implemented:
1. **Data Inspection & Cleaning:** Removal of duplicate records and constant columns.
2. **Exploratory Data Analysis (EDA):** Visualizing class distributions, feature histograms, and correlation matrices to understand underlying patterns.
3. **Feature Engineering:** Creating domain-specific derived features (e.g., total lines changed, global collaboration intensity, change velocity).
4. **Data Preprocessing Pipelines:** Using `scikit-learn` Pipelines and `ColumnTransformer` combined with `SimpleImputer` (median) and `StandardScaler` to safely preprocess features and prevent data leakage.
5. **Model Training & Hyperparameter Tuning:** Exploring algorithms like **Random Forest** and **HistGradientBoostingClassifier**, applying `class_weight='balanced'` to handle severe class imbalance, and using `GridSearchCV` to optimize hyperparameters.
6. **Model Evaluation:** Using performance metrics suited for imbalanced classification (primarily **F1-Score** and confusion matrices) alongside learning curves to diagnose and mitigate overfitting.
7. **Explainable AI (XAI):** Using **SHAP (SHapley Additive exPlanations)** to interpret model predictions and uncover which code metrics drive conflict risk.
8. **Model Persistence:** Serializing the complete end-to-end pipeline using `joblib` for deployment readiness.
<br>

## Dataset Description

The dataset (`MergeConflictsDataset.csv`) consists of historical merge events containing 37 columns and 26,973 records. Key feature categories include:
* **Pull Request Status (`is pr`):** Indicator of whether a pull request was created prior to merging.
* **Code Modification Metrics:** Number of added lines, deleted lines, and files modified, added, deleted, or renamed.
* **Collaboration Metrics:** Number of distinct developers and commit counts on each parent branch (`devs parent1`, `devs parent2`, `nr commits1`, `nr commits2`).
* **Activity Density:** Commit density in the preceding week (`density1`, `density2`) and development duration (`time`).
* **Commit Message Text Features:** Keyword frequencies (e.g., `fix`, `bug`, `feature`, `refactor`, `update`) and statistical lengths (min, max, mean, median).
* **Target Variable (`conflict`):** Binary label indicating whether a merge conflict occurred (`1`) or not (`0`).
<br>

## Repository Structure

* `MergeConflictPredictor.ipynb`: The primary Jupyter Notebook containing the full end-to-end implementation.
* `MergeConflictsDataset.csv`: The underlying dataset.
<br>

## Getting Started & Usage

1. Clone the repository and ensure you have Python 3.8+ installed.
2. Install the required dependencies by running:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn shap joblib
   ```
   
<br>

## Loading and Using the Saved Model

You can load the trained pipeline in your own scripts to make predictions on new merge data without re-running the training loop:

```python
import joblib
import pandas as pd

# Load the saved pipeline
model = joblib.load("my_model.pkl")

# Load new raw merge data
new_data = pd.read_csv("new_merge_sample.csv")

# Predict conflict outcomes automatically (preprocessing included)
predictions = model.predict(new_data)
print(predictions)


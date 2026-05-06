# Malicious URL Detection

This repository contains a pipeline for detecting malicious URLs and classifying them into categories such as phishing, defacement, and benign.

## Project Structure

- `data/raw/malicious_phish.csv` - Main dataset containing `url` and `type` columns.
- `src/data/load_data.py` - Utility functions for loading the dataset.
- `src/features/structural_features.py` - Structural URL feature extraction (length, dots, query parameters, entropy, etc.).
- `src/features/nlp_features.py` - Character and token TF-IDF vectorizers for URL text features.
- `src/features/feature_pipeline.py` - Feature extraction pipeline and helper functions.
- `src/models/train_model.py` - Training pipeline with proper train/validation/test splits and leakage prevention.
- `src/models/evaluate_model.py` - Model evaluation helpers for reports, confusion matrices, feature names, and SHAP values.
- `src/visualization/plots.py` - Plotting utilities for confusion matrices, feature importance, and SHAP summaries.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap

## Workflow

1. Load the dataset from `data/raw/malicious_phish.csv`.
2. Extract structural URL features using `extract_basic_features`.
3. Fit TF-IDF vectorizers on the training split only, then transform validation and test data.
4. Combine structural and NLP features into a single feature matrix.
5. Train a `RandomForestClassifier` on the training set.
6. Evaluate on validation and test sets to check for overfitting and data leakage.

## Example Usage

```python
import pandas as pd
from src.models.train_model import train_model_with_validation

path = "data/raw/malicious_phish.csv"
df = pd.read_csv(path)

model, scaler, vectorizers, metrics = train_model_with_validation(df)

print("train accuracy:", metrics["train"]["accuracy"])
print("validation accuracy:", metrics["validation"]["accuracy"])
print("test accuracy:", metrics["test"]["accuracy"])
```

## Evaluation

Use `src.models.evaluate_model` for:
- Generating a classification report (`generate_classification_report`)
- Computing a confusion matrix (`compute_confusion_matrix`)
- Building combined feature names from the vectorizers (`get_feature_names`)
- Extracting model feature importance (`compute_feature_importance`)
- Sampling data for SHAP analysis (`sample_for_shap`)
- Computing SHAP values for tree-based models (`compute_shap_values`)

## Visualization

`src/visualization/plots.py` provides plots for:
- Confusion matrix
- Feature importance
- SHAP summary plots

## Notes

- Install the required packages before use.
- `train_model_with_validation` is designed to avoid data leakage by fitting vectorizers and scalers only on the training split.
- The project can be extended with additional models, feature sets, or analysis workflows.

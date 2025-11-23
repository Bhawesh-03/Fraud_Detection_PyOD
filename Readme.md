# Fraud Detection with PyOD AutoEncoder

This project trains an AutoEncoder model from the [PyOD](https://pyod.readthedocs.io/) library to detect fraudulent transactions on the Kaggle **Credit Card Fraud Detection** dataset.

## Project Overview

- **Dataset**: `creditcard.csv` (anonymized credit card transactions with severe class imbalance).
- **Model**: PyOD `AutoEncoder` (deep learning–based outlier detector).
- **Task**: Detect fraudulent transactions treated as anomalies.
- **Metrics**:
  - ROC-AUC
  - Precision-Recall AUC
  - Confusion matrix and classification report at a threshold based on the contamination rate.

## Files

- `train_pyod_autoencoder.py` – Main training and evaluation script.
- `requirements.txt` – Python dependencies (manifest file).
- `outputs/` – Created after running the script; contains:
  - `roc_curve.png` – ROC curve.
  - `pr_curve.png` – Precision-Recall curve.
  - `metrics.json` – ROC/PR AUC, threshold, confusion matrix, classification report.
  - `metrics.csv` – Confusion matrix as a CSV.
  - `autoencoder_pyod.joblib` – Trained AutoEncoder model.
  - `robust_scaler.joblib` – Fitted RobustScaler for preprocessing.

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraud Detection with AutoEncoder (PyOD) on the Kaggle Credit Card dataset.

- Loads data from data/creditcard.csv (you must place the file there), or pass --data <path>.
- Trains a PyOD AutoEncoder to detect anomalies (fraud).
- Evaluates with ROC-AUC and PR-AUC; saves confusion matrix at a chosen threshold.
- Writes metrics to outputs/metrics.json and outputs/metrics.csv
- Saves example plots to outputs/ (ROC, PR, CM)
- Saves trained model to outputs/autoencoder_pyod.joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from pyod.models.auto_encoder import AutoEncoder

RANDOM_STATE = 42


# -------------------- Data -------------------- #
def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}. "
            "Download 'creditcard.csv' from Kaggle and place it under the 'data' directory."
        )
    return pd.read_csv(csv_path)


"""
    Separate features/labels and scale features with RobustScaler.

    The dataset has a 'Class' column: 0 = normal, 1 = fraud.
    We scale all other columns to make the AutoEncoder training more stable.
    """
def prepare_features(df: pd.DataFrame):
    
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in the dataset.")
    y = df["Class"].astype(int).values
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, y, feature_cols, scaler


# -------------------- Plots -------------------- #
def plot_roc_pr(y_true, scores, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AutoEncoder (PyOD)")
    plt.legend(loc="lower right")
    plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - AutoEncoder (PyOD)")
    plt.legend(loc="lower left")
    plt.savefig(out_dir / "pr_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    return roc_auc, pr_auc


# -------------------- Eval -------------------- #
"""
    Choose a threshold based on the contamination rate.

    We take the (1 - contamination) quantile of the anomaly scores so that
    approximately `contamination` fraction of samples are flagged as fraud.
    """
def evaluate_threshold(y_true, scores, contamination=0.002):
    threshold = float(np.quantile(scores, 1 - contamination))
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return threshold, cm, report


# -------------------- Main -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/creditcard.csv")
    parser.add_argument("--outputs", type=str, default="outputs")
    # Keep your original arg name but map to PyOD 2.x parameter
    parser.add_argument("--epochs_num", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--contamination", type=float, default=0.002)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    df = load_data(data_path)
    X, y, feature_cols, scaler = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("[INFO] Training AutoEncoder...")

    # ---- PyOD 2.x first; fallback to old Keras-based version if needed ----
    try:
        # PyOD >= 2.x (Torch-based AutoEncoder)
        ae = AutoEncoder(
            hidden_neuron_list=[64, 32],
            epoch_num=args.epochs_num,
            batch_size=args.batch_size,
            contamination=args.contamination,
            dropout_rate=0.0,
            verbose=1,
            random_state=RANDOM_STATE,
        )
    except TypeError:
        # PyOD 1.x (Keras-based AutoEncoder)
        ae = AutoEncoder(
            hidden_neurons=[64, 32],
            epochs=args.epochs_num,
            batch_size=args.batch_size,
            contamination=args.contamination,
            dropout_rate=0.0,
            verbose=1,
            random_state=RANDOM_STATE,
        )


    ae.fit(X_train)

    # PyOD decision_function: higher score = more abnormal
    scores = ae.decision_function(X_test)

    roc_auc, pr_auc = plot_roc_pr(y_test, scores, out_dir)
    threshold, cm, report = evaluate_threshold(y_test, scores, args.contamination)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(out_dir / "metrics.csv", index=False)

    joblib.dump(ae, out_dir / "autoencoder_pyod.joblib")
    joblib.dump(scaler, out_dir / "robust_scaler.joblib")
    print(f"[DONE] ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()

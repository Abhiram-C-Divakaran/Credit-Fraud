"""
Real-Data Benchmark Evaluation Pipeline: Kaggle / ULB Credit Card Fraud Dataset
==============================================================================
Evaluates baseline and Random Forest models on real-world credit card transactions
using a leakage-conscious chronological train/validation/test split.

Important Notice:
    The Kaggle/ULB dataset uses 28 anonymized PCA components (V1–V28) plus Time and Amount.
    These anonymized features do NOT correspond to the semantic inputs in the browser demo
    (distance, hour, foreign txn, frequency). This benchmark is an offline empirical pipeline
    and does not replace the interactive browser demo.

Usage:
    python evaluate_real_data.py [--data-path data/creditcard.csv] [--report-dir reports/]
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Expected dataset specification (Kaggle / ULB Machine Learning Group)
REQUIRED_PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time"] + REQUIRED_PCA_COLUMNS + ["Amount", "Class"]
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
DATASET_CITATION = "Credit Card Fraud Detection Dataset, Worldline and the Machine Learning Group (ULB), Kaggle, 2018."


def compute_file_sha256(filepath: str, block_size: int = 65536) -> str:
    """Compute the SHA256 checksum of the local dataset file for auditability."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()


def print_missing_dataset_help(data_path: str):
    """Print clear, actionable instructions when the dataset file is absent."""
    print("=" * 75)
    print("  [INFO] Real-World Dataset Not Found")
    print(f"  Target path: {data_path}")
    print("=" * 75)
    print("\nTo evaluate models against the real Kaggle / ULB Credit Card Fraud dataset:\n")
    print("  1. Download 'creditcard.csv' from Kaggle:")
    print(f"     {DATASET_SOURCE_URL}")
    print("\n  2. Create a local 'data/' directory (this folder is ignored by git):")
    print("     mkdir data")
    print("\n  3. Place the downloaded file at:")
    print(f"     {os.path.abspath(data_path)}")
    print("\n  4. Re-run this evaluation benchmark:")
    print(f"     python evaluate_real_data.py --data-path {data_path}")
    print("\n" + "=" * 75)
    print("Note: The browser demo remains active and runs on the synthetic model.")
    print("=" * 75)


def validate_dataset_columns(df: pd.DataFrame) -> None:
    """Validate that the dataset contains all required columns and valid classes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} required columns: {missing}\n"
            f"Expected 31 columns: Time, V1-V28, Amount, Class."
        )

    if len(df) == 0:
        raise ValueError("Dataset is empty (0 rows).")

    unique_classes = set(df["Class"].dropna().unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(f"Target column 'Class' contains invalid labels: {unique_classes}. Expected only 0 and 1.")


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.60, val_ratio: float = 0.20):
    """
    Split data chronologically by the 'Time' column to prevent future temporal leakage.

    Why chronological splitting:
        Credit card transactions are sequential events. In real banking systems, models
        are trained on historical data and tested on future transactions. A random split
        leaks future fraud attacks and transient card-testing bursts into the training set,
        drastically overstating real-world detection ability.
    """
    df_sorted = df.sort_values(by="Time").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    return train_df, val_df, test_df


def tune_decision_threshold(y_true: np.ndarray, y_probs: np.ndarray, num_steps: int = 50) -> float:
    """Find the probability threshold that maximizes F1 score strictly on validation data."""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.05, 0.95, num_steps):
        preds = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return best_thresh


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate comprehensive evaluation metrics suited for severe class imbalance."""
    preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    pr_auc = float(average_precision_score(y_true, y_probs))
    roc_auc = float(roc_auc_score(y_true, y_probs))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def run_benchmark(data_path: str = "data/creditcard.csv", report_dir: str = "reports", fast_mode: bool = False):
    """Run the complete real-data benchmark pipeline."""
    if not os.path.exists(data_path):
        print_missing_dataset_help(data_path)
        return None

    print("=" * 75)
    print("  Evaluating Kaggle / ULB Real-World Credit Card Fraud Benchmark")
    print("=" * 75)

    # 1. Inspect and validate file
    print(f"\n[1/5] Loading and verifying {data_path}...")
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    file_hash = compute_file_sha256(data_path)
    print(f"      • File size: {file_size_mb:.2f} MB")
    print(f"      • SHA256:   {file_hash[:16]}...{file_hash[-16:]}")

    df = pd.read_csv(data_path)
    validate_dataset_columns(df)
    n_total = len(df)
    n_fraud = int(df["Class"].sum())
    n_legit = n_total - n_fraud
    fraud_pct = (n_fraud / n_total) * 100
    print(f"      • Total rows: {n_total:,}")
    print(f"      • Legitimate: {n_legit:,} ({100 - fraud_pct:.3f}%)")
    print(f"      • Fraudulent: {n_fraud:,} ({fraud_pct:.3f}%)")
    print(f"      • Class imbalance ratio: ~1 fraud per {int(n_legit / max(1, n_fraud)):,} transactions")

    # 2. Chronological Train / Val / Test Split
    print("\n[2/5] Performing chronological split by 'Time' (60% Train, 20% Val, 20% Test)...")
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.60, val_ratio=0.20)

    feature_cols = [c for c in df.columns if c != "Class"]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["Class"].values
    X_val = val_df[feature_cols].copy()
    y_val = val_df["Class"].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df["Class"].values

    print(f"      • Train split: {len(X_train):,} samples (Frauds: {y_train.sum():,})")
    print(f"      • Val split:   {len(X_val):,} samples (Frauds: {y_val.sum():,})")
    print(f"      • Test split:  {len(X_test):,} samples (Frauds: {y_test.sum():,})")

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split does not contain both legitimate and fraud classes. Ensure dataset has fraud examples in the early timeline.")
    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation split does not contain both legitimate and fraud classes.")
    if len(np.unique(y_test)) < 2:
        raise ValueError("Test split does not contain both legitimate and fraud classes.")

    # 3. Leakage-Free Preprocessing (Fit on train only)
    print("\n[3/5] Preprocessing: Fitting RobustScaler strictly on training data...")
    # Amount and Time have high skewness/outliers; PCA components are already normalized
    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_val_scaled[["Amount", "Time"]] = scaler.transform(X_val[["Amount", "Time"]])
    X_test_scaled[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    # 4. Train Models
    print("\n[4/5] Training models...")
    # Model A: Baseline Class-Weighted Logistic Regression
    print("      • Training Baseline (Logistic Regression, class_weight='balanced')...")
    baseline = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    baseline.fit(X_train_scaled, y_train)

    val_probs_base = baseline.predict_proba(X_val_scaled)[:, 1]
    best_thresh_base = tune_decision_threshold(y_val, val_probs_base)
    test_probs_base = baseline.predict_proba(X_test_scaled)[:, 1]
    eval_base_std = evaluate_predictions(y_test, test_probs_base, threshold=0.5)
    eval_base_tuned = evaluate_predictions(y_test, test_probs_base, threshold=best_thresh_base)

    # Model B: Random Forest Classifier
    n_trees = 30 if fast_mode else 100
    print(f"      • Training Random Forest ({n_trees} trees, max_depth=10, class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    val_probs_rf = rf.predict_proba(X_val_scaled)[:, 1]
    best_thresh_rf = tune_decision_threshold(y_val, val_probs_rf)
    test_probs_rf = rf.predict_proba(X_test_scaled)[:, 1]
    eval_rf_std = evaluate_predictions(y_test, test_probs_rf, threshold=0.5)
    eval_rf_tuned = evaluate_predictions(y_test, test_probs_rf, threshold=best_thresh_rf)

    # 5. Display & Save Results
    print("\n[5/5] Final Test Results (Held-Out Future Test Split):")
    print("-" * 75)
    print(f"{'Model':<24} {'PR-AUC':<9} {'ROC-AUC':<9} {'Precision':<11} {'Recall':<9} {'F1':<8} {'FPR':<8}")
    print("-" * 75)
    print(f"{'Baseline (thr=0.5)':<24} {eval_base_std['pr_auc']:<9} {eval_base_std['roc_auc']:<9} {eval_base_std['precision']:<11} {eval_base_std['recall']:<9} {eval_base_std['f1_score']:<8} {eval_base_std['false_positive_rate']:<8}")
    print(f"{f'Baseline (thr={best_thresh_base})':<24} {eval_base_tuned['pr_auc']:<9} {eval_base_tuned['roc_auc']:<9} {eval_base_tuned['precision']:<11} {eval_base_tuned['recall']:<9} {eval_base_tuned['f1_score']:<8} {eval_base_tuned['false_positive_rate']:<8}")
    print(f"{'Random Forest (thr=0.5)':<24} {eval_rf_std['pr_auc']:<9} {eval_rf_std['roc_auc']:<9} {eval_rf_std['precision']:<11} {eval_rf_std['recall']:<9} {eval_rf_std['f1_score']:<8} {eval_rf_std['false_positive_rate']:<8}")
    print(f"{f'Random Forest (thr={best_thresh_rf})':<24} {eval_rf_tuned['pr_auc']:<9} {eval_rf_tuned['roc_auc']:<9} {eval_rf_tuned['precision']:<11} {eval_rf_tuned['recall']:<9} {eval_rf_tuned['f1_score']:<8} {eval_rf_tuned['false_positive_rate']:<8}")
    print("-" * 75)

    os.makedirs(report_dir, exist_ok=True)
    report_json_path = os.path.join(report_dir, "real_data_evaluation.json")
    report_md_path = os.path.join(report_dir, "real_data_evaluation.md")

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_metadata": {
            "source_name": "Kaggle / ULB Machine Learning Group",
            "source_url": DATASET_SOURCE_URL,
            "citation": DATASET_CITATION,
            "file_sha256": file_hash,
            "file_size_mb": round(file_size_mb, 2),
            "row_count": n_total,
            "features_count": len(feature_cols),
            "feature_names": feature_cols,
            "class_distribution": {
                "legitimate_count": n_legit,
                "fraudulent_count": n_fraud,
                "fraud_percentage": round(fraud_pct, 4),
            },
        },
        "split_methodology": {
            "method": "Chronological split by 'Time'",
            "rationale": "Prevents temporal data leakage and look-ahead bias in sequential transactions.",
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "preprocessing": {
            "scaler": "RobustScaler applied to Amount and Time",
            "fit_scope": "Exclusively on X_train",
        },
        "results": {
            "baseline_logistic_regression": {
                "validation_tuned_threshold": best_thresh_base,
                "test_at_standard_threshold": eval_base_std,
                "test_at_tuned_threshold": eval_base_tuned,
            },
            "random_forest": {
                "validation_tuned_threshold": best_thresh_rf,
                "test_at_standard_threshold": eval_rf_std,
                "test_at_tuned_threshold": eval_rf_tuned,
            },
        },
        "disclaimer": (
            "Empirical evaluation on historical European cardholder transactions (Sept 2013). "
            "Results reflect specific historical PCA components and do not establish production readiness, "
            "adversarial robustness against modern evasions, or transferability across financial institutions."
        ),
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Real-World Benchmark Evaluation Report\n\n")
        f.write(f"- **Evaluated**: {report_payload['timestamp']}\n")
        f.write(f"- **Dataset**: {DATASET_CITATION}\n")
        f.write(f"- **SHA256**: `{file_hash}`\n")
        f.write(f"- **Samples**: {n_total:,} ({n_fraud:,} frauds, {fraud_pct:.3f}%)\n")
        f.write(f"- **Split**: Chronological 60% Train / 20% Val / 20% Test\n\n")
        f.write("## Test Performance (Held-out chronological test split)\n\n")
        f.write("| Model | PR-AUC | ROC-AUC | Precision | Recall | F1-Score | False Positive Rate |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| Baseline LR (0.50) | {eval_base_std['pr_auc']} | {eval_base_std['roc_auc']} | {eval_base_std['precision']} | {eval_base_std['recall']} | {eval_base_std['f1_score']} | {eval_base_std['false_positive_rate']} |\n")
        f.write(f"| Baseline LR (Tuned: {best_thresh_base}) | {eval_base_tuned['pr_auc']} | {eval_base_tuned['roc_auc']} | {eval_base_tuned['precision']} | {eval_base_tuned['recall']} | {eval_base_tuned['f1_score']} | {eval_base_tuned['false_positive_rate']} |\n")
        f.write(f"| Random Forest (0.50) | {eval_rf_std['pr_auc']} | {eval_rf_std['roc_auc']} | {eval_rf_std['precision']} | {eval_rf_std['recall']} | {eval_rf_std['f1_score']} | {eval_rf_std['false_positive_rate']} |\n")
        f.write(f"| Random Forest (Tuned: {best_thresh_rf}) | {eval_rf_tuned['pr_auc']} | {eval_rf_tuned['roc_auc']} | {eval_rf_tuned['precision']} | {eval_rf_tuned['recall']} | {eval_rf_tuned['f1_score']} | {eval_rf_tuned['false_positive_rate']} |\n\n")
        f.write("## Production Disclaimer\n\n")
        f.write(f"> {report_payload['disclaimer']}\n")

    print(f"\nSaved benchmark reports:")
    print(f"  • {report_json_path}")
    print(f"  • {report_md_path}")
    print("=" * 75)
    return report_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate real-world Kaggle/ULB Credit Card Fraud dataset using a chronological split."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/creditcard.csv",
        help="Local path to creditcard.csv (default: data/creditcard.csv)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation reports (default: reports)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode with reduced estimators for quick sanity checks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_benchmark(
        data_path=args.data_path,
        report_dir=args.report_dir,
        fast_mode=args.fast,
    )
    if result is None:
        sys.exit(1)

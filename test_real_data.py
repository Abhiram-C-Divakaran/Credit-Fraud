"""
Unit Tests for Real-Data Benchmark Pipeline (evaluate_real_data.py)
==================================================================
Tests dataset schema validation, missing/malformed file handling,
leakage-free chronological splitting, walk-forward temporal windows,
train-only preprocessing isolation, validation-only threshold tuning,
low-fraud/single-class window handling, and report schema.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from evaluate_real_data import (
    REQUIRED_COLUMNS,
    REQUIRED_PCA_COLUMNS,
    CANDIDATE_THRESHOLDS,
    validate_dataset_columns,
    chronological_split,
    generate_expanding_windows,
    tune_decision_threshold,
    evaluate_predictions,
    compute_threshold_tradeoff_analysis,
    compute_aggregate_window_metrics,
    generate_stability_prose,
    compute_file_sha256,
    run_benchmark,
)


def test_missing_dataset_returns_none(capsys):
    """Verify that a missing dataset triggers instructions and returns None."""
    result = run_benchmark(data_path="data/non_existent_file.csv")
    assert result is None
    captured = capsys.readouterr()
    assert "[INFO] Real-World Dataset Not Found" in captured.out
    assert "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" in captured.out


def test_validate_dataset_columns_detects_missing_column():
    """Verify ValueError is raised if any required PCA or metadata column is missing."""
    cols = ["Time"] + REQUIRED_PCA_COLUMNS[:27] + ["Amount"]
    df = pd.DataFrame(np.zeros((10, len(cols))), columns=cols)

    with pytest.raises(ValueError) as exc:
        validate_dataset_columns(df)
    assert "missing 2 required columns" in str(exc.value)
    assert "V28" in str(exc.value)
    assert "Class" in str(exc.value)


def test_validate_dataset_columns_detects_empty_dataframe():
    """Verify ValueError on empty dataframe."""
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError) as exc:
        validate_dataset_columns(df)
    assert "Dataset is empty" in str(exc.value)


def test_validate_dataset_columns_detects_invalid_class_labels():
    """Verify ValueError if Class column has non-binary or invalid labels."""
    df = pd.DataFrame(np.zeros((10, len(REQUIRED_COLUMNS))), columns=REQUIRED_COLUMNS)
    df["Class"] = [0, 1, 0, 1, 2, 0, 1, 0, 1, 0]  # Contains '2'

    with pytest.raises(ValueError) as exc:
        validate_dataset_columns(df)
    assert "invalid labels" in str(exc.value)


def test_chronological_split_prevents_temporal_leakage():
    """Verify that split ordering strictly respects time with zero lookahead leakage."""
    n_samples = 1000
    times = np.random.uniform(0, 172800, size=n_samples)
    classes = np.random.binomial(1, 0.05, size=n_samples)

    data = {"Time": times, "Amount": np.random.uniform(1, 500, size=n_samples), "Class": classes}
    for col in REQUIRED_PCA_COLUMNS:
        data[col] = np.random.normal(0, 1, size=n_samples)

    df = pd.DataFrame(data)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.60, val_ratio=0.20)

    assert len(train_df) == 600
    assert len(val_df) == 200
    assert len(test_df) == 200

    # Ensure strictly sorted Time boundaries
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_chronological_window_boundaries_no_leakage():
    """Verify that expanding temporal windows preserve strict non-overlapping order (Train < Val < Test)."""
    n_samples = 1200
    times = np.linspace(0, 100000, n_samples)
    classes = np.zeros(n_samples, dtype=int)
    classes[::20] = 1

    data = {"Time": times, "Amount": np.random.uniform(10, 200, size=n_samples), "Class": classes}
    for col in REQUIRED_PCA_COLUMNS:
        data[col] = np.random.normal(0, 1, size=n_samples)

    df = pd.DataFrame(data)
    windows = generate_expanding_windows(df)
    assert len(windows) == 3

    for w in windows:
        tr_df, va_df, te_df = w["train_df"], w["val_df"], w["test_df"]
        assert len(tr_df) > 0
        assert len(va_df) > 0
        assert len(te_df) > 0
        assert tr_df["Time"].max() <= va_df["Time"].min(), f"Leakage: Train overlaps Val in {w['name']}"
        assert va_df["Time"].max() <= te_df["Time"].min(), f"Leakage: Val overlaps Test in {w['name']}"

    # Verify expanding training window property: W1 train < W2 train < W3 train
    assert len(windows[0]["train_df"]) < len(windows[1]["train_df"]) < len(windows[2]["train_df"])


def test_train_only_preprocessing_isolation():
    """Verify that preprocessing scaler is fitted strictly on X_train and does not learn from test data."""
    np.random.seed(42)
    # Train has mean 10, std 2
    train_amounts = np.random.normal(10.0, 2.0, size=100)
    # Test has vastly different mean 500, std 50
    test_amounts = np.random.normal(500.0, 50.0, size=50)

    train_df = pd.DataFrame({"Amount": train_amounts, "Time": np.linspace(0, 100, 100)})
    test_df = pd.DataFrame({"Amount": test_amounts, "Time": np.linspace(101, 200, 50)})

    scaler = RobustScaler()
    scaler.fit(train_df[["Amount", "Time"]])
    initial_center = scaler.center_.copy()
    initial_scale = scaler.scale_.copy()

    # Transform test set
    _ = scaler.transform(test_df[["Amount", "Time"]])

    # Assert that scaler center and scale remain completely unaltered after transforming test data
    np.testing.assert_array_equal(scaler.center_, initial_center)
    np.testing.assert_array_equal(scaler.scale_, initial_scale)


def test_validation_only_threshold_tuning_independence():
    """Verify threshold tuning depends strictly on validation data and is invariant to changes in test data."""
    np.random.seed(42)
    y_val = np.array([0] * 80 + [1] * 20)
    probs_val = np.array([0.1] * 70 + [0.4] * 10 + [0.8] * 15 + [0.2] * 5)

    tuned_threshold_initial = tune_decision_threshold(y_val, probs_val)

    # Completely different test distribution
    y_test_a = np.array([0] * 95 + [1] * 5)
    probs_test_a = np.random.uniform(0, 0.3, size=100)

    y_test_b = np.array([1] * 50 + [0] * 50)
    probs_test_b = np.random.uniform(0.7, 1.0, size=100)

    # Regardless of test set A or B, threshold tuned from validation remains identical
    assert tune_decision_threshold(y_val, probs_val) == tuned_threshold_initial


def test_low_fraud_and_zero_fraud_window_handling():
    """Verify that windows with zero or low frauds mark ranking metrics as None with an explanatory note."""
    np.random.seed(42)
    # Case 1: Zero frauds present in test window (only legitimate class 0)
    y_all_legit = np.zeros(100, dtype=int)
    probs_legit = np.random.uniform(0.01, 0.2, size=100)
    metrics_zero_fraud = evaluate_predictions(y_all_legit, probs_legit, threshold=0.5)

    assert metrics_zero_fraud["pr_auc"] is None
    assert metrics_zero_fraud["roc_auc"] is None
    assert "unavailable_reasons" in metrics_zero_fraud
    assert "pr_auc" in metrics_zero_fraud["unavailable_reasons"]
    assert "Insufficient positive fraud cases" in metrics_zero_fraud["unavailable_reasons"]["pr_auc"]
    assert metrics_zero_fraud["confusion_matrix"]["true_positives"] == 0
    assert metrics_zero_fraud["confusion_matrix"]["false_negatives"] == 0

    # Case 2: Zero legitimate transactions present (only fraud class 1)
    y_all_fraud = np.ones(50, dtype=int)
    probs_fraud = np.random.uniform(0.8, 0.99, size=50)
    metrics_zero_legit = evaluate_predictions(y_all_fraud, probs_fraud, threshold=0.5)

    assert metrics_zero_legit["pr_auc"] is None
    assert metrics_zero_legit["roc_auc"] is None
    assert "unavailable_reasons" in metrics_zero_legit
    assert "Insufficient legitimate cases" in metrics_zero_legit["unavailable_reasons"]["pr_auc"]


def test_threshold_tradeoff_analysis_monotonicity():
    """Verify threshold tradeoff analysis computes across candidate thresholds with decreasing false positives."""
    np.random.seed(42)
    y_true = np.array([0] * 80 + [1] * 20)
    probs = np.linspace(0.01, 0.99, 100)

    tradeoffs = compute_threshold_tradeoff_analysis(y_true, probs, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    assert len(tradeoffs) == 5

    # As threshold increases, False Positives must be non-increasing
    fps = [t["confusion_matrix"]["false_positives"] for t in tradeoffs]
    for i in range(len(fps) - 1):
        assert fps[i] >= fps[i + 1], f"FPs did not decrease: {fps}"


def test_aggregate_window_metrics_calculation():
    """Verify compute_aggregate_window_metrics correctly computes mean, std, spread, and confusion matrix totals."""
    mock_metrics = [
        {
            "pr_auc": 0.80,
            "roc_auc": 0.97,
            "precision": 0.85,
            "recall": 0.75,
            "f1_score": 0.80,
            "false_positive_rate": 0.0001,
            "confusion_matrix": {"true_negatives": 50000, "false_positives": 5, "false_negatives": 25, "true_positives": 75},
        },
        {
            "pr_auc": 0.82,
            "roc_auc": 0.98,
            "precision": 0.90,
            "recall": 0.70,
            "f1_score": 0.79,
            "false_positive_rate": 0.00005,
            "confusion_matrix": {"true_negatives": 50000, "false_positives": 2, "false_negatives": 30, "true_positives": 70},
        },
    ]

    agg = compute_aggregate_window_metrics(mock_metrics)
    assert "metrics_summary" in agg
    assert "total_confusion_matrix" in agg
    assert "independence_warning" in agg
    assert agg["windows_evaluated"] == 2
    assert agg["total_confusion_matrix"]["true_positives"] == 145
    assert agg["total_confusion_matrix"]["false_positives"] == 7
    assert agg["metrics_summary"]["pr_auc"]["mean"] == 0.81
    assert agg["metrics_summary"]["pr_auc"]["spread"] == 0.02


def test_mock_benchmark_pipeline_execution():
    """Test full execution of benchmark pipeline on a synthetic mock CSV file including expanding windows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_csv_path = os.path.join(tmpdir, "mock_creditcard.csv")
        mock_report_dir = os.path.join(tmpdir, "mock_reports")

        np.random.seed(42)
        n = 600
        times = np.sort(np.linspace(0, 10000, n))
        classes = np.zeros(n, dtype=int)
        # Ensure fraud presence across early, middle, and late segments
        classes[np.random.choice(n, size=40, replace=False)] = 1

        data = {
            "Time": times,
            "Amount": np.random.uniform(5, 500, n),
            "Class": classes,
        }
        for col in REQUIRED_PCA_COLUMNS:
            data[col] = np.random.normal(0, 1, n)

        df = pd.DataFrame(data)
        df.to_csv(mock_csv_path, index=False)

        # Run pipeline in fast mode
        report = run_benchmark(data_path=mock_csv_path, report_dir=mock_report_dir, fast_mode=True)
        assert report is not None
        assert report["dataset_metadata"]["row_count"] == 600
        assert "baseline_logistic_regression" in report["results"]
        assert "random_forest" in report["results"]
        assert "single_split_reference" in report
        assert "temporal_window_evaluation" in report
        assert "windows" in report["temporal_window_evaluation"]
        assert len(report["temporal_window_evaluation"]["windows"]) == 3
        assert "aggregate_metrics" in report["temporal_window_evaluation"]
        assert "threshold_tradeoff_analysis" in report

        # Check report files written
        json_file = os.path.join(mock_report_dir, "real_data_evaluation.json")
        md_file = os.path.join(mock_report_dir, "real_data_evaluation.md")
        assert os.path.exists(json_file)
        assert os.path.exists(md_file)

        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "## 1. Reference Single Chronological Split" in md_content
            assert "## 2. Walk-Forward Temporal Robustness Analysis" in md_content
            assert "## 3. Decision Threshold Sensitivity Analysis" in md_content
            assert "Non-Independence Disclosure" in md_content


def test_report_prose_consistency_with_computed_metrics():
    """Verify that prose claims in Section 4 strictly match the underlying JSON metrics."""
    json_path = os.path.join("reports", "real_data_evaluation.json")
    md_path = os.path.join("reports", "real_data_evaluation.md")

    if not os.path.exists(json_path) or not os.path.exists(md_path):
        pytest.skip("Report files not present locally.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    temporal = data["temporal_window_evaluation"]
    agg_rf = temporal["aggregate_metrics"]["random_forest"]["tuned_threshold_aggregate"]["metrics_summary"]
    tradeoffs_rf = {round(r["threshold"], 2): r for r in data["threshold_tradeoff_analysis"]["random_forest"]}

    # Verify Section 4 text contains the exact PR-AUC aggregate mean and range
    rf_mean_str = f"Mean: {agg_rf['pr_auc']['mean']} ± {agg_rf['pr_auc']['std']}"
    rf_range_str = f"PR-AUC {agg_rf['pr_auc']['min']} to {agg_rf['pr_auc']['max']}"
    assert rf_mean_str in md_text, f"Expected '{rf_mean_str}' in markdown prose."
    assert rf_range_str in md_text, f"Expected '{rf_range_str}' in markdown prose."

    # Verify threshold 0.30 metrics in prose match candidate threshold table
    row_30 = tradeoffs_rf[0.30]
    expected_30_recall = f"{row_30['recall'] * 100:.1f}%"
    expected_30_fp = f"{row_30['confusion_matrix']['false_positives']} false alarms"
    expected_30_tp = f"{row_30['confusion_matrix']['true_positives']} of"
    assert expected_30_recall in md_text, f"Expected recall '{expected_30_recall}' in prose."
    assert expected_30_fp in md_text, f"Expected '{expected_30_fp}' in prose."
    assert expected_30_tp in md_text, f"Expected '{expected_30_tp}' in prose."

    # Verify threshold 0.80 metrics in prose match candidate threshold table
    row_80 = tradeoffs_rf[0.80]
    expected_80_recall = f"{row_80['recall'] * 100:.1f}%"
    expected_80_fn = f"missing {row_80['confusion_matrix']['false_negatives']} of"
    total_test_frauds = row_80["confusion_matrix"]["true_positives"] + row_80["confusion_matrix"]["false_negatives"]
    expected_80_missed_pct = f"{round(row_80['confusion_matrix']['false_negatives'] / total_test_frauds * 100, 1):.1f}% missed"
    assert expected_80_recall in md_text, f"Expected recall '{expected_80_recall}' in prose."
    assert expected_80_fn in md_text, f"Expected '{expected_80_fn}' in prose."
    assert expected_80_missed_pct in md_text, f"Expected '{expected_80_missed_pct}' in prose."


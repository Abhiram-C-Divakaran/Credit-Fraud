"""
Unit Tests for Real-Data Benchmark Pipeline (evaluate_real_data.py)
==================================================================
Tests dataset schema validation, missing/malformed file handling,
leakage-free chronological splitting, and threshold tuning on mock data.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

from evaluate_real_data import (
    REQUIRED_COLUMNS,
    REQUIRED_PCA_COLUMNS,
    validate_dataset_columns,
    chronological_split,
    tune_decision_threshold,
    evaluate_predictions,
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
    # Dataframe missing 'V28' and 'Class'
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
    times = np.random.uniform(0, 172800, size=n_samples)  # 2 days in seconds
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


def test_threshold_tuning_and_metric_evaluation():
    """Verify threshold tuning chooses valid threshold and metrics compute correctly."""
    np.random.seed(42)
    y_true = np.array([0] * 90 + [1] * 10)
    # Simulated probabilities: higher for frauds with noise
    y_probs = np.array([0.05] * 80 + [0.45] * 10 + [0.80] * 8 + [0.20] * 2)

    best_thresh = tune_decision_threshold(y_true, y_probs)
    assert 0.05 <= best_thresh <= 0.95

    metrics = evaluate_predictions(y_true, y_probs, threshold=best_thresh)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "false_positive_rate" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_mock_benchmark_pipeline_execution():
    """Test full execution of benchmark pipeline on a synthetic mock CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_csv_path = os.path.join(tmpdir, "mock_creditcard.csv")
        mock_report_dir = os.path.join(tmpdir, "mock_reports")

        np.random.seed(42)
        n = 500
        times = np.sort(np.linspace(0, 10000, n))
        classes = np.zeros(n, dtype=int)
        # Distribute frauds across all chronological splits
        classes[np.random.choice(n, size=25, replace=False)] = 1

        data = {
            "Time": times,
            "Amount": np.random.uniform(5, 500, n),
            "Class": classes,
        }
        for col in REQUIRED_PCA_COLUMNS:
            data[col] = np.random.normal(0, 1, n)

        df = pd.DataFrame(data)
        df.to_csv(mock_csv_path, index=False)

        # Run pipeline
        report = run_benchmark(data_path=mock_csv_path, report_dir=mock_report_dir, fast_mode=True)
        assert report is not None
        assert report["dataset_metadata"]["row_count"] == 500
        assert "baseline_logistic_regression" in report["results"]
        assert "random_forest" in report["results"]

        # Check report files written
        json_file = os.path.join(mock_report_dir, "real_data_evaluation.json")
        md_file = os.path.join(mock_report_dir, "real_data_evaluation.md")
        assert os.path.exists(json_file)
        assert os.path.exists(md_file)


def test_evaluate_predictions_single_class_edge_cases():
    """Verify evaluate_predictions safely handles extreme thresholds with zero predicted positives or single class."""
    y_true = np.array([0, 0, 0, 1])
    # Extreme threshold 1.0 -> no positive predictions
    y_probs = np.array([0.1, 0.2, 0.1, 0.8])
    metrics = evaluate_predictions(y_true, y_probs, threshold=0.99)
    assert metrics["confusion_matrix"]["true_positives"] == 0
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["true_negatives"] == 3
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0

    # Extreme threshold 0.0 -> all positive predictions
    metrics_low = evaluate_predictions(y_true, y_probs, threshold=0.01)
    assert metrics_low["confusion_matrix"]["true_positives"] == 1
    assert metrics_low["confusion_matrix"]["false_positives"] == 3
    assert metrics_low["confusion_matrix"]["false_negatives"] == 0
    assert metrics_low["confusion_matrix"]["true_negatives"] == 0


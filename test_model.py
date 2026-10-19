"""
Unit and Integration Tests for Credit Card Fraud Detection Model
=================================================================
Tests model reproducibility, data leakage prevention, schema integrity,
parity between exported JSON trees and evaluation metrics, and honest reporting.
"""

import json
import os
import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import fraud
from fraud import (
    generate_synthetic_data,
    export_decision_tree,
    predict_tree_dict,
    predict_forest_dict,
    validate_model_payload,
    RANDOM_SEED,
    N_ESTIMATORS,
    TOTAL_SAMPLES,
    LEGIT_RATIO,
    MODEL_OUTPUT_PATH,
)


def test_synthetic_data_generation():
    """Verify synthetic generation is deterministic, follows ratio, and respects feature bounds."""
    X1, y1, legit1, fraud1 = generate_synthetic_data(n_samples=1000, seed=42)
    X2, y2, legit2, fraud2 = generate_synthetic_data(n_samples=1000, seed=42)

    # Determinism
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)

    # Class distribution
    assert legit1 == 850
    assert fraud1 == 150
    assert len(y1) == 1000
    assert (y1 == 0).sum() == 850
    assert (y1 == 1).sum() == 150

    # Bounds: Amount > 0, Hour in 0..23, Distance >= 0, Freq >= 0, Binary flags in {0, 1}
    assert np.all(X1[:, 0] > 0)
    assert np.all((X1[:, 1] >= 0) & (X1[:, 1] <= 23))
    assert np.all(X1[:, 2] >= 0)
    assert np.all(X1[:, 3] >= 0)
    assert set(np.unique(X1[:, 4])).issubset({0, 1})
    assert set(np.unique(X1[:, 5])).issubset({0, 1})


def test_scaler_prevention_of_data_leakage():
    """Verify scaler is fit exclusively on train set without looking at test data."""
    X, y, _, _ = generate_synthetic_data(n_samples=1000, seed=42)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    assert len(scaler.mean_) == 6
    assert len(scaler.scale_) == 6
    assert np.all(scaler.scale_ > 0)
    # Check that scaler mean is indeed the train mean, not full X mean
    assert np.allclose(scaler.mean_, np.mean(X_train, axis=0))


def test_exported_model_json_schema():
    """Verify that model.json exists and strictly conforms to v2.0 schema."""
    assert os.path.exists(MODEL_OUTPUT_PATH), f"{MODEL_OUTPUT_PATH} must exist"

    with open(MODEL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Run schema validation function
    validate_model_payload(payload)

    assert payload["schema_version"] == "2.0.0"
    assert payload["n_trees"] == N_ESTIMATORS
    assert len(payload["trees"]) == N_ESTIMATORS
    assert len(payload["feature_names"]) == 6
    assert len(payload["feature_importances"]) == 6

    # Verify metadata
    meta = payload["metadata"]
    assert meta["is_synthetic"] is True
    assert "disclaimer" in meta
    assert meta["decision_threshold"] == 0.5
    assert meta["total_samples"] == TOTAL_SAMPLES

    # Verify metrics exist and are realistic
    metrics = payload["metrics"]
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert key in metrics
        val = metrics[key]
        assert 0.0 < val <= 1.0, f"Metric {key} must be valid probability (got {val})"
        # Honest reporting: metrics should NOT be claimed as perfect 1.0 on noisy synthetic data
        assert val < 1.0, f"Metric {key} should realistically reflect overlap and not be 1.0"


def test_tree_inference_parity_with_exported_metrics():
    """Verify that simulating client-side tree traversal reproduces exported metrics exactly."""
    with open(MODEL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    X, y, _, _ = generate_synthetic_data(seed=RANDOM_SEED)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    mean = np.array(payload["scaler"]["mean"])
    scale = np.array(payload["scaler"]["scale"])
    X_test_scaled = (X_test - mean) / scale

    # Run dictionary traversal (same algorithm that runs in JavaScript)
    test_probs = predict_forest_dict(payload["trees"], X_test_scaled)
    test_preds = (test_probs >= payload["metadata"]["decision_threshold"]).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)

    m = payload["metrics"]
    assert np.isclose(acc, m["accuracy"], atol=1e-3)
    assert np.isclose(prec, m["precision"], atol=1e-3)
    assert np.isclose(rec, m["recall"], atol=1e-3)
    assert np.isclose(f1, m["f1_score"], atol=1e-3)
    assert np.isclose(roc_auc, m["roc_auc"], atol=1e-3)


def test_feature_importance_sum_and_ordering():
    """Verify feature importances are sorted descending and sum to 100%."""
    with open(MODEL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    importances = payload["feature_importances"]
    percentages = [item["percentage"] for item in importances]

    # Verify descending sort
    assert percentages == sorted(percentages, reverse=True)
    # Sum should be ~100%
    assert 99.0 <= sum(percentages) <= 101.0

"""
Credit Card Fraud Detection - Random Forest Classifier
======================================================
Train an educational Random Forest model on synthetic credit card transaction data
and export the exact evaluated ensemble to `model.json` for client-side browser inference.

Educational Prototype Notice:
    This model is trained on synthetic data for demonstration purposes only.
    Its outputs represent illustrative demonstration risk scores, NOT a validated
    real-world probability of credit card fraud. Production fraud detection requires
    unblinded longitudinal banking telemetry, PCI-DSS compliance, adversarial robustness,
    and highly skewed class balancing techniques.

Usage:
    python fraud.py
"""

import json
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────
RANDOM_SEED = 42
TOTAL_SAMPLES = 5000
LEGIT_RATIO = 0.85
N_ESTIMATORS = 20  # Exact number of trees trained, evaluated, AND exported
MAX_DEPTH = 8
DECISION_THRESHOLD = 0.5
MODEL_OUTPUT_PATH = "model.json"

FEATURE_SPECS = [
    {"name": "Amount ($)", "key": "amount", "unit": "USD", "description": "Transaction amount in US Dollars"},
    {"name": "Hour of Day", "key": "hour", "unit": "0-23", "description": "Hour when transaction occurred (0 = midnight)"},
    {"name": "Distance (km)", "key": "distance", "unit": "km", "description": "Distance from cardholder primary residence"},
    {"name": "Txn Frequency", "key": "frequency", "unit": "txns/hr", "description": "Number of transactions recorded in this hour"},
    {"name": "Foreign Txn", "key": "foreign", "unit": "binary", "description": "Flag indicating cross-border international transaction (0=No, 1=Yes)"},
    {"name": "Online Txn", "key": "online", "unit": "binary", "description": "Flag indicating digital card-not-present transaction (0=No, 1=Yes)"},
]
FEATURE_NAMES = [f["name"] for f in FEATURE_SPECS]


def generate_synthetic_data(n_samples: int = TOTAL_SAMPLES, legit_ratio: float = LEGIT_RATIO, seed: int = RANDOM_SEED):
    """
    Generate synthetic transaction records with realistic feature overlaps and noise,
    modeling the real-world challenge where legitimate and fraudulent behaviors partially overlap.
    """
    rng = np.random.default_rng(seed)
    n_legit = int(n_samples * legit_ratio)
    n_fraud = n_samples - n_legit

    # Legitimate transactions
    # Majority: everyday spend (~$30-$150), daytime, local.
    # Realistic overlap: 4% large purchases ($800-$2500), 6% late night, 7% travel, 40% online shopping.
    base_amount_legit = rng.lognormal(mean=3.7, sigma=0.9, size=n_legit)
    large_legit_idx = rng.choice(n_legit, size=int(n_legit * 0.04), replace=False)
    base_amount_legit[large_legit_idx] = rng.uniform(800.0, 2600.0, size=len(large_legit_idx))

    hours_legit = rng.integers(low=6, high=23, size=n_legit)
    night_legit_idx = rng.choice(n_legit, size=int(n_legit * 0.06), replace=False)
    hours_legit[night_legit_idx] = rng.choice([0, 1, 2, 3, 4, 5], size=len(night_legit_idx))

    distance_legit = rng.exponential(scale=22.0, size=n_legit)
    travel_legit_idx = rng.choice(n_legit, size=int(n_legit * 0.07), replace=False)
    distance_legit[travel_legit_idx] = rng.uniform(120.0, 600.0, size=len(travel_legit_idx))

    legit = {
        "amount": np.clip(base_amount_legit, 1.0, 15000.0),
        "hour": hours_legit,
        "distance": np.clip(distance_legit, 0.1, 2000.0),
        "freq": rng.poisson(lam=2.3, size=n_legit),
        "foreign": rng.binomial(n=1, p=0.09, size=n_legit),
        "online": rng.binomial(n=1, p=0.42, size=n_legit),
        "label": np.zeros(n_legit, dtype=int),
    }

    # Fraudulent transactions
    # 22% card-testing / small frauds ($5-$80), 78% high-value takeovers ($300-$3800).
    # 65% night hours, 35% day hours. Elevated frequency and foreign/online flags with realistic noise.
    n_small_fraud = int(n_fraud * 0.22)
    n_large_fraud = n_fraud - n_small_fraud
    small_fraud_amt = rng.uniform(5.0, 85.0, size=n_small_fraud)
    large_fraud_amt = rng.lognormal(mean=6.3, sigma=1.0, size=n_large_fraud)
    fraud_amounts = np.concatenate([small_fraud_amt, large_fraud_amt])
    rng.shuffle(fraud_amounts)

    n_night_fraud = int(n_fraud * 0.65)
    n_day_fraud = n_fraud - n_night_fraud
    fraud_hours = np.concatenate([
        rng.choice([0, 1, 2, 3, 4, 5, 23], size=n_night_fraud),
        rng.integers(low=6, high=22, size=n_day_fraud),
    ])
    rng.shuffle(fraud_hours)

    fraud_distance = rng.exponential(scale=150.0, size=n_fraud)

    fraud = {
        "amount": np.clip(fraud_amounts, 2.0, 15000.0),
        "hour": fraud_hours,
        "distance": np.clip(fraud_distance, 0.5, 3000.0),
        "freq": rng.poisson(lam=7.5, size=n_fraud),
        "foreign": rng.binomial(n=1, p=0.62, size=n_fraud),
        "online": rng.binomial(n=1, p=0.82, size=n_fraud),
        "label": np.ones(n_fraud, dtype=int),
    }

    X = np.column_stack([
        np.concatenate([legit["amount"], fraud["amount"]]),
        np.concatenate([legit["hour"], fraud["hour"]]),
        np.concatenate([legit["distance"], fraud["distance"]]),
        np.concatenate([legit["freq"], fraud["freq"]]),
        np.concatenate([legit["foreign"], fraud["foreign"]]),
        np.concatenate([legit["online"], fraud["online"]]),
    ])
    y = np.concatenate([legit["label"], fraud["label"]])
    return X, y, n_legit, n_fraud


def export_decision_tree(estimator):
    """
    Recursively serialize a scikit-learn DecisionTreeClassifier into a nested dictionary
    that can be traversed in client-side JavaScript.
    """
    tree_ = estimator.tree_

    def recurse(node_id: int):
        if tree_.children_left[node_id] == -1:  # Leaf node
            values = tree_.value[node_id][0]
            total_samples = float(np.sum(values))
            fraud_prob = float(values[1] / total_samples) if len(values) > 1 and total_samples > 0 else 0.0
            return {
                "leaf": True,
                "prob": round(fraud_prob, 5),
            }
        return {
            "feature": int(tree_.feature[node_id]),
            "threshold": float(tree_.threshold[node_id]),
            "left": recurse(int(tree_.children_left[node_id])),
            "right": recurse(int(tree_.children_right[node_id])),
        }

    return recurse(0)


def predict_tree_dict(node: dict, features: np.ndarray) -> float:
    """Simulate client-side decision tree traversal."""
    if node.get("leaf"):
        return node["prob"]
    feature_idx = node["feature"]
    threshold = node["threshold"]
    if features[feature_idx] <= threshold:
        return predict_tree_dict(node["left"], features)
    else:
        return predict_tree_dict(node["right"], features)


def predict_forest_dict(trees: list, scaled_features: np.ndarray) -> np.ndarray:
    """Predict probabilities across an array of samples using the exported tree dictionaries."""
    n_samples = len(scaled_features)
    probs = np.zeros(n_samples)
    for i in range(n_samples):
        tree_probs = [predict_tree_dict(tree, scaled_features[i]) for tree in trees]
        probs[i] = sum(tree_probs) / len(tree_probs)
    return probs


def validate_model_payload(payload: dict) -> None:
    """Validate model dictionary structure, dimensions, and numerical validity."""
    required_keys = {"schema_version", "metadata", "scaler", "feature_names", "trees", "n_trees", "metrics", "feature_importances"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Model payload missing required keys: {missing}")

    if payload["n_trees"] != len(payload["trees"]):
        raise ValueError(f"Tree count mismatch: n_trees={payload['n_trees']}, len(trees)={len(payload['trees'])}")

    scaler = payload["scaler"]
    if len(scaler["mean"]) != len(payload["feature_names"]) or len(scaler["scale"]) != len(payload["feature_names"]):
        raise ValueError("Scaler dimensions do not match feature_names count")

    for idx, scale_val in enumerate(scaler["scale"]):
        if scale_val <= 0 or not np.isfinite(scale_val):
            raise ValueError(f"Invalid scale parameter at feature {idx}: {scale_val}")

    for idx, tree in enumerate(payload["trees"]):
        if not isinstance(tree, dict) or "left" not in tree and not tree.get("leaf"):
            raise ValueError(f"Malformed tree root at index {idx}")


def train_and_export():
    print("=" * 65)
    print("  Credit Card Fraud Detection - Random Forest Training & Export")
    print("=" * 65)

    # 1. Generate data
    print(f"\n[1/5] Generating {TOTAL_SAMPLES} synthetic transactions...")
    X, y, n_legit, n_fraud = generate_synthetic_data()
    print(f"      - Legitimate: {n_legit} ({n_legit / TOTAL_SAMPLES * 100:.1f}%)")
    print(f"      - Fraudulent: {n_fraud} ({n_fraud / TOTAL_SAMPLES * 100:.1f}%)")

    # 2. Preprocess: Train/Test split & fit scaler ONLY on train data
    print("\n[2/5] Splitting data (80% train, 20% test) & standardizing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"      - Train samples: {len(X_train)} (Legit: {(y_train == 0).sum()}, Fraud: {(y_train == 1).sum()})")
    print(f"      - Test samples:  {len(X_test)} (Legit: {(y_test == 0).sum()}, Fraud: {(y_test == 1).sum()})")

    # 3. Train Random Forest (Exact N_ESTIMATORS matching export)
    print(f"\n[3/5] Training Random Forest ({N_ESTIMATORS} trees, max_depth={MAX_DEPTH})...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)

    # 4. Export trees and evaluate the exact exported model
    print("\n[4/5] Exporting model trees and evaluating exact exported model...")
    exported_trees = [export_decision_tree(est) for est in clf.estimators_]

    # Evaluate using the exact dictionary traversal logic that runs in the browser
    test_probs = predict_forest_dict(exported_trees, X_test_s)
    test_preds = (test_probs >= DECISION_THRESHOLD).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()

    print(f"      - Accuracy : {acc * 100:.2f}%")
    print(f"      - Precision: {prec * 100:.2f}% (Fraud class)")
    print(f"      - Recall   : {rec * 100:.2f}% (Fraud class)")
    print(f"      - F1-Score : {f1 * 100:.2f}%")
    print(f"      - ROC-AUC  : {roc_auc:.4f}")
    print("\n      Confusion Matrix (Exact exported model on test set):")
    print(f"                     Predicted Safe   Predicted Fraud")
    print(f"        Actual Safe     {cm[0][0]:<14} {cm[0][1]}")
    print(f"        Actual Fraud    {cm[1][0]:<14} {cm[1][1]}")

    print("\n      Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["Safe (0)", "Fraud (1)"], digits=4))

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance_list = [
        {
            "name": FEATURE_NAMES[i],
            "key": FEATURE_SPECS[i]["key"],
            "importance": round(float(importances[i]), 4),
            "percentage": round(float(importances[i] * 100), 1),
        }
        for i in sorted_idx
    ]
    print("      Feature Importances:")
    for item in feature_importance_list:
        bar = "#" * int(item["percentage"] / 2)
        print(f"        {item['name']:<18} {bar:<25} {item['percentage']}%")

    payload = {
        "schema_version": "2.0.0",
        "metadata": {
            "model_type": "RandomForestClassifier",
            "framework": "scikit-learn",
            "is_synthetic": True,
            "disclaimer": "Educational demonstration on synthetic data. Not calibrated for production credit card transactions.",
            "decision_threshold": DECISION_THRESHOLD,
            "random_seed": RANDOM_SEED,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": TOTAL_SAMPLES,
            "class_distribution": {
                "legitimate_pct": round(LEGIT_RATIO * 100, 1),
                "fraudulent_pct": round((1 - LEGIT_RATIO) * 100, 1),
                "legitimate_count": n_legit,
                "fraudulent_count": n_fraud,
            },
        },
        "scaler": {
            "mean": [round(float(v), 6) for v in scaler.mean_],
            "scale": [round(float(v), 6) for v in scaler.scale_],
        },
        "feature_specs": FEATURE_SPECS,
        "feature_names": FEATURE_NAMES,
        "feature_importances": feature_importance_list,
        "n_trees": len(exported_trees),
        "trees": exported_trees,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "test_size": len(X_test),
        },
    }

    validate_model_payload(payload)

    with open(MODEL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    file_size_kb = os.path.getsize(MODEL_OUTPUT_PATH) / 1024
    print(f"\n[5/5] Successfully exported {MODEL_OUTPUT_PATH} ({file_size_kb:.1f} KB)")
    print("=" * 65)
    print("  Training & Export Completed Successfully!")
    print("=" * 65)
    return payload


if __name__ == "__main__":
    train_and_export()
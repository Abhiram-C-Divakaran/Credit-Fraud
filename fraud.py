"""
Credit Card Fraud Detection - Random Forest Classifier
======================================================
Run this script to train the model and export model.json
Then open fraud_detection.html in a browser for live inference.

Requirements:
    pip install scikit-learn numpy pandas

Usage:
    python fraud_ml.py
"""

import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC TRANSACTION DATA
# ─────────────────────────────────────────────
np.random.seed(42)
N = 5000

print("=" * 55)
print("  Credit Card Fraud Detection — Random Forest")
print("=" * 55)
print(f"\n[1/4] Generating {N} synthetic transactions...")

# Legitimate transactions (85%)
n_legit = int(N * 0.85)
legit = {
    "amount":   np.random.lognormal(3.5, 1.2, n_legit),    # avg ~$150
    "hour":     np.random.randint(6, 23, n_legit),          # daytime hours
    "distance": np.random.exponential(30, n_legit),         # close to home
    "freq":     np.random.poisson(3, n_legit),              # normal frequency
    "foreign":  np.random.binomial(1, 0.10, n_legit),       # 10% foreign
    "online":   np.random.binomial(1, 0.30, n_legit),       # 30% online
    "label":    np.zeros(n_legit)
}

# Fraudulent transactions (15%)
n_fraud = int(N * 0.15)
fraud = {
    "amount":   np.random.lognormal(5.5, 1.5, n_fraud),    # avg ~$4500 (higher)
    "hour":     np.random.choice([0,1,2,3,4,23], n_fraud), # late night hours
    "distance": np.random.exponential(200, n_fraud),        # far from home
    "freq":     np.random.poisson(10, n_fraud),             # high frequency
    "foreign":  np.random.binomial(1, 0.70, n_fraud),       # 70% foreign
    "online":   np.random.binomial(1, 0.80, n_fraud),       # 80% online
    "label":    np.ones(n_fraud)
}

# Stack into feature matrix
X = np.column_stack([
    np.concatenate([legit["amount"],   fraud["amount"]]),
    np.concatenate([legit["hour"],     fraud["hour"]]),
    np.concatenate([legit["distance"], fraud["distance"]]),
    np.concatenate([legit["freq"],     fraud["freq"]]),
    np.concatenate([legit["foreign"],  fraud["foreign"]]),
    np.concatenate([legit["online"],   fraud["online"]]),
])
y = np.concatenate([legit["label"], fraud["label"]])

feature_names = ["Amount ($)", "Hour of Day", "Distance (km)",
                 "Txn Frequency", "Foreign Txn", "Online Txn"]

print(f"    Legitimate transactions : {n_legit} ({n_legit/N*100:.0f}%)")
print(f"    Fraudulent transactions : {n_fraud} ({n_fraud/N*100:.0f}%)")

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
print("\n[2/4] Preprocessing & splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"    Train set : {len(X_train)} samples")
print(f"    Test set  : {len(X_test)} samples")

# ─────────────────────────────────────────────
# 3. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n[3/4] Training Random Forest Classifier...")
print("    n_estimators=50, max_depth=8, random_state=42")

clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
clf.fit(X_train_s, y_train)

# Evaluate
preds      = clf.predict(X_test_s)
probs      = clf.predict_proba(X_test_s)[:, 1]
accuracy   = accuracy_score(y_test, preds)
cm         = confusion_matrix(y_test, preds)

print(f"\n    ✓ Accuracy  : {accuracy * 100:.2f}%")
print(f"\n    Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Safe   Fraud")
print(f"    Actual Safe  {cm[0][0]:<6} {cm[0][1]}")
print(f"    Actual Fraud {cm[1][0]:<6} {cm[1][1]}")
print(f"\n    Classification Report:")
print(classification_report(y_test, preds, target_names=["Safe", "Fraud"]))

# Feature importances
importances = clf.feature_importances_
print("    Feature Importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"    {name:<18} {bar} {imp*100:.1f}%")

# ─────────────────────────────────────────────
# 4. EXPORT MODEL → model.json
# ─────────────────────────────────────────────
print("\n[4/4] Exporting model to model.json...")

def export_tree(estimator):
    """Recursively export a decision tree to a JSON-serialisable dict."""
    t = estimator.tree_
    def recurse(node):
        if t.children_left[node] == -1:   # leaf node
            vals  = t.value[node][0]
            total = sum(vals)
            fraud_prob = vals[1] / total if len(vals) > 1 else 0.0
            return {"leaf": True, "prob": round(float(fraud_prob), 4)}
        return {
            "feature":   int(t.feature[node]),
            "threshold": round(float(t.threshold[node]), 6),
            "left":      recurse(int(t.children_left[node])),
            "right":     recurse(int(t.children_right[node]))
        }
    return recurse(0)

# Export first 20 trees (good accuracy, keeps file small)
trees = [export_tree(est) for est in clf.estimators_[:20]]

model_data = {
    "scaler": {
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    },
    "feature_names": feature_names,
    "trees":   trees,
    "n_trees": len(trees),
    "accuracy": round(float(accuracy), 4)
}

output_path = "model.json"
with open(output_path, "w") as f:
    json.dump(model_data, f, indent=2)

size_kb = len(json.dumps(model_data)) / 1024
print(f"    Saved  → {output_path}")
print(f"    Trees  : {len(trees)}")
print(f"    Size   : {size_kb:.1f} KB")
print(f"\n{'='*55}")
print("  Done! Open fraud_detection.html in your browser.")
print("="*55)
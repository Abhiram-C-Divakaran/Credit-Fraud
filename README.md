# 🛡️ Credit Card Fraud Detection — Machine Learning System & Benchmark

[![CI](https://github.com/Abhiram-C-Divakaran/Credit-Fraud/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhiram-C-Divakaran/Credit-Fraud/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Important Architecture Distinction**  
> This repository contains two distinct components:
> 1. **Default Application — Interactive Browser Demo**: A zero-server, client-side web application running a 20-tree Random Forest model on 6 intuitive behavioral signals (Amount, Hour, Distance, Frequency, Foreign, Online) trained on synthetic data for educational exploration.
> 2. **Offline Benchmark Pipeline (`evaluate_real_data.py`)**: A separate, leakage-conscious evaluation script that benchmarks baseline and Random Forest models on the real-world Kaggle/ULB credit card dataset (28 anonymized PCA features `V1`–`V28`, plus `Time` and `Amount`).

---

## 📌 Repository Structure

```text
├── fraud.py                 # Training & JSON export for browser demo model
├── model.json               # Exported 20-tree Random Forest & scaler for browser inference
├── core.js                  # Shared JavaScript module (validation, scaling, tree inference)
├── app.js                   # Client-side UI controller & state management
├── index.html               # Semantic, accessible HTML5 dashboard structure
├── styles.css               # Design system with dark/light modes & responsive grid
├── run.bat                  # Windows one-click local server and browser launcher
│
├── evaluate_real_data.py    # Offline benchmark pipeline for Kaggle/ULB dataset
├── test_model.py            # Pytest suite for synthetic model integrity & parity
├── test_real_data.py        # Pytest suite for real-data benchmark pipeline
├── test_frontend.js         # Node.js test suite for shared core.js module
│
├── .github/workflows/ci.yml # Continuous Integration (pytest + node tests)
├── requirements.txt         # Reproducible Python dependencies manifest
├── LICENSE                  # MIT License
├── .gitignore               # Excludes python artifacts, test caches, and raw data
└── README.md                # System documentation
```

---

## 🌐 Part 1: Interactive Browser Demo (Default App)

### How It Works
Traditional ML deployments rely on server-side APIs that introduce network latency, API costs, and data privacy risks. This demo runs **100% client-side**:
- User enters transaction parameters.
- [core.js](core.js) standardizes inputs using precomputed `scaler.mean` and `scaler.scale`.
- The browser traverses 20 pruned decision trees in [model.json](model.json) and averages their class probabilities.
- Risk scores, probability meters, and transaction history update instantaneously without sending data to any external server.

### Features & Behavioral Signals

| Feature Name | Key | Unit | Description | Typical Normal Spend | Suspicious Indicator |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Amount** | `amount` | USD ($) | Total transaction purchase value | Moderate spend (~$30–$150) | High spend (>$1,500) or test probes ($1–$5) |
| **Hour of Day** | `hour` | 0–23 | Hour transaction occurred (0 = midnight) | Normal daytime (07:00–22:00) | Late night / early morning (00:00–05:00) |
| **Distance** | `distance` | km | Distance from primary residence | Local spend (< 30 km) | Remote distance (> 200 km) without notice |
| **Txn Frequency** | `frequency` | count | Transactions recorded in the current hour | 1–3 txns/hour | Rapid bursts (> 8 txns/hour) |
| **Foreign Txn** | `foreign` | Binary | Cross-border transaction indicator | Domestic (0) | Unexpected foreign merchant (1) |
| **Online Txn** | `online` | Binary | Card-not-present digital transaction | In-store or routine web | Uncharacteristic online authorization burst (1) |

### Running the Browser Demo Locally
Due to browser **CORS restrictions**, `model.json` cannot be fetched via direct `file:///` opening. Start a local HTTP server:

```bash
# Option A: Windows shortcut
double-click run.bat

# Option B: Python HTTP server
python -m http.server 5500

# Option C: Node npx
npx serve .
```
Then open [http://localhost:5500](http://localhost:5500) in your browser.

---

## 🔬 Part 2: Real-World Benchmark Pipeline (`evaluate_real_data.py`)

The offline benchmark evaluates models against the real-world **Kaggle / ULB Machine Learning Group** credit card fraud dataset.

### Why It Is Kept Separate from the Browser Demo
1. **PCA Features vs Semantic Inputs**: The Kaggle dataset features `V1`–`V28` are anonymized principal components resulting from PCA transformation (to protect cardholder confidentiality). They do not correspond to the interactive demo's intuitive inputs (`distance`, `frequency`, `hour`).
2. **Class Imbalance Realities**: The real dataset contains **492 frauds among 284,807 transactions (0.172%)**. Production models must optimize Precision-Recall AUC (PR-AUC) rather than raw accuracy.

### Setup Instructions

1. **Download the Dataset**:
   Download `creditcard.csv` from Kaggle:  
   🔗 [Kaggle / ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

2. **Place in the Data Folder**:
   Create a `data/` directory and place `creditcard.csv` inside:
   ```bash
   mkdir data
   # Place creditcard.csv at: data/creditcard.csv
   ```
   *(Note: `data/` and `*.csv` are excluded by `.gitignore` to prevent committing raw data).*

3. **Run the Benchmark**:
   ```bash
   # Standard benchmark (100-tree Random Forest + Baseline Logistic Regression)
   python evaluate_real_data.py

   # Quick sanity run (fast mode)
   python evaluate_real_data.py --fast
   ```

### Evaluation Methodology & Leakage Prevention

- **Chronological Split by Time**: Transactions are ordered by the `Time` column:
  - **60% Train** (earliest transactions)
  - **20% Validation** (intermediate transactions)
  - **20% Test** (latest unseen transactions)
  *Rationale: Random cross-validation in temporal fraud datasets creates massive look-ahead data leakage because fraud campaigns occur in clustered time bursts. A chronological split reflects real operational deployment.*
- **Strict Preprocessing Isolation**: `RobustScaler` is fitted **strictly** on training data (`X_train`) and applied downstream to validation and test sets.
- **Validation-Tuned Decision Threshold**: Rather than assuming an arbitrary `0.5` cutoff, decision thresholds are selected on the validation set to optimize the precision-recall trade-off before final evaluation on the untouched test split.
- **Report Generation**: Outputs are saved to:
  - `reports/real_data_evaluation.json` (machine-readable metrics and metadata)
  - `reports/real_data_evaluation.md` (formatted markdown summary)

### Benchmark Metrics Tracked
- **PR-AUC (Average Precision)**: Primary metric for severe class imbalance.
- **ROC-AUC**: Evaluates ranking across all discrimination thresholds.
- **Precision, Recall, F1-Score**: Evaluated at standard (0.50) and validation-tuned thresholds.
- **False Positive Rate (FPR)**: Critical for measuring customer friction / false alarms.
- **Confusion Matrix**: Full breakdown of TP, FP, TN, FN.

---

## 🧪 Testing & Verification

The repository includes test suites for both Python and Node.js environments:

```bash
# 1. Run Python unit and integration tests (synthetic model + real benchmark pipeline)
pytest -v

# 2. Run Node.js client-side inference and validation tests
node test_frontend.js

# 3. Test real-data missing-file handling
python evaluate_real_data.py
```

All tests run in headless environments and do **not** require the optional private Kaggle dataset.

---

## ⚙️ Continuous Integration

A GitHub Actions workflow is configured at [.github/workflows/ci.yml](.github/workflows/ci.yml). On every push and pull request to `main`:
1. Sets up Python 3.11 with pip caching and Node.js 20.
2. Installs dependencies from [requirements.txt](requirements.txt).
3. Executes `pytest -v` across all test suites.
4. Executes `node test_frontend.js` using the shared [core.js](core.js) engine.
5. Verifies graceful missing-dataset handling in [evaluate_real_data.py](evaluate_real_data.py).

---

## ⚠️ Real-World Deployment Limitations

1. **Dataset Scope**: The Kaggle/ULB dataset consists of European cardholder transactions across a single 2-day period in September 2013. Model metrics obtained on this dataset do not guarantee performance on modern transactions or other geographical regions.
2. **Adversarial Evolution & Concept Drift**: Fraud rings continuously adapt payment routing, card testing velocities, and proxy evasion techniques. Static models require automated drift monitoring, retrain schedules, and integration with real-time velocity engines.
3. **Multi-Factor Context**: Production fraud systems combine machine learning scores with device fingerprinting, behavioral biometrics, 3D-Secure (3DS) challenges, and IP consortium intelligence.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

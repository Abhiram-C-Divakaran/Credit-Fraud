# 🛡️ Credit Card Fraud Detection — Client-Side Machine Learning Prototype

> **Educational Demonstration Notice**  
> This repository is an educational prototype demonstrating client-side machine learning inference. The model is trained on synthetic transaction data to illustrate how standard scaling and Random Forest decision tree ensembles can execute entirely in the browser.  
> **The outputs are illustrative demonstration risk scores, not calibrated real-world probabilities of financial fraud.**

---

## 📌 Overview

Traditional machine learning deployments rely on cloud servers, REST APIs, and microservices to run inference. This project showcases an alternative **zero-server architecture**:
1. **Model Training & Export (`fraud.py`)**: A Random Forest classifier is trained in Python using scikit-learn on standardized transaction features. The exact 20 trained decision trees and StandardScaler coefficients are exported into a portable, structured `model.json`.
2. **Client-Side Browser Inference (`app.js`)**: The browser downloads `model.json`, normalizes user input with the saved scaling parameters, traverses each decision tree client-side, and averages tree probabilities in milliseconds.
3. **Full Privacy & Zero Latency**: No card details or transaction figures are sent to any remote server. Everything executes locally in the user's browser sandbox.

---

## 📁 Repository Structure

```text
├── fraud.py            # Python training script, evaluation & model.json export
├── model.json          # Exported Random Forest (scaler, 20 decision trees, metrics)
├── index.html          # Semantic, accessible HTML5 dashboard structure
├── styles.css          # CSS tokens, light/dark themes, responsive grid, animations
├── app.js              # Client-side inference engine, validation, & UI controller
├── run.bat             # Windows one-click local server and browser launcher
├── test_model.py       # Pytest suite (data leakage, parity, schema, reproducibility)
├── test_frontend.js    # Node.js test suite (tree traversal, presets, validation)
└── README.md           # Project documentation and engineering guide
```

---

## 📊 Features & Behavioral Signals

The demonstration model processes 6 transaction features:

| Feature Name | Key | Unit | Description | Typical Legitimate Profile | Suspicious Indicators |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Amount** | `amount` | USD ($) | Total monetary charge value | Median ~$50–$150 | High value (>$1,500) or micro-amounts ($1–$5 test charges) |
| **Hour of Day** | `hour` | 0–23 | Local hour of transaction (0 = midnight) | Normal daytime (07:00–22:00) | Late night / early morning (00:00–05:00) |
| **Distance** | `distance` | km | Distance from cardholder primary residence | Local spend (< 30 km) | Large distance (> 200 km) without travel notice |
| **Txn Frequency** | `frequency` | count | Transactions recorded within the current hour | 1–3 transactions/hour | Rapid transaction bursts (> 8 transactions/hour) |
| **Foreign Txn** | `foreign` | Binary (0/1) | Cross-border transaction flag | Domestic (0) | Unexpected foreign billing location (1) |
| **Online Txn** | `online` | Binary (0/1) | Card-not-present e-commerce flag | Physical card or routine web spend | Uncharacteristic online authorization burst |

---

## ⚙️ Model Architecture & Evaluation

- **Algorithm**: Random Forest Classifier (`n_estimators=20`, `max_depth=8`, `class_weight='balanced'`).
- **Data Splitting**: 80% train (4,000 samples), 20% test (1,000 samples), stratified by class label.
- **Preprocessing**: `StandardScaler` fit **exclusively** on training data to prevent data leakage.
- **Parity Guarantee**: The exact 20 trees exported to `model.json` are evaluated on the held-out test set, ensuring the metrics reported in the UI match browser inference.

### Evaluation Metrics (Held-Out Test Set)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Test Accuracy** | **97.10%** | Overall correct classifications on test split |
| **Fraud Precision** | **87.58%** | Ratio of true frauds among all transactions flagged as fraud |
| **Fraud Recall** | **94.00%** | Percentage of actual frauds caught by the model |
| **F1-Score** | **90.68%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9792** | Area under the Receiver Operating Characteristic curve |

> *Confusion Matrix on 1,000 Test Samples (850 Legitimate, 150 Fraudulent):*
> - **True Negatives (Safe correctly identified)**: 830
> - **False Positives (False alarms)**: 20
> - **False Negatives (Missed frauds)**: 9
> - **True Positives (Frauds caught)**: 141

---

## ⚠️ Synthetic Data & Real-World Limitations

1. **Synthetic Feature Separation**: The training script generates synthetic distributions where fraudulent transactions have higher probabilities of night hours, elevated frequencies, and cross-border locations. While realistic overlap and noise are injected, real-world credit card data is significantly more complex.
2. **Extreme Class Imbalance**: In production banking telemetry, fraud prevalence is typically **0.1% to 0.2%** (1 in 1,000 transactions), not the 15% ratio used here. Production models optimize Precision-Recall AUC (PR-AUC) rather than raw accuracy.
3. **Adversarial Adaptation & Concept Drift**: Fraud syndicates continuously alter spend amounts, payment gateways, and geolocation cloaking to bypass fixed rule sets. Real production systems require continuous online learning, anomaly detection, velocity counters, and device fingerprinting.

---

## 🛠️ Setup & Local Development

### 1. Prerequisites
- **Python 3.9+** (for training and evaluation)
- **Node.js 18+** (optional, for frontend automated tests)
- Any modern web browser (Chrome, Edge, Firefox, Safari)

### 2. Install Python Dependencies
```bash
pip install scikit-learn numpy pandas pytest
```

### 3. Retrain and Export Model
To generate synthetic data, train the Random Forest, verify parity, and export `model.json`:
```bash
python fraud.py
```

### 4. Run Automated Tests
Run both the Python model test suite and the Node frontend verification suite:
```bash
# Run Python model integrity & schema tests
pytest -v test_model.py

# Run JavaScript inference & input validation tests
node test_frontend.js
```

### 5. Launch the Web Application
Because modern browsers enforce **CORS (Cross-Origin Resource Sharing)** restrictions on the `file:///` protocol, `model.json` cannot be fetched via direct double-click on `index.html`. You must run a lightweight HTTP server.

**Option A — Windows Shortcut:**
Double-click `run.bat` in the project root. It will start a local server at `http://localhost:5500/` and open your default browser automatically.

**Option B — Python HTTP Server:**
```bash
python -m http.server 5500
```
Then navigate to [http://localhost:5500](http://localhost:5500) in your web browser.

**Option C — Node.js npx:**
```bash
npx serve .
```

---

## 🌐 Deploying as a Static Website

Because the application requires no backend server or database, it can be deployed for free to any static hosting provider:

### Deploying to GitHub Pages
1. Push this repository to GitHub.
2. Go to **Settings** > **Pages** in your repository.
3. Under **Branch**, select `main` (or `master`) and folder `/ (root)`.
4. Click **Save**. Within a minute, your detector will be live at `https://<username>.github.io/<repository-name>/`.

---

## 🔬 Integrating a Real-World Dataset

To adapt this codebase to a real-world dataset (such as the [Kaggle / ULB Machine Learning Group Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)):

1. **Obtain Data**: Download `creditcard.csv` (contains 284,807 European cardholder transactions with PCA components `V1`–`V28`, `Time`, `Amount`, and `Class`).
2. **Update Feature Extraction in `fraud.py`**:
   ```python
   import pandas as pd
   df = pd.read_csv("creditcard.csv")
   X = df[["Amount", "Time", "V1", "V2", "V3", "V4"]].values # Select salient features
   y = df["Class"].values
   ```
3. **Address 0.17% Class Imbalance**:
   - Use `class_weight='balanced_subsample'`.
   - Evaluate using `average_precision_score` (PR-AUC) and set the classification threshold via ROC / PR curve analysis rather than a default `0.5`.
4. **Update `FEATURE_SPECS`**: Ensure `FEATURE_SPECS` in `fraud.py` matches the names and indices of the selected columns so the exported `model.json` schema remains aligned with `app.js`.

---

## 🔒 Privacy & Data Ethics

- **Zero Data Collection**: No transaction inputs, amounts, locations, or identifiers are logged to any external server or third-party analytics tool.
- **Fictional Visuals**: All credit card visuals shown in the interface use specimen card numbers (`•••• •••• •••• 4096`) and dummy names (`DEMO SPECIMEN`) to adhere to security best practices.

---

## 📄 License
This project is open-source under the MIT License.

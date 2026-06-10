# 🛡️ Credit Card Fraud Detection – Random Forest Classifier

An end-to-end fraud detection system that uses a **Random Forest machine learning model** to classify credit card transactions as **safe** or **fraudulent**. The model runs entirely in the browser – no server, no API keys, no recurring costs.

## 🔍 Live Demo

Once deployed (e.g., on GitHub Pages), you can test transactions instantly:

- **Normal transaction** → $85 at 2 PM, 12 km from home → ✅ Safe
- **Suspicious transaction** → $4,500 at 2 AM, 350 km away, foreign, online → 🚨 Fraud

## 📁 Project Structure

├── fraud.py # Python script to generate data, train model, and export model.json
├── index.html # Frontend UI + JavaScript inference engine
├── model.json # Exported Random Forest (scaler + 20 decision trees)
└── README.md # This file


## 🚀 How It Works

### 1. Model Training (Python)

- **Synthetic dataset** of 5,000 transactions (85% legitimate, 15% fraudulent).
- Features:
  - `Amount` (USD)
  - `Hour of day`
  - `Distance from home` (km)
  - `Transaction frequency` (txns per hour)
  - `Foreign transaction` (0/1)
  - `Online transaction` (0/1)
- A **Random Forest classifier** (50 trees, max depth 8) is trained.
- The model is exported to `model.json` containing:
  - Feature scaler (mean & standard deviation)
  - First 20 decision trees (pruned for browser efficiency)

### 2. Browser Inference (JavaScript)

- `index.html` fetches `model.json`.
- User input is **standardised** using the saved scaler.
- Each decision tree is traversed to produce a fraud probability.
- Probabilities from all trees are **averaged**.
- If average ≥ 0.5 → **FRAUD**, otherwise **SAFE**.

### 3. User Interface

- Clean landing page and fraud detection dashboard.
- Interactive form with presets (normal, suspicious, large amount, night transaction).
- Real-time results:
  - Fraud probability bar
  - Live risk meter
  - Transaction history
- **Dark / light mode** toggle with persistent preference.
- Fully responsive and mobile-friendly.

## 🛠️ Requirements & Setup

### For local development

1. **Clone or download** this repository.
2. **Install Python dependencies** (only needed if you want to retrain the model):
   ```bash
   pip install scikit-learn numpy pandas


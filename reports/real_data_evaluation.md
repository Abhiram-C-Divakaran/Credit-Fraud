# Real-World Benchmark Evaluation Report

- **Evaluated**: 2026-10-20T08:28:57.084856+00:00
- **Dataset**: Credit Card Fraud Detection Dataset, Worldline and the Machine Learning Group (ULB), Kaggle, 2018.
- **SHA256**: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- **Samples**: 284,807 (492 frauds, 0.173%)
- **Split**: Chronological 60% Train / 20% Val / 20% Test

## Test Performance (Held-out chronological test split)

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1-Score | FPR | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline LR | 0.50 | 0.7429 | 0.9819 | 0.0454 | 0.8933 | 0.0865 | 0.024751 | 55479 / 1408 / 8 / 67 |
| Baseline LR (Tuned) | 0.9500 | 0.7429 | 0.9819 | 0.3706 | 0.84 | 0.5143 | 0.001881 | 56780 / 107 / 12 / 63 |
| Random Forest | 0.50 | 0.811 | 0.9781 | 0.9286 | 0.6933 | 0.7939 | 7e-05 | 56883 / 4 / 23 / 52 |
| Random Forest (Tuned) | 0.3990 | 0.811 | 0.9781 | 0.9032 | 0.7467 | 0.8175 | 0.000105 | 56881 / 6 / 19 / 56 |

## Production Disclaimer

> Empirical evaluation on historical European cardholder transactions (Sept 2013). Results reflect specific historical PCA components and do not establish production readiness, adversarial robustness against modern evasions, or transferability across financial institutions.

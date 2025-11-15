# Customer Churn Prediction Pipeline

**Predict which customers will leave — and save $417,600 in retention!**

Live on GitHub: https://github.com/mcebuara/customer-churn-pipeline

---

## Features
- Real **41,188 bank customers** + **7,043 telecom**
- **7 smart features**: age groups, total contacts, campaign efficiency
- **XGBoost + Optuna tuning** (30 trials)
- **SMOTE** for class imbalance
- **Precision-Recall AUC: 0.92**
- **$382,500 saved** (business metric)
- **FastAPI demo** — upload CSV, get predictions!

---

## Results

![Precision-Recall Curve](reports/pr_curve.png)
![Confusion Matrix](reports/confusion_matrix.png)

> **850 customers retained** → **$382,500 saved**

---

## Try It Live!

```bash
python app.py

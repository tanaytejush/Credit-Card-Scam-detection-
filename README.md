# Credit Card Fraud Detection

A machine learning notebook that detects fraudulent credit card transactions, and compares how different class-balancing strategies change the results on a heavily imbalanced dataset.

## The problem

Fraud is rare. In the standard credit card dataset, only a tiny fraction of transactions are fraudulent, so a model can score over 99% accuracy while missing most fraud. This project shows that effect and how to fix it.

## Approach

1. **Baseline on the raw data**: Logistic Regression and Decision Tree on the imbalanced data.
2. **Undersampling**: randomly sample 473 normal transactions to match the fraud cases.
3. **Oversampling with SMOTE**: synthesise new fraud examples using `imbalanced-learn`.
4. **Export** a trained Decision Tree with `joblib` (`credit_card_model.pkl`).

## Results (Logistic Regression)

| Strategy | Accuracy | Precision | Recall | F1 |
|----------|----------|-----------|--------|----|
| Raw (imbalanced) | 0.999 | 0.887 | 0.604 | 0.719 |
| Undersampling | 0.926 | 0.949 | 0.912 | 0.930 |
| SMOTE | 0.944 | 0.973 | 0.913 | 0.942 |

Accuracy alone is misleading here. Balancing the classes raises recall from about 60% to about 91%, so far more fraud is caught.

## Files

| File | Description |
|------|-------------|
| `Credit Card Fraud Detection.ipynb` | Full analysis and model training |
| `credit_card_model.pkl` | Exported Decision Tree model |

## Run it

The notebook reads `creditcard.csv`, which is not included in this repo. Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place it next to the notebook.

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib jupyter
jupyter notebook "Credit Card Fraud Detection.ipynb"
```

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, seaborn, matplotlib


---

<sub>Made by [tanaytejush](https://github.com/tanaytejush)</sub>

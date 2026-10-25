# PaySim Fraud Detection

An exploratory analysis and machine learning study of fraud detection in the synthetic [PaySim transaction dataset](https://www.kaggle.com/datasets/ealaxi/paysim1). The project compares Logistic Regression with XGBoost, with attention to class imbalance, transaction time, and evaluation on later transactions.

## Results

The final XGBoost model was evaluated on a held-out period beginning at simulation step 378. Its classification threshold (0.3061) was selected using validation data.

| Test metric | Result |
| --- | ---: |
| Average precision | 0.3604 |
| ROC-AUC | 0.9501 |
| Fraud precision | 34.82% |
| Fraud recall | 44.66% |
| Fraud F1 | 0.3913 |

The test confusion matrix contains 1,791 true positives, 3,352 false positives, 2,219 false negatives, and 948,382 true negatives. The threshold met a 50% precision target on validation data, but test precision was lower. These are the saved results in `03_xgboost.ipynb`; they have not been rerun as part of publishing this repository.

## Dataset and approach

PaySim contains 6,362,620 synthetic transactions, of which 8,213 (0.129%) are labeled fraudulent. Because the classes are highly imbalanced, this project emphasizes average precision, fraud precision, recall, and false alarms rather than accuracy alone.

The EDA examines transaction types, amounts, account balances, and time. The modeling notebooks use `step`, `type`, and `amount` as primary predictors. Balance fields are excluded because their strong relationship with the fraud label may reflect simulation behavior or information unavailable at decision time. The final XGBoost model adds `hour_of_day = (step - 1) % 24`.

Transactions are split by simulation step: training uses steps before 323, validation uses steps 323–377, and test uses step 378 onward. The test period is used for final evaluation after model and threshold selection.

## Repository contents

```text
.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_logistic_regression.ipynb
│   └── 03_xgboost.ipynb
├── data/
│   └── raw/                  # Place the downloaded CSV here; Git ignores it
├── README.md
├── requirements.txt
└── .gitignore
```

Read the notebooks in numbered order. The XGBoost notebook includes the final test evaluation, error analysis, and a function demonstrating inference. When run, it saves its trained pipeline and threshold under `models/`; these generated files are not included in Git.

## Run locally

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). Place `PS_20174392719_1491204439457_log.csv` at `data/raw/PS_20174392719_1491204439457_log.csv`.
2. From the repository root, create an environment and install the dependencies:

   ```bash
   python -m venv .venv
   # Activate .venv using your shell's activation command.
   python -m pip install -r requirements.txt
   python -m jupyter lab
   ```

3. Open the notebooks in `notebooks/` and run them in order. The dataset-loading cells support kernels started from either the repository root or the `notebooks/` directory.

Full reruns use a 6.36-million-row dataset and may require substantial memory and time. Saved notebook outputs are included so the existing analysis can be read without rerunning it.

## Limitations

PaySim is synthetic; results may not generalize to real transactions. The hour-of-day feature can reflect patterns specific to the simulation. Fraud prevalence changes across the time periods, and the final model still misses 2,219 fraud cases on the test set. Model scores are not calibrated real-world fraud probabilities. This is a research project, not a production fraud prevention system.

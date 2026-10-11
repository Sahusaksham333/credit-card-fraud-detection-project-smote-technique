# Credit Card Fraud Detection — Imbalance Learning

A machine learning project that detects potentially fraudulent credit card transactions while addressing severe class imbalance using **SMOTE** and comparing **Logistic Regression** with **Random Forest**.

## Project Overview

Credit card fraud datasets are highly imbalanced: legitimate transactions vastly outnumber fraudulent ones. A classifier can therefore achieve deceptively high accuracy while failing to identify fraud. This project evaluates approaches that pay closer attention to the minority fraud class.

The assignment was completed for **Imbalance Learning (2000095)** at **Madhav Institute of Technology & Science, Gwalior**.

## Objectives

- Explore the transaction dataset and examine the target-class distribution.
- Check for missing values and prepare the data for modelling.
- Standardize the `Time` and `Amount` features.
- Preserve the class ratio in an 80/20 train-test split.
- Compare Logistic Regression without balancing, Logistic Regression with SMOTE, and Random Forest with SMOTE.
- Evaluate performance using precision, recall, F1-score, and ROC-AUC.

## Dataset

The assignment report describes **284,807 transactions**, including **492 fraudulent transactions** (`Class = 1`) and **284,315 legitimate transactions** (`Class = 0`). The dataset contains 31 columns: `Time`, anonymized features `V1`–`V28`, `Amount`, and the target `Class`.

The notebook loads the dataset from `creditcard.csv`. The dataset is not included in this repository by default. Obtain it from an authorized source and place it in the project directory before running the notebook. Do not commit sensitive or restricted data.

## Methods and Workflow

1. **Exploratory data analysis**
   - Inspect the first rows and dataset schema.
   - Visualize the class distribution.
   - Check for missing values.
2. **Preprocessing**
   - Standardize `Time` and `Amount` using `StandardScaler`.
   - Separate features (`X`) from the target (`y`).
   - Split into training and test sets with an 80/20 ratio and stratification.
3. **Imbalance handling**
   - Apply `SMOTE` to the training set only, avoiding resampling the test set.
4. **Model training**
   - Logistic Regression without balancing (baseline).
   - Logistic Regression trained on SMOTE-resampled data.
   - Random Forest trained on SMOTE-resampled data.
5. **Evaluation**
   - Review classification reports, especially the minority-class precision, recall, and F1-score.
   - Calculate ROC-AUC using Random Forest prediction probabilities.

## Reported Results

The following values are transcribed from the supplied notebook output; results can vary if the dataset, package versions, split, or random seeds differ.

| Model | Fraud precision | Fraud recall | Fraud F1-score | Notes |
|---|---:|---:|---:|---|
| Logistic Regression (no balancing) | 0.83 | 0.64 | 0.72 | Baseline; misses some fraud cases |
| Logistic Regression + SMOTE | 0.06 | 0.92 | 0.11 | Higher recall, but many false positives |
| Random Forest + SMOTE | 0.83 | 0.81 | 0.82 | Best balance among the three reported models |

**Reported Random Forest ROC-AUC:** `0.9688`

Accuracy alone is not a sufficient metric for this problem. In a fraud-detection setting, precision and recall represent a practical trade-off: false positives can inconvenience legitimate customers, while false negatives allow fraud to go undetected.

## Technology Stack

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn (`SMOTE`)
- Jupyter Notebook / Google Colab

## Repository Structure

```text
credit-card-fraud-detection/
├── A6.ipynb              # Analysis and model training notebook
├── README.md
├── requirements.txt      # Project dependencies
└── creditcard.csv        # Dataset file; obtain separately, do not commit if restricted
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
cd <YOUR-REPOSITORY>
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Create a `requirements.txt` containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

Then run:

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset and run the notebook

Place `creditcard.csv` beside `A6.ipynb`, then launch:

```bash
jupyter notebook A6.ipynb
```

You can also upload the notebook and dataset to Google Colab and update the dataset path if needed.

## Important Implementation Note

For a production-quality experiment, fit preprocessing steps only on the training data and use an `imblearn` pipeline so resampling occurs within each training fold during cross-validation. The notebook standardizes `Time` and `Amount` before the train-test split; fitting the scaler before splitting can leak information from the test set. A revised implementation should split first, fit the scaler on training data only, and transform the test data with that fitted scaler.

## Future Improvements

- Tune the classification threshold based on the cost of false positives and false negatives.
- Compare class-weighted models and majority-class undersampling.
- Use cross-validation with an `imblearn` pipeline.
- Evaluate Precision-Recall AUC, confusion matrices, and fraud-class metrics.
- Investigate anomaly-detection approaches such as autoencoders.
- Track experiments and model parameters for reproducibility.

## Author

**Saksham Sahu**  
B.Tech — Mathematics and Computing  
Madhav Institute of Technology & Science, Gwalior

## Academic Context

Assignment 6 — Imbalance Learning (2000095), Session January–June 2026.

---

*This repository documents an academic machine-learning experiment. The reported metrics are from the submitted notebook and should not be interpreted as a guarantee of real-world fraud-detection performance.*

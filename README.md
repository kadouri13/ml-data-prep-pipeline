# End-to-End Machine Learning Data Preparation & Modeling Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, leak-free machine learning workflow demonstrating industry and academic best practices for tabular data preprocessing, model selection, evaluation, and interpretability using **Scikit-Learn**.

---

## 📌 Project Overview

Data leakage and improper preprocessing are common pitfalls in applied machine learning. This repository illustrates how to build a modular, reproducible, and robust classification pipeline that:
* Prevents data leakage between training and testing splits.
* Combines disparate feature transformations (numerical scaling, imputation, categorical encoding) into a single `ColumnTransformer`.
* Integrates estimators within `Pipeline` objects for leak-free cross-validation.
* Evaluates model performance using a comprehensive suite of diagnostic metrics and curves.
* Interprets model decisions via feature coefficients and feature importances.

---

## 🚀 Workflow Architecture

```
Raw Data
   │
   ▼
[Exploratory Data Inspection]
   │
   ▼
[Stratified Train/Test Split] ── (80% Train / 20% Test, stratify=y)
   │
   ▼
[ColumnTransformer Preprocessor]
   ├── Numerical Pipeline: SimpleImputer(median) ➔ StandardScaler
   └── Categorical Pipeline: SimpleImputer(most_frequent) ➔ OneHotEncoder
   │
   ▼
[Scikit-Learn Pipeline]
   ├── Preprocessor
   └── Estimator (LogisticRegression / RandomForestClassifier)
   │
   ▼
[Stratified 5-Fold Cross-Validation] ── Leak-free validation across folds
   │
   ▼
[Final Evaluation on Unseen Test Set] ── Accuracy, Precision, Recall, F1, ROC-AUC
   │
   ▼
[Model Interpretability] ── Feature Coefficients & Feature Importances
```

---

## 📊 Benchmark Results

Evaluated on the Breast Cancer diagnostic dataset (569 samples, 30 features):

| Model | 5-Fold CV Accuracy | Test Accuracy | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | **97.80% ± 0.98%** | **98.25%** | **0.98** | **0.9954** |
| **Random Forest (100 Trees)** | 96.26% ± 1.79% | 96.49% | 0.96 | 0.9921 |

---

## 🛠️ Repository Structure

```
ml-data-prep-pipeline/
├── learn_preparing_data.ipynb  # Interactive Jupyter Notebook with complete pipeline & plots
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 💻 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kadouri13/ml-data-prep-pipeline.git
cd ml-data-prep-pipeline
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook learn_preparing_data.ipynb
```
or open it directly inside **VS Code** with the Jupyter extension.

---

## 📚 Key Concepts Demonstrated
* **Preventing Data Leakage**: Transformers are fit strictly on training splits.
* **`ColumnTransformer`**: Clean parallel handling of heterogeneous feature spaces.
* **`Pipeline`**: Encapsulating data transformations and model estimators.
* **Stratified K-Fold CV**: Maintaining class ratio distributions across evaluation folds.
* **Evaluation Suite**: Confusion Matrix Display, ROC-AUC Curve, and Classification Reports.

# Text Classification System using BERT and Ensemble Models

## Overview

This project focuses on building a robust **text classification system** using **BERT-based embeddings** and ensemble classifiers (Logistic Regression, Random Forest, XGBoost). The system extracts semantic features from textual data using a pre-trained transformer and trains several models for evaluation, comparison, and eventual ensemble-based inference.

---

## ✅ Work Completed

### 🔍 1. Data Loading
- Loads data from `balanced_100k_dataset.csv`, expecting columns `text` and `label`.

### 🧠 2. Feature Extraction via BERT
- A custom `BertFeatureExtractor` class uses `distilbert-base-uncased` to:
  - Tokenize text
  - Extract `[CLS]` and mean-pooled embeddings
  - Concatenate them to form a rich vector representation
- Embeddings are cached in `bert_embeddings.pt` for speedup.

### 🏗️ 3. Model Initialization
- Three classifiers initialized:
  - **Logistic Regression** with hyperparameter tuning using `GridSearchCV`
  - **Random Forest** with `GridSearchCV`
  - **XGBoost** with GPU acceleration (`gpu_hist`) if available

### 🏋️‍♂️ 4. Training & Evaluation
- Each model is trained on a train/val split.
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Confusion matrices are plotted and saved.

### 📊 5. Visualizations
- Confusion matrices (`{model}_confusion_matrix.png`)
- Metrics comparison bar chart (`model_metrics_comparison.png`)
- XGBoost accuracy/loss curves
- Tree visualization for both XGBoost and RandomForest (1st tree)

### 📦 6. Saving Artifacts
- Trained models saved as `.pkl` files with timestamp
- Results saved to:
  - `model_metrics.json`
  - `model_results.csv`
  - `model_report.md`

### 🤝 7. Ensemble Predictions
- Supports weighted ensemble predictions based on `predict_proba`
- Default weight is uniform
- Automatically handles models without `predict_proba`

---

## 🚧 Work to be Done

### 🧪 1. **Final Evaluation on Test Set**
- A separate, unseen test set should be evaluated to validate generalization.

### 🛠️ 2. **Model Export for Inference**
- Convert best-performing model(s) to ONNX / TorchScript for production deployment (optional).

### 🧠 3. **Interpretability and Explanation**
- Use SHAP or LIME to provide model explanations, especially for XGBoost and Random Forest.

### 🔁 4. **Support for Multiclass / Multilabel Tasks**
- Currently optimized for binary classification. Extend to handle multiclass cases.

### 🧪 5. **Error Analysis**
- Implement tools to analyze misclassified samples and improve model/data iteratively.

### 🖥️ 6. **GUI or CLI Frontend**
- Optional interactive interface for:
  - Uploading new datasets
  - Running predictions
  - Comparing model performance visually

---

## 🧰 Requirements

```bash
pip install torch transformers pandas scikit-learn xgboost seaborn matplotlib tqdm joblib

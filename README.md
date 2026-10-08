
# 🎯 Binary Classification Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org)
[![Google Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

An end-to-end machine learning pipeline for binary classification using multiple models, feature engineering, hyperparameter tuning, and ensemble methods.

---

## 🛠️ Pipeline Steps

| # | Step | Description |
|---|------|-------------|
| 1 | Data Loading | Load & explore dataset (500 samples, 2 features) |
| 2 | Train/Test Split | 80/20 split with stratification |
| 3 | Feature Engineering | Interaction terms, ratios, squared features |
| 4 | Model Comparison | Logistic Regression, SVM, Random Forest, Gradient Boosting |
| 5 | Hyperparameter Tuning | Grid Search with 5-fold CV |
| 6 | Ensemble | Soft Voting Classifier combining top 3 models |
| 7 | Threshold Optimization | Find optimal decision threshold |
| 8 | Visualization | Confusion Matrix, CV scores, Feature Importance |

---

## 📊 Results

| Model | CV F1 Score | Test Accuracy |
|-------|------------|---------------|
| Logistic Regression | ~94% | ~95% |
| SVM (RBF) | ~89% | ~90% |
| Random Forest | ~99.7% | 100% |
| Gradient Boosting | ~100% | 100% |
| **Ensemble** | — | **100%** |

---

## 🚀 Quick Start

```bash
# Open in Google Colab
# Run cells from top to bottom
# View results and charts
```

Or clone and run locally:
```bash
git clone https://github.com/mazyad-alrashidi/binary-classification-pipeline.git
cd binary-classification-pipeline
jupyter notebook notebook.ipynb
```

---

## 📈 Key Insights

- **Feature interaction** (`F1 × F2`) is the most important predictor
- **Gradient Boosting** outperforms other single models
- **Ensemble voting** achieves maximum accuracy
- Pipeline is ready to adapt for real-world datasets

---

## 🛠️ Tech Stack

- Python 3.x
- Scikit-learn
- Pandas · NumPy
- Matplotlib · Seaborn

---

## 📝 Note

Built as a learning project demonstrating ML best practices. Ready to adapt for real-world classification tasks.

---

<div align="center">

**⭐ If you find this useful, consider giving it a star!**

Made by [Mazyad Alrashidi](https://github.com/mazyad-alrashidi) | Riyadh, Saudi Arabia 🇸🇦

</div>

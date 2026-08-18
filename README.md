# 📊 Student Digital Habits & Wellness Analysis
> A Hybrid Supervised & Unsupervised Machine Learning Framework for Stress Risk Prediction & Behavioral Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Executive Summary
This project presents an end-to-end Machine Learning pipeline developed to analyze student lifestyle choices—including daily screen time, sleep duration, and study patterns—to accurately predict mental stress risk and identify underlying behavioral student segments. 

By deploying both **Supervised Classification Algorithms** and **Unsupervised Clustering & Anomaly Detection Techniques**, this study provides actionable insights for campus wellness programs and student intervention systems.

### Core Workflow Architecture
* **Data Preprocessing & Balancing:** Feature normalization using `StandardScaler` and class distribution alignment.
* **Supervised Risk Classification:** Evaluation of Decision Tree, Random Forest, and Hyperparameter-Tuned XGBoost via 5-Fold Cross-Validation.
* **Unsupervised Pattern Mining:** K-Means Clustering ($k=3$), Principal Component Analysis (PCA) for 2D visualization, Isolation Forest for anomaly detection, and Agglomerative Hierarchical Clustering.

---

## 📈 2. Supervised Learning Performance
Three classification models were trained and benchmarked to determine the optimal predictor for student stress risk. The hyperparameter-tuned **XGBoost Classifier** outperformed baseline models across all primary evaluation metrics.

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 82.0% | 80.0% | 81.0% | 80.5% | 0.85 |
| **Random Forest Classifier** | 88.0% | 86.0% | 87.0% | 86.5% | 0.91 |
| **Tuned XGBoost (GridSearchCV)** | **93.0%** | **91.0%** | **92.0%** | **91.5%** | **0.96** |

> **Key Feature Importance:** Feature importance metrics derived from XGBoost confirm that **Average Daily Usage Hours (Screen Time)** and **Sleep Hours Per Night** serve as the primary predictive features driving student stress classification.

---

## 🔍 3. Unsupervised Learning & Behavioral Segmentation

### A. K-Means Clustering & Persona Profiling
The optimal number of behavioral groups ($k=3$) was established using the **Elbow Method (Inertia Curve)** and validated with **Silhouette Scores**.

* **Cluster 0 (High Risk Group):** High daily digital usage (>7.5 hrs/day) paired with low sleep duration (<5 hrs/night).
* **Cluster 1 (Balanced Lifestyle):** Moderate screen time, optimal sleep (>7 hrs/night), and regular physical activity.
* **Cluster 2 (Low Risk / High Academic Focus):** Low daily screen time with higher dedicated study hours and stable sleep.

### B. Dimensionality Reduction & Anomaly Detection
* **PCA (Principal Component Analysis):** Compressed multidimensional feature space into a 2D plane for clean cluster boundary rendering.
* **Isolation Forest:** Flagged extreme outlier students suffering from severe sleep deprivation combined with excessive screen exposure.
* **Hierarchical Clustering:** Rendered dendrogram structures reflecting multi-level hierarchical relationships across student profiles.

---

## 🚀 4. Recommendations & Future Scope
1. **Real-time Web Deployment:** Export the trained XGBoost model (`.pkl` file) into a Streamlit interactive dashboard for self-administered wellness evaluations.
2. **Targeted Campus Interventions:** Enable academic counselors to offer early digital detox and sleep hygiene guidance to students flagged in Cluster 0 or identified by Isolation Forest.
3. **Feature Dataset Expansion:** Incorporate additional academic parameters (GPA), daily caffeine consumption, and structured exercise logs in future iterations.

# 🎓 Student Lifestyle & Stress Risk Analysis

## 🌐 Live Website
[🚀 Click Here to Visit Live Website](https://student-lifestyle-clustering-8zns.onrender.com)

## 💻 Project Repository
[View Source Code on GitHub](https://github.com/kartikcodes03/student-lifestyle-clustering)



# Student Lifestyle & Stress Risk Analysis

A machine learning project that studies students' digital habits and
daily routines to explore stress-risk patterns and discover
lifestyle-based student segments.

> **Project status:** Exploratory / portfolio project. The reported
> metrics below are the saved outputs from the current notebook and
> should be treated as provisional until the model pipeline and
> cross-validation procedure are corrected and rerun. This project is
> not a medical or psychological diagnostic tool.

## Overview

This project uses both supervised and unsupervised machine learning:

-   **Classification:** Predicts a binary stress-risk label derived from
    the dataset's `Stress_Level` field.
-   **Clustering:** Groups students using screen usage, sleep, study,
    and physical-activity hours.
-   **Dimensionality reduction:** Uses PCA to visualize student groups
    in two dimensions.
-   **Anomaly detection:** Uses Isolation Forest to flag unusual
    lifestyle profiles.

## Objectives

1.  Explore relationships between students' digital habits and lifestyle
    indicators.
2.  Create a binary target: `High_Stress_Risk = 1` for `High` or
    `Very High` stress; otherwise `0`.
3.  Compare Decision Tree, Random Forest, and Gradient Boosting
    classifiers.
4.  Identify lifestyle segments with K-Means and inspect hierarchical
    groupings.
5.  Visualize clusters with PCA and flag potential outliers with
    Isolation Forest.

## Dataset

The notebook expects a CSV file named:

`Student Social Media And Mental Health Impact.csv`

Place the dataset in the project folder (or update the path in the
notebook). The notebook uses fields including:

-   `Stress_Level`
-   `Avg_Daily_Usage_Hours`
-   `Daily_Unlocks`
-   `Sleep_Hours_Per_Night`
-   `Physical_Activity_Hours`
-   `Study_Hours`
-   `Gender`
-   `Country`
-   `Academic_Level`
-   `Most_Used_Platform`
-   `Purpose_Of_Use`

The dataset is not included in this repository by default. Obtain it
from its original source and check its license and usage terms before
redistribution. Add the dataset source and license details here before
publishing the project.

## Feature Engineering

The notebook creates three derived features:

-   `Digital_Overload_Index`
-   `Wellness_Ratio`
-   `Screen_To_Sleep_Ratio`

Categorical columns are one-hot encoded using `pandas.get_dummies()`.

## Machine Learning Workflow

### Supervised learning

The current notebook trains and evaluates:

-   Decision Tree Classifier
-   Random Forest Classifier
-   Gradient Boosting Classifier with `GridSearchCV`

**Implementation note:** The notebook's third estimator is
`sklearn.ensemble.GradientBoostingClassifier`, not `XGBClassifier`. Some
existing variable names and plot labels call it "XGBoost"; those labels
should be corrected unless a real XGBoost implementation is added.

The notebook uses an 80/20 train-test split with `stratify=y` and random
state 42. It balances the training data by resampling the minority
class.

### Current saved test-set results

The following values are transcribed from the notebook's existing saved
output. The third column is Gradient Boosting, despite the notebook's
current "XGBoost" label.

  Metric        Decision Tree   Random Forest   Gradient Boosting
  ----------- --------------- --------------- -------------------
  Accuracy             0.9130          0.9420              0.9510
  Precision            0.9268          0.9497              0.9490
  Recall               0.9314          0.9559              0.9722
  F1-score             0.9291          0.9528              0.9605
  ROC-AUC              0.9153          0.9895              0.9915

The notebook also records a mean 5-fold accuracy of `0.9673` (standard
deviation `0.0065`) for the tuned Gradient Boosting estimator on the
resampled training data. Because resampling is performed before
cross-validation in the current notebook, this cross-validation result
may be optimistic. Move resampling inside each training fold with an
imbalanced-learn pipeline, then rerun tuning and evaluation before using
these numbers as final portfolio claims.

### Unsupervised learning

K-Means uses these four lifestyle features:

-   `Avg_Daily_Usage_Hours`
-   `Sleep_Hours_Per_Night`
-   `Study_Hours`
-   `Physical_Activity_Hours`

The notebook selects `k=3`, examines the Elbow curve and Silhouette
Score, and visualizes the resulting groups.

#### Saved cluster averages

  -----------------------------------------------------------------------
  Cluster        Daily usage  Sleep (hours)  Study (hours)       Physical
                     (hours)                                     activity
                                                                  (hours)
  ----------- -------------- -------------- -------------- --------------
  0                     6.75           5.46           1.47           1.27

  1                     3.08           8.04           5.02           2.39

  2                     4.75           6.87           3.19           1.80
  -----------------------------------------------------------------------

These are descriptive averages, not clinical categories. Interpret and
name clusters from their measured characteristics rather than assigning
a stress diagnosis.

### PCA and anomaly detection

-   PCA reduces the four scaled clustering features to two components
    for visualization. The saved notebook output reports **90.53%**
    total variance explained by the two components.
-   Isolation Forest is configured with `contamination=0.05`. The saved
    output flags 250 of 5,000 records as outliers. These are statistical
    outliers, not necessarily students experiencing a health problem.

### Hierarchical clustering

The notebook creates a Ward-linkage dendrogram. For a clearer and more
scalable visualization, consider generating it from a documented
representative sample rather than all records.

## Repository Structure

``` text
student-lifestyle-clustering/
├── data/
│   └── README.md                 # Dataset source, license and setup notes
├── notebooks/
│   └── student-lifestyle-clustering.ipynb
├── README.md
└── requirements.txt
```

Adjust the structure to match the files actually committed to the
repository. Do not commit private, restricted, or unlicensed data.

## Installation

``` bash
git clone https://github.com/kartikcodes03/student-lifestyle-clustering.git
cd student-lifestyle-clustering

python -m venv .venv
```

Activate the environment:

**macOS / Linux**

``` bash
source .venv/bin/activate
```

**Windows**

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy imbalanced-learn jupyter
```

Launch Jupyter:

``` bash
jupyter notebook
```

Open the notebook and update the CSV path to the location where you
saved the dataset. The current notebook contains a Google Colab path
(`/content/...`), which will need changing when running locally.

## Evaluation and Visualizations

The notebook includes:

-   Classification metric comparison
-   Confusion matrices
-   ROC curves
-   Classification reports
-   Correlation heatmap
-   K-Means Elbow curve
-   Silhouette score comparison
-   Cluster scatter plot
-   PCA cluster visualization
-   Isolation Forest outlier counts
-   Hierarchical clustering dendrogram

## Limitations and Next Steps

-   Correct the Gradient Boosting / XGBoost naming throughout the
    notebook and README.
-   Put resampling inside cross-validation using an `imblearn` pipeline
    to prevent information leakage between folds.
-   Re-run hyperparameter tuning and all test metrics after correcting
    the pipeline.
-   Add the dataset's original source, license, and citation.
-   Add a complete `requirements.txt`, saved model artifacts only after
    validation, and screenshots of actual notebook outputs.
-   Consider explainability (for example, SHAP) after the modeling
    pipeline is stable.
-   Validate cluster quality and interpret cluster profiles carefully.
-   Build a small interactive dashboard only after the analysis and
    evaluation are reproducible.

## Responsible Use

The target is a dataset-derived stress-risk label, not a diagnosis.
Predictions and outlier flags must not be used to label, rank, penalize,
or make high-stakes decisions about individual students. Any real-world
use would require appropriate consent, privacy safeguards, domain-expert
review, and validation on suitable data.

## Author

**Kartik Sharma**

GitHub: [kartikcodes03](https://github.com/kartikcodes03)

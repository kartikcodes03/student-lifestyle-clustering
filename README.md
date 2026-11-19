# 📊 Student Digital Habits & Wellness Analysis
> A Hybrid Supervised & Unsupervised Machine Learning Framework for Stress Risk Prediction & Behavioral Segmentation

Overview
This project studies the relationship between students' social-media/digital usage, sleep, study habits and physical activity, and their reported stress levels. It combines supervised classification for stress-risk prediction with unsupervised methods to explore behavioral groups and detect unusual observations.
Important: The notebook explores associations in the dataset. It is an educational ML project, not a clinical assessment, diagnosis, or substitute for professional support.

Objectives
- Prepare and encode student lifestyle data for machine learning.
- Create interpretable derived features representing digital usage and lifestyle balance.
- Compare three supervised classifiers for a binary stress-risk target.
- Evaluate models using accuracy, classification reports, confusion matrices and ROC-AUC curves.
- Explore student behavioral groups with K-Means and PCA.
- Use Isolation Forest and hierarchical clustering to explore unusual observations and similarity structure.
Dataset
Notebook input file: Student Social Media And Mental Health Impact.csv.

The notebook loads the CSV from a Google Colab path. Place the CSV in your working directory or update the loading path before running locally. The source notebook expects columns such as Stress_Level, Avg_Daily_Usage_Hours, Daily_Unlocks, Sleep_Hours_Per_Night, Physical_Activity_Hours, Study_Hours, and categorical variables including Gender, Country, Academic_Level, Most_Used_Platform, and Purpose_Of_Use.
Feature engineering
The notebook creates these derived variables:
- High_Stress_Risk: mapped to 1 for High or Very High stress levels and 0 otherwise.
- Digital_Overload_Index: (Avg_Daily_Usage_Hours × Daily_Unlocks) / (Sleep_Hours_Per_Night + 1).
- Wellness_Ratio: (Sleep_Hours_Per_Night + Physical_Activity_Hours + Study_Hours) / (Avg_Daily_Usage_Hours + 1).
- Screen_To_Sleep_Ratio: Avg_Daily_Usage_Hours / (Sleep_Hours_Per_Night + 0.1).
Categorical features are one-hot encoded using pandas.get_dummies(..., drop_first=True). The classification target is High_Stress_Risk; Stress_Level and the target column itself are removed from the predictor matrix as specified in the notebook.
Supervised learning workflow
1. Create the binary stress-risk target.
2. Engineer the lifestyle/digital-use features.
3. One-hot encode categorical variables.
4. Split into training and testing sets (80/20, stratified, random state 42).
5. Balance training classes through resampling (training set only).
6. Train and compare classifiers.
Models
- Decision Tree: entropy criterion, min_samples_split=5, random_state=42.
- Random Forest: 300 estimators, max_depth=14, random_state=42.
- Gradient Boosting (tuned): the notebook uses GradientBoostingClassifier, grid search and stratified 5-fold cross-validation. The notebook labels this tuned model as XGBoost in some plots/text, but the estimator used in the shown code is scikit-learn Gradient Boosting, not the external XGBoost library.
Evaluation
The notebook compares the models using:
- Accuracy and other classification metrics
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- ROC curves and ROC-AUC
- Feature-importance plots
- Correlation heatmap
Results depend on the CSV, train/test split and run environment. Consult the notebook outputs for exact metrics; this README does not claim specific performance scores.
Unsupervised learning and exploratory analysis
The notebook scales these four variables before clustering:
- Avg_Daily_Usage_Hours
- Sleep_Hours_Per_Night
- Study_Hours
- Physical_Activity_Hours
K-Means
- Use the elbow/inertia plot and silhouette scores to explore possible cluster counts.
- Fit the final K-Means model with k = 3, random_state=42, and n_init=10.
- Summarize cluster averages and visualize clusters with screen-use vs sleep plots.
PCA
Principal Component Analysis projects the scaled cluster features into two components for a 2-D visualization. PCA is used for visualization/exploration here, not as the stress classifier in the supervised pipeline.
Anomaly detection
Isolation Forest is applied to the scaled lifestyle variables with contamination=0.05 and random_state=42. In its output, -1 indicates an observation flagged as an anomaly and 1 indicates a regular observation. An anomaly is not automatically a problem or a negative label.
Hierarchical clustering
Ward linkage is used to generate a dendrogram, allowing the structure and similarity of observations to be explored.
Repository structure
student-wellness-project/
├── README.md
├── student-lifestyle-clustering.ipynb
├── Student Social Media And Mental Health Impact.csv  # add dataset if permitted
└── index.html                                         # optional static demo UI
Run the notebook
1. Install Python 3.10+ and Jupyter, or open the notebook in Google Colab.
2. Install the notebook dependencies:pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
3. Place the CSV at the path used in the notebook, or edit the loading cell.
4. Run notebook cells from top to bottom.
5. Review the generated model evaluation and clustering charts.
Static frontend demo
The optional index.html is a standalone interactive UI demonstration. It runs in the browser and has no server requirements. It uses client-side illustrative calculations/rules, not the notebook's trained estimators. Do not present its output as an actual model prediction. To use the notebook's trained model in a real application, serialize the preprocessing pipeline and estimator, then connect them through a backend or a compatible browser inference pipeline.
Limitations and ethical considerations
- Self-reported lifestyle and stress data can be noisy, incomplete or biased.
- Model performance on one dataset does not guarantee performance on other students or populations.
- A predicted risk category is not a diagnosis or a label for a person.
- Use the results for learning and cautious exploration; avoid consequential decisions based on these outputs.
Conclusion
The notebook demonstrates an end-to-end machine-learning workflow: data preparation, feature engineering, class balancing, supervised model comparison, evaluation, behavioral clustering, PCA visualization, anomaly detection and hierarchical analysis. Combining prediction and exploratory pattern discovery offers several views of the dataset while requiring careful interpretation.
2. **Targeted Campus Interventions:** Enable academic counselors to offer early digital detox and sleep hygiene guidance to students flagged in Cluster 0 or identified by Isolation Forest.
3. **Feature Dataset Expansion:** Incorporate additional academic parameters (GPA), daily caffeine consumption, and structured exercise logs in future iterations.

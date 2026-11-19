import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(
    page_title="Student Lifestyle Intelligence",
    page_icon="🎓",
    layout="wide"
)
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #132743, #245b82);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: white;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6ebf2;
        padding: 14px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] * {
        color: #172033 !important;
    }
</style>
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Student Lifestyle Intelligence</h1>
  <p>Explore student digital habits, lifestyle segments and dataset-derived stress-risk patterns.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational portfolio demo only. The stress-risk output reflects a label learned from the uploaded "
    "dataset; it is not a medical or psychological diagnosis."
)

# These are the columns used in the supplied notebook's feature engineering.
NUMERIC_FEATURES = [
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Sleep_Hours_Per_Night",
    "Physical_Activity_Hours",
    "Study_Hours",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
]
REQUIRED = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Stress_Level"]
CLUSTER_FEATURES = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Study_Hours",
    "Physical_Activity_Hours",
]

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def prepare_data(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    df = raw.copy()
    df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip()
    df = df[df["Stress_Level"].str.lower().isin(
        ["low", "medium", "moderate", "high", "very high"]
    )].copy()

    stress = df["Stress_Level"].str.lower()
    y = stress.isin(["high", "very high"]).astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Recreate the notebook's three engineered numeric features.
    X["Digital_Overload_Index"] = (
        X["Avg_Daily_Usage_Hours"] * X["Daily_Unlocks"]
    ) / (X["Sleep_Hours_Per_Night"] + 1)
    X["Wellness_Ratio"] = (
        X["Sleep_Hours_Per_Night"] +
        X["Physical_Activity_Hours"] +
        X["Study_Hours"]
    ) / (X["Avg_Daily_Usage_Hours"] + 1)
    X["Screen_To_Sleep_Ratio"] = (
        X["Avg_Daily_Usage_Hours"] /
        (X["Sleep_Hours_Per_Night"] + 0.1)
    )

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)

    return df, X, y

def make_model(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    # Oversampling occurs only inside each training fold / training split.
    return ImbPipeline([
        ("preprocess", preprocessor),
        ("sampler", RandomOverSampler(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_split=5,
            random_state=42,
            class_weight=None,
            n_jobs=-1
        ))
    ])

@st.cache_resource(show_spinner="Training the model on your dataset...")
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_model(X_train)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_test, prob)
    except ValueError:
        metrics["ROC-AUC"] = np.nan

    return model, X_test, y_test, pred, metrics

def add_engineered_features(values):
    row = dict(values)
    usage = float(row["Avg_Daily_Usage_Hours"])
    unlocks = float(row["Daily_Unlocks"])
    sleep = float(row["Sleep_Hours_Per_Night"])
    activity = float(row["Physical_Activity_Hours"])
    study = float(row["Study_Hours"])
    row["Digital_Overload_Index"] = usage * unlocks / (sleep + 1)
    row["Wellness_Ratio"] = (sleep + activity + study) / (usage + 1)
    row["Screen_To_Sleep_Ratio"] = usage / (sleep + 0.1)
    return row

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader(
        "Upload the project CSV",
        type=["csv"],
        help="Expected columns are listed below."
    )
    st.caption("Required: " + ", ".join(REQUIRED))
    st.divider()
    st.caption("Built from the feature definitions in the project's Jupyter notebook.")

if uploaded is None:
    st.info("Upload **Student Social Media And Mental Health Impact.csv** in the sidebar to start.")
    st.subheader("What this app provides")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 📊 Model evaluation\nTrain/test metrics and a confusion matrix.")
    c2.markdown("### 🧩 Lifestyle groups\nK-Means segments and cluster profiles.")
    c3.markdown("### 🔎 Risk estimate\nA dataset-derived estimate from entered lifestyle details.")
    st.stop()

try:
    raw_df = load_data(uploaded.getvalue())
    df, X, y = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Could not read this dataset: {exc}")
    st.stop()

if len(X) < 30:
    st.error("The uploaded dataset has too few valid rows for a meaningful train/test demonstration (need at least 30).")
    st.stop()
if y.nunique() < 2:
    st.error("The target must contain both low/medium and high/very high stress records.")
    st.stop()

st.success(f"Dataset loaded: {len(df):,} records | {int(y.sum()):,} high-risk labels | {int((1-y).sum()):,} other labels")

try:
    model, X_test, y_test, y_pred, metrics = train_model(X, y)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Overview & Model", "🧩 Lifestyle Clusters", "🧪 Try a Student Profile"])

with tab1:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Students", f"{len(df):,}")
    b.metric("High-stress label", f"{y.mean()*100:.1f}%")
    c.metric("Features used", str(X.shape[1]))

    st.subheader("Held-out test performance")
    metric_cols = st.columns(len(metrics))
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, "N/A" if pd.isna(value) else f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm)
        ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["Other stress levels", "High / Very High"],
            output_dict=True, zero_division=0
        )
        report.pop("accuracy", None)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.caption("Metrics are computed at app runtime on a stratified held-out split. They may differ from the notebook's saved outputs.")

with tab2:
    st.subheader("K-Means lifestyle segmentation")
    k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)
    cluster_df = df[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    cluster_df = cluster_df.fillna(cluster_df.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    profile = cluster_df.assign(Cluster=labels).groupby("Cluster").mean().round(2)
    st.dataframe(profile, use_container_width=True)

    try:
        sil = silhouette_score(scaled, labels)
        st.metric("Silhouette score", f"{sil:.3f}")
    except Exception:
        st.info("Silhouette score is not available for this dataset.")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        cluster_df["Avg_Daily_Usage_Hours"],
        cluster_df["Sleep_Hours_Per_Night"],
        c=labels, alpha=0.65
    )
    ax.set_xlabel("Average daily usage (hours)")
    ax.set_ylabel("Sleep per night (hours)")
    ax.set_title("Student lifestyle clusters")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Cluster IDs are arbitrary. Interpret groups using the averages above; they are not clinical categories.")

with tab3:
    st.subheader("Explore a student lifestyle profile")
    st.write("Enter values to estimate which dataset-derived stress label the trained model associates with this profile.")

    cat_options = {}
    for col in CATEGORICAL_FEATURES:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        cat_options[col] = vals if vals else ["Unknown"]

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            usage = st.number_input("Average daily screen usage (hours)", 0.0, 24.0, 5.0, 0.5)
            unlocks = st.number_input("Daily phone unlocks", 0, 500, 50, 1)
            sleep = st.number_input("Sleep per night (hours)", 0.0, 24.0, 7.0, 0.5)
            study = st.number_input("Study hours per day", 0.0, 24.0, 3.0, 0.5)
            activity = st.number_input("Physical activity hours per day", 0.0, 24.0, 1.0, 0.5)
        with col2:
            gender = st.selectbox("Gender", cat_options["Gender"])
            country = st.selectbox("Country", cat_options["Country"])
            academic = st.selectbox("Academic level", cat_options["Academic_Level"])
            platform = st.selectbox("Most used platform", cat_options["Most_Used_Platform"])
            purpose = st.selectbox("Purpose of use", cat_options["Purpose_Of_Use"])
        submitted = st.form_submit_button("Estimate dataset label", type="primary")

    if submitted:
        row = add_engineered_features({
            "Avg_Daily_Usage_Hours": usage,
            "Daily_Unlocks": unlocks,
            "Sleep_Hours_Per_Night": sleep,
            "Physical_Activity_Hours": activity,
            "Study_Hours": study,
            "Gender": gender,
            "Country": country,
            "Academic_Level": academic,
            "Most_Used_Platform": platform,
            "Purpose_Of_Use": purpose,
        })
        sample = pd.DataFrame([row])[X.columns]
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0, 1])
        if prediction == 1:
            st.warning(f"Model-assigned label: High / Very High stress category (probability: {probability:.1%})")
        else:
            st.success(f"Model-assigned label: Low / Medium stress category (probability of high-risk label: {probability:.1%})")
        st.caption("This is a model output based on patterns in the uploaded dataset, not an assessment of a person's mental health.")

st.divider()
st.caption("Student Lifestyle Intelligence | Machine Learning portfolio demonstration | Not for clinical or high-stakes use.")

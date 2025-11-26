import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="IoT Malware Detection Dashboard",
    layout="wide"
)

# ----------------- Dark mode toggle (simple CSS) -----------------
dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=True)

if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #111111; color: #EEEEEE; }
        .stApp { background-color: #111111; color: #EEEEEE; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# LOAD DATA & MODELS
# ============================================================

@st.cache_resource
def load_models_and_data():
    # Test split (preprocessed, scenario-based)
    X_test = pd.read_csv("data/processed/X_test_scenario.csv")
    y_test = pd.read_csv("data/processed/y_test_scenario.csv")["label"]

    # Encoders & scaler – used for upload
    ord_enc = joblib.load("saved_models/ordinal_encoder_scenario.pkl")
    scaler = joblib.load("saved_models/scaler_scenario.pkl")

    # Models
    models = {
        "Decision Tree": joblib.load("saved_models/decision_tree_scenario.pkl"),
        "Decision Tree (tuned)": joblib.load("saved_models/decision_tree_scenario_tuned.pkl"),
        "Random Forest": joblib.load("saved_models/random_forest_scenario.pkl"),
        "Random Forest (tuned)": joblib.load("saved_models/random_forest_scenario_tuned.pkl"),
    }

    return X_test, y_test, models, ord_enc, scaler


X_test, y_test, MODELS, ORD_ENCODER, SCALER = load_models_and_data()

CAT_COLS = ["proto", "conn_state", "history"]
CAT_COLS = [c for c in CAT_COLS if c in X_test.columns]
NUM_COLS = [c for c in X_test.columns if c not in CAT_COLS]


# ============================================================
# METRIC HELPERS
# ============================================================

def compute_metrics(model, X, y, threshold=0.5):
    """Compute classification metrics with adjustable threshold."""
    # Get probabilities if supported
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]  # P(Malicious)
        preds = np.where(probs >= threshold, "Malicious", "Benign")
    else:
        probs = None
        preds = model.predict(X)

    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, preds, average="weighted"
    )
    cm = confusion_matrix(y, preds)

    roc_auc = None
    fpr = tpr = pr_prec = pr_rec = None

    if probs is not None:
        y_bin = (y == "Malicious").astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs)
        roc_auc = auc(fpr, tpr)
        pr_prec, pr_rec, _ = precision_recall_curve(y_bin, probs)

    return {
        "preds": preds,
        "probs": probs,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "pr_prec": pr_prec,
        "pr_rec": pr_rec,
    }


def plot_confusion_matrix(cm, dark=False):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticklabels(["Benign", "Malicious"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr(rec, prec):
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_k=15):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    feat = np.array(feature_names)[idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feat[::-1], vals[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    return fig


# ============================================================
# STREAMLIT LAYOUT
# ============================================================

st.title("📡 IoT Malware Detection — Scenario-based Evaluation Dashboard")

st.markdown(
    """
This dashboard visualizes the **scenario-based models** trained on the IoT-23 dataset.

- Train: CTU-IoT-Malware-Capture-3, 8, 20, 21  
- Test: Somfy-01 + CTU-IoT-Malware-Capture-34, 42, 44 + CTU-Honeypot-Capture-4, 5  
- Labels: **Benign vs Malicious** (flagging any attack as Malicious)
"""
)

tab_overview, tab_model, tab_threshold, tab_importance, tab_upload = st.tabs(
    ["🔎 Overview", "📊 Model Details", "🎚 Threshold Analysis", "🧬 Feature Importance", "📁 Upload & Test"]
)

# ============================================================
# TAB 1 — OVERVIEW: Multi-model comparison
# ============================================================

with tab_overview:
    st.subheader("Overall Comparison (Scenario Test Set)")

    rows = []
    for name, model in MODELS.items():
        metrics = compute_metrics(model, X_test, y_test, threshold=0.5)
        rows.append(
            {
                "Model": name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1": round(metrics["f1"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4) if metrics["roc_auc"] is not None else None,
            }
        )

    res_df = pd.DataFrame(rows)
    st.dataframe(res_df, use_container_width=True)

    st.markdown("### Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(res_df["Model"], res_df["Accuracy"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(res_df["Model"], rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)


# ============================================================
# TAB 2 — MODEL DETAILS
# ============================================================

with tab_model:
    st.subheader("Detailed Metrics for a Single Model")

    model_name = st.selectbox("Select model", list(MODELS.keys()))
    model = MODELS[model_name]

    metrics = compute_metrics(model, X_test, y_test, threshold=0.5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(metrics["cm"])
        st.pyplot(fig_cm)

    with c2:
        if metrics["roc_auc"] is not None:
            st.markdown("#### ROC Curve")
            fig_roc = plot_roc(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])
            st.pyplot(fig_roc)
        else:
            st.info("ROC not available (model has no probability output).")

    st.markdown("#### Precision–Recall Curve")
    if metrics["pr_prec"] is not None:
        fig_pr = plot_pr(metrics["pr_rec"], metrics["pr_prec"])
        st.pyplot(fig_pr)
    else:
        st.info("PR curve not available.")


# ============================================================
# TAB 3 — THRESHOLD ANALYSIS
# ============================================================

with tab_threshold:
    st.subheader("Threshold Analysis (Malicious / Benign Trade-off)")

    model_name = st.selectbox("Choose model for threshold tuning", list(MODELS.keys()), key="threshold_model")
    model = MODELS[model_name]

    st.markdown("Move the slider to change the decision threshold on P(Malicious).")
    th = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    metrics = compute_metrics(model, X_test, y_test, threshold=th)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")

    st.markdown("#### Confusion Matrix at current threshold")
    fig_cm = plot_confusion_matrix(metrics["cm"])
    st.pyplot(fig_cm)

    # Show FP / FN explicitly
    cm = metrics["cm"]
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"""
        - True Benign (TN): **{tn}**  
        - False Positive (FP): **{fp}**  (benign flagged as malicious)  
        - False Negative (FN): **{fn}**  (malicious missed)  
        - True Malicious (TP): **{tp}**
        """
    )


# ============================================================
# TAB 4 — FEATURE IMPORTANCE
# ============================================================

with tab_importance:
    st.subheader("Feature Importance (Explainability)")

    model_name = st.selectbox("Choose model", list(MODELS.keys()), key="importance_model")
    model = MODELS[model_name]

    fig_imp = plot_feature_importance(model, list(X_test.columns), top_k=15)
    if fig_imp is None:
        st.info("This model does not expose feature_importances_.")
    else:
        st.pyplot(fig_imp)


# ============================================================
# TAB 5 — UPLOAD & PREDICT
# ============================================================

with tab_upload:
    st.subheader("Upload CSV to Test the Models")

    st.markdown(
        """
        The uploaded CSV should have **the same columns as the original flow-level dataset**  
        (before encoding/scaling): `proto`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `missed_bytes`, `history`, `orig_pkts`, `orig_ip_bytes`, `resp_pkts`, `resp_ip_bytes`, etc.

        For this project, the safest way is to:
        - take a subset of the raw scenarios,
        - run them through the same preprocessing pipeline you used for training,
        - and upload the resulting dataset.
        """
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.write("Raw uploaded data (first 5 rows):")
        st.write(raw.head())

        # Try to align columns with training schema
        missing_cols = [c for c in X_test.columns if c not in raw.columns]
        extra_cols = [c for c in raw.columns if c not in X_test.columns]

        st.markdown("#### Column check")
        st.write("Missing columns (expected but not found):", missing_cols)
        st.write("Extra columns (ignored):", extra_cols)

        # Only keep known columns, fill missing with 0
        df = raw.copy()
        for c in missing_cols:
            df[c] = 0

        df = df[X_test.columns]  # align order

        # Assume uploaded CSV is already numeric & encoded if it matches X_test schema.
        # If it's raw Zeek-style, user should preprocess it offline with the same pipeline.
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        model_name = st.selectbox("Model for prediction", list(MODELS.keys()), key="upload_model")
        model = MODELS[model_name]

        preds = model.predict(df)
        st.markdown("#### Predictions (first 20 rows)")
        out_df = pd.DataFrame({"prediction": preds})
        st.write(out_df.head(20))

        mal_rate = (preds == "Malicious").mean() * 100.0
        st.metric("Malicious traffic percentage", f"{mal_rate:.2f}%")

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import os

# Load Data

X_test = pd.read_csv("data/processed/X_test_scenario.csv")
y_test = pd.read_csv("data/processed/y_test_scenario.csv")["label"]

# Load models
models = {
    "Decision Tree": joblib.load("saved_models/decision_tree_scenario.pkl"),
    "Decision Tree (tuned)": joblib.load("saved_models/decision_tree_scenario_tuned.pkl"),
    "Random Forest": joblib.load("saved_models/random_forest_scenario.pkl"),
    "Random Forest (tuned)": joblib.load("saved_models/random_forest_scenario_tuned.pkl"),
}

os.makedirs("reports", exist_ok=True)

# For ROC curve storage
roc_dir = "reports/roc_curves"
os.makedirs(roc_dir, exist_ok=True)

results = []

print("\n=== Scenario Model Evaluation ===\n")

# Evaluate each model

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Predictions
    y_pred = model.predict(X_test)

    # Probability scores for ROC (if available)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve((y_test == "Malicious"), y_proba)
        roc_auc = auc(fpr, tpr)
    except:
        y_proba = None
        roc_auc = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if roc_auc is not None:
        print(f"AUC      : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save ROC plot
    if y_proba is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.title(f"ROC Curve: {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f"{roc_dir}/roc_{name.replace(' ', '_')}.png")
        plt.close()

    # Save results to list
    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": roc_auc
    })

# Export comparison table

results_df = pd.DataFrame(results)
results_df.to_csv("reports/model_comparison_scenario.csv", index=False)

print("\n=== Evaluation complete ===")
print("Results saved to reports/model_comparison_scenario.csv")
print("ROC curves saved to reports/roc_curves/")

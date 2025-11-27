import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV


# Load Scenario Split

X_train = pd.read_csv("data/processed/X_train_scenario.csv")
X_test  = pd.read_csv("data/processed/X_test_scenario.csv")
y_train = pd.read_csv("data/processed/y_train_scenario.csv")["label"]
y_test  = pd.read_csv("data/processed/y_test_scenario.csv")["label"]

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# Baseline Random Forest

print("\n=== Baseline Random Forest (Scenario-Based) ===")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

start = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start

print(f"Training time: {train_time:.2f}s")

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(rf, "saved_models/random_forest_scenario.pkl")
print("\nSaved baseline RF scenario model → saved_models/random_forest_scenario.pkl")


# Tuned Random Forest

print("\n=== Tuned Random Forest (Scenario-Based) ===")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=1
)

start = time.time()
grid.fit(X_train, y_train)
tune_time = time.time() - start

print(f"Tuning time: {tune_time:.2f}s")
print("Best parameters:", grid.best_params_)

best_rf = RandomForestClassifier(
    **grid.best_params_,
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train, y_train)

y_pred2 = best_rf.predict(X_test)

acc2 = accuracy_score(y_test, y_pred2)
prec2, rec2, f12, _ = precision_recall_fscore_support(
    y_test, y_pred2, average="weighted"
)
#Metrics
print(f"Accuracy: {acc2:.4f}")
print(f"Precision: {prec2:.4f}")
print(f"Recall: {rec2:.4f}")
print(f"F1 Score: {f12:.4f}")

print("\nConfusion Matrix (Tuned):")
print(confusion_matrix(y_test, y_pred2))

joblib.dump(best_rf, "saved_models/random_forest_scenario_tuned.pkl")
print("\nSaved tuned RF scenario model → saved_models/random_forest_scenario_tuned.pkl")

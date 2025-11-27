"""
NOTE:
This preprocessing script was part of the initial full_dataset approach.
It performs a random 80/20 train-test split, which is NOT appropriate for IoT-23
because flows from the same capture scenario can leak into both sets.

This script is kept for documentation purposes only.
The final project uses scenario-based preprocessing:
  → preprocess_scenario_split.py
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

INPUT_FILE = "data/processed/final_dataset.csv"
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

print("\n=== LOADING CLEAN DATASET ===")
df = pd.read_csv(INPUT_FILE)

# FIX LABEL INCONSISTENCY
df["label"] = df["label"].replace({"benign": "Benign"})

print("\nLabel distribution AFTER fix:")
print(df["label"].value_counts())

# TRAIN/TEST SPLIT BEFORE ANY ENCODING
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# LABEL ENCODING (FIT ONLY ON TRAIN)
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_test_enc = label_enc.transform(y_test)

# Save label encoder for models
import joblib
joblib.dump(label_enc, "saved_models/label_encoder.pkl")

# ENCODE CATEGORICAL FEATURES
cat_cols = ["proto", "conn_state", "history"]
num_cols = [c for c in X.columns if c not in cat_cols]

from sklearn.preprocessing import OrdinalEncoder

# OrdinalEncoder supports unseen categories
ord_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[cat_cols] = ord_enc.fit_transform(X_train[cat_cols].astype(str))
X_test[cat_cols] = ord_enc.transform(X_test[cat_cols].astype(str))

joblib.dump(ord_enc, "saved_models/ordinal_encoder.pkl")
print("\nCategorical features encoded with OrdinalEncoder.")

# SCALE NUMERIC FEATURES
# CLEAN NUMERIC COLUMNS (Zeek uses "-" for missing data)
for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Replace NaN with 0
X_train[num_cols] = X_train[num_cols].fillna(0)
X_test[num_cols] = X_test[num_cols].fillna(0)

print("\nNumeric cleaning complete.")

# SCALE NUMERIC FEATURES
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, "saved_models/scaler.pkl")

print("\nNumeric scaling complete.")


# SAVE FINAL SPLITS
X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
pd.Series(y_train_enc).to_csv(f"{OUT_DIR}/y_train.csv", index=False)
pd.Series(y_test_enc).to_csv(f"{OUT_DIR}/y_test.csv", index=False)

print("\nSaved:")
print(" - X_train.csv")
print(" - X_test.csv")
print(" - y_train.csv")
print(" - y_test.csv")

print("\n=== PREPROCESSING COMPLETE ===")

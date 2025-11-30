import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib

INPUT_DIR = "data/intermediate"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# SCENARIO DEFINITIONS (FINAL)

TRAIN_SCENARIOS = [
    "CTU-IoT-Malware-Capture-3-1.csv",
    "CTU-IoT-Malware-Capture-8-1.csv",
    "CTU-IoT-Malware-Capture-20-1.csv",
    "CTU-IoT-Malware-Capture-21-1.csv"
]

TEST_SCENARIOS = [
    "Somfy-01.csv",
    "CTU-IoT-Malware-Capture-34-1.csv",
    "CTU-IoT-Malware-Capture-42-1.csv",
    "CTU-IoT-Malware-Capture-44-1.csv",
    "CTU-Honeypot-Capture-4-1.csv",
    "CTU-Honeypot-Capture-5-1.csv"
]


print("\n=== USING SCRIPT:", __file__, "===")
print("TRAIN SET:", TRAIN_SCENARIOS)
print("TEST SET:", TEST_SCENARIOS)

# LOAD + LABEL FIXING

def load_and_fix(files):
    dfs = []
    for fname in files:
        path = f"{INPUT_DIR}/{fname}"
        print("Loading:", path)
        df = pd.read_csv(path, low_memory=False)

        # Fix merged last column from Zeek logs
        last_col = df.columns[-1]
        if ("label" in last_col) or ("tunnel" in last_col):
            split_cols = df[last_col].astype(str).str.split(expand=True)

            while split_cols.shape[1] < 3:
                split_cols[split_cols.shape[1]] = np.nan

            split_cols = split_cols.iloc[:, :3]
            split_cols.columns = ["tunnel_parents", "label", "detailed_label"]

            df = df.drop(columns=[last_col])
            df = pd.concat([df, split_cols], axis=1)

        # prefer detailed label
        if "detailed_label" in df.columns:
            df["label"] = df["detailed_label"]

        # drop useless columns
        DROP = [
            "uid", "id.orig_h", "id.resp_h",
            "local_orig", "local_resp", "service",
            "tunnel_parents", "detailed_label"
        ]
        df = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


train_df = load_and_fix(TRAIN_SCENARIOS)
test_df  = load_and_fix(TEST_SCENARIOS)

print("\nAfter loading & fixing:")
print("Train:", train_df.shape)
print("Test:", test_df.shape)


# CLEAN LABELS — MALICIOUS FLAGGING

# remove missing labels
train_df = train_df[train_df["label"].notna()]
test_df  = test_df[test_df["label"].notna()]

# known malicious keywords (IoT-23 official)
MALICIOUS_KEYWORDS = [
    "okiru", "mirai", "gafgyt",
    "ddos", "attack", "malware",
    "scan", "botnet", "c&c", "c2",
    "trojan", "exploit", "infection",
    "unknown_malicious", "malicious"
]

def map_label(label):
    lbl = str(label).lower().strip()
    return "Malicious" if any(k in lbl for k in MALICIOUS_KEYWORDS) else "Benign"

train_df["label"] = train_df["label"].apply(map_label)
test_df["label"]  = test_df["label"].apply(map_label)

print("\nLabel distribution AFTER malicious-flag mapping:")
print("\nTrain:\n", train_df["label"].value_counts())
print("\nTest:\n", test_df["label"].value_counts())

y_train = train_df["label"]
y_test  = test_df["label"]

X_train = train_df.drop(columns=["label"])
X_test  = test_df.drop(columns=["label"])


# CATEGORICAL ENCODING

cat_cols = ["proto", "conn_state", "history"]
cat_cols = [c for c in cat_cols if c in X_train.columns]

print("\nCategorical columns:", cat_cols)

# Clean categorical junk
for col in cat_cols:
    X_train[col] = X_train[col].replace(['-', '', ' ', np.nan], 'unknown')
    X_test[col]  = X_test[col].replace(['-', '', ' ', np.nan], 'unknown')

X_train[cat_cols] = X_train[cat_cols].astype(str)
X_test[cat_cols]  = X_test[cat_cols].astype(str)

ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = ord_enc.fit_transform(X_train[cat_cols])
X_test[cat_cols]  = ord_enc.transform(X_test[cat_cols])

joblib.dump(ord_enc, "saved_models/ordinal_encoder_scenario.pkl")


# NUMERIC SCALING

num_cols = [c for c in X_train.columns if c not in cat_cols and c != "ts"]

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col]  = pd.to_numeric(X_test[col], errors="coerce")

X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

joblib.dump(scaler, "saved_models/scaler_scenario.pkl")


# SAVE SPLIT

X_train.to_csv(f"{OUTPUT_DIR}/X_train_scenario.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test_scenario.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train_scenario.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test_scenario.csv", index=False)

print("\n=== Scenario split completed ===")
print("Final Train:", X_train.shape)
print("Final Test:", X_test.shape)

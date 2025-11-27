import pandas as pd
import glob
import os

INPUT_DIR = "data/intermediate"
OUTPUT_FILE = "data/processed/final_dataset.csv"

os.makedirs("data/processed", exist_ok=True)

print("\n=== LOADING INTERMEDIATE CSV FILES ===")
files = glob.glob(f"{INPUT_DIR}/*.csv")
dfs = []

for f in files:
    print("Loading:", os.path.basename(f))
    dfs.append(pd.read_csv(f, low_memory=False))

df = pd.concat(dfs, ignore_index=True)
print("\nCombined raw shape:", df.shape)

# STEP 1 — IDENTIFY AND SPLIT THE ZEek-LABELED COLUMN SAFELY

# The labeled Zeek column is always the LAST column
last_col = df.columns[-1]

print(f"\nLast column detected as: {last_col}")

# Split it into as many pieces as available
split_cols = df[last_col].astype(str).str.split(expand=True)

# Zeek format usually:
# <tunnel> <label> <detailed_label>
# But we keep ONLY the middle column ("label") for ML

label_col = None

if split_cols.shape[1] >= 2:
    label_col = split_cols[1]
else:
    # fallback: entire column is the label
    label_col = split_cols[0]

df["label"] = label_col

# Drop the entire merged column
df = df.drop(columns=[last_col])

print("\nExtracted clean label column.")

# STEP 2 — DROP NON-USABLE COLUMNS

DROP_COLS = [
    "ts", "uid", "id.orig_h", "id.resp_h",
    "local_orig", "local_resp", "service",
    "tunnel_parents", "detailed_label"
]

existing = [c for c in DROP_COLS if c in df.columns]
df = df.drop(columns=existing, errors="ignore")

print("\nDropped unused fields:", existing)

# STEP 3 — DROP ROWS WITH NO LABEL

df = df[df["label"].notna()]
df = df[df["label"].astype(str).str.strip() != ""]

print("\nAfter dropping empty-label rows:", df.shape)

# STEP 4 — DROP ALL DUPLICATES BEFORE ANY PROCESSING

before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"\nDropped {before - after} duplicate rows.")
print("Final deduped shape:", df.shape)

# STEP 5 — SAVE CLEAN DATASET

df.to_csv(OUTPUT_FILE, index=False)
print("\nSaved CLEAN final_dataset.csv →", OUTPUT_FILE)
print("\n=== DONE ===")

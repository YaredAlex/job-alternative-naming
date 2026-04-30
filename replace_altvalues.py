import pandas as pd

# -----------------------------
# FILE PATHS
# -----------------------------
SOURCE_CSV = "D:\LMIS\job data\mapped_Tibarek_alternatives.csv"
REFERENCE_CSV = "unmatched_titles_4k.csv"
OUTPUT_CSV = "D:\LMIS\job data\mapped_Tibarek_alternatives.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df_source = pd.read_csv(SOURCE_CSV)
df_ref = pd.read_csv(REFERENCE_CSV)

# Column names
source_col = "informal work in eng"
match_col = "title_en"
replace_col = "alternative_position"

# -----------------------------
# CLEAN (avoid mismatch issues)
# -----------------------------
df_source[source_col] = df_source[source_col].astype(str).str.strip()
df_ref[match_col] = df_ref[match_col].astype(str).str.strip()

# -----------------------------
# CREATE MAPPING DICTIONARY
# -----------------------------
mapping = dict(zip(df_ref[match_col], df_ref[replace_col]))

# -----------------------------
# REPLACE VALUES (EXACT MATCH)
# -----------------------------
df_source[source_col] = df_source[source_col].map(mapping).fillna(df_source[source_col])

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df_source.to_csv(OUTPUT_CSV, index=False)

print("✅ Replacement completed. Saved to:", OUTPUT_CSV)
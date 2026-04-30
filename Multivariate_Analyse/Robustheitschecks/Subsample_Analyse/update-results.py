import pandas as pd
import numpy as np

# =========================
# 1. DATEIEN LADEN
# =========================
df_harmonized = pd.read_excel("Harmonized_Results.xlsx")
df_split = pd.read_excel("Ergebnisse_H3_split.xlsx")
print("Harmonized_Results Spalten:", df_harmonized.columns.tolist())
print("Ergebnisse_H3_split Spalten:", df_split.columns.tolist())
print(f"\nH3-Zeilen vorher: {df_harmonized['Hypothese'].str.startswith('H3').sum()}")

# =========================
# 2. SPLIT-ERGEBNISSE IN HARMONIZED-FORMAT UMWANDELN
# =========================
hyp_meta = {
    "H3a1": {
        "step1_var": "Protests_log_lag2",
        "step2_var": "Posts_log_lag1",
        "step1_dv":  "Posts",
        "step2_dv":  "Protests",
    },
    "H3a2": {
        "step1_var": "Protests_log_lag2",
        "step2_var": "Active_Accounts_log_lag1",
        "step1_dv":  "Active_Accounts",
        "step2_dv":  "Protests",
    },
    "H3a3": {
        "step1_var": "participants_log_lag2",
        "step2_var": "Posts_log_lag1",
        "step1_dv":  "Posts",
        "step2_dv":  "participants",
    },
    "H3b1": {
        "step1_var": "Posts_log_lag2",
        "step2_var": "Protests_log_lag1",
        "step1_dv":  "Protests",
        "step2_dv":  "Posts",
    },
    "H3b2": {
        "step1_var": "Active_Accounts_log_lag2",
        "step2_var": "Protests_log_lag1",
        "step1_dv":  "Protests",
        "step2_dv":  "Active_Accounts",
    },
    "H3b3": {
        "step1_var": "Posts_log_lag2",
        "step2_var": "participants_log_lag1",
        "step1_dv":  "participants",
        "step2_dv":  "Posts",
    },
}

new_rows = []
for _, row in df_split.iterrows():
    label = row["label"]  # z.B. "FirstHalf-Daily-H3a1"

    # Half extrahieren
    if label.startswith("FirstHalf"):
        half = "FirstHalf"
    elif label.startswith("SecondHalf"):
        half = "SecondHalf"
    else:
        continue

    # Aggregationsebene extrahieren
    for agg in ["Daily", "Weekly", "Monthly"]:
        if f"-{agg}-" in label:
            aggregation = agg
            break
    else:
        continue

    # Hypothese extrahieren
    for hyp in hyp_meta.keys():
        if hyp in label:
            hypothese = hyp
            break
    else:
        continue

    meta = hyp_meta[hypothese]

    # Step1-Eintrag
    new_rows.append({
        "Hypothese":  hypothese,
        "variable":   meta["step1_var"],
        "coef":       np.nan,
        "std_err":    np.nan,
        "model":      f"{half}-{hypothese}-{aggregation}-Step1",
        "dv":         meta["step1_dv"],
        "effect_pct": row["step1"],
        "ci_low":     row["step1_low"],
        "ci_high":    row["step1_high"],
    })

    # Step2-Eintrag
    new_rows.append({
        "Hypothese":  hypothese,
        "variable":   meta["step2_var"],
        "coef":       np.nan,
        "std_err":    np.nan,
        "model":      f"{half}-{hypothese}-{aggregation}-Step2",
        "dv":         meta["step2_dv"],
        "effect_pct": row["step2"],
        "ci_low":     row["step2_low"],
        "ci_high":    row["step2_high"],
    })

df_new_h3 = pd.DataFrame(new_rows)

# =========================
# 3. ALTE H3-ZEILEN ENTFERNEN & NEUE EINFÜGEN
# =========================
df_non_h3 = df_harmonized[~df_harmonized["Hypothese"].str.startswith("H3")].copy()
df_result = pd.concat([df_non_h3, df_new_h3], ignore_index=True)
print(f"H3-Zeilen nachher: {df_result['Hypothese'].str.startswith('H3').sum()}")
print(f"Gesamtzeilen: {len(df_result)}")

# =========================
# 4. SPEICHERN
# =========================
output_file = "Harmonized_Results_updated.xlsx"
df_result.to_excel(output_file, index=False)
print(f"\nFertig! Gespeichert als '{output_file}'")
print("\nVorschau neue H3-Zeilen:")
print(df_result[df_result["Hypothese"].str.startswith("H3")].head(10).to_string())

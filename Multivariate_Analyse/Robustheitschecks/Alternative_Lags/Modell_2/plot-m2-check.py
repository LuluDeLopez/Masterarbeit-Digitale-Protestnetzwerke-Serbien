import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Excel-Dateien laden
# =========================
df_short = pd.read_excel("Model_Results_M2.xlsx")
df_middle = pd.read_excel("Model_Results_M2_Lag2.xlsx")
df_long = pd.read_excel("Model_Results_M2_Lag3.xlsx")

# =========================
# 2. Lag-Spalte hinzufügen
# =========================
df_short["Lag"] = "Short (1 Lag)"
df_middle["Lag"] = "Middle (2 Lags)"
df_long["Lag"] = "Long (3 Lags)"

# =========================
# 3. Alle zusammenführen
# =========================
df_all = pd.concat([df_short, df_middle, df_long], ignore_index=True)

# =========================
# 4. Label erstellen (falls nicht vorhanden)
# =========================
if "label" not in df_all.columns:
    # Kombiniere Modellname (falls vorhanden) und DV (falls vorhanden)
    cols = []
    if "model" in df_all.columns:
        cols.append(df_all["model"])
    if "dv" in df_all.columns:
        cols.append(df_all["dv"])
    df_all["label"] = pd.Series([" | ".join(map(str, t)) for t in zip(*cols)]) if cols else pd.Series([f"Row {i+1}" for i in range(len(df_all))])

# =========================
# 5. Kategorien definieren
# =========================
lag_order = ["Short (1 Lag)", "Middle (2 Lags)", "Long (3 Lags)"]
df_all["Lag"] = pd.Categorical(df_all["Lag"], categories=lag_order, ordered=True)

# =========================
# 6. Signifikanz markieren
# =========================
df_all["sig"] = np.where(
    ("ci_low" in df_all.columns) & ("ci_high" in df_all.columns) & ((df_all["ci_low"] > 0) | (df_all["ci_high"] < 0)),
    "*",
    ""
)

# =========================
# 7. Effekt formatieren
# =========================
def format_effect(row):
    if "effect_pct" in row:
        val = row["effect_pct"]
        ci_low = row["ci_low"] if "ci_low" in row else np.nan
        ci_high = row["ci_high"] if "ci_high" in row else np.nan
        sig = row["sig"] if "sig" in row else ""
        return f"{val:.2f}{sig} ({ci_low:.2f}, {ci_high:.2f})"
    else:
        return ""

df_all["Effekt (%)"] = df_all.apply(format_effect, axis=1)

# =========================
# 8. Tabelle vorbereiten
# =========================
table_data = df_all[["label", "Lag", "Effekt (%)"]].values
column_labels = ["Modell", "Lag", "Effekt in % (95% CI)"]

# =========================
# 9. Farben für Signifikanz
# =========================
cell_colors = []
for idx, row in df_all.iterrows():
    if row["sig"] == "*":
        color = [0.9, 1.0, 0.9] if "effect_pct" in row and row["effect_pct"] > 0 else [1.0, 0.8, 0.8]
    else:
        color = [1, 1, 1]
    cell_colors.append([color]*3)

# =========================
# 10. Plot erstellen
# =========================
fig, ax = plt.subplots(figsize=(16, max(2, 0.5*len(df_all))))
ax.axis('off')

table = ax.table(
    cellText=table_data,
    colLabels=column_labels,
    cellLoc='center',
    colLoc='center',
    loc='center',
    cellColours=cell_colors
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Spaltenbreite anpassen
for key, cell in table.get_celld().items():
    row, col = key
    if col == 0:
        cell.set_width(0.5)
    elif col == 1:
        cell.set_width(0.2)
    else:
        cell.set_width(0.3)

plt.tight_layout()
plt.savefig("results_table_M2_combined.png", dpi=300)
plt.show()

print("Tabelle gespeichert als: results_table_M2_combined.png")

# =========================
# 11. Optional: Excel speichern
# =========================
df_all.to_excel("results_M2_combined.xlsx", index=False)
print("Ergebnisse auch als Excel gespeichert: results_M2_combined.xlsx")

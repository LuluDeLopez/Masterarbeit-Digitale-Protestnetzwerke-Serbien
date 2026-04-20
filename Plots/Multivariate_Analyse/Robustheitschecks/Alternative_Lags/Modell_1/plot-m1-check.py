import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Excel-Dateien laden
# =========================
df_short = pd.read_excel("Model_Results.xlsx")
df_middle = pd.read_excel("Model_Results_Lag2.xlsx")
df_long = pd.read_excel("Model_Results_Lag3.xlsx")

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
# 4. Lesbare Labels erstellen
# =========================
df_all["Variable"] = df_all["variable"].replace({
    "Protests_lag1": "Protests",
    "Protests_lag2": "Protests",
    "Protests_lag3": "Protests",
    "participants_lag1": "Participants",
    "participants_lag2": "Participants",
    "participants_lag3": "Participants"
})

df_all["DV"] = df_all["dv"]
df_all["Aggregation"] = df_all["model"]

# Kombiniertes Label (ähnlich deiner alten "Hypothese")
df_all["label"] = df_all["Aggregation"] + " | " + df_all["DV"] + " | " + df_all["Variable"]

# =========================
# 5. Kategorien definieren
# =========================
lag_order = ["Short (1 Lag)", "Middle (2 Lags)", "Long (3 Lags)"]
agg_order = ["Daily", "Weekly", "Monthly"]

df_all["Lag"] = pd.Categorical(df_all["Lag"], categories=lag_order, ordered=True)
df_all["Aggregation"] = pd.Categorical(df_all["Aggregation"], categories=agg_order, ordered=True)

# =========================
# 6. Sortieren
# =========================
df_all = df_all.sort_values(by=["Lag", "Aggregation", "DV", "Variable"]).reset_index(drop=True)

# =========================
# 7. Signifikanz markieren (CI basiert)
# =========================
df_all["sig"] = np.where(
    (df_all["ci_low"] > 0) | (df_all["ci_high"] < 0),
    "*",
    ""
)

# =========================
# 8. Formatierte Effekt-Spalte
# =========================
df_all["Effekt (%)"] = df_all.apply(
    lambda x: f"{x['effect_pct']:.2f}{x['sig']} ({x['ci_low']:.2f}, {x['ci_high']:.2f})",
    axis=1
)

# =========================
# 9. Tabelle vorbereiten
# =========================
table_data = df_all[["label", "Lag", "Effekt (%)"]].values
column_labels = ["Modell", "Lag", "Effekt in % (95% CI)"]

# =========================
# 10. Farben (Signifikanz)
# =========================
cell_colors = []
for idx, row in df_all.iterrows():
    if row["sig"] == "*":
        color = [0.9, 1.0, 0.9] if row["effect_pct"] > 0 else [1.0, 0.8, 0.8]
    else:
        color = [1, 1, 1]
    cell_colors.append([color]*3)

# =========================
# 11. Plot
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
plt.savefig("results_table_nb_models.png", dpi=300)
plt.show()

print("Tabelle gespeichert als: results_table_nb_models.png")

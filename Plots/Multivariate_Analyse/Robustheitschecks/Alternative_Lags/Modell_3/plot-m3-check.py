import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# =========================
# 1. Excel-Dateien laden
# =========================
df_short = pd.read_excel("Ergebnisse_H3.xlsx")
df_middle = pd.read_excel("Ergebnisse_H3_middle.xlsx")
df_long = pd.read_excel("Ergebnisse_H3_3lags.xlsx")

# =========================
# 2. Lag-Spalte hinzufügen
# =========================
df_short["Lag"] = "Short (1 Lag)"
df_middle["Lag"] = "Middle (2 Lags)"
df_long["Lag"] = "Long (3 Lags)"

# =========================
# 3. Einheitliche Hypothesen-Titel aus Short übernehmen
# =========================
hyp_mapping = dict(zip(df_short["label"], df_short["label"]))
for df in [df_middle, df_long]:
    df["label"] = df["label"].replace(hyp_mapping)

# =========================
# 4. Alle zusammenführen
# =========================
df_all = pd.concat([df_short, df_middle, df_long], ignore_index=True)

# =========================
# 5. Aggregationsebene extrahieren (Daily, Weekly, Monthly)
# =========================
df_all["Aggregation"] = df_all["label"].str.extract(r'^(Daily|Weekly|Monthly)')[0]

# =========================
# 6. Hypothesen-Reihenfolge extrahieren (H3a1, H3a2, H3b1, ...)
def extract_hyp_order(label):
    match = re.search(r'H3[ab]\d+', label)
    return match.group(0) if match else "zzz"

df_all["hyp_order"] = df_all["label"].apply(extract_hyp_order)

# =========================
# 7. Kategorien für Sortierung definieren
# =========================
lag_order = ["Short (1 Lag)", "Middle (2 Lags)", "Long (3 Lags)"]
agg_order = ["Daily", "Weekly", "Monthly"]

df_all["Lag"] = pd.Categorical(df_all["Lag"], categories=lag_order, ordered=True)
df_all["Aggregation"] = pd.Categorical(df_all["Aggregation"], categories=agg_order, ordered=True)

# =========================
# 8. Sortieren: Lag → Aggregation → Hypothese
# =========================
df_all = df_all.sort_values(by=["Lag", "Aggregation", "hyp_order"]).reset_index(drop=True)

# =========================
# 9. Signifikanz markieren
# =========================
df_all["step1_sig"] = np.where((df_all["step1_ci_low"] > 0) | (df_all["step1_ci_high"] < 0), "*", "")
df_all["step2_sig"] = np.where((df_all["step2_ci_low"] > 0) | (df_all["step2_ci_high"] < 0), "*", "")
df_all["indirect_sig"] = np.where((df_all["indirect_ci_low"] > 0) | (df_all["indirect_ci_high"] < 0), "*", "")

# =========================
# 10. Formatierte Spalten erstellen
# =========================
df_all["Step1 Effekt"] = df_all.apply(lambda x: f"{x['step1_effect']:.2f}{x['step1_sig']}", axis=1)
df_all["Step2 Effekt"] = df_all.apply(lambda x: f"{x['step2_effect']:.2f}{x['step2_sig']}", axis=1)
df_all["Indirekter Effekt (CI)"] = df_all.apply(
    lambda x: f"{x['indirect_effect']:.2f}{x['indirect_sig']} ({x['indirect_ci_low']:.2f}, {x['indirect_ci_high']:.2f})",
    axis=1
)

# =========================
# 11. Tabelle plotten mit angepasster Hypothesen-Spalte
# =========================
table_data = df_all[["label", "Lag", "Step1 Effekt", "Step2 Effekt", "Indirekter Effekt (CI)"]].values
column_labels = ["Hypothese", "Lag", "Step1 Effekt", "Step2 Effekt", "Indirekter Effekt (CI)"]

cell_colors = []
for idx, row in df_all.iterrows():
    if row["indirect_sig"] == "*":
        color = [0.9, 1.0, 0.9] if row["indirect_effect"] > 0 else [1.0, 0.8, 0.8]
    else:
        color = [1, 1, 1]
    cell_colors.append([color]*5)

fig, ax = plt.subplots(figsize=(18, max(2, 0.5*len(df_all))))
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
table.scale(1, 1.5)  # Höhe skalieren, Breite lassen wir normal

# =========================
# Nur erste Spalte verbreitern
# =========================
for key, cell in table.get_celld().items():
    row, col = key
    if col == 0:  # nur Hypothesen-Spalte
        cell.set_width(0.30)  # hier anpassen, z.B. 0.45 → 45% der Gesamtbreite
    else:
        cell.set_width(0.13)  # andere Spalten etwas kleiner

plt.tight_layout()
plt.savefig("results_table_hypotheses_fullwidth.png", dpi=300)
plt.show()
print("Tabelle gespeichert als: results_table_hypotheses_fullwidth.png")
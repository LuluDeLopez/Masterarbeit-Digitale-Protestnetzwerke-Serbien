import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# 3. Labels vereinheitlichen
# Mittlere und lange Labels auf das Short-Format bringen,
# indem der Präfix (z.B. "Middle-") entfernt wird.
# =========================
def clean_label(label):
    return re.sub(r'^(Daily|Weekly|Monthly)-(Middle-|Long-)?', r'\1-', label)

for df in [df_short, df_middle, df_long]:
    df["label_clean"] = df["label"].apply(clean_label)

# =========================
# 4. Alle zusammenführen
# =========================
df_all = pd.concat([df_short, df_middle, df_long], ignore_index=True)

# =========================
# 5. Aggregationsebene & Hypothese extrahieren
# =========================
df_all["Aggregation"] = df_all["label_clean"].str.extract(r'^(Daily|Weekly|Monthly)')[0]

def extract_hyp(label):
    match = re.search(r'(H3[ab]\d+.*)', label)
    return match.group(1) if match else label

df_all["Hypothese"] = df_all["label_clean"].apply(extract_hyp)

def extract_hyp_order(label):
    match = re.search(r'H3[ab]\d+', label)
    return match.group(0) if match else "zzz"

df_all["hyp_order"] = df_all["label_clean"].apply(extract_hyp_order)

# =========================
# 6. Kategorien & Sortierung
# =========================
lag_order = ["Short (1 Lag)", "Middle (2 Lags)", "Long (3 Lags)"]
agg_order = ["Daily", "Weekly", "Monthly"]

df_all["Lag"] = pd.Categorical(df_all["Lag"], categories=lag_order, ordered=True)
df_all["Aggregation"] = pd.Categorical(df_all["Aggregation"], categories=agg_order, ordered=True)
df_all = df_all.sort_values(by=["Aggregation", "hyp_order", "Lag"]).reset_index(drop=True)

# =========================
# 7. Signifikanz markieren
# =========================
df_all["step1_sig"] = np.where((df_all["step1_ci_low"] > 0) | (df_all["step1_ci_high"] < 0), "*", "")
df_all["step2_sig"] = np.where((df_all["step2_ci_low"] > 0) | (df_all["step2_ci_high"] < 0), "*", "")
df_all["indirect_sig"] = np.where((df_all["indirect_ci_low"] > 0) | (df_all["indirect_ci_high"] < 0), "*", "")

# =========================
# 8. PLOT A: Übersichtstabelle (alle Hypothesen × Lags)
# =========================
df_all["Step1 Effekt"] = df_all.apply(lambda x: f"{x['step1_effect']:.2f}{x['step1_sig']}", axis=1)
df_all["Step2 Effekt"] = df_all.apply(lambda x: f"{x['step2_effect']:.2f}{x['step2_sig']}", axis=1)
df_all["Indirekter Effekt (CI)"] = df_all.apply(
    lambda x: f"{x['indirect_effect']:.2f}{x['indirect_sig']} ({x['indirect_ci_low']:.2f}, {x['indirect_ci_high']:.2f})",
    axis=1
)

table_data = df_all[["Hypothese", "Aggregation", "Lag", "Step1 Effekt", "Step2 Effekt", "Indirekter Effekt (CI)"]].values
column_labels = ["Hypothese", "Aggregation", "Lag", "Step1 Effekt (%)", "Step2 Effekt (%)", "Indirekter Effekt (CI, %)"]

cell_colors = []
for _, row in df_all.iterrows():
    if row["indirect_sig"] == "*":
        color = [0.85, 1.0, 0.85] if row["indirect_effect"] > 0 else [1.0, 0.8, 0.8]
    else:
        color = [1, 1, 1]
    cell_colors.append([color] * len(column_labels))

fig, ax = plt.subplots(figsize=(20, max(3, 0.45 * len(df_all))))
ax.axis("off")
table = ax.table(
    cellText=table_data,
    colLabels=column_labels,
    cellLoc="center",
    loc="center",
    cellColours=cell_colors
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

col_widths = [0.28, 0.08, 0.12, 0.10, 0.10, 0.22]
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_widths[col])

plt.title("H3 – Übersicht aller sequenziellen Rückkopplungseffekte", fontsize=13, weight="bold", pad=12)
plt.tight_layout()
plt.savefig("H3_tabelle_gesamt.png", dpi=300, bbox_inches="tight")
plt.show()
print("Tabelle gespeichert: H3_tabelle_gesamt.png")

# =========================
# 9. PLOT B: Forest Plot – Indirekter Effekt je Hypothese & Lag
# Aufgeteilt nach Aggregationsebene (3 Subplots)
# =========================
hypothesen = df_all["Hypothese"].unique()
lag_colors = {
    "Short (1 Lag)": "tab:blue",
    "Middle (2 Lags)": "tab:orange",
    "Long (3 Lags)": "tab:green"
}
lag_offsets = {"Short (1 Lag)": -0.2, "Middle (2 Lags)": 0.0, "Long (3 Lags)": 0.2}

fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.6 * len(hypothesen))), sharey=True)

for ax, agg in zip(axes, agg_order):
    df_agg = df_all[df_all["Aggregation"] == agg]
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    for i, hyp in enumerate(hypothesen):
        for lag, color in lag_colors.items():
            row = df_agg[(df_agg["Hypothese"] == hyp) & (df_agg["Lag"] == lag)]
            if row.empty:
                continue
            row = row.iloc[0]
            y = i + lag_offsets[lag]
            ax.errorbar(
                row["indirect_effect"], y,
                xerr=[[row["indirect_effect"] - row["indirect_ci_low"]],
                      [row["indirect_ci_high"] - row["indirect_effect"]]],
                fmt="o", color=color, capsize=4, markersize=6,
                label=lag if i == 0 else ""
            )

    ax.set_yticks(range(len(hypothesen)))
    ax.set_yticklabels(hypothesen, fontsize=8)
    ax.set_title(agg, fontsize=11, weight="bold")
    ax.set_xlabel("Indirekter Effekt (%)", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

axes[0].legend(title="Lag-Struktur", fontsize=8, title_fontsize=9, loc="lower left")
fig.suptitle("H3 – Forest Plot: Indirekte Effekte nach Aggregationsebene", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("H3_forestplot_indirekt.png", dpi=300, bbox_inches="tight")
plt.show()
print("Forest Plot gespeichert: H3_forestplot_indirekt.png")

# =========================
# 10. PLOT C: Heatmap – Signifikante indirekte Effekte
# Zeilen: Hypothesen, Spalten: Aggregation × Lag
# =========================
df_all["Agg_Lag"] = df_all["Aggregation"].astype(str) + " / " + df_all["Lag"].astype(str)
pivot_effect = df_all.pivot_table(index="Hypothese", columns="Agg_Lag", values="indirect_effect", aggfunc="first")
pivot_sig = df_all.pivot_table(index="Hypothese", columns="Agg_Lag", values="indirect_sig", aggfunc="first")

# Spalten sortieren
col_sort = [f"{a} / {l}" for a in agg_order for l in lag_order]
col_sort = [c for c in col_sort if c in pivot_effect.columns]
pivot_effect = pivot_effect[col_sort]
pivot_sig = pivot_sig[col_sort]

fig, ax = plt.subplots(figsize=(max(10, len(col_sort) * 1.2), max(3, len(pivot_effect) * 0.6)))
vmax = max(abs(pivot_effect.values[~np.isnan(pivot_effect.values)])) if pivot_effect.notna().any().any() else 1
im = ax.imshow(pivot_effect.values.astype(float), cmap="RdYlGn", aspect="auto",
               vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(col_sort)))
ax.set_xticklabels(col_sort, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot_effect.index)))
ax.set_yticklabels(pivot_effect.index, fontsize=9)

# Werte und Sternchen in Zellen
for i in range(pivot_effect.shape[0]):
    for j in range(pivot_effect.shape[1]):
        val = pivot_effect.values[i, j]
        sig = pivot_sig.values[i, j]
        if not np.isnan(float(val)) if val is not None else False:
            ax.text(j, i, f"{float(val):.1f}{sig}", ha="center", va="center", fontsize=7.5)

plt.colorbar(im, ax=ax, label="Indirekter Effekt (%)")
ax.set_title("H3 – Heatmap: Indirekte Effekte (grün = positiv, rot = negativ, * = sig.)",
             fontsize=11, weight="bold")
plt.tight_layout()
plt.savefig("H3_heatmap_indirekt.png", dpi=300, bbox_inches="tight")
plt.show()
print("Heatmap gespeichert: H3_heatmap_indirekt.png")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. DATEN LADEN
# =========================
results = pd.read_excel("Harmonized_Results_updated.xlsx")

# =========================
# 2. SIGNIFIKANZ BESTIMMEN
# ci_low und ci_high sind bereits in der Datei vorberechnet
# =========================
results['signif'] = results.apply(
    lambda row: '*' if (row['ci_low'] > 0 or row['ci_high'] < 0) else '',
    axis=1
)

# =========================
# 3. EFFEKT-TEXT ERSTELLEN
# =========================
results['effect_text'] = results.apply(
    lambda row: f"{row['effect_pct']:.1f}% [{row['ci_low']:.1f}%, {row['ci_high']:.1f}%]{row['signif']}",
    axis=1
)

# =========================
# 4. EBENE & HYPOTHESE EXTRAHIEREN
# =========================
results['Level'] = results['model'].str.extract(r'(Daily|Weekly|Monthly)')[0].str.strip()
results['Hyp'] = results['Hypothese'].str.strip()

# =========================
# 5. SPLIT: ERSTE / ZWEITE HÄLFTE
# =========================
first = results[results['model'].str.contains("FirstHalf")].copy()
second = results[results['model'].str.contains("SecondHalf")].copy()

# =========================
# 6. AGGREGATION PRO HYPOTHESE + LEVEL
# Bei H3 gibt es Step1 und Step2 pro Hypothese/Level –
# wir nehmen den Eintrag mit dem größten absoluten Effekt
# (alternativ: mean, je nach Präferenz)
# =========================
def aggregate_effects(df):
    df = df.copy()
    df['abs_effect'] = df['effect_pct'].abs()
    df_agg = (
        df.sort_values('abs_effect', ascending=False)
          .groupby(['Hyp', 'Level'])
          .first()
          .reset_index()
    )[['Hyp', 'Level', 'effect_text', 'effect_pct', 'signif']]
    return df_agg

first_agg = aggregate_effects(first)
second_agg = aggregate_effects(second)

# =========================
# 7. MERGE FIRST/SECOND HALF
# =========================
df_plot = pd.merge(
    first_agg, second_agg,
    on=['Hyp', 'Level'],
    suffixes=('_FirstHalf', '_SecondHalf'),
    how='outer'
)

# =========================
# 8. SORTIERUNG
# =========================
hyp_order = ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c',
             'H3a1', 'H3a2', 'H3a3', 'H3b1', 'H3b2', 'H3b3']
level_order = ['Daily', 'Weekly', 'Monthly']

df_plot['Hyp_sort'] = df_plot['Hyp'].apply(
    lambda x: hyp_order.index(x) if x in hyp_order else len(hyp_order)
)
df_plot['Level_sort'] = df_plot['Level'].apply(
    lambda x: level_order.index(x) if x in level_order else len(level_order)
)
df_plot = df_plot.sort_values(['Level_sort', 'Hyp_sort']).reset_index(drop=True)
df_plot['Hypothese_Level'] = df_plot['Hyp'] + ' (' + df_plot['Level'] + ')'

# =========================
# 9. FARBMATRIX
# grün = signifikant positiv, rot = signifikant negativ, weiß = nicht signifikant
# =========================
def cell_color(signif, effect_pct):
    if signif == '*':
        return '#a6d96a' if effect_pct > 0 else '#f46d43'
    return '#ffffff'

colors = []
for _, row in df_plot.iterrows():
    colors.append([
        '#f0f0f0',  # Hypothese-Spalte grau hinterlegen
        cell_color(row.get('signif_FirstHalf', ''), row.get('effect_pct_FirstHalf', 0)),
        cell_color(row.get('signif_SecondHalf', ''), row.get('effect_pct_SecondHalf', 0)),
    ])

# =========================
# 10. TABELLE PLOTTEN
# =========================
table_data = df_plot[['Hypothese_Level', 'effect_text_FirstHalf', 'effect_text_SecondHalf']].fillna('—').values

fig, ax = plt.subplots(figsize=(14, max(3, len(df_plot) * 0.45)))
ax.axis('off')

the_table = ax.table(
    cellText=table_data,
    colLabels=['Unterhypothese', 'Erste Hälfte', 'Zweite Hälfte'],
    cellColours=colors,
    cellLoc='center',
    loc='center'
)

the_table.auto_set_font_size(False)
the_table.set_fontsize(9)

# Spaltenbreiten anpassen
col_widths = [0.25, 0.37, 0.37]
for (row_idx, col_idx), cell in the_table.get_celld().items():
    cell.set_width(col_widths[col_idx])
    cell.set_height(0.055)
    if row_idx == 0:
        cell.set_text_props(weight='bold')

plt.tight_layout()
plt.savefig("Effekte_Hypothesen.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Tabelle gespeichert als 'Effekte_Hypothesen.png'")

# =========================
# 11. FOREST PLOT – Effekte mit KI
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(df_plot) * 0.35)), sharey=True)

for ax, level in zip(axes, level_order):
    df_level = df_plot[df_plot['Level'] == level].copy()
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

    for i, (_, row) in enumerate(df_level.iterrows()):
        # Erste Hälfte
        eff_f = row.get('effect_pct_FirstHalf', np.nan)
        # ci_low/ci_high nicht direkt in df_plot – aus results holen
        mask_f = (results['Hyp'] == row['Hyp']) & \
                 (results['Level'] == level) & \
                 results['model'].str.contains("FirstHalf")
        r_f = results[mask_f].sort_values('effect_pct', key=abs, ascending=False)
        if not r_f.empty and not np.isnan(eff_f):
            r_f = r_f.iloc[0]
            ax.errorbar(
                eff_f, i + 0.15,
                xerr=[[eff_f - r_f['ci_low']], [r_f['ci_high'] - eff_f]],
                fmt='o', color='tab:blue', capsize=3, markersize=5,
                label='Erste Hälfte' if i == 0 else ''
            )

        # Zweite Hälfte
        eff_s = row.get('effect_pct_SecondHalf', np.nan)
        mask_s = (results['Hyp'] == row['Hyp']) & \
                 (results['Level'] == level) & \
                 results['model'].str.contains("SecondHalf")
        r_s = results[mask_s].sort_values('effect_pct', key=abs, ascending=False)
        if not r_s.empty and not np.isnan(eff_s):
            r_s = r_s.iloc[0]
            ax.errorbar(
                eff_s, i - 0.15,
                xerr=[[eff_s - r_s['ci_low']], [r_s['ci_high'] - eff_s]],
                fmt='s', color='tab:orange', capsize=3, markersize=5,
                label='Zweite Hälfte' if i == 0 else ''
            )

    ax.set_yticks(range(len(df_level)))
    ax.set_yticklabels(df_level['Hyp'].values, fontsize=8)
    ax.set_title(level, fontsize=11, weight='bold')
    ax.set_xlabel('Effekt (%)', fontsize=9)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.legend(fontsize=8)

fig.suptitle("Forest Plot: Effekte nach Hypothese und Aggregationsebene", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig("Forest_Plot_Effekte.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Forest Plot gespeichert als 'Forest_Plot_Effekte.png'")

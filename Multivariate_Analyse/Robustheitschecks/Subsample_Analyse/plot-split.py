import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Daten laden
results = pd.read_excel("Harmonized_Results_Neu.xlsx")



# =========================
# CI für alle Hypothesen berechnen
# =========================

# H1/H2: direkte Effekte
mask_direct = results['coef'].notna()
results.loc[mask_direct, 'ci_low'] = (np.exp(results.loc[mask_direct, 'coef'] - 1.96 * results.loc[mask_direct, 'std_err']) - 1) * 100
results.loc[mask_direct, 'ci_high'] = (np.exp(results.loc[mask_direct, 'coef'] + 1.96 * results.loc[mask_direct, 'std_err']) - 1) * 100

# H3: indirekte Effekte
mask_indirect = results['indirect_coef'].notna()
results.loc[mask_indirect, 'ci_low'] = (np.exp(results.loc[mask_indirect, 'indirect_coef'] - 1.96 * results.loc[mask_indirect, 'indirect_se']) - 1) * 100
results.loc[mask_indirect, 'ci_high'] = (np.exp(results.loc[mask_indirect, 'indirect_coef'] + 1.96 * results.loc[mask_indirect, 'indirect_se']) - 1) * 100


# 2. Prozent-Effekt berechnen
def compute_effect(row):
    if pd.notnull(row.get('indirect_pct')):
        return row['indirect_pct']
    elif pd.notnull(row.get('coef')):
        return (np.exp(row['coef']) - 1) * 100
    else:
        return 0

results['effect_pct'] = results.apply(compute_effect, axis=1)

# 3. Signifikanz bestimmen
def compute_signif(row):
    if 'H3' in row['Hypothese']:
        return '*' if row['ci_low'] > 0 or row['ci_high'] < 0 else ''
    else:
        return '*' if row['ci_low'] > 0 or row['ci_high'] < 0 else ''

results['signif'] = results.apply(compute_signif, axis=1)

# 4. Effekt-Text erstellen
results['effect_text'] = results.apply(
    lambda row: f"{row['effect_pct']:.1f}% [{row['ci_low']:.1f}%, {row['ci_high']:.1f}%]{row['signif']}", axis=1
)

# 5. Ebene extrahieren
results['Level'] = results['model'].str.extract(r'(Daily|Weekly|Monthly)')[0].str.strip()
results['Hyp'] = results['Hypothese'].str.strip()

# 6. Split: erste / zweite Hälfte
first = results[results['model'].str.contains("FirstHalf")]
second = results[results['model'].str.contains("SecondHalf")]

# 7. Aggregation pro Unterhypothese + Level
def aggregate_effects(df):
    df_agg = df.groupby(['Hyp','Level']).agg({
        'effect_text':' | '.join,
        'effect_pct':'mean',
        'signif':'first'
    }).reset_index()
    return df_agg

first_agg = aggregate_effects(first)
second_agg = aggregate_effects(second)

# 8. Merge First/Second Half
df_plot = pd.merge(
    first_agg, second_agg, on=['Hyp','Level'], suffixes=('_FirstHalf','_SecondHalf'), how='outer'
)

# 9. Reihenfolge definieren
hyp_order = ['H1a','H1b','H1c','H2a','H2b','H2c','H3a1','H3a2','H3a3','H3b1','H3b2','H3b3']
level_order = ['Daily','Weekly','Monthly']

df_plot['Hyp_sort'] = df_plot['Hyp'].apply(lambda x: hyp_order.index(x) if x in hyp_order else len(hyp_order))
df_plot['Level_sort'] = df_plot['Level'].apply(lambda x: level_order.index(x) if x in level_order else len(level_order))
df_plot = df_plot.sort_values(['Level_sort','Hyp_sort']).reset_index(drop=True)
df_plot['Hypothese_Level'] = df_plot['Hyp'] + ' ' + df_plot['Level']

# 10. Farbmatrix erstellen
colors = []
for _, row in df_plot.iterrows():
    color_row = ['#ffffff']  # Hypothese immer weiß
    # FirstHalf
    if row.get('signif_FirstHalf','')=='*':
        color_row.append('#a6d96a' if row.get('effect_pct_FirstHalf',0)>0 else '#f46d43')
    else:
        color_row.append('#ffffff')
    # SecondHalf
    if row.get('signif_SecondHalf','')=='*':
        color_row.append('#a6d96a' if row.get('effect_pct_SecondHalf',0)>0 else '#f46d43')
    else:
        color_row.append('#ffffff')
    colors.append(color_row)

# 11. Tabelle plotten
fig, ax = plt.subplots(figsize=(12, max(2, len(df_plot)*0.5)))
ax.axis('off')
ax.axis('tight')

table_data = df_plot[['Hypothese_Level','effect_text_FirstHalf','effect_text_SecondHalf']].fillna('').values

the_table = ax.table(
    cellText=table_data,
    colLabels=['Unterhypothese','First Half','Second Half'],
    cellColours=colors,
    cellLoc='center',
    loc='center'
)

the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.auto_set_column_width([0,1,2])

plt.title("Effekte pro Unterhypothese und Aggregationsebene\n(nur signifikante Effekte eingefärbt)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("Effekte_Hypothesen_RichtigSortiert.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Tabelle als 'Effekte_Hypothesen_RichtigSortiert.png' gespeichert.")
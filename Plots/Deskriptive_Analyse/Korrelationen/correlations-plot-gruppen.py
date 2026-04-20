import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------
# 1. Daten laden
# ---------------------------
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_merged")

# Spaltennamen bereinigen
df.columns = df.columns.str.strip()

# ---------------------------
# 2. Orte ausschließen
# ---------------------------
exclude_orte = [
    "USA", "Deutschland", "England", "Österreich", "Belgien",
    "Frankreich", "Italien", "Kanada", "Luxemburg",
    "Schweiz", "Spanien", "Zypern", 0
]
df = df[~df["Ort"].isin(exclude_orte)]

# ---------------------------
# 3. Teilnehmerzahl berechnen
# ---------------------------
df["Teilnehmerzahl"] = df["Crowd Durchschnitt  Pro"].fillna(0) + df["Crowd Durchschnitt Anti"].fillna(0)

# ---------------------------
# 4. Accounts pro Gruppe berechnen
# ---------------------------
groups = ["student", "schueler", "zbor", "none"]
for g in groups:
    df[f"Accounts_{g}"] = df["n_accounts"] * df[f"share_{g}"]



# ---------------------------
# 4b. Posts pro Gruppe aus Instas
# ---------------------------
instas = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spalten bereinigen
instas.columns = instas.columns.str.strip()

# Orte bereinigen (wichtig für Merge!)
df["Ort_clean"] = df["Ort"].str.strip().str.lower()
instas["ORT_clean"] = instas["ORT"].str.strip().str.lower()

# Posts bestimmen: Zeitraum oder Gesamt
instas["Posts_final"] = instas["POSTS ZEITRAUM"].replace(0, np.nan).fillna(instas["POSTS GESAMT"])

# Gruppen definieren (0=none, 1=student, 2=schueler, 3=zbor)
group_map = {
    0: "none",
    1: "student",
    2: "schueler",
    3: "zbor"
}

instas["Gruppe"] = instas["Gruppe"].map(group_map)

# Nur gültige Gruppen behalten
instas = instas[instas["Gruppe"].notna()]

# Aggregation: Posts pro Ort & Gruppe
posts_agg = instas.groupby(["ORT_clean", "Gruppe"])["Posts_final"].sum().reset_index()

# Pivot → Spalten je Gruppe
posts_pivot = posts_agg.pivot(index="ORT_clean", columns="Gruppe", values="Posts_final").fillna(0)

# Spalten umbenennen
posts_pivot.columns = [f"Posts_{col}" for col in posts_pivot.columns]

# Merge
df = df.merge(posts_pivot, left_on="Ort_clean", right_index=True, how="left")

# Fehlende Werte = 0
for g in ["student", "schueler", "zbor", "none"]:
    col = f"Posts_{g}"
    if col in df.columns:
        df[col] = df[col].fillna(0)




# ---------------------------
# 5. Funktionen für partielle Korrelation
# ---------------------------
def regress_residuals(y, X):
    X = np.column_stack([np.ones(len(X)), X])  # Intercept
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    return y - y_hat

def partial_corr(df, x, y, controls=None):
    if controls is None:
        r = np.corrcoef(df[x], df[y])[0, 1]
    else:
        X_controls = df[controls].values
        resid_x = regress_residuals(df[x].values, X_controls)
        resid_y = regress_residuals(df[y].values, X_controls)
        r = np.corrcoef(resid_x, resid_y)[0, 1]
    return r, len(df)

def corr_p_value(r, n, k=0):
    df = n - k - 2
    t_stat = r * np.sqrt(df / (1 - r**2))
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return p

def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# ---------------------------
# 6. Korrelationen für jede Account-Gruppe berechnen
# ---------------------------
results_groups = []

for g in groups:
    acc_col = f"Accounts_{g}"
    
    r_raw_proteste, n = partial_corr(df, "Anzahl Proteste", acc_col, controls=None)
    r_part_proteste, _ = partial_corr(df, "Anzahl Proteste", acc_col, controls=["Einwohner", "Wahl Opposition"])
    
    r_raw_teilnehmer, _ = partial_corr(df, "Teilnehmerzahl", acc_col, controls=None)
    r_part_teilnehmer, _ = partial_corr(df, "Teilnehmerzahl", acc_col, controls=["Einwohner", "Wahl Opposition"])
    
    # p-Werte
    p_raw_proteste = corr_p_value(r_raw_proteste, n, k=0)
    p_part_proteste = corr_p_value(r_part_proteste, n, k=2)
    
    p_raw_teilnehmer = corr_p_value(r_raw_teilnehmer, n, k=0)
    p_part_teilnehmer = corr_p_value(r_part_teilnehmer, n, k=2)

    results_groups.append({
        "Gruppe": g.capitalize(),
        "Korrelation": "Proteste ↔ Accounts",
        "Roh r": f"{round(r_raw_proteste,3)}{significance_stars(p_raw_proteste)}",
        "Partiell r": f"{round(r_part_proteste,3)}{significance_stars(p_part_proteste)}",
        "Roh p": round(p_raw_proteste,4),
        "Partiell p": round(p_part_proteste,4),
        "N": n
    })
    
    results_groups.append({
        "Gruppe": g.capitalize(),
        "Korrelation": "Teilnehmer ↔ Accounts",
        "Roh r": f"{round(r_raw_teilnehmer,3)}{significance_stars(p_raw_teilnehmer)}",
        "Partiell r": f"{round(r_part_teilnehmer,3)}{significance_stars(p_part_teilnehmer)}",
        "Roh p": round(p_raw_teilnehmer,4),
        "Partiell p": round(p_part_teilnehmer,4),
        "N": n
    })

results_groups_df = pd.DataFrame(results_groups)
print(results_groups_df)

# ---------------------------
# 7. Tabelle als Grafik speichern
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table = ax.table(
    cellText=results_groups_df.values,
    colLabels=results_groups_df.columns,
    loc='center'
)

# Schriftgröße und Spaltenbreite
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(results_groups_df.columns))))

# Header fett
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

# Überschrift näher an der Tabelle
plt.title("Korrelationen nach Account-Inhaber-Gruppe (Roh vs. Partiell)", pad=5)

# Tabelle speichern
plt.savefig("korrelationen_account_groups.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

# ---------------------------
# 8. Korrelationen für Posts nach Gruppen
# ---------------------------
results_posts = []

for g in groups:
    posts_col = f"Posts_{g}"
    
    if posts_col not in df.columns:
        continue
    
    r_raw_proteste, n = partial_corr(df, "Anzahl Proteste", posts_col, controls=None)
    r_part_proteste, _ = partial_corr(df, "Anzahl Proteste", posts_col, controls=["Einwohner", "Wahl Opposition"])
    
    r_raw_teilnehmer, _ = partial_corr(df, "Teilnehmerzahl", posts_col, controls=None)
    r_part_teilnehmer, _ = partial_corr(df, "Teilnehmerzahl", posts_col, controls=["Einwohner", "Wahl Opposition"])
    
    # p-Werte
    p_raw_proteste = corr_p_value(r_raw_proteste, n, k=0)
    p_part_proteste = corr_p_value(r_part_proteste, n, k=2)
    
    p_raw_teilnehmer = corr_p_value(r_raw_teilnehmer, n, k=0)
    p_part_teilnehmer = corr_p_value(r_part_teilnehmer, n, k=2)

    results_posts.append({
        "Gruppe": g.capitalize(),
        "Korrelation": "Proteste ↔ Posts",
        "Roh r": f"{round(r_raw_proteste,3)}{significance_stars(p_raw_proteste)}",
        "Partiell r": f"{round(r_part_proteste,3)}{significance_stars(p_part_proteste)}",
        "Roh p": round(p_raw_proteste,4),
        "Partiell p": round(p_part_proteste,4),
        "N": n
    })
    
    results_posts.append({
        "Gruppe": g.capitalize(),
        "Korrelation": "Teilnehmer ↔ Posts",
        "Roh r": f"{round(r_raw_teilnehmer,3)}{significance_stars(p_raw_teilnehmer)}",
        "Partiell r": f"{round(r_part_teilnehmer,3)}{significance_stars(p_part_teilnehmer)}",
        "Roh p": round(p_raw_teilnehmer,4),
        "Partiell p": round(p_part_teilnehmer,4),
        "N": n
    })

results_posts_df = pd.DataFrame(results_posts)
print(results_posts_df)


# ---------------------------
# 9. Posts-Tabelle als Grafik speichern
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table = ax.table(
    cellText=results_posts_df.values,
    colLabels=results_posts_df.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(results_posts_df.columns))))

# Header fett
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

plt.title("Korrelationen nach Account-Gruppen: Posts (Roh vs. Partiell)", pad=5)

plt.savefig("korrelationen_posts_groups.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()


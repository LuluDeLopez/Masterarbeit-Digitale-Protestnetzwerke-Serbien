import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Daten laden
# ---------------------------
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_merged")
instas = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spaltennamen bereinigen (Leerzeichen am Anfang/Ende entfernen)
df.columns = df.columns.str.strip()
instas.columns = instas.columns.str.strip()

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
# 3. Teilnehmerzahl Pro/Anti
# ---------------------------
df["Teilnehmer Pro"] = df["Crowd Durchschnitt  Pro"].fillna(0)
df["Teilnehmer Anti"] = df["Crowd Durchschnitt Anti"].fillna(0)

# ---------------------------
# 4. Accounts Pro/Anti
# ---------------------------
df["Accounts Pro"] = df["n_accounts"] * df["share_pro"]
df["Accounts Anti"] = df["n_accounts"] * (1 - df["share_pro"])

# ---------------------------
# 5. Posts Pro/Anti aus Instas
# ---------------------------

# Orte bereinigen
df["Ort_clean"] = df["Ort"].str.strip().str.lower()
instas["ORT_clean"] = instas["ORT"].str.strip().str.lower()

# Posts final bestimmen (Zeitraum oder Gesamt)
instas["Posts_final"] = instas["POSTS ZEITRAUM"].replace(0, np.nan).fillna(instas["POSTS GESAMT"])

# Dafür-Spalte in int umwandeln und nur 0/1 behalten
instas["Dafür"] = instas["Dafür"].astype(int)
instas = instas[instas["Dafür"].isin([0,1])]  # 2 wird entfernt

# Summe pro Ort und Pro/Anti
posts_agg = instas.groupby(["ORT_clean", "Dafür"])["Posts_final"].sum().reset_index()

# Pivot für Pro/Anti
posts_pivot = posts_agg.pivot(index="ORT_clean", columns="Dafür", values="Posts_final").fillna(0)
posts_pivot.rename(columns={0: "Posts Anti", 1: "Posts Pro"}, inplace=True)

# Merge mit df
df = df.merge(posts_pivot, left_on="Ort_clean", right_index=True, how="left")

# NaNs nach merge auf 0 setzen
df["Posts Pro"] = df["Posts Pro"].fillna(0)
df["Posts Anti"] = df["Posts Anti"].fillna(0)

# ---------------------------
# 6. Funktionen für partielle Korrelation
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

# ---------------------------
# 7. Korrelationen für Pro/Anti berechnen
# ---------------------------
results = []

pairs = [
    ("Anzahl Pro", "Accounts Pro", "Posts Pro", "Teilnehmer Pro", "Pro"),
    ("Anzahl Anti", "Accounts Anti", "Posts Anti", "Teilnehmer Anti", "Anti")
]

for anz, acc, posts, teil, label in pairs:
    # Roh- und partielle Korrelationen
    r_raw_acc, n = partial_corr(df, anz, acc, controls=None)
    r_part_acc, _ = partial_corr(df, anz, acc, controls=["Einwohner","Wahl Opposition"])
    
    r_raw_posts, _ = partial_corr(df, anz, posts, controls=None)
    r_part_posts, _ = partial_corr(df, anz, posts, controls=["Einwohner","Wahl Opposition"])
    
    r_raw_teil_acc, _ = partial_corr(df, teil, acc, controls=None)
    r_part_teil_acc, _ = partial_corr(df, teil, acc, controls=["Einwohner","Wahl Opposition"])
    
    r_raw_teil_posts, _ = partial_corr(df, teil, posts, controls=None)
    r_part_teil_posts, _ = partial_corr(df, teil, posts, controls=["Einwohner","Wahl Opposition"])
    
    # Ergebnisse speichern
    results.append({
        "Pro/Anti": label,
        "Korrelation": "Proteste ↔ Accounts",
        "Roh r": round(r_raw_acc,3),
        "Partiell r": round(r_part_acc,3),
        "N": n
    })
    results.append({
        "Pro/Anti": label,
        "Korrelation": "Proteste ↔ Posts",
        "Roh r": round(r_raw_posts,3),
        "Partiell r": round(r_part_posts,3),
        "N": n
    })
    results.append({
        "Pro/Anti": label,
        "Korrelation": "Teilnehmer ↔ Accounts",
        "Roh r": round(r_raw_teil_acc,3),
        "Partiell r": round(r_part_teil_acc,3),
        "N": n
    })
    results.append({
        "Pro/Anti": label,
        "Korrelation": "Teilnehmer ↔ Posts",
        "Roh r": round(r_raw_teil_posts,3),
        "Partiell r": round(r_part_teil_posts,3),
        "N": n
    })

results_df = pd.DataFrame(results)
print(results_df)

# ---------------------------
# 8. Tabelle als Grafik speichern
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(results_df.columns))))

# Header fett
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

# Überschrift näher an der Tabelle
plt.title("Pro/Anti Korrelationen (Roh vs. Partiell)", pad=3)

# Tabelle speichern
plt.savefig("korrelationen_pro_anti.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

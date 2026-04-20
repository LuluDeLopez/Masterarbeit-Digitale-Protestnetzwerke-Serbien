import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------
# 1. Daten laden
# ---------------------------
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_merged")
instas = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

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

# 1. Orte bereinigen
df["Ort_clean"] = df["Ort"].str.strip().str.lower()
instas["ORT_clean"] = instas["ORT"].str.strip().str.lower()

# 2. Posts final bestimmen (Zeitraum oder Gesamt)
instas["Posts_final"] = instas["POSTS ZEITRAUM"].replace(0, np.nan).fillna(instas["POSTS GESAMT"])

# 3. Dafür-Spalte in int und nur 0/1 behalten
instas["Dafür"] = instas["Dafür"].astype(int)
instas = instas[instas["Dafür"].isin([0,1])]  # entfernt alles, z.B. 2

# 4. Summe pro Ort und Pro/Anti
posts_agg = instas.groupby(["ORT_clean", "Dafür"])["Posts_final"].sum().reset_index()

# 5. Pivot für Pro/Anti
posts_pivot = posts_agg.pivot(index="ORT_clean", columns="Dafür", values="Posts_final").fillna(0)
posts_pivot.rename(columns={0: "Posts Anti", 1: "Posts Pro"}, inplace=True)

# 6. Merge mit df
df = df.merge(posts_pivot, left_on="Ort_clean", right_index=True, how="left")

# 7. NaNs nach merge auf 0 setzen
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
# 7. Korrelationen für Pro/Anti berechnen
# ---------------------------
results = []

pairs = [
    ("Anzahl Pro", "Accounts Pro", "Posts Pro", "Teilnehmer Pro", "Pro"),
    ("Anzahl Anti", "Accounts Anti", "Posts Anti", "Teilnehmer Anti", "Anti")
]

for anz, acc, posts, teil, label in pairs:
    
    # --- Proteste ↔ Accounts ---
    r_raw_acc, n = partial_corr(df, anz, acc, controls=None)
    r_part_acc, _ = partial_corr(df, anz, acc, controls=["Einwohner","Wahl Opposition"])
    
    p_raw_acc = corr_p_value(r_raw_acc, n, k=0)
    p_part_acc = corr_p_value(r_part_acc, n, k=2)

    # --- Proteste ↔ Posts ---
    r_raw_posts, _ = partial_corr(df, anz, posts, controls=None)
    r_part_posts, _ = partial_corr(df, anz, posts, controls=["Einwohner","Wahl Opposition"])
    
    p_raw_posts = corr_p_value(r_raw_posts, n, k=0)
    p_part_posts = corr_p_value(r_part_posts, n, k=2)

    # --- Teilnehmer ↔ Accounts ---
    r_raw_teil_acc, _ = partial_corr(df, teil, acc, controls=None)
    r_part_teil_acc, _ = partial_corr(df, teil, acc, controls=["Einwohner","Wahl Opposition"])
    
    p_raw_teil_acc = corr_p_value(r_raw_teil_acc, n, k=0)
    p_part_teil_acc = corr_p_value(r_part_teil_acc, n, k=2)

    # --- Teilnehmer ↔ Posts ---
    r_raw_teil_posts, _ = partial_corr(df, teil, posts, controls=None)
    r_part_teil_posts, _ = partial_corr(df, teil, posts, controls=["Einwohner","Wahl Opposition"])
    
    p_raw_teil_posts = corr_p_value(r_raw_teil_posts, n, k=0)
    p_part_teil_posts = corr_p_value(r_part_teil_posts, n, k=2)

    # --- Append ---
    results.append({
        "Pro/Anti": label,
        "Korrelation": "Proteste ↔ Accounts",
        "Roh r": f"{round(r_raw_acc,3)}{significance_stars(p_raw_acc)}",
        "Partiell r": f"{round(r_part_acc,3)}{significance_stars(p_part_acc)}",
        "Roh p": round(p_raw_acc,4),
        "Partiell p": round(p_part_acc,4),
        "N": n
    })

    results.append({
        "Pro/Anti": label,
        "Korrelation": "Proteste ↔ Posts",
        "Roh r": f"{round(r_raw_posts,3)}{significance_stars(p_raw_posts)}",
        "Partiell r": f"{round(r_part_posts,3)}{significance_stars(p_part_posts)}",
        "Roh p": round(p_raw_posts,4),
        "Partiell p": round(p_part_posts,4),
        "N": n
    })

    results.append({
        "Pro/Anti": label,
        "Korrelation": "Teilnehmer ↔ Accounts",
        "Roh r": f"{round(r_raw_teil_acc,3)}{significance_stars(p_raw_teil_acc)}",
        "Partiell r": f"{round(r_part_teil_acc,3)}{significance_stars(p_part_teil_acc)}",
        "Roh p": round(p_raw_teil_acc,4),
        "Partiell p": round(p_part_teil_acc,4),
        "N": n
    })

    results.append({
        "Pro/Anti": label,
        "Korrelation": "Teilnehmer ↔ Posts",
        "Roh r": f"{round(r_raw_teil_posts,3)}{significance_stars(p_raw_teil_posts)}",
        "Partiell r": f"{round(r_part_teil_posts,3)}{significance_stars(p_part_teil_posts)}",
        "Roh p": round(p_raw_teil_posts,4),
        "Partiell p": round(p_part_teil_posts,4),
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

plt.title("Pro/Anti Korrelationen (Roh vs. Partiell)", pad=2)

plt.savefig("korrelationen_pro_anti.png", dpi=300, bbox_inches='tight')
plt.close()

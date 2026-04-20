import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------
# 1. Daten vorbereiten
# ---------------------------
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_merged")

exclude_orte = [
    "USA", "Deutschland", "England", "Österreich", "Belgien",
    "Frankreich", "Italien", "Kanada", "Luxemburg",
    "Schweiz", "Spanien", "Zypern", 0
]
df = df[~df["Ort"].isin(exclude_orte)]

df["Teilnehmerzahl"] = (
    df["Crowd Durchschnitt  Pro"].fillna(0) +
    df["Crowd Durchschnitt Anti"].fillna(0)
)

cols = [
    "Anzahl Proteste",
    "n_accounts",
    "posts_period_sum",
    "Teilnehmerzahl",
    "Einwohner",
    "Wahl Opposition"
]
df = df[cols].dropna()

# ---------------------------
# 2. Funktionen für partielle Korrelation
# ---------------------------
def regress_residuals(y, X):
    X = np.column_stack([np.ones(len(X)), X])  # Intercept
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    return y - y_hat

def partial_corr(df, x, y, controls=None):
    if controls is None:
        # Roh-Korrelation
        r = np.corrcoef(df[x], df[y])[0, 1]
    else:
        X_controls = df[controls].values
        resid_x = regress_residuals(df[x].values, X_controls)
        resid_y = regress_residuals(df[y].values, X_controls)
        r = np.corrcoef(resid_x, resid_y)[0, 1]
    return r, len(df)


def corr_p_value(r, n, k=0):
    """
    r = Korrelation
    n = Stichprobengröße
    k = Anzahl Kontrollvariablen (für partielle Korrelation)
    """
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
# 3. Korrelationen berechnen
# ---------------------------
pairs = [
    ("Anzahl Proteste", "n_accounts", "Proteste ↔ Accounts"),
    ("Anzahl Proteste", "posts_period_sum", "Proteste ↔ Posts"),
    ("Teilnehmerzahl", "n_accounts", "Teilnehmer ↔ Accounts"),
    ("Teilnehmerzahl", "posts_period_sum", "Teilnehmer ↔ Posts"),
]

results = []
for x, y, label in pairs:
    r_raw, n = partial_corr(df, x, y, controls=None)
    r_partial, _ = partial_corr(df, x, y, controls=["Einwohner", "Wahl Opposition"])
    
    p_raw = corr_p_value(r_raw, n, k=0)
    p_partial = corr_p_value(r_partial, n, k=2)

    results.append({
    "Korrelation": label,
    "Roh r": f"{round(r_raw, 3)}{significance_stars(p_raw)}",
    "Partiell r": f"{round(r_partial, 3)}{significance_stars(p_partial)}",
    "Roh p": round(p_raw, 4),
    "Partiell p": round(p_partial, 4),
    "N": n
    })

results_df = pd.DataFrame(results)
print(results_df)

# ---------------------------
# 4. Tabelle als Grafik speichern
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc='center'
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))

# Header fett
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

plt.title("Roh- vs. Partielle Korrelationen (kontrolliert für Einwohner & Wahl Opposition)", pad=10)

plt.savefig("korrelationen_raw_vs_partial.png", dpi=300, bbox_inches='tight')
plt.close()
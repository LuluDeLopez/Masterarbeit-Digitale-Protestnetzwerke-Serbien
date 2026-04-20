import pandas as pd

# Datei laden
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_merged")

# Orte ausschließen
exclude_orte = [
    "USA", "Deutschland", "England", "Österreich", "Belgien",
    "Frankreich", "Italien", "Kanada", "Luxemburg",
    "Schweiz", "Spanien", "Zypern", 0
]

df = df[~df["Ort"].isin(exclude_orte)]

# Teilnehmerzahl berechnen (Pro + Anti)
df["Teilnehmerzahl"] = (
    df["Crowd Durchschnitt  Pro"].fillna(0) +
    df["Crowd Durchschnitt Anti"].fillna(0)
)

# Relevante Variablen auswählen
data = df[[
    "Anzahl Proteste",
    "n_accounts",
    "posts_period_sum",
    "Teilnehmerzahl"
]]

# Korrelationen berechnen
correlations = pd.DataFrame({
    "Korrelation": [
        data["Anzahl Proteste"].corr(data["n_accounts"]),
        data["Anzahl Proteste"].corr(data["posts_period_sum"]),
        data["Teilnehmerzahl"].corr(data["n_accounts"]),
        data["Teilnehmerzahl"].corr(data["posts_period_sum"])
    ]
}, index=[
    "Proteste ↔ Accounts",
    "Proteste ↔ Posts",
    "Teilnehmer ↔ Accounts",
    "Teilnehmer ↔ Posts"
])

print(correlations)

import pandas as pd
import matplotlib.pyplot as plt

# Excel laden
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spaltenname angleichen
df_accounts.rename(columns={'ORT': 'Ort'}, inplace=True)

# -----------------------------
# Orte ausschließen
# -----------------------------
exclude_orte = [
    "USA", "Deutschland", "England", "Österreich", "Belgien",
    "Frankreich", "Italien", "Kanada", "Luxemburg",
    "Schweiz", "Spanien", "Zypern"
]

df_filtered = df_accounts[~df_accounts['Ort'].isin(exclude_orte)]

# -----------------------------
# Accounts pro Ort zählen
# -----------------------------
accounts_per_ort = df_filtered['Ort'].value_counts()

# -----------------------------
# Verteilung erstellen:
# Wie viele Orte haben X Accounts?
# -----------------------------
distribution = accounts_per_ort.value_counts().sort_index()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))

plt.bar(distribution.index, distribution.values, color='skyblue')

plt.xlabel("Anzahl Accounts pro Ort")
plt.ylabel("Anzahl der Orte")
plt.title("Verteilung der Accounts pro Ort")

plt.xticks(distribution.index)
plt.tight_layout()
plt.show()

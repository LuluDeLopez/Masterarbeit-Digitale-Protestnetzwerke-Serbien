import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Excel laden
# -----------------------------
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spaltennamen anpassen
df_accounts.rename(columns={'ORT': 'Ort'}, inplace=True)

# -----------------------------
# Orte ausschließen
# -----------------------------
exclude_orte = [
    "USA", "Deutschland", "England", "Österreich", "Belgien",
    "Frankreich", "Italien", "Kanada", "Luxemburg",
    "Schweiz", "Spanien", "Zypern", 0
]

df_filtered = df_accounts[~df_accounts['Ort'].isin(exclude_orte)]

# -----------------------------
# Gruppen-Labels
# -----------------------------
gruppen_labels = {0: "Nicht spezifiziert", 1: "Studenten", 2: "Schüler", 3: "Zbor"}

# -----------------------------
# Accounts pro Ort zählen
# -----------------------------
accounts_per_ort = df_filtered['Ort'].value_counts()

# Top & Bottom 5 Orte nach Anzahl Accounts
top5_orte = accounts_per_ort.head(5).index.tolist()
bottom5_orte = accounts_per_ort.tail(5).index.tolist()
selected_orte = top5_orte + bottom5_orte

df_selected = df_filtered[df_filtered['Ort'].isin(selected_orte)]

# -----------------------------
# Pivot-Tabelle erstellen und Reihenfolge fixieren
# -----------------------------
pivot = pd.crosstab(df_selected['Ort'], df_selected['Gruppe'])
pivot.rename(columns=gruppen_labels, inplace=True)

# Reihenfolge der Orte wie bei Top/Bottom-5 festlegen
pivot = pivot.reindex(selected_orte)

# -----------------------------
# Terminal-Ausgabe
# -----------------------------
print("\nGruppenverteilung für Top & Bottom 5 Orte:\n")
print(pivot)

# -----------------------------
# Plot (gestapelt)
# -----------------------------
pivot.plot(kind='bar', stacked=True, figsize=(12,6), color=['grey', 'blue', 'green', 'red'])

plt.title("Gruppenverteilung in Top & Bottom 5 Orten")
plt.ylabel("Anzahl Accounts")
plt.xlabel("Ort")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
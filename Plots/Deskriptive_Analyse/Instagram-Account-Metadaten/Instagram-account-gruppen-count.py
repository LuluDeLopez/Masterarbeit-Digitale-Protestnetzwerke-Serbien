import pandas as pd

# Excel laden
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spalten anpassen
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
# Accounts pro Gruppe
# -----------------------------
group_counts = df_filtered['Gruppe'].value_counts().sort_index()

# Mapping für bessere Lesbarkeit
gruppen_labels = {
    0: "Nicht spezifiziert",
    1: "Studenten",
    2: "Schüler",
    3: "Zbor"
}

print("Accounts pro Gruppe:\n")
for gruppe, count in group_counts.items():
    label = gruppen_labels.get(gruppe, f"Gruppe {gruppe}")
    print(f"{label}: {count}")

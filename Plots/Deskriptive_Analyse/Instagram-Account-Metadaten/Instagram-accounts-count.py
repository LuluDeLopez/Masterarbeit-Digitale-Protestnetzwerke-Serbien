import pandas as pd

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
# Gesamtanzahl Accounts
# -----------------------------
total_accounts = len(df_accounts)

# -----------------------------
# Anzahl Accounts pro Ort
# -----------------------------
accounts_per_ort = df_accounts['Ort'].value_counts()

# -----------------------------
# Ausgabe
# -----------------------------
print(f"Gesamtanzahl Accounts: {total_accounts}\n")

print("Accounts pro Ort:")
for ort, count in accounts_per_ort.items():
    print(f"{ort}: {count}")

# -----------------------------
# Anzahl der Orte (nach Ausschluss)
# -----------------------------
anzahl_orte = df_filtered['Ort'].nunique()

print(f"\nAnzahl der Orte (ohne ausgeschlossene Länder): {anzahl_orte}")

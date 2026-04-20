import pandas as pd
import matplotlib.pyplot as plt

# Excel laden
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spaltennamen angleichen
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
# Gesamt-Follower
# -----------------------------
total_followers = df_filtered['FOLLOWER'].sum()

print(f"Gesamtsumme der Follower (gefiltert): {int(total_followers)}")

# -----------------------------
# Follower pro Ort
# -----------------------------
followers_per_ort = df_filtered.groupby('Ort')['FOLLOWER'].sum().reset_index()

# Sortieren
followers_sorted = followers_per_ort.sort_values(by='FOLLOWER', ascending=False)

# Top & Bottom 5
top5 = followers_sorted.head(5)
bottom5 = followers_sorted.tail(5)

combined = pd.concat([top5, bottom5])

# -----------------------------
# Terminal-Ausgabe
# -----------------------------
print("\nTop 5 Orte nach Followern:")
for _, row in top5.iterrows():
    print(f"{row['Ort']}: {int(row['FOLLOWER'])}")

print("\nBottom 5 Orte nach Followern:")
for _, row in bottom5.iterrows():
    print(f"{row['Ort']}: {int(row['FOLLOWER'])}")

# -----------------------------
# Plot
# -----------------------------
x = range(len(combined))

plt.figure(figsize=(12,6))

plt.bar(x, combined['FOLLOWER'], color='purple')

plt.xticks(x, combined['Ort'], rotation=45, ha='right')
plt.ylabel("Follower")
plt.title("Top & Bottom 5 Orte nach Followern")

plt.yscale('log')
plt.ylabel("Follower (log-Skala)")

plt.tight_layout()
plt.show()

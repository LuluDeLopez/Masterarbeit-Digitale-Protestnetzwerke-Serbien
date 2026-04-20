import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Excel laden
# -----------------------------
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Instas")
df_coords   = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Proteste")

print(df_accounts.columns)

# Spaltennamen angleichen
df_accounts.rename(columns={'ORT':'Ort'}, inplace=True)

# Tabellen zusammenführen
df = pd.merge(df_coords, df_accounts, on='Ort', how='inner')

# -----------------------------
# Pro/Anti Posts direkt aus Instas berechnen
# -----------------------------

# Sicherstellen, dass Spalten korrekt heißen
df_accounts.rename(columns={
    'POSTS ZEITRAUM': 'Posts_Zeitraum',
    'Dafür': 'Dafür'
}, inplace=True)

# Pro Posts (Dafuer = 1)
df_pro_posts = df_accounts[df_accounts['Dafür'] == 1] \
    .groupby('Ort')['Posts_Zeitraum'] \
    .sum() \
    .reset_index(name='Pro_Posts')

# Anti Posts (Dafür = 0)
df_anti_posts = df_accounts[df_accounts['Dafuer'] == 0] \
    .groupby('Ort')['Posts_Zeitraum'] \
    .sum() \
    .reset_index(name='Anti_Posts')

# Zusammenführen
df_posts = pd.merge(df_pro_posts, df_anti_posts, on='Ort', how='outer').fillna(0)

# Mit Haupt-df mergen
df = pd.merge(df, df_posts, on='Ort', how='left')

# Falls Orte ohne Eintrag existieren
df['Pro_Posts'] = df['Pro_Posts'].fillna(0)
df['Anti_Posts'] = df['Anti_Posts'].fillna(0)

# -----------------------------
# Funktion: Top/Bottom 5 visualisieren
# -----------------------------
def plot_compare(df, col_pro, col_anti, title):
    df['total'] = df[col_pro] + df[col_anti]
    df_sorted = df.sort_values(by='total', ascending=False)
    
    top5 = df_sorted.head(5)
    bottom5 = df_sorted.tail(5)
    combined = pd.concat([top5, bottom5])
    
    x = range(len(combined))
    
    plt.figure(figsize=(12,6))
    
    # Balken: Pro = blau, Anti = rot
    plt.bar(x, combined[col_pro], width=0.4, label='Pro', color='blue')
    plt.bar([i+0.4 for i in x], combined[col_anti], width=0.4, label='Anti', color='red')
    
    plt.xticks([i+0.2 for i in x], combined['Ort'], rotation=45, ha='right')
    plt.ylabel("Anzahl")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Top/Bottom 5 nach Accounts
# -----------------------------
plot_compare(df.copy(), "Pro_Accounts", "Anti_Accounts", "Pro vs. Anti Accounts (Top & Bottom 5 Orte)")

# -----------------------------
# Top/Bottom 5 nach Posts
# -----------------------------
plot_compare(df.copy(), "Pro_Posts", "Anti_Posts", "Pro vs. Anti Posts (Top & Bottom 5 Orte)")

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Excel laden
# -----------------------------
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spaltennamen angleichen
df_accounts.rename(columns={
    'ORT': 'Ort',
    'POSTS ZEITRAUM': 'Posts_Zeitraum',
    'Dafür': 'Dafuer'
}, inplace=True)

# -----------------------------
# Ort 0 ausschließen
# -----------------------------
df_accounts = df_accounts[df_accounts['Ort'] != 0]
df_accounts = df_accounts[df_accounts['Ort'] != "Kanada"]
df_accounts = df_accounts[df_accounts['Ort'] != "Frankreich"]
# -----------------------------
# Pro/Anti Posts pro Ort berechnen
# -----------------------------

df_accounts['Posts_Final'] = df_accounts['Posts_Zeitraum'].fillna(df_accounts['POSTS GESAMT'])

# Berechnung Pro-Posts
df_pro_posts = df_accounts[df_accounts['Dafuer'] == 1].groupby('Ort')['Posts_Final'].sum().reset_index()
df_pro_posts.rename(columns={'Posts_Final': 'Pro_Posts'}, inplace=True)

# Berechnung Anti-Posts
df_anti_posts = df_accounts[df_accounts['Dafuer'] == 0].groupby('Ort')['Posts_Final'].sum().reset_index()
df_anti_posts.rename(columns={'Posts_Final': 'Anti_Posts'}, inplace=True)

# Zusammenführen
df_posts = pd.merge(df_pro_posts, df_anti_posts, on='Ort', how='outer').fillna(0)


def plot_top_bottom(df, col_pro, col_anti, title):
    df['total_posts'] = df[col_pro] + df[col_anti]
    
    df_sorted = df.sort_values(by='total_posts', ascending=False)
    
    top5 = df_sorted.head(5)
    bottom5 = df_sorted.tail(5)
    combined = pd.concat([top5, bottom5])
    
    # -----------------------------
    # Terminal-Ausgabe
    # -----------------------------
    print("\nTop 5 Orte nach Gesamtposts:")
    for idx, row in top5.iterrows():
        print(f"{row['Ort']}: Pro = {int(row[col_pro])}, Anti = {int(row[col_anti])}, Gesamt = {int(row['total_posts'])}")
    
    print("\nBottom 5 Orte nach Gesamtposts:")
    for idx, row in bottom5.iterrows():
        print(f"{row['Ort']}: Pro = {int(row[col_pro])}, Anti = {int(row[col_anti])}, Gesamt = {int(row['total_posts'])}")
    
    # -----------------------------
    # Gestapeltes Balkendiagramm
    # -----------------------------
    x = range(len(combined))
    plt.figure(figsize=(12,6))
    
    # Pro-Posts
    plt.bar(x, combined[col_pro], label='Pro', color='blue')
    # Anti-Posts gestapelt oben drauf
    plt.bar(x, combined[col_anti], bottom=combined[col_pro], label='Anti', color='red')
    
    plt.xticks(x, combined['Ort'], rotation=45, ha='right')
    plt.ylabel("Anzahl Posts")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # -----------------------------
    # Plot
    # -----------------------------
    x = range(len(combined))
    plt.figure(figsize=(12,6))
    
    plt.bar(x, combined[col_pro], width=0.4, label='Pro', color='blue')
    plt.bar([i+0.4 for i in x], combined[col_anti], width=0.4, label='Anti', color='red')
    
    plt.xticks([i+0.2 for i in x], combined['Ort'], rotation=45, ha='right')
    plt.ylabel("Anzahl Posts")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Top/Bottom 5 Orte nach Posts
# -----------------------------
plot_top_bottom(df_posts, "Pro_Posts", "Anti_Posts", "Pro vs. Anti Posts (Top & Bottom 5 Orte)")

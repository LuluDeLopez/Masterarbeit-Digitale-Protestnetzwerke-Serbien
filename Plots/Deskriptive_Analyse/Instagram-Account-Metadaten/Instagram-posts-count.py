import pandas as pd

# Excel laden
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Spaltennamen angleichen
df_accounts.rename(columns={'Dafür': 'Dafuer', 'POSTS ZEITRAUM': 'Posts_Zeitraum'}, inplace=True)

# -----------------------------
# Richtige Post-Zahl wählen
# -----------------------------
df_accounts['Posts_Final'] = df_accounts['Posts_Zeitraum'].fillna(df_accounts['POSTS GESAMT'])

# -----------------------------
# Pro-Accounts (Dafuer=1)
# -----------------------------
pro_accounts = df_accounts[df_accounts['Dafuer'] == 1]
sum_pro_posts = pro_accounts['Posts_Final'].sum()
avg_pro_posts = sum_pro_posts / len(pro_accounts) if len(pro_accounts) > 0 else 0

# -----------------------------
# Anti-Accounts (Dafuer=0)
# -----------------------------
anti_accounts = df_accounts[df_accounts['Dafuer'] == 0]
sum_anti_posts = anti_accounts['Posts_Final'].sum()
avg_anti_posts = sum_anti_posts / len(anti_accounts) if len(anti_accounts) > 0 else 0

# -----------------------------
# Ausgabe
# -----------------------------
print(f"Pro-Accounts (Dafuer=1): Summe = {int(sum_pro_posts)}, Durchschnitt = {avg_pro_posts:.2f} Posts pro Account")
print(f"Anti-Accounts (Dafuer=0): Summe = {int(sum_anti_posts)}, Durchschnitt = {avg_anti_posts:.2f} Posts pro Account")
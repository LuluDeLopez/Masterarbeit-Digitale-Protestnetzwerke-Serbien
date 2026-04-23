import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Daten laden
# -----------------------------
df = pd.read_excel("Zeitreihe_final.xlsx")
df['Date'] = pd.to_datetime(df['Date'])

# Spalten: Location, Date, Posts, Active_Accounts, Protests, participants, log_participants

# -----------------------------
# 2️⃣ Lag-Funktion
# -----------------------------
def create_lags(df, cols, lag_days=1):
    for col in cols:
        df[f'{col}_lag{lag_days}'] = df.groupby('Location')[col].shift(lag_days)
    return df

# -----------------------------
# 3️⃣ Tagesdaten (Original)
# -----------------------------
daily = df.copy()
daily = create_lags(daily, ['Posts','Active_Accounts','Protests','participants'], lag_days=1)
daily = create_lags(daily, ['Posts','Active_Accounts','Protests','participants'], lag_days=2)

# Korrelation t vs t+1
print("=== Tagesdaten: Protest heute vs Instagram morgen ===")
print(daily[['Protests','Posts_lag1']].corr().iloc[0,1])
print(daily[['Protests','Active_Accounts_lag1']].corr().iloc[0,1])

# -----------------------------
# 4️⃣ Wochenaggregation
# -----------------------------
df['Week'] = df['Date'].dt.to_period('W')
weekly = df.groupby(['Location','Week']).agg({
    'Posts':'sum',
    'Active_Accounts':'sum',
    'Protests':'sum',
    'participants':'sum'
}).reset_index()

weekly['log_participants'] = np.log1p(weekly['participants'])

weekly = create_lags(weekly, ['Posts','Active_Accounts','Protests','participants'], lag_days=1)
weekly = create_lags(weekly, ['Posts','Active_Accounts','Protests','participants'], lag_days=2)

print("\n=== Wochendaten: Protest diese Woche vs Instagram nächste Woche ===")
print(weekly[['Protests','Posts_lag1']].corr().iloc[0,1])
print(weekly[['Protests','Active_Accounts_lag1']].corr().iloc[0,1])

# -----------------------------
# 5️⃣ Monatsaggregation
# -----------------------------
df['Month'] = df['Date'].dt.to_period('M')
monthly = df.groupby(['Location','Month']).agg({
    'Posts':'sum',
    'Active_Accounts':'sum',
    'Protests':'sum',
    'participants':'sum'
}).reset_index()

monthly['log_participants'] = np.log1p(monthly['participants'])

monthly = create_lags(monthly, ['Posts','Active_Accounts','Protests','participants'], lag_days=1)
monthly = create_lags(monthly, ['Posts','Active_Accounts','Protests','participants'], lag_days=2)

print("\n=== Monatsdaten: Protest diesen Monat vs Instagram nächsten Monat ===")
print(monthly[['Protests','Posts_lag1']].corr().iloc[0,1])
print(monthly[['Protests','Active_Accounts_lag1']].corr().iloc[0,1])

# -----------------------------
# 6️⃣ Ergebnis: Speichern
# -----------------------------
daily.to_excel("Tagesdaten_mit_Lags.xlsx", index=False)
weekly.to_excel("Wochendaten_mit_Lags.xlsx", index=False)
monthly.to_excel("Monatsdaten_mit_Lags.xlsx", index=False)

print("\nFertig! Alle Aggregationen mit Lags gespeichert.")

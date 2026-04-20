# =============================================
# Zeitreihenanalyse Proteste - Masterarbeit
# Datensatz: Datensatz-Master-Final.xlsx
# Sheet: Zeitreihe-Tag
# =============================================

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Excel-Datei laden
# -----------------------------
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Zeitreihe-Tag")
# 1. Alle Spaltennamen anzeigen, um zu sehen, wie die Spalte wirklich heißt
print(df.columns.tolist())
# -----------------------------
# 2. Datum konvertieren
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'])

# -----------------------------
# 3. Duplikate am gleichen Datum aggregieren
# Summe für Protests & participants, Mittelwert für log_participants
# -----------------------------
df = df.groupby('Date', as_index=False).agg({
    'Protests': 'sum',
    'participants': 'sum',
    'log_participants': 'mean'
})

# -----------------------------
# 4. Sortieren nach Datum
# -----------------------------
df = df.sort_values('Date')

# -----------------------------
# 5. Vollständigen Datumsbereich erstellen
# Fehlende Tage = 0
# -----------------------------
date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
df = df.set_index('Date').reindex(date_range).fillna(0).rename_axis('Date').reset_index()

# -----------------------------
# 6. 7-Tage-Durchschnitt berechnen
# -----------------------------
df['protests_7d'] = df['Protests'].rolling(window=7).mean()
df['participants_7d'] = df['participants'].rolling(window=7).mean()
df['log_participants_7d'] = df['log_participants'].rolling(window=7).mean()

# =============================================
# 7. Plot 1: Tageszeitreihe Proteste
# =============================================
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Protests'], color='lightgray', label='Tägliche Proteste')
plt.plot(df['Date'], df['protests_7d'], color='darkblue', linewidth=2, label='7-Tage-Durchschnitt')
plt.xlabel('Zeit')
plt.ylabel('Anzahl Proteste')
plt.title('Zeitliche Entwicklung der Protestaktivität')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================
# 8. Plot 2: Teilnehmerzahlen
# =============================================
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['participants'], color='lightgray', label='Tägliche Teilnehmerzahl')
plt.plot(df['Date'], df['participants_7d'], color='darkgreen', linewidth=2, label='7-Tage-Durchschnitt')
plt.xlabel('Zeit')
plt.ylabel('Teilnehmerzahl')
plt.title('Zeitliche Entwicklung der Protestintensität')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================
# 9. Plot 3: Logarithmierte Teilnehmerzahlen
# =============================================
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['log_participants'], color='lightgray', label='Log(Teilnehmer)')
plt.plot(df['Date'], df['log_participants_7d'], color='purple', linewidth=2, label='7-Tage-Durchschnitt')
plt.xlabel('Zeit')
plt.ylabel('Logarithmierte Teilnehmerzahl')
plt.title('Zeitliche Entwicklung der Protestintensität (log)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================
# 10. Wochenaggregation (optional für Robustheit)
# =============================================
weekly = df.set_index('Date').resample('W').sum()

plt.figure(figsize=(12,6))
plt.plot(weekly.index, weekly['Protests'], color='darkred', marker='o', label='Wöchentliche Proteste')
plt.xlabel('Zeit')
plt.ylabel('Anzahl Proteste')
plt.title('Wöchentliche Protestaktivität')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

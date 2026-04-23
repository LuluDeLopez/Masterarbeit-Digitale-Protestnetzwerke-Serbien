import pandas as pd
import re
import numpy as np

# -----------------------------
# 1️⃣ Funktionen
# -----------------------------

# Funktion zum Zählen aktiver Instagram-Accounts
def count_active_accounts(row, accounts_df):
    city_accounts = accounts_df[accounts_df["ORT"] == row["Location"]]
    active = city_accounts[
        (city_accounts["ERSTER POST"] <= row["Date"]) &
        (city_accounts["LETZTER POST"] >= row["Date"])
    ]
    return len(active)

# Funktion zur Umwandlung von "tags" in numerische Teilnehmerzahl
def parse_participants(text):
    if pd.isna(text):
        return None
    text = str(text).lower()
    
    # Sonderfälle textuell
    if "between several hundred and a thousand" in text:
        return (300 + 1000)/2
    if "tens of thousands" in text:
        return 20000
    if "several tens of thousands" in text:
        return 30000
    if "several thousand" in text:
        return 3000
    if "thousands" in text:
        return 2000
    if "more than thousand" in text or "more than a thousand" in text:
        return 1000
    if "several hundred" in text:
        return 300
    if "hundreds" in text:
        return 200
    if "several tens" in text:
        return 30
    if "dozens" in text:
        return 24
    if "more than a hundred" in text:
        return 100
    if "at least several hundred" in text:
        return 300

    # Bereich: between X and Y
    m = re.search(r'between ([\d\.,]+) and ([\d\.,]+)', text)
    if m:
        a = float(m.group(1).replace(".", "").replace(",", ""))
        b = float(m.group(2).replace(".", "").replace(",", ""))
        return (a + b)/2

    # direkte Zahlen erkennen
    m = re.search(r'([\d\.,]+)', text)
    if m:
        return float(m.group(1).replace(".", "").replace(",", ""))

    # kleine ausgeschriebene Zahlen
    words = {
        "one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10
    }
    for w,n in words.items():
        if w in text:
            return n

    return None

# -----------------------------
# 2️⃣ Daten laden
# -----------------------------
# Zeitreihe-Tabelle
timeseries = pd.read_excel("Zeitreihe-Tage.xlsx")

# Instagram-Accounts
accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Proteste
protests = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Proteste")

# Datumsformate
timeseries["Date"] = pd.to_datetime(timeseries["Date"])
accounts["ERSTER POST"] = pd.to_datetime(accounts["ERSTER POST"])
accounts["LETZTER POST"] = pd.to_datetime(accounts["LETZTER POST"])
protests["event_date"] = pd.to_datetime(protests["event_date"])

# fehlende LETZTER POST → Enddatum setzen
accounts["LETZTER POST"] = accounts["LETZTER POST"].fillna(pd.Timestamp("2025-12-31"))

# fehlende Posts → 0
timeseries["Posts"] = timeseries["Posts"].fillna(0)

# -----------------------------
# 3️⃣ Active Instagram Accounts
# -----------------------------
timeseries["Active_Accounts"] = timeseries.apply(lambda row: count_active_accounts(row, accounts), axis=1)

# -----------------------------
# 4️⃣ Protestanzahl pro Stadt/Tag
# -----------------------------
protest_counts = protests.groupby(["location","event_date"]).size().reset_index(name="Protests")

timeseries = timeseries.merge(
    protest_counts,
    left_on=["Location","Date"],
    right_on=["location","event_date"],
    how="left"
)

timeseries["Protests"] = timeseries["Protests"].fillna(0).astype(int)
timeseries = timeseries.drop(columns=["location","event_date"])

# -----------------------------
# 5️⃣ Crowd Size aus tags extrahieren
# -----------------------------
protests["participants"] = protests["tags"].apply(parse_participants)

participants_day = protests.groupby(["location","event_date"])["participants"].sum().reset_index()

timeseries = timeseries.merge(
    participants_day,
    left_on=["Location","Date"],
    right_on=["location","event_date"],
    how="left"
)

timeseries["participants"] = timeseries["participants"].fillna(0)
timeseries = timeseries.drop(columns=["location","event_date"])

# -----------------------------
# 6️⃣ Optional: log(Participants+1) erstellen
# -----------------------------
timeseries["log_participants"] = np.log1p(timeseries["participants"])

# -----------------------------
# 7️⃣ Ergebnis speichern
# -----------------------------
timeseries.to_excel("Zeitreihe_final.xlsx", index=False)

print("Fertig! Zeitreihe_final.xlsx wurde erstellt.")


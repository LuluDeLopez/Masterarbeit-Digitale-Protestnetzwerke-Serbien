import pandas as pd

# Dateien laden
timeseries = pd.read_excel("Zeitreihe.xlsx", sheet_name="Tabellenblatt2")  
accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Instas")

# Datumsformat sicherstellen
timeseries["Date"] = pd.to_datetime(timeseries["Date"])
accounts["ERSTER POST"] = pd.to_datetime(accounts["ERSTER POST"])
accounts["LETZTER POST"] = pd.to_datetime(accounts["LETZTER POST"])

# Funktion: aktive Accounts zählen
def count_active(row):
    city_accounts = accounts[accounts["ORT"] == row["Location"]]
    
    active = city_accounts[
        (city_accounts["ERSTER POST"] <= row["Date"]) &
        (city_accounts["LETZTER POST"] >= row["Date"])
    ]
    
    return len(active)

# neue Spalte berechnen
timeseries["Active_Accounts"] = timeseries.apply(count_active, axis=1)

# speichern
timeseries.to_excel("timeseries_with_accounts.xlsx", index=False)

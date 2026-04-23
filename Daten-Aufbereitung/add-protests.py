import pandas as pd

# Zeitreihen-Datensatz laden
timeseries = pd.read_excel("Zeitreihe-Tage.xlsx")

# Protest-Datensatz laden (Tabellenblatt "Proteste")
protests = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Proteste")

# Datumsformat vereinheitlichen
timeseries["Date"] = pd.to_datetime(timeseries["Date"])
protests["event_date"] = pd.to_datetime(protests["event_date"])

# Proteste pro Stadt und Tag zählen
protest_counts = (
    protests
    .groupby(["location", "event_date"])
    .size()
    .reset_index(name="Protests")
)

# Merge mit Zeitreihe
timeseries = timeseries.merge(
    protest_counts,
    left_on=["Location", "Date"],
    right_on=["location", "event_date"],
    how="left"
)

# fehlende Werte = 0 Proteste
timeseries["Protests"] = timeseries["Protests"].fillna(0).astype(int)

# Hilfsspalten entfernen
timeseries = timeseries.drop(columns=["location", "event_date"])

# speichern
timeseries.to_excel("Zeitreihe-Tage_mit_Protesten.xlsx", index=False)

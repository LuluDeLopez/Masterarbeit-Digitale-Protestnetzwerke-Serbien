import pandas as pd

# Zeitreihe laden
timeseries = pd.read_excel("Zeitreihe-Tage_mit_Protesten.xlsx")

# Protestdatensatz laden
protests = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Proteste")

# Datum konvertieren
timeseries["Date"] = pd.to_datetime(timeseries["Date"])
protests["event_date"] = pd.to_datetime(protests["event_date"])

# -----------------------------
# 1. Fehlende Posts = 0
# -----------------------------
timeseries["Posts"] = timeseries["Posts"].fillna(0)

# -----------------------------
# 2. Teilnehmerzahl aus tags extrahieren
# -----------------------------
protests["participants"] = (
    protests["tags"]
    .astype(str)
    .str.extract(r"(\d+)")   # erste Zahl im Text
    .astype(float)
)

# -----------------------------
# 3. Teilnehmer pro Stadt/Tag aggregieren
# -----------------------------
participants_day = (
    protests
    .groupby(["location", "event_date"])["participants"]
    .sum()
    .reset_index()
)

# -----------------------------
# 4. In Zeitreihe mergen
# -----------------------------
timeseries = timeseries.merge(
    participants_day,
    left_on=["Location", "Date"],
    right_on=["location", "event_date"],
    how="left"
)

# fehlende Werte → 0
timeseries["participants"] = timeseries["participants"].fillna(0)

# Hilfsspalten entfernen
timeseries = timeseries.drop(columns=["location", "event_date"])

# speichern
timeseries.to_excel("Zeitreihe_final.xlsx", index=False)

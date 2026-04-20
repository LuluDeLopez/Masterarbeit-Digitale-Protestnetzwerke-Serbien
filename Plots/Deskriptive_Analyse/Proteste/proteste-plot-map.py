import pandas as pd
import folium

# Excel laden
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Proteste")

# Komma in Dezimalpunkt umwandeln und in float konvertieren
df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.').astype(float)
df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.').astype(float)

# Fehlende Koordinaten entfernen
df = df.dropna(subset=['Latitude', 'Longitude', 'Anzahl Proteste'])

# Karte erstellen
karte = folium.Map(location=[44.0, 21.0], zoom_start=6, tiles='cartodbpositron')

for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=max(3, row['Anzahl Proteste'] / 10),
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"{row['Ort']}: {row['Anzahl Proteste']} Proteste"
    ).add_to(karte)

karte.save("Proteste_Serbien.html")
print("Karte erstellt! Öffne Proteste_Serbien.html im Browser.")

import pandas as pd
import folium

# 1. Excel-Datei laden
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Proteste")

# Komma in Dezimalpunkt umwandeln, falls nötig
df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.').astype(float)
df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.').astype(float)

# Fehlende Koordinaten entfernen
df = df.dropna(subset=['Latitude', 'Longitude', 'Anzahl Pro', 'Anzahl Anti'])

# 2. Basis-Karte Serbien
karte = folium.Map(location=[44.0, 21.0], zoom_start=6, tiles='cartodbpositron')

# 3. Marker pro Ort hinzufügen (Pro = Blau, Anti = Rot)
for idx, row in df.iterrows():
    # Pro-Posts (blau)
    if row['Anzahl Pro'] > 0:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(3, row['Anzahl Pro'] / 10),
            color='blue',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"{row['Ort']}: {row['Anzahl Pro']} Pro-Proteste"
        ).add_to(karte)
    
    # Anti-Posts (rot)
    if row['Anzahl Anti'] > 0:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(3, row['Anzahl Anti'] / 10),
            color='blue',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"{row['Ort']}: {row['Anzahl Anti']} Anti-Proteste"
        ).add_to(karte)

# 4. Karte speichern
karte.save("Pro_Anti_Serbien.html")
print("Karte erstellt! Öffne Pro_Anti_Serbien.html im Browser.")

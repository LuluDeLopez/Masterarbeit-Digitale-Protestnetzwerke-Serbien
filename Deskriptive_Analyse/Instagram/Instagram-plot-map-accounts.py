import pandas as pd
import folium

# -----------------------------
# 1. Excel-Dateien laden
# -----------------------------
df_accounts = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Instas")
df_coords   = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Proteste")

# Spaltennamen angleichen
df_accounts.rename(columns={'ORT': 'Ort'}, inplace=True)

# -----------------------------
# 2. Latitude/Longitude bereinigen
# -----------------------------
df_coords['Latitude'] = df_coords['Latitude'].astype(str).str.replace(',', '.').astype(float)
df_coords['Longitude'] = df_coords['Longitude'].astype(str).str.replace(',', '.').astype(float)

# -----------------------------
# 3. Tabellen zusammenführen
# -----------------------------
df = pd.merge(df_coords, df_accounts, on='Ort', how='inner')
df = df.dropna(subset=['Latitude', 'Longitude', 'n_accounts', 'share_pro'])

# -----------------------------
# 4. Basis-Karte Serbien
# -----------------------------
karte = folium.Map(location=[44.0, 21.0], zoom_start=6, tiles='cartodbpositron')

# -----------------------------
# 5. Hilfsfunktion: Radius skalieren (Min-Max)
# -----------------------------
min_radius = 3
max_radius = 20

def scale_radius(val, min_val, max_val, min_r=min_radius, max_r=max_radius):
    if max_val == min_val:
        return (min_r + max_r) / 2
    return min_r + (val - min_val) / (max_val - min_val) * (max_r - min_r)

# Min/Max Accounts für Skalierung bestimmen
min_accounts = df['n_accounts'].min()
max_accounts = df['n_accounts'].max()

# -----------------------------
# 6. Marker pro Ort hinzufügen
# -----------------------------
for idx, row in df.iterrows():
    # Pro- und Anti-Accounts berechnen
    pro_accounts  = row['n_accounts'] * row['share_pro']
    anti_accounts = row['n_accounts'] - pro_accounts
    
    # Radius skalieren
    pro_radius  = scale_radius(pro_accounts, min_accounts, max_accounts)
    anti_radius = scale_radius(anti_accounts, min_accounts, max_accounts)
    
    # Leichter Offset, damit Marker nicht exakt überlappen
    lat_pro = row['Latitude'] + 0.005
    lat_anti = row['Latitude'] - 0.005
    lon = row['Longitude']
    
    # Pro-Accounts (blau)
    if pro_accounts > 0:
        folium.CircleMarker(
            location=[lat_pro, lon],
            radius=pro_radius,
            color=None,
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"{row['Ort']}: {pro_accounts:.0f} Pro-Accounts"
        ).add_to(karte)
    
    # Anti-Accounts (rot)
    if anti_accounts > 0:
        folium.CircleMarker(
            location=[lat_anti, lon],
            radius=anti_radius,
            color=None,
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"{row['Ort']}: {anti_accounts:.0f} Anti-Accounts"
        ).add_to(karte)

# -----------------------------
# 7. Karte speichern
# -----------------------------
karte.save("Accounts_Pro_Anti_Serbien.html")
print("Karte erstellt! Öffne Accounts_Pro_Anti_Serbien.html im Browser.")

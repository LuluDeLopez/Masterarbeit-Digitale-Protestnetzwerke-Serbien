import pandas as pd
import matplotlib.pyplot as plt

# Excel laden
df = pd.read_excel("Datensatz-Master-Final.xlsx", sheet_name="Stadt_Proteste")
print(df.columns)
# -----------------------------
# Funktion: Pro vs Anti vergleichen
# -----------------------------
def plot_compare(df, col_pro, col_anti, title):
    
    # Gesamtwert zur Sortierung (Pro + Anti)
    df['total'] = df[col_pro] + df[col_anti]
    
    df_sorted = df.sort_values(by='total', ascending=False)
    
    top5 = df_sorted.head(5)
    bottom5 = df_sorted.tail(5)
    
    combined = pd.concat([top5, bottom5])

    x = range(len(combined))

    plt.figure(figsize=(12,6))
    
    # Balken
    plt.bar(x, combined[col_pro], width=0.4, label='Pro', color='blue')
    plt.bar([i+0.4 for i in x], combined[col_anti], width=0.4, label='Anti', color='red')

    # Achsen & Layout
    plt.xticks([i+0.2 for i in x], combined['Ort'], rotation=45, ha='right')
    plt.ylabel("Wert")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 1. Protestanzahl vergleichen
# -----------------------------
plot_compare(
    df.copy(),
    "Anzahl Pro",
    "Anzahl Anti",
    "Pro vs. Anti Proteste (Top & Bottom 5 Orte)"
)

# -----------------------------
# 2. Teilnehmerzahlen vergleichen
# -----------------------------
plot_compare(
    df.copy(),
    "Crowd Durchschnitt Pro",
    "Crowd Durchschnitt Anti",
    "Pro vs. Anti Teilnehmer (Top & Bottom 5 Orte)"
)

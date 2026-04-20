import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"

# Hauptdaten
sheet_name = "Lag-Tag"
df = pd.read_excel(file_path, sheet_name=sheet_name)
df = df.fillna(0)
df["Date"] = pd.to_datetime(df["Date"])

# Kontrollvariablen laden
df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")

# Spalte "Ort" in "Location" umbenennen für den Merge
df_controls = df_controls.rename(columns={"Ort": "Location"})

# Merge
df = df.merge(df_controls[["Location","Einwohner","Wahl Opposition"]],
              on="Location", how="left")

# Fehlende Werte auffüllen
df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

# Zeitspalten
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

df = df.sort_values(["Location", "Date"])

# =========================
# 2. Lagged Variablen erstellen
# =========================
lag_vars = ["Protests","participants","Posts","Active_Accounts"]
for var in lag_vars:
    df[f"{var}_lag1"] = df.groupby("Location")[var].shift(1).fillna(0)

# =========================
# 3. Aggregationen vorbereiten
# =========================
# Tag
aggregations = [("Daily", df)]

# Woche
df_week = df.groupby(["Location", "week"]).agg({
    "Posts": "sum",
    "Active_Accounts": "sum",
    "Protests": "sum",
    "participants": "sum",
    "Einwohner": "first",
    "Wahl Opposition": "first"
}).reset_index()

for var in lag_vars:
    df_week[f"{var}_lag1"] = df_week.groupby("Location")[var].shift(1).fillna(0)
aggregations.append(("Weekly", df_week))

# Monat
df_month = df.groupby(["Location", "month"]).agg({
    "Posts": "sum",
    "Active_Accounts": "sum",
    "Protests": "sum",
    "participants": "sum",
    "Einwohner": "first",
    "Wahl Opposition": "first"
}).reset_index()

for var in lag_vars:
    df_month[f"{var}_lag1"] = df_month.groupby("Location")[var].shift(1).fillna(0)
aggregations.append(("Monthly", df_month))

# =========================
# 4. Überdispersion prüfen
# =========================
dispersion_data = []
dvs = ["Posts","Active_Accounts","Protests","participants"]
for label, df_agg in aggregations:
    for dv in dvs:
        mean_val = df_agg[dv].mean()
        var_val = df_agg[dv].var()
        dispersion_data.append({
            "DV": dv,
            "Aggregation": label,
            "Mean": mean_val,
            "Variance": var_val,
            "Var/Mean": var_val/mean_val
        })

disp_df = pd.DataFrame(dispersion_data).round(2)
print("\n===== Überdispersion der Zählvariablen =====")
print(disp_df.to_string(index=False))



# Für Tages-, Wochen- und Monats-Fixed Effects Kategorie-Spalten erstellen
df["Date_cat"] = df["Date"].dt.strftime("%Y-%m-%d")  # Tages-FE als String
df_week["week_cat"] = df_week["week"].dt.strftime("%Y-%m-%d")  # Wochen-FE
df_month["month_cat"] = df_month["month"].dt.strftime("%Y-%m")  # Monats-FE

# =========================
# 5. Modellfunktion mit umgedrehter Richtung
# =========================
def run_model_insta_to_protest_nb(data, time_var, dv, label):
    """
    Negative-Binomial-Modell: Protestanzahl/Teilnehmer ~ Instagram (Posts + Active Accounts)
    time_var: kategoriale Zeiteffekte (Date_cat, week_cat, month_cat)
    dv: abhängige Variable ("Protests" oder "participants")
    """
    formula = f"{dv} ~ Posts_lag1 + Active_Accounts_lag1 + C(Location) + C({time_var})"
    model = smf.glm(formula=formula, data=data,
                    family=sm.families.NegativeBinomial()).fit()
    results = pd.DataFrame({
        "variable": ["Posts_lag1","Active_Accounts_lag1"],
        "coef": model.params[["Posts_lag1","Active_Accounts_lag1"]],
        "std_err": model.bse[["Posts_lag1","Active_Accounts_lag1"]],
        "dv": dv,
        "model": label
    })
    return results, model

# =========================
# 6. Negative-Binomial-Modelle ausführen (umgedrehte Richtung)
# =========================
res_day_protests, _ = run_model_insta_to_protest_nb(df,"Date_cat","Protests","Daily")
res_day_participants, _ = run_model_insta_to_protest_nb(df,"Date_cat","participants","Daily")

res_week_protests, _ = run_model_insta_to_protest_nb(df_week,"week_cat","Protests","Weekly")
res_week_participants, _ = run_model_insta_to_protest_nb(df_week,"week_cat","participants","Weekly")

res_month_protests, _ = run_model_insta_to_protest_nb(df_month,"month_cat","Protests","Monthly")
res_month_participants, _ = run_model_insta_to_protest_nb(df_month,"month_cat","participants","Monthly")

# Ergebnisse zusammenführen
results_nb = pd.concat([
    res_day_protests,res_day_participants,
    res_week_protests,res_week_participants,
    res_month_protests,res_month_participants
])

# Effekte in Prozent
results_nb["effect_pct"] = (np.exp(results_nb["coef"])-1)*100
results_nb["ci_low"] = (np.exp(results_nb["coef"]-1.96*results_nb["std_err"])-1)*100
results_nb["ci_high"] = (np.exp(results_nb["coef"]+1.96*results_nb["std_err"])-1)*100

# =========================
# 7. Plots & Tabellen bleiben gleich
# =========================
time_levels = ["Daily","Weekly","Monthly"]
variables = ["Posts_lag1","Active_Accounts_lag1"]
dvs = ["Protests","participants"]
colors = {"Protests":"tab:red","participants":"tab:green"}
markers = {"Posts_lag1":"o","Active_Accounts_lag1":"s"}

plt.figure(figsize=(12,6))
for i, model in enumerate(time_levels):
    df_plot = results_nb[results_nb["model"]==model]
    for j, variable in enumerate(variables):
        for k, dv in enumerate(dvs):
            row = df_plot[(df_plot["variable"]==variable)&(df_plot["dv"]==dv)]
            if not row.empty:
                x = i + (j*2 + k - 1.5)*0.15
                y = row["effect_pct"].values[0]
                yerr_low = y - row["ci_low"].values[0]
                yerr_high = row["ci_high"].values[0] - y
                plt.errorbar(x, y,
                             yerr=[[yerr_low],[yerr_high]],
                             fmt=markers[variable],
                             color=colors[dv],
                             capsize=5,
                             markersize=8,
                             label=f"{variable} × {dv}")

plt.xticks(ticks=range(len(time_levels)), labels=["Tag","Woche","Monat"])
plt.axhline(0,color="grey",linestyle="--")
plt.ylabel("Effekt auf Protest (%)")
plt.xlabel("Zeitebene")
plt.title("Negative-Binomial-Modelle: Instagram (t-1) → Protestaktivität")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(),by_label.keys(),title="Variable × DV", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()



# =========================
# Funktion: Terminal-Ausgabe der NB-Ergebnisse
# =========================
def print_nb_results_table(df_results, title):
    df_print = df_results[["variable","dv","effect_pct","ci_low","ci_high"]].copy().round(2)
    print(f"\n===== {title} =====")
    print(df_print.to_string(index=False))

# Funktion: Farbiges Plotten der NB-Ergebnisse
def plot_colored_results_table(df_results, title):
    df_plot = df_results[["variable","dv","effect_pct","ci_low","ci_high"]].copy().round(2)
    row_colors = []
    for val in df_plot["effect_pct"]:
        if val > 0:
            row_colors.append([0.8,1.0,0.8])  # grün = positiver Effekt
        elif val < 0:
            row_colors.append([1.0,0.8,0.8])  # rot = negativer Effekt
        else:
            row_colors.append([1.0,1.0,1.0])  # weiß = kein Effekt
    cell_colors = [[c for _ in range(df_plot.shape[1])] for c in row_colors]

    fig, ax = plt.subplots(figsize=(8, df_plot.shape[0]*0.5+1))
    ax.axis("off")
    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(list(range(df_plot.shape[1])))
    plt.title(title, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

# Terminal-Tabellen
for agg in ["Daily","Weekly","Monthly"]:
    df_agg = results_nb[results_nb["model"]==agg]
    print_nb_results_table(df_agg, f"{agg} Aggregation: Instagram → Proteste")
    plot_colored_results_table(df_agg, f"{agg} Aggregation: Instagram → Proteste")




# =========================
# Terminal-Ausgabe der sequenziellen Ergebnisse
# =========================
def print_feedback_results_table(df_results, title):
    df_print = df_results[["label","step1_coef","step2_coef","indirect_effect"]].copy().round(3)
    print(f"\n===== {title} =====")
    print(df_print.to_string(index=False))

# =========================
# Farbiges Plotten der sequenziellen Ergebnisse
# =========================
def plot_colored_feedback_table(df_results, title):
    df_plot = df_results[["label","step1_coef","step2_coef","indirect_effect"]].copy().round(3)
    
    # Farblogik: grün=positiv, rot=negativ, weiß=0
    row_colors = []
    for val in df_plot["indirect_effect"]:
        if val > 0:
            row_colors.append([0.8,1.0,0.8])  # grün = positiver Effekt
        elif val < 0:
            row_colors.append([1.0,0.8,0.8])  # rot = negativer Effekt
        else:
            row_colors.append([1.0,1.0,1.0])  # weiß = kein Effekt
    cell_colors = [[c for _ in range(df_plot.shape[1])] for c in row_colors]

    fig, ax = plt.subplots(figsize=(8, df_plot.shape[0]*0.6+1))
    ax.axis("off")
    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(list(range(df_plot.shape[1])))
    plt.title(title, fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

# =========================
# H3a: Offline initiiert
# =========================
df_h3a = results_df[results_df["label"].str.contains("H3a")]
print_feedback_results_table(df_h3a, "H3a: Offline initiiert (Protest → Instagram → Protest)")
plot_colored_feedback_table(df_h3a, "H3a: Offline initiiert (Protest → Instagram → Protest)")

# =========================
# H3b: Online initiiert
# =========================
df_h3b = results_df[results_df["label"].str.contains("H3b")]
print_feedback_results_table(df_h3b, "H3b: Online initiiert (Instagram → Protest → Instagram)")
plot_colored_feedback_table(df_h3b, "H3b: Online initiiert (Instagram → Protest → Instagram)")




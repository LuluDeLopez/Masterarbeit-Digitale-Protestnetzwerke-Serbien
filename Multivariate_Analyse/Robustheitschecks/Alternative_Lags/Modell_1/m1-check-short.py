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
# 5. Modellfunktion mit Kontrollvariablen und lagged DV
# =========================
def run_model_protest_to_insta_nb(data, time_var, dv, label):
    # time_var sollte jetzt die Kategorie-Spalte sein: Date_cat, week_cat, month_cat
    formula = f"{dv} ~ Protests_lag1 + participants_lag1 + C(Location) + C({time_var})"
    model = smf.glm(formula=formula, data=data,
                    family=sm.families.NegativeBinomial()).fit()
    results = pd.DataFrame({
        "variable": ["Protests_lag1","participants_lag1"],
        "coef": model.params[["Protests_lag1","participants_lag1"]],
        "std_err": model.bse[["Protests_lag1","participants_lag1"]],
        "dv": dv,
        "model": label
    })
    return results, model

# =========================
# 6. Negative-Binomial-Modelle ausführen
# =========================
res_day_posts, _ = run_model_protest_to_insta_nb(df,"Date_cat","Posts","Daily")
res_day_accounts, _ = run_model_protest_to_insta_nb(df,"Date_cat","Active_Accounts","Daily")

res_week_posts, _ = run_model_protest_to_insta_nb(df_week,"week_cat","Posts","Weekly")
res_week_accounts, _ = run_model_protest_to_insta_nb(df_week,"week_cat","Active_Accounts","Weekly")

res_month_posts, _ = run_model_protest_to_insta_nb(df_month,"month_cat","Posts","Monthly")
res_month_accounts, _ = run_model_protest_to_insta_nb(df_month,"month_cat","Active_Accounts","Monthly")

results_nb = pd.concat([
    res_day_posts,res_day_accounts,
    res_week_posts,res_week_accounts,
    res_month_posts,res_month_accounts
])

# Effekte in Prozent
results_nb["effect_pct"] = (np.exp(results_nb["coef"])-1)*100
results_nb["ci_low"] = (np.exp(results_nb["coef"]-1.96*results_nb["std_err"])-1)*100
results_nb["ci_high"] = (np.exp(results_nb["coef"]+1.96*results_nb["std_err"])-1)*100

# =========================
# 7. Ergebnisse in Excel speichern
# =========================
output_file = "Model_Results.xlsx"

# Schreiben der Überdispersionstabelle und der Modell-Ergebnisse in verschiedene Sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_nb.to_excel(writer, sheet_name="NB_Model_Results", index=False)

print(f"Ergebnisse wurden in '{output_file}' gespeichert.")

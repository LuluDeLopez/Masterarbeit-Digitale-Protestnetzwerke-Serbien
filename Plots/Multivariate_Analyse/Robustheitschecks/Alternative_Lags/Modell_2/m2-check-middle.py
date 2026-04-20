import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"

df = pd.read_excel(file_path, sheet_name="Lag-Tag")
df = df.fillna(0)
df["Date"] = pd.to_datetime(df["Date"])

# Kontrollvariablen
df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")
df_controls = df_controls.rename(columns={"Ort": "Location"})

df = df.merge(
    df_controls[["Location", "Einwohner", "Wahl Opposition"]],
    on="Location",
    how="left"
)

df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

# Zeitvariablen
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

df = df.sort_values(["Location", "Date"])

# =========================
# 2. Lagged Variablen (Lag 2)
# =========================
lag_vars = ["Protests","participants","Posts","Active_Accounts"]
for var in lag_vars:
    df[f"{var}_lag2"] = df.groupby("Location")[var].shift(2).fillna(0)

# =========================
# 3. Aggregationen
# =========================
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
    df_week[f"{var}_lag2"] = df_week.groupby("Location")[var].shift(2).fillna(0)

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
    df_month[f"{var}_lag2"] = df_month.groupby("Location")[var].shift(2).fillna(0)

# =========================
# 4. Fixed Effects
# =========================
df["Date_cat"] = df["Date"].dt.strftime("%Y-%m-%d")
df_week["week_cat"] = df_week["week"].dt.strftime("%Y-%m-%d")
df_month["month_cat"] = df_month["month"].dt.strftime("%Y-%m")

# =========================
# 5. Modellfunktion (M2 Lag 2)
# =========================
def run_model_insta_to_protest_nb(data, time_var, dv, label):
    formula = f"{dv} ~ Posts_lag2 + Active_Accounts_lag2 + C(Location) + C({time_var})"
    
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.NegativeBinomial()
    ).fit()
    
    results = pd.DataFrame({
        "variable": ["Posts_lag2","Active_Accounts_lag2"],
        "coef": model.params[["Posts_lag2","Active_Accounts_lag2"]],
        "std_err": model.bse[["Posts_lag2","Active_Accounts_lag2"]],
        "dv": dv,
        "model": label
    })
    
    return results

# =========================
# 6. Modelle ausführen
# =========================
results_nb = pd.concat([
    run_model_insta_to_protest_nb(df,"Date_cat","Protests","Daily"),
    run_model_insta_to_protest_nb(df,"Date_cat","participants","Daily"),
    run_model_insta_to_protest_nb(df_week,"week_cat","Protests","Weekly"),
    run_model_insta_to_protest_nb(df_week,"week_cat","participants","Weekly"),
    run_model_insta_to_protest_nb(df_month,"month_cat","Protests","Monthly"),
    run_model_insta_to_protest_nb(df_month,"month_cat","participants","Monthly")
])

# =========================
# 7. Effekte berechnen
# =========================
results_nb["effect_pct"] = (np.exp(results_nb["coef"]) - 1) * 100
results_nb["ci_low"] = (np.exp(results_nb["coef"] - 1.96 * results_nb["std_err"]) - 1) * 100
results_nb["ci_high"] = (np.exp(results_nb["coef"] + 1.96 * results_nb["std_err"]) - 1) * 100

# =========================
# 8. Labels
# =========================
results_nb["Variable"] = results_nb["variable"].replace({
    "Posts_lag2": "Posts",
    "Active_Accounts_lag2": "Active Accounts"
})

results_nb["DV"] = results_nb["dv"]
results_nb["Aggregation"] = results_nb["model"]

results_nb["label"] = (
    results_nb["Aggregation"] + " | " +
    results_nb["DV"] + " | " +
    results_nb["Variable"]
)

# Signifikanz
results_nb["sig"] = np.where(
    (results_nb["ci_low"] > 0) | (results_nb["ci_high"] < 0),
    "*",
    ""
)

# Formatierte Spalte
results_nb["Effekt (%)"] = results_nb.apply(
    lambda x: f"{x['effect_pct']:.2f}{x['sig']} ({x['ci_low']:.2f}, {x['ci_high']:.2f})",
    axis=1
)

# Finale Tabelle
final_table = results_nb[[
    "label",
    "Aggregation",
    "DV",
    "Variable",
    "effect_pct",
    "ci_low",
    "ci_high",
    "Effekt (%)"
]].round(2)

# =========================
# 9. Excel Export
# =========================
output_file = "Model_Results_M2_Lag2.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    final_table.to_excel(writer, sheet_name="NB_Model_Results_M2_Lag2", index=False)

print(f"Ergebnisse wurden in '{output_file}' gespeichert.")

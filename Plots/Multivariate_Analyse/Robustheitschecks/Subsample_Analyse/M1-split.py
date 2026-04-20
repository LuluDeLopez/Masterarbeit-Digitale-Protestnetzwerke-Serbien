import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"

df = pd.read_excel(file_path, sheet_name="Lag-Tag").fillna(0)
df["Date"] = pd.to_datetime(df["Date"])

df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")
df_controls = df_controls.rename(columns={"Ort": "Location"})

df = df.merge(df_controls[["Location","Einwohner","Wahl Opposition"]],
              on="Location", how="left")

df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

df = df.sort_values(["Location","Date"])

# =========================
# 2. ZEIT-SPLIT
# =========================
cutoff_date = df["Date"].median()

df_first = df[df["Date"] <= cutoff_date].copy()
df_second = df[df["Date"] > cutoff_date].copy()

# =========================
# 3. FUNKTION: PREPARE DATA
# =========================
def prepare_data(df_input):

    df_input = df_input.copy()

    # Zeitvariablen
    df_input["week"] = df_input["Date"].dt.to_period("W").dt.start_time
    df_input["month"] = df_input["Date"].dt.to_period("M").dt.start_time

    # Lags
    lag_vars = ["Protests","participants","Posts","Active_Accounts"]
    for var in lag_vars:
        df_input[f"{var}_lag1"] = df_input.groupby("Location")[var].shift(1).fillna(0)

    # Aggregationen
    df_week = df_input.groupby(["Location","week"]).agg({
        "Posts":"sum","Active_Accounts":"sum","Protests":"sum","participants":"sum",
        "Einwohner":"first","Wahl Opposition":"first"
    }).reset_index()

    for var in lag_vars:
        df_week[f"{var}_lag1"] = df_week.groupby("Location")[var].shift(1).fillna(0)

    df_month = df_input.groupby(["Location","month"]).agg({
        "Posts":"sum","Active_Accounts":"sum","Protests":"sum","participants":"sum",
        "Einwohner":"first","Wahl Opposition":"first"
    }).reset_index()

    for var in lag_vars:
        df_month[f"{var}_lag1"] = df_month.groupby("Location")[var].shift(1).fillna(0)

    # FE Kategorien
    df_input["Date_cat"] = df_input["Date"].dt.strftime("%Y-%m-%d")
    df_week["week_cat"] = df_week["week"].dt.strftime("%Y-%m-%d")
    df_month["month_cat"] = df_month["month"].dt.strftime("%Y-%m")

    return {
        "Daily": df_input,
        "Weekly": df_week,
        "Monthly": df_month
    }

# =========================
# 4. MODELLFUNKTION
# =========================
def run_model(data, time_var, dv, label):

    formula = f"{dv} ~ Protests_lag1 + participants_lag1 + C(Location) + C({time_var})"

    model = smf.glm(formula=formula, data=data,
                    family=sm.families.NegativeBinomial()).fit()

    res = pd.DataFrame({
        "variable": ["Protests_lag1","participants_lag1"],
        "coef": model.params[["Protests_lag1","participants_lag1"]],
        "std_err": model.bse[["Protests_lag1","participants_lag1"]],
        "dv": dv,
        "model": label
    })

    return res

# =========================
# 5. RUN PIPELINE
# =========================
def run_all(df_input, sample_name):

    datasets = prepare_data(df_input)
    results = []

    for level, data in datasets.items():

        if level == "Daily":
            time_var = "Date_cat"
        elif level == "Weekly":
            time_var = "week_cat"
        else:
            time_var = "month_cat"

        for dv in ["Posts","Active_Accounts"]:
            res = run_model(
                data,
                time_var,
                dv,
                f"{sample_name}-{level}-{dv}"
            )
            results.append(res)

    return pd.concat(results)

# =========================
# 6. MODELLE AUSFÜHREN
# =========================
results_first = run_all(df_first, "FirstHalf")
results_second = run_all(df_second, "SecondHalf")

results_nb = pd.concat([results_first, results_second])

# Effekte in %
results_nb["effect_pct"] = (np.exp(results_nb["coef"]) - 1) * 100
results_nb["ci_low"] = (np.exp(results_nb["coef"] - 1.96*results_nb["std_err"]) - 1) * 100
results_nb["ci_high"] = (np.exp(results_nb["coef"] + 1.96*results_nb["std_err"]) - 1) * 100

# =========================
# 7. SPEICHERN
# =========================
output_file = "Model_Results_split.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_nb.to_excel(writer, sheet_name="NB_Results_Split", index=False)

print("Fertig! Datei gespeichert:", output_file)

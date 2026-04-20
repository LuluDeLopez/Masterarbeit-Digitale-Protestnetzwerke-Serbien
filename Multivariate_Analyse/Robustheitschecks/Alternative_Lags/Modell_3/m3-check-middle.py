import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Zeitreihe-Tag")
df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")

df_raw = df_raw.fillna(0)
df_controls = df_controls.rename(columns={"Ort": "Location"})
df = df_raw.merge(
    df_controls[["Location", "Einwohner", "Wahl Opposition"]],
    on="Location",
    how="left"
)
df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

# =========================
# 2. LAGS ERSTELLEN FÜR MIDDLE
# =========================
lag_vars = ["Protests","Posts","participants","Active_Accounts"]
for var in lag_vars:
    # alle benötigten Lags erstellen
    df[f"{var}_lag1"] = df.groupby("Location")[var].shift(1)
    df[f"{var}_lag2"] = df.groupby("Location")[var].shift(2)
    df[f"{var}_lag4"] = df.groupby("Location")[var].shift(4)

# Log-Transformationen
for var in lag_vars:
    df[f"{var}_log"] = np.log1p(df[var])
    df[f"{var}_log_lag1"] = np.log1p(df[f"{var}_lag1"])
    df[f"{var}_log_lag2"] = np.log1p(df[f"{var}_lag2"])
    df[f"{var}_log_lag4"] = np.log1p(df[f"{var}_lag4"])

df = df.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# 3. ZEIT-AGGREGATION
# =========================
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

df_week = df.groupby(["Location","week"]).agg({
    "Posts":"sum","Active_Accounts":"sum","Protests":"sum","participants":"sum",
    "Einwohner":"first","Wahl Opposition":"first"
}).reset_index()

df_month = df.groupby(["Location","month"]).agg({
    "Posts":"sum","Active_Accounts":"sum","Protests":"sum","participants":"sum",
    "Einwohner":"first","Wahl Opposition":"first"
}).reset_index()

agg_dfs = {"Daily": df, "Weekly": df_week, "Monthly": df_month}

# Lags für aggregierte Daten ebenfalls erstellen
for name, df_agg in agg_dfs.items():
    for var in lag_vars:
        df_agg[f"{var}_lag1"] = df_agg.groupby("Location")[var].shift(1)
        df_agg[f"{var}_lag2"] = df_agg.groupby("Location")[var].shift(2)
        df_agg[f"{var}_lag4"] = df_agg.groupby("Location")[var].shift(4)
        df_agg[f"{var}_log"] = np.log1p(df_agg[var])
        df_agg[f"{var}_log_lag1"] = np.log1p(df_agg[f"{var}_lag1"])
        df_agg[f"{var}_log_lag2"] = np.log1p(df_agg[f"{var}_lag2"])
        df_agg[f"{var}_log_lag4"] = np.log1p(df_agg[f"{var}_lag4"])
    df_agg.dropna(inplace=True)
agg_dfs["Daily"], agg_dfs["Weekly"], agg_dfs["Monthly"] = df, df_week, df_month

# =========================
# 4. SEQUENTIELLE RÜCKKOPPLUNG MIDDLE
# =========================
def sequential_feedback_middle(df, step1_dv, step1_iv, step2_dv, step2_iv,
                               control_vars=None, time_var=None, label=""):
    if control_vars is None:
        control_vars = []

    formula1 = f"{step1_dv} ~ {step1_iv} + " + " + ".join(control_vars)
    formula2 = f"{step2_dv} ~ {step2_iv} + " + " + ".join(control_vars)

    if time_var:
        formula1 += f" + C(Location) + C({time_var})"
        formula2 += f" + C(Location) + C({time_var})"
    else:
        formula1 += " + C(Location)"
        formula2 += " + C(Location)"

    step1_model = smf.glm(formula=formula1, data=df, family=sm.families.NegativeBinomial()).fit()
    step2_model = smf.glm(formula=formula2, data=df, family=sm.families.NegativeBinomial()).fit()

    step1_effect = (np.exp(step1_model.params[step1_iv]) - 1) * 100
    step2_effect = (np.exp(step2_model.params[step2_iv]) - 1) * 100
    indirect_effect = (np.exp(step1_model.params[step1_iv] * step2_model.params[step2_iv]) - 1) * 100

    se1, se2 = step1_model.bse[step1_iv], step2_model.bse[step2_iv]
    z = norm.ppf(0.975)
    step1_ci = ((np.exp(step1_model.params[step1_iv] - z*se1) - 1)*100,
                (np.exp(step1_model.params[step1_iv] + z*se1) - 1)*100)
    step2_ci = ((np.exp(step2_model.params[step2_iv] - z*se2) - 1)*100,
                (np.exp(step2_model.params[step2_iv] + z*se2) - 1)*100)
    indirect_se = np.sqrt((step2_model.params[step2_iv]**2 * se1**2) + (step1_model.params[step1_iv]**2 * se2**2))
    indirect_ci = ((np.exp(step1_model.params[step1_iv]*step2_model.params[step2_iv] - z*indirect_se) - 1)*100,
                   (np.exp(step1_model.params[step1_iv]*step2_model.params[step2_iv] + z*indirect_se) - 1)*100)

    return {
        "label": label,
        "step1_effect": step1_effect,
        "step1_ci_low": step1_ci[0],
        "step1_ci_high": step1_ci[1],
        "step2_effect": step2_effect,
        "step2_ci_low": step2_ci[0],
        "step2_ci_high": step2_ci[1],
        "indirect_effect": indirect_effect,
        "indirect_ci_low": indirect_ci[0],
        "indirect_ci_high": indirect_ci[1]
    }

# =========================
# 5. RUN MIDDLE MODELS
# =========================
controls = ["Protests_log_lag2", "participants_log_lag2", "Posts_log_lag2", "Active_Accounts_log_lag2"]

h3_middle_hypotheses = [
    {"label":"Middle-H3a1: Protests → Posts → Protests", "step1_dv":"Posts_log_lag4", "step1_iv":"Protests_log_lag2",
     "step2_dv":"Protests_log", "step2_iv":"Posts_log_lag4"},
    {"label":"Middle-H3a2: Protests → ActiveAccounts → Protests", "step1_dv":"Active_Accounts_log_lag4", "step1_iv":"Protests_log_lag2",
     "step2_dv":"Protests_log", "step2_iv":"Active_Accounts_log_lag4"},
    {"label":"Middle-H3a3: Participants → Posts → Participants", "step1_dv":"Posts_log_lag4", "step1_iv":"participants_log_lag2",
     "step2_dv":"participants_log", "step2_iv":"Posts_log_lag4"},
]

def run_middle_all_levels():
    results = []
    for level, df_level in agg_dfs.items():
        time_var = None
        if level == "Weekly":
            df_level["week_cat"] = df_level["week"].dt.strftime("%Y-%m-%d")
            time_var = "week_cat"
        elif level == "Monthly":
            df_level["month_cat"] = df_level["month"].dt.strftime("%Y-%m")
            time_var = "month_cat"

        for hyp in h3_middle_hypotheses:
            res = sequential_feedback_middle(
                df=df_level,
                step1_dv=hyp["step1_dv"],
                step1_iv=hyp["step1_iv"],
                step2_dv=hyp["step2_dv"],
                step2_iv=hyp["step2_iv"],
                control_vars=controls,
                time_var=time_var,
                label=f"{level}-{hyp['label']}"
            )
            results.append(res)
    return pd.DataFrame(results)

results_middle_df = run_middle_all_levels()

# =========================
# 6. SPEICHERN
# =========================
results_middle_df.to_excel("Ergebnisse_H3_middle.xlsx", index=False)
print("Middle-Ergebnisse in 'Ergebnisse_H3_middle.xlsx' gespeichert.")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt

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
# 2. LAGS ERSTELLEN
# AV: rohe Zählwerte (_lag1 für verzögerte AV in Step1)
# Prädiktoren: log-transformierte Lags (_log_lag1, _log_lag2)
# =========================
count_vars = ["Protests", "Posts", "participants", "Active_Accounts"]

for var in count_vars:
    df[f"{var}_lag1"] = df.groupby("Location")[var].shift(1)
    df[f"{var}_lag2"] = df.groupby("Location")[var].shift(2)
    df[f"{var}_log_lag1"] = np.log1p(df[f"{var}_lag1"])
    df[f"{var}_log_lag2"] = np.log1p(df[f"{var}_lag2"])

df = df.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# 3. ZEIT-AGGREGATION
# =========================
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

df_week = df.groupby(["Location", "week"]).agg({
    "Posts": "sum", "Active_Accounts": "sum", "Protests": "sum", "participants": "sum",
    "Einwohner": "first", "Wahl Opposition": "first"
}).reset_index()

df_month = df.groupby(["Location", "month"]).agg({
    "Posts": "sum", "Active_Accounts": "sum", "Protests": "sum", "participants": "sum",
    "Einwohner": "first", "Wahl Opposition": "first"
}).reset_index()

# Lags für aggregierte Daten
agg_dfs = {"Daily": df, "Weekly": df_week, "Monthly": df_month}
for name, df_agg in agg_dfs.items():
    for var in count_vars:
        df_agg[f"{var}_lag1"] = df_agg.groupby("Location")[var].shift(1)
        df_agg[f"{var}_lag2"] = df_agg.groupby("Location")[var].shift(2)
        df_agg[f"{var}_log_lag1"] = np.log1p(df_agg[f"{var}_lag1"])
        df_agg[f"{var}_log_lag2"] = np.log1p(df_agg[f"{var}_lag2"])
    df_agg.dropna(inplace=True)
agg_dfs["Daily"], agg_dfs["Weekly"], agg_dfs["Monthly"] = df, df_week, df_month

# =========================
# 4. SEQUENTIELLE RÜCKKOPPLUNG MIT FE
#
# AV:          rohe Zählwerte (NegBin erwartet nicht-negative Ganzzahlen)
# Prädiktoren: log-transformierte Lags
# Interpretation: exp(beta) - 1 = prozentualer Effekt auf den erwarteten Zählwert
# =========================
def sequential_feedback_fe(df, step1_dv, step1_iv, step2_dv, step2_iv,
                            control_vars=None, time_var=None, label=""):
    if control_vars is None:
        control_vars = []

    # STEP 1: Prädiktor (t-2) → AV (t-1)
    formula1 = f"{step1_dv} ~ {step1_iv} + " + " + ".join(control_vars)
    formula1 += f" + C(Location) + C({time_var})" if time_var else " + C(Location)"
    step1_model = smf.glm(formula=formula1, data=df, family=sm.families.NegativeBinomial()).fit()

    # STEP 2: AV aus Step1 (t-1) → AV (t)
    formula2 = f"{step2_dv} ~ {step2_iv} + " + " + ".join(control_vars)
    formula2 += f" + C(Location) + C({time_var})" if time_var else " + C(Location)"
    step2_model = smf.glm(formula=formula2, data=df, family=sm.families.NegativeBinomial()).fit()

    # Effekte in % (IRR-basiert)
    step1_effect = (np.exp(step1_model.params[step1_iv]) - 1) * 100
    step2_effect = (np.exp(step2_model.params[step2_iv]) - 1) * 100
    indirect_effect = (np.exp(step1_model.params[step1_iv] * step2_model.params[step2_iv]) - 1) * 100

    se1 = step1_model.bse[step1_iv]
    se2 = step2_model.bse[step2_iv]
    z = norm.ppf(0.975)

    step1_ci = (
        (np.exp(step1_model.params[step1_iv] - z * se1) - 1) * 100,
        (np.exp(step1_model.params[step1_iv] + z * se1) - 1) * 100
    )
    step2_ci = (
        (np.exp(step2_model.params[step2_iv] - z * se2) - 1) * 100,
        (np.exp(step2_model.params[step2_iv] + z * se2) - 1) * 100
    )
    indirect_se = np.sqrt(
        (step2_model.params[step2_iv] ** 2 * se1 ** 2) +
        (step1_model.params[step1_iv] ** 2 * se2 ** 2)
    )
    indirect_ci = (
        (np.exp(step1_model.params[step1_iv] * step2_model.params[step2_iv] - z * indirect_se) - 1) * 100,
        (np.exp(step1_model.params[step1_iv] * step2_model.params[step2_iv] + z * indirect_se) - 1) * 100
    )

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
# 5. HYPOTHESEN
#
# step1_dv / step2_dv: rohe Zählwerte (AV für NegBin)
# step1_iv / step2_iv: log-transformierte Lags (Prädiktoren)
# =========================
controls = [
    "Protests_log_lag2", "participants_log_lag2",
    "Posts_log_lag2", "Active_Accounts_log_lag2"
]

h3a_hypotheses = [
    {"label": "H3a1: Protests → Posts → Protests",
     "step1_dv": "Posts_lag1",
     "step1_iv": "Protests_log_lag2",
     "step2_dv": "Protests",
     "step2_iv": "Posts_log_lag1"},
    {"label": "H3a2: Protests → ActiveAccounts → Protests",
     "step1_dv": "Active_Accounts_lag1",
     "step1_iv": "Protests_log_lag2",
     "step2_dv": "Protests",
     "step2_iv": "Active_Accounts_log_lag1"},
    {"label": "H3a3: Participants → Posts → Participants",
     "step1_dv": "Posts_lag1",
     "step1_iv": "participants_log_lag2",
     "step2_dv": "participants",
     "step2_iv": "Posts_log_lag1"},
]

h3b_hypotheses = [
    {"label": "H3b1: Posts → Protests → Posts",
     "step1_dv": "Protests_lag1",
     "step1_iv": "Posts_log_lag2",
     "step2_dv": "Posts",
     "step2_iv": "Protests_log_lag1"},
    {"label": "H3b2: ActiveAccounts → Protests → ActiveAccounts",
     "step1_dv": "Protests_lag1",
     "step1_iv": "Active_Accounts_log_lag2",
     "step2_dv": "Active_Accounts",
     "step2_iv": "Protests_log_lag1"},
    {"label": "H3b3: Posts → Participants → Posts",
     "step1_dv": "participants_lag1",
     "step1_iv": "Posts_log_lag2",
     "step2_dv": "Posts",
     "step2_iv": "participants_log_lag1"},
]

# =========================
# 6. MODELLE LAUFEN LASSEN
# =========================
def run_h3_all_levels():
    results = []
    for level, df_level in agg_dfs.items():
        time_var = None
        if level == "Weekly":
            df_level["week_cat"] = df_level["week"].dt.strftime("%Y-%m-%d")
            time_var = "week_cat"
        elif level == "Monthly":
            df_level["month_cat"] = df_level["month"].dt.strftime("%Y-%m")
            time_var = "month_cat"

        for hyp in h3a_hypotheses + h3b_hypotheses:
            res = sequential_feedback_fe(
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

# =========================
# 7. ERGEBNISSE SPEICHERN
# =========================
results_df = run_h3_all_levels()
print(results_df.round(1))

output_file = "Ergebnisse_H3.xlsx"
results_df.to_excel(output_file, index=False)
print(f"Ergebnisse erfolgreich in '{output_file}' gespeichert.")

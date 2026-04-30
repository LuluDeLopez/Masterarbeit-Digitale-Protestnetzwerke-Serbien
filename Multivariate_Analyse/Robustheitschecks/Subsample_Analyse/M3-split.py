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
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# =========================
# 2. SPLIT NACH ZEIT
# =========================
cutoff_date = df["Date"].median()
df_first_half = df[df["Date"] <= cutoff_date].copy()
df_second_half = df[df["Date"] > cutoff_date].copy()

# =========================
# 3. FUNKTION: LAGS + LOGS
# AV: rohe Zählwerte (_lag1 für verzögerte AV in Step1)
# Prädiktoren: log-transformierte Lags (_log_lag1, _log_lag2)
# =========================
count_vars = ["Protests", "Posts", "participants", "Active_Accounts"]

def prepare_lags(df_input):
    for var in count_vars:
        df_input[f"{var}_lag1"] = df_input.groupby("Location")[var].shift(1)
        df_input[f"{var}_lag2"] = df_input.groupby("Location")[var].shift(2)
        df_input[f"{var}_log_lag1"] = np.log1p(df_input[f"{var}_lag1"])
        df_input[f"{var}_log_lag2"] = np.log1p(df_input[f"{var}_lag2"])
    df_input = df_input.replace([np.inf, -np.inf], np.nan).dropna()
    return df_input

# =========================
# 4. FUNKTION: AGGREGATION
# =========================
def create_agg_dfs(df_input):
    df_input = prepare_lags(df_input.copy())

    df_input["week"] = df_input["Date"].dt.to_period("W").dt.start_time
    df_input["month"] = df_input["Date"].dt.to_period("M").dt.start_time

    df_week = df_input.groupby(["Location", "week"]).agg({
        "Posts": "sum", "Active_Accounts": "sum", "Protests": "sum", "participants": "sum",
        "Einwohner": "first", "Wahl Opposition": "first"
    }).reset_index()

    df_month = df_input.groupby(["Location", "month"]).agg({
        "Posts": "sum", "Active_Accounts": "sum", "Protests": "sum", "participants": "sum",
        "Einwohner": "first", "Wahl Opposition": "first"
    }).reset_index()

    agg_dfs_local = {"Daily": df_input, "Weekly": df_week, "Monthly": df_month}

    for name, df_agg in agg_dfs_local.items():
        for var in count_vars:
            df_agg[f"{var}_lag1"] = df_agg.groupby("Location")[var].shift(1)
            df_agg[f"{var}_lag2"] = df_agg.groupby("Location")[var].shift(2)
            df_agg[f"{var}_log_lag1"] = np.log1p(df_agg[f"{var}_lag1"])
            df_agg[f"{var}_log_lag2"] = np.log1p(df_agg[f"{var}_lag2"])
        df_agg.dropna(inplace=True)

    return agg_dfs_local

# =========================
# 5. MODEL FUNKTION
#
# AV:          rohe Zählwerte (NegBin erwartet nicht-negative Ganzzahlen)
# Prädiktoren: log-transformierte Lags
# Interpretation: exp(beta) - 1 = prozentualer Effekt auf den erwarteten Zählwert
# =========================
def sequential_feedback_fe(df, step1_dv, step1_iv, step2_dv, step2_iv,
                            control_vars=None, time_var=None, label=""):
    if control_vars is None:
        control_vars = []

    formula1 = f"{step1_dv} ~ {step1_iv} + " + " + ".join(control_vars)
    formula1 += f" + C(Location) + C({time_var})" if time_var else " + C(Location)"

    model1 = smf.glm(formula=formula1, data=df,
                     family=sm.families.NegativeBinomial()).fit()

    formula2 = f"{step2_dv} ~ {step2_iv} + " + " + ".join(control_vars)
    formula2 += f" + C(Location) + C({time_var})" if time_var else " + C(Location)"

    model2 = smf.glm(formula=formula2, data=df,
                     family=sm.families.NegativeBinomial()).fit()

    b1, b2 = model1.params[step1_iv], model2.params[step2_iv]
    se1, se2 = model1.bse[step1_iv], model2.bse[step2_iv]
    z = norm.ppf(0.975)

    step1 = (np.exp(b1) - 1) * 100
    step2 = (np.exp(b2) - 1) * 100
    indirect = (np.exp(b1 * b2) - 1) * 100

    ci1 = ((np.exp(b1 - z * se1) - 1) * 100, (np.exp(b1 + z * se1) - 1) * 100)
    ci2 = ((np.exp(b2 - z * se2) - 1) * 100, (np.exp(b2 + z * se2) - 1) * 100)

    indirect_se = np.sqrt((b2 ** 2 * se1 ** 2) + (b1 ** 2 * se2 ** 2))
    ci_ind = (
        (np.exp(b1 * b2 - z * indirect_se) - 1) * 100,
        (np.exp(b1 * b2 + z * indirect_se) - 1) * 100
    )

    return {
        "label": label,
        "step1": step1, "step1_low": ci1[0], "step1_high": ci1[1],
        "step2": step2, "step2_low": ci2[0], "step2_high": ci2[1],
        "indirect": indirect, "ind_low": ci_ind[0], "ind_high": ci_ind[1]
    }

# =========================
# 6. HYPOTHESEN
#
# step1_dv / step2_dv: rohe Zählwerte (AV für NegBin)
# step1_iv / step2_iv: log-transformierte Lags (Prädiktoren)
# =========================
controls = [
    "Protests_log_lag2", "participants_log_lag2",
    "Posts_log_lag2", "Active_Accounts_log_lag2"
]

h3a = [
    {"label": "H3a1",
     "step1_dv": "Posts_lag1",            # AV Step1: rohe Zählwerte
     "step1_iv": "Protests_log_lag2",     # Prädiktor: log-transformiert
     "step2_dv": "Protests",              # AV Step2: rohe Zählwerte
     "step2_iv": "Posts_log_lag1"},       # Prädiktor: log-transformiert
    {"label": "H3a2",
     "step1_dv": "Active_Accounts_lag1",
     "step1_iv": "Protests_log_lag2",
     "step2_dv": "Protests",
     "step2_iv": "Active_Accounts_log_lag1"},
]

h3b = [
    {"label": "H3b1",
     "step1_dv": "Protests_lag1",
     "step1_iv": "Posts_log_lag2",
     "step2_dv": "Posts",
     "step2_iv": "Protests_log_lag1"},
]

# =========================
# 7. RUN FUNKTION
# =========================
def run_models(agg_dfs, sample_name):
    results = []
    for level, df_level in agg_dfs.items():
        time_var = None
        if level == "Weekly":
            df_level["week_cat"] = df_level["week"].astype(str)
            time_var = "week_cat"
        elif level == "Monthly":
            df_level["month_cat"] = df_level["month"].astype(str)
            time_var = "month_cat"

        for hyp in h3a + h3b:
            res = sequential_feedback_fe(
                df_level,
                hyp["step1_dv"], hyp["step1_iv"],
                hyp["step2_dv"], hyp["step2_iv"],
                controls, time_var,
                label=f"{sample_name}-{level}-{hyp['label']}"
            )
            results.append(res)
    return pd.DataFrame(results)

# =========================
# 8. PIPELINE AUSFÜHREN
# =========================
agg_first = create_agg_dfs(df_first_half)
agg_second = create_agg_dfs(df_second_half)

results_first = run_models(agg_first, "FirstHalf")
results_second = run_models(agg_second, "SecondHalf")

results_df = pd.concat([results_first, results_second])
print(results_df.round(1))

# =========================
# 9. SPEICHERN
# =========================
output_file = "Ergebnisse_H3_split.xlsx"
results_df.to_excel(output_file, index=False)
print("Fertig! Datei gespeichert:", output_file)

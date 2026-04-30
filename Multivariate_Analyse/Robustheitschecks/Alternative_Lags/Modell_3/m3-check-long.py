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
# 2. LAGS ERSTELLEN
# =========================
count_vars = ["Protests", "Posts", "participants", "Active_Accounts"]

for var in count_vars:
    df[f"{var}_lag1"] = df.groupby("Location")[var].shift(1)
    df[f"{var}_lag2"] = df.groupby("Location")[var].shift(2)
    df[f"{var}_lag3"] = df.groupby("Location")[var].shift(3)
    df[f"{var}_lag6"] = df.groupby("Location")[var].shift(6)
    df[f"{var}_log_lag1"] = np.log1p(df[f"{var}_lag1"])
    df[f"{var}_log_lag2"] = np.log1p(df[f"{var}_lag2"])
    df[f"{var}_log_lag3"] = np.log1p(df[f"{var}_lag3"])
    df[f"{var}_log_lag6"] = np.log1p(df[f"{var}_lag6"])

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

agg_dfs = {"Daily": df, "Weekly": df_week, "Monthly": df_month}

for name, df_agg in agg_dfs.items():
    for var in count_vars:
        df_agg[f"{var}_lag1"] = df_agg.groupby("Location")[var].shift(1)
        df_agg[f"{var}_lag2"] = df_agg.groupby("Location")[var].shift(2)
        df_agg[f"{var}_lag3"] = df_agg.groupby("Location")[var].shift(3)
        df_agg[f"{var}_lag6"] = df_agg.groupby("Location")[var].shift(6)
        df_agg[f"{var}_log_lag1"] = np.log1p(df_agg[f"{var}_lag1"])
        df_agg[f"{var}_log_lag2"] = np.log1p(df_agg[f"{var}_lag2"])
        df_agg[f"{var}_log_lag3"] = np.log1p(df_agg[f"{var}_lag3"])
        df_agg[f"{var}_log_lag6"] = np.log1p(df_agg[f"{var}_lag6"])
    df_agg.dropna(inplace=True)
agg_dfs["Daily"], agg_dfs["Weekly"], agg_dfs["Monthly"] = df, df_week, df_month

# =========================
# 4. SEQUENTIELLE RÜCKKOPPLUNG 3-LAGS
# =========================
def sequential_feedback_3lags(df, step1_dv, step1_iv, step2_dv, step2_iv,
                               control_vars=None, time_var=None, label=""):
    if control_vars is None:
        control_vars = []

    # IV aus Controls entfernen (verhindert Kollinearität / KeyError)
    controls1 = [c for c in control_vars if c != step1_iv]
    controls2 = [c for c in control_vars if c != step2_iv]

    # Relevante Spalten isolieren und bereinigen
    cols1 = [step1_dv, step1_iv] + controls1 + (["Location", time_var] if time_var else ["Location"])
    cols2 = [step2_dv, step2_iv] + controls2 + (["Location", time_var] if time_var else ["Location"])

    df1 = df[[c for c in cols1 if c in df.columns]].copy()
    df2 = df[[c for c in cols2 if c in df.columns]].copy()

    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    # DV: nicht-negativ, endlich, ganzzahlig (NegBin-Voraussetzung)
    df1 = df1[df1[step1_dv] >= 0]
    df2 = df2[df2[step2_dv] >= 0]
    df1 = df1[np.isfinite(df1[step1_dv])]
    df2 = df2[np.isfinite(df2[step2_dv])]
    df1[step1_dv] = df1[step1_dv].round().astype(int)
    df2[step2_dv] = df2[step2_dv].round().astype(int)

    if len(df1) < 10 or len(df2) < 10:
        print(f"⚠️  Zu wenige Beobachtungen für {label} – übersprungen.")
        return {
            "label": label,
            "step1_effect": np.nan, "step1_ci_low": np.nan, "step1_ci_high": np.nan,
            "step2_effect": np.nan, "step2_ci_low": np.nan, "step2_ci_high": np.nan,
            "indirect_effect": np.nan, "indirect_ci_low": np.nan, "indirect_ci_high": np.nan
        }

    formula1 = f"{step1_dv} ~ {step1_iv} + " + " + ".join(controls1)
    formula2 = f"{step2_dv} ~ {step2_iv} + " + " + ".join(controls2)

    if time_var:
        formula1 += f" + C(Location) + C({time_var})"
        formula2 += f" + C(Location) + C({time_var})"
    else:
        formula1 += " + C(Location)"
        formula2 += " + C(Location)"

    try:
        step1_model = smf.glm(formula=formula1, data=df1,
                              family=sm.families.NegativeBinomial()).fit(disp=False)
        step2_model = smf.glm(formula=formula2, data=df2,
                              family=sm.families.NegativeBinomial()).fit(disp=False)
    except Exception as e:
        print(f"⚠️  Modell-Fehler bei {label}: {e}")
        return {
            "label": label,
            "step1_effect": np.nan, "step1_ci_low": np.nan, "step1_ci_high": np.nan,
            "step2_effect": np.nan, "step2_ci_low": np.nan, "step2_ci_high": np.nan,
            "indirect_effect": np.nan, "indirect_ci_low": np.nan, "indirect_ci_high": np.nan
        }

    step1_effect = (np.exp(step1_model.params[step1_iv]) - 1) * 100
    step2_effect = (np.exp(step2_model.params[step2_iv]) - 1) * 100
    indirect_effect = (np.exp(step1_model.params[step1_iv] * step2_model.params[step2_iv]) - 1) * 100

    se1, se2 = step1_model.bse[step1_iv], step2_model.bse[step2_iv]
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
# Zeitliche Logik (Daily):
#   Step1: IV bei t-3 → Mediator-DV bei t-6 (Lücke: 3 Perioden)
#   Step2: Mediator bei t-6 → Outcome-DV bei t   (Lücke: 6 Perioden)
#
# H3a: Offline → Online → Offline
# H3b: Online → Offline → Online
# =========================
controls = [
    "Protests_log_lag2", "participants_log_lag2",
    "Posts_log_lag2", "Active_Accounts_log_lag2"
]

h3a_3lags_hypotheses = [
    # Protests(t-3) → Posts(t-6) → Protests(t)
    {"label": "H3a1: Protests → Posts → Protests",
     "step1_dv": "Posts_lag6",
     "step1_iv": "Protests_log_lag3",
     "step2_dv": "Protests",
     "step2_iv": "Posts_log_lag6"},
    # Protests(t-3) → ActiveAccounts(t-6) → Protests(t)
    {"label": "H3a2: Protests → ActiveAccounts → Protests",
     "step1_dv": "Active_Accounts_lag6",
     "step1_iv": "Protests_log_lag3",
     "step2_dv": "Protests",
     "step2_iv": "Active_Accounts_log_lag6"},
    # Participants(t-3) → Posts(t-6) → Participants(t)
    {"label": "H3a3: Participants → Posts → Participants",
     "step1_dv": "Posts_lag6",
     "step1_iv": "participants_log_lag3",
     "step2_dv": "participants",
     "step2_iv": "Posts_log_lag6"},
]

h3b_3lags_hypotheses = [
    # Posts(t-3) → Protests(t-6) → Posts(t)
    {"label": "H3b1: Posts → Protests → Posts",
     "step1_dv": "Protests_lag6",          # ← KORRIGIERT: war Protests_lag6 mit falschem IV
     "step1_iv": "Posts_log_lag3",         # ← KORRIGIERT: Prädiktor bei t-3
     "step2_dv": "Posts",
     "step2_iv": "Protests_log_lag6"},
    # ActiveAccounts(t-3) → Protests(t-6) → ActiveAccounts(t)
    {"label": "H3b2: ActiveAccounts → Protests → ActiveAccounts",
     "step1_dv": "Protests_lag6",          # ← KORRIGIERT: Mediator-DV
     "step1_iv": "Active_Accounts_log_lag3", # ← KORRIGIERT: Prädiktor bei t-3
     "step2_dv": "Active_Accounts",
     "step2_iv": "Protests_log_lag6"},
    # Posts(t-3) → Participants(t-6) → Posts(t)
    {"label": "H3b3: Posts → Participants → Posts",
     "step1_dv": "participants_lag6",      # ← KORRIGIERT: Mediator-DV
     "step1_iv": "Posts_log_lag3",         # ← KORRIGIERT: Prädiktor bei t-3
     "step2_dv": "Posts",
     "step2_iv": "participants_log_lag6"},
]

# =========================
# 6. MODELLE LAUFEN LASSEN
# =========================
def run_3lags_all_levels():
    results = []
    for level, df_level in agg_dfs.items():
        time_var = None
        if level == "Weekly":
            df_level["week_cat"] = df_level["week"].dt.strftime("%Y-%m-%d")
            time_var = "week_cat"
        elif level == "Monthly":
            df_level["month_cat"] = df_level["month"].dt.strftime("%Y-%m")
            time_var = "month_cat"

        for hyp in h3a_3lags_hypotheses + h3b_3lags_hypotheses:
            res = sequential_feedback_3lags(
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
results_3lags_df = run_3lags_all_levels()
print(results_3lags_df.round(1))

results_3lags_df.to_excel("Ergebnisse_H3_3lags.xlsx", index=False)
print("3-Lags-Ergebnisse in 'Ergebnisse_H3_3lags.xlsx' gespeichert.")

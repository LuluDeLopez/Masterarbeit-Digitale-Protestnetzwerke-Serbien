import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"

df = pd.read_excel(file_path, sheet_name="Zeitreihe-Tag").fillna(0)
df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")

df_controls = df_controls.rename(columns={"Ort": "Location"})

df = df.merge(
    df_controls[["Location","Einwohner","Wahl Opposition"]],
    on="Location",
    how="left"
)

df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Location","Date"])

# =========================
# 2. ZEIT-SPLIT
# =========================
cutoff_date = df["Date"].median()

df_first = df[df["Date"] <= cutoff_date].copy()
df_second = df[df["Date"] > cutoff_date].copy()

# =========================
# 3. DATEN VORBEREITUNG
# =========================
def prepare_data(df_input):

    df_input = df_input.copy()

    # Zeitvariablen
    df_input["week"] = df_input["Date"].dt.to_period("W").dt.start_time
    df_input["month"] = df_input["Date"].dt.to_period("M").dt.start_time

    vars_all = ["Protests","participants","Posts","Active_Accounts"]

    # Lags
    for var in vars_all:
        df_input[f"{var}_lag1"] = df_input.groupby("Location")[var].shift(1)
        df_input[f"{var}_lag2"] = df_input.groupby("Location")[var].shift(2)

    # Logs
    for var in vars_all:
        df_input[f"{var}_log"] = np.log1p(df_input[var])
        df_input[f"{var}_log_lag1"] = np.log1p(df_input[f"{var}_lag1"])
        df_input[f"{var}_log_lag2"] = np.log1p(df_input[f"{var}_lag2"])

    df_input = df_input.replace([np.inf,-np.inf],np.nan).dropna()

    # Aggregationen
    def aggregate(df_base, time_col):
        df_agg = df_base.groupby(["Location",time_col]).agg({
            "Posts":"sum","Active_Accounts":"sum",
            "Protests":"sum","participants":"sum",
            "Einwohner":"first","Wahl Opposition":"first"
        }).reset_index()

        for var in vars_all:
            df_agg[f"{var}_lag1"] = df_agg.groupby("Location")[var].shift(1)
            df_agg[f"{var}_lag2"] = df_agg.groupby("Location")[var].shift(2)

            df_agg[f"{var}_log"] = np.log1p(df_agg[var])
            df_agg[f"{var}_log_lag1"] = np.log1p(df_agg[f"{var}_lag1"])
            df_agg[f"{var}_log_lag2"] = np.log1p(df_agg[f"{var}_lag2"])

        return df_agg.replace([np.inf,-np.inf],np.nan).dropna()

    df_week = aggregate(df_input,"week")
    df_month = aggregate(df_input,"month")

    # FE Kategorien
    df_input["time_cat"] = df_input["Date"].astype(str)
    df_week["time_cat"] = df_week["week"].astype(str)
    df_month["time_cat"] = df_month["month"].astype(str)

    return {
        "Daily": df_input,
        "Weekly": df_week,
        "Monthly": df_month
    }

# =========================
# 4. GENERISCHE MODELLFUNKTION
# =========================
# alte Version:
# def run_nb_model(df, dv, ivs, controls, label):

# neue Version:
def run_nb_model(df, dv, ivs, controls, label, hypothesis=None):
    formula = f"{dv} ~ " + " + ".join(ivs + controls) + " + C(Location) + C(time_cat)"

    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()

        res = pd.DataFrame({
            "Hypothese": hypothesis if hypothesis else "",
            "variable": ivs,
            "coef": model.params[ivs],
            "std_err": model.bse[ivs],
            "model": label,
            "dv": dv
        })

        return res
    except Exception as e:
        print(f"⚠️ Fehler in {label}: {e}")
        return pd.DataFrame()


def run_mediation(df, x, mediator, y, controls, label, hypothesis):

    try:
        # =====================
        # STEP 1: X → Mediator
        # =====================
        formula_a = f"{mediator} ~ {x} + " + " + ".join(controls) + " + C(Location) + C(time_cat)"
        model_a = smf.glm(
            formula=formula_a,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()

        # =====================
        # STEP 2: Mediator → Y (inkl. X)
        # =====================
        formula_b = f"{y} ~ {mediator} + {x} + " + " + ".join(controls) + " + C(Location) + C(time_cat)"
        model_b = smf.glm(
            formula=formula_b,
            data=df,
            family=sm.families.NegativeBinomial()
        ).fit()

        a = model_a.params[x]
        b = model_b.params[mediator]

        # indirekter Effekt
        indirect = a * b

        # SE (Delta-Methode)
        se_a = model_a.bse[x]
        se_b = model_b.bse[mediator]
        se_indirect = np.sqrt((b**2 * se_a**2) + (a**2 * se_b**2))

        return pd.DataFrame({
            "Hypothese": hypothesis,
            "model": label,
            "a_path": a,
            "b_path": b,
            "indirect_coef": indirect,
            "indirect_se": se_indirect
        }, index=[0])
    except Exception as e:
        print(f"⚠️ Mediation Fehler {label}: {e}")
        return pd.DataFrame()

# =========================
# 5. M1, M2, H3 RUNNER
# =========================
controls = [
    "Protests_log_lag2","participants_log_lag2",
    "Posts_log_lag2","Active_Accounts_log_lag2"
]

def run_all_models(df_input, sample_name):
    datasets = prepare_data(df_input)
    results = []

    for level, data in datasets.items():

        # =====================
        # M1: Protest → Online-Kommunikation (H1)
        # =====================

        # H1a: Protests → Posts
        results.append(run_nb_model(
            data,
            dv="Posts_log",
            ivs=["Protests_log_lag1"],
            controls=controls,
            label=f"{sample_name}-M1-{level}-Posts",
            hypothesis="H1a"
        ))

        # H1b: Protests → Active_Accounts
        results.append(run_nb_model(
            data,
            dv="Active_Accounts_log",
            ivs=["Protests_log_lag1"],
            controls=controls,
            label=f"{sample_name}-M1-{level}-Accounts",
            hypothesis="H1b"
        ))

        # H1c: participants → Posts
        results.append(run_nb_model(
            data,
            dv="Posts_log",
            ivs=["participants_log_lag1"],
            controls=controls,
            label=f"{sample_name}-M1-{level}-Posts-participants",
            hypothesis="H1c"
        ))

        # =====================
        # M2: Online → Protest (H2)
        # =====================

        # H2a: Posts → Protests
        results.append(run_nb_model(
            data,
            dv="Protests_log",
            ivs=["Posts_log_lag1"],
            controls=controls,
            label=f"{sample_name}-M2-{level}-Protests-Posts",
            hypothesis="H2a"
        ))

        # H2b: Active_Accounts → Protests
        results.append(run_nb_model(
            data,
            dv="Protests_log",
            ivs=["Active_Accounts_log_lag1"],
            controls=controls,
            label=f"{sample_name}-M2-{level}-Protests-Accounts",
            hypothesis="H2b"
        ))

        # H2c: Posts → participants
        results.append(run_nb_model(
            data,
            dv="participants_log",
            ivs=["Posts_log_lag1"],
            controls=controls,
            label=f"{sample_name}-M2-{level}-Participants-Posts",
            hypothesis="H2c"
        ))

        # =====================
        # H3: echte Mediation (t-2 → t-1 → t)
        # =====================

        # H3a1: Protests → Posts → Protests
        results.append(run_mediation(
            data,
            x="Protests_log_lag2",
            mediator="Posts_log_lag1",
            y="Protests_log",
            controls=["participants_log_lag2","Active_Accounts_log_lag2"],
            label=f"{sample_name}-H3a1-{level}",
            hypothesis="H3a1"
        ))

        # H3a2: Protests → Active_Accounts → Protests
        results.append(run_mediation(
            data,
            x="Protests_log_lag2",
            mediator="Active_Accounts_log_lag1",
            y="Protests_log",
            controls=["participants_log_lag2","Posts_log_lag2"],
            label=f"{sample_name}-H3a2-{level}",
            hypothesis="H3a2"
        ))

        # H3a3: participants → Posts → participants
        results.append(run_mediation(
            data,
            x="participants_log_lag2",
            mediator="Posts_log_lag1",
            y="participants_log",
            controls=["Protests_log_lag2","Active_Accounts_log_lag2"],
            label=f"{sample_name}-H3a3-{level}",
            hypothesis="H3a3"
        ))

        # H3b1: Posts → Protests → Posts
        results.append(run_mediation(
            data,
            x="Posts_log_lag2",
            mediator="Protests_log_lag1",
            y="Posts_log",
            controls=["participants_log_lag2","Active_Accounts_log_lag2"],
            label=f"{sample_name}-H3b1-{level}",
            hypothesis="H3b1"
        ))

        # H3b2: Active_Accounts → Protests → Active_Accounts
        results.append(run_mediation(
            data,
            x="Active_Accounts_log_lag2",
            mediator="Protests_log_lag1",
            y="Active_Accounts_log",
            controls=["participants_log_lag2","Posts_log_lag2"],
            label=f"{sample_name}-H3b2-{level}",
            hypothesis="H3b2"
        ))

        # H3b3: Posts → participants → Posts
        results.append(run_mediation(
            data,
            x="Posts_log_lag2",
            mediator="participants_log_lag1",
            y="Posts_log",
            controls=["Protests_log_lag2","Active_Accounts_log_lag2"],
            label=f"{sample_name}-H3b3-{level}",
            hypothesis="H3b3"
        ))

    return pd.concat(results)

# =========================
# 6. AUSFÜHRUNG
# =========================
results_first = run_all_models(df_first,"FirstHalf")
results_second = run_all_models(df_second,"SecondHalf")

results = pd.concat([results_first,results_second])

# =========================
# 7. EFFEKTE
# =========================
results["indirect_pct"] = (np.exp(results["indirect_coef"]) - 1) * 100
results["ci_low"] = (np.exp(results["indirect_coef"] - 1.96 * results["indirect_se"]) - 1) * 100
results["ci_high"] = (np.exp(results["indirect_coef"] + 1.96 * results["indirect_se"]) - 1) * 100

# =========================
# 8. EXPORT
# =========================
results.to_excel("Harmonized_Results_Neu.xlsx", index=False)

print("✅ Fertig! Harmonisiertes Modell geschätzt.")

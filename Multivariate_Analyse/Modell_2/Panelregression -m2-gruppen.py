import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"

df = pd.read_excel(file_path, sheet_name="Zeitreihe-Tag")
df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")

df = df.fillna(0)
df["Date"] = pd.to_datetime(df["Date"])

# Zeitvariablen
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

# Kontrollvariablen mergen
df_controls = df_controls.rename(columns={"Ort": "Location"})
df = df.merge(
    df_controls[["Location", "Einwohner", "Wahl Opposition"]],
    on="Location",
    how="left"
)

df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

# =========================
# 2. SORTIEREN + LAGS
# =========================
df = df.sort_values(["Location", "Young_Account", "Date"])

lag_vars = ["Protests", "participants", "Posts", "Active_Accounts"]
for var in lag_vars:
    df[f"{var}_lag1"] = df.groupby(["Location", "Young_Account"])[var].shift(1).fillna(0)

# Log-Transformation (wichtig!)
df["participants_log"] = np.log1p(df["participants"])
df["participants_log_lag1"] = df.groupby(["Location","Young_Account"])["participants_log"].shift(1).fillna(0)

df["Protests_log"] = np.log1p(df["Protests"])
df["Protests_log_lag1"] = df.groupby(["Location","Young_Account"])["Protests_log"].shift(1).fillna(0)

# =========================
# 3. AGGREGATIONEN
# =========================
def prepare_aggregation(data, time_var):

    df_agg = data.groupby(["Location", time_var, "Young_Account"]).agg({
        "Posts": "sum",
        "Active_Accounts": "sum",
        "Protests": "sum",
        "participants": "sum",
        "Einwohner": "first",
        "Wahl Opposition": "first"
    }).reset_index().sort_values(["Location", time_var])

    # Lags
    for var in ["Posts", "Active_Accounts"]:
        df_agg[f"{var}_lag1"] = df_agg.groupby(["Location","Young_Account"])[var].shift(1).fillna(0)

    # Logs
    df_agg["participants_log"] = np.log1p(df_agg["participants"])
    df_agg["participants_log_lag1"] = df_agg.groupby(["Location","Young_Account"])["participants_log"].shift(1).fillna(0)

    df_agg["Protests_log"] = np.log1p(df_agg["Protests"])
    df_agg["Protests_log_lag1"] = df_agg.groupby(["Location","Young_Account"])["Protests_log"].shift(1).fillna(0)

    return df_agg


df_week = prepare_aggregation(df, "week")
df_month = prepare_aggregation(df, "month")

# =========================
# 4. MODELLFUNKTION (STABIL)
# =========================
def run_model_nb(df_group, dv, label):

    df_group = df_group.copy().reset_index(drop=True)
    df_group["t_numeric"] = np.arange(len(df_group))

    # Clean
    df_group = df_group.replace([np.inf, -np.inf], np.nan).dropna()

    if dv == "Protests_log":
        formula = """
        Protests_log ~ Posts_lag1 + Active_Accounts_lag1 + Protests_log_lag1
        + Einwohner + Q('Wahl Opposition') + t_numeric
        """

    elif dv == "participants_log":
        formula = """
        participants_log ~ Posts_lag1 + Active_Accounts_lag1 + participants_log_lag1
        + Einwohner + Q('Wahl Opposition') + t_numeric
        """

    try:
        model = smf.glm(
            formula=formula,
            data=df_group,
            family=sm.families.NegativeBinomial()
        ).fit(maxiter=100)

    except:
        print(f"⚠️ Modell fehlgeschlagen: {label} | {dv}")
        return pd.DataFrame()

    variables = ["Posts_lag1", "Active_Accounts_lag1"]

    results = pd.DataFrame({
        "variable": variables,
        "coef": model.params[variables],
        "std_err": model.bse[variables],
        "dv": dv,
        "model": label
    })

    return results

# =========================
# 5. ERGEBNISSE BERECHNEN
# =========================
def compute_results(df, label_prefix):

    results_all = []

    for group, df_group in df.groupby("Young_Account"):

        label = f"{label_prefix}_Y{group}"

        res1 = run_model_nb(df_group, "Protests_log", label)
        res2 = run_model_nb(df_group, "participants_log", label)

        if res1.empty or res2.empty:
            continue

        res = pd.concat([res1, res2])
        res["Young_Account"] = group

        # Effekte in %
        res["effect_pct"] = (np.exp(res["coef"]) - 1) * 100
        res["ci_low"] = (np.exp(res["coef"] - 1.96 * res["std_err"]) - 1) * 100
        res["ci_high"] = (np.exp(res["coef"] + 1.96 * res["std_err"]) - 1) * 100

        results_all.append(res)

    return pd.concat(results_all)


results = pd.concat([
    compute_results(df, "Daily"),
    compute_results(df_week, "Weekly"),
    compute_results(df_month, "Monthly")
]).reset_index(drop=True)

# =========================
# 6. TERMINAL OUTPUT
# =========================
def print_table(df, title):
    df_print = df[["model","Young_Account","variable","dv","effect_pct","ci_low","ci_high"]].round(2)
    print(f"\n===== {title} =====")
    print(df_print.to_string(index=False))


for model in ["Daily", "Weekly", "Monthly"]:
    for y in [0,1]:
        subset = results[
            (results["model"].str.contains(model)) &
            (results["Young_Account"] == y)
        ]
        print_table(subset, f"{model} Y{y}")

# =========================
# 7. VISUALISIERUNG MIT LEGENDEN
# =========================
plt.figure(figsize=(12,6))

time_order = ["Daily_Y0","Daily_Y1","Weekly_Y0","Weekly_Y1","Monthly_Y0","Monthly_Y1"]
variables = ["Posts_lag1","Active_Accounts_lag1"]
dvs = ["Protests_log","participants_log"]

# Farben: DV × Young_Account
colors = {
    ("Protests_log",0): "tab:green",
    ("Protests_log",1): "lightgreen",
    ("participants_log",0): "tab:red",
    ("participants_log",1): "salmon"
}

# Marker: Variable
markers = {
    "Posts_lag1": "o",
    "Active_Accounts_lag1": "s"
}

# Zeichnen
for i, model in enumerate(time_order):
    df_plot = results[results["model"] == model]
    young = 1 if "_Y1" in model else 0

    for var in variables:
        for dv in dvs:
            row = df_plot[(df_plot["variable"]==var) & (df_plot["dv"]==dv)]
            if not row.empty:
                x = i + (variables.index(var)*2 + dvs.index(dv) - 1.5)*0.15
                y = row["effect_pct"].values[0]
                yerr_low = y - row["ci_low"].values[0]
                yerr_high = row["ci_high"].values[0] - y

                plt.errorbar(
                    x, y,
                    yerr=[[yerr_low],[yerr_high]],
                    fmt=markers[var],
                    color=colors[(dv,young)],
                    capsize=5,
                    markersize=8,
                    label=f"{var} × {dv} × Y{young}"
                )

# Achsen & Titel
plt.xticks(range(len(time_order)), ["D0","D1","W0","W1","M0","M1"])
plt.axhline(0, linestyle="--", color="grey")
plt.ylabel("Effekt (%)")
plt.xlabel("Zeitebene × Young_Account")
plt.title("Instagram (t-1) → Protestaktivität")

# Legende nur eindeutige Einträge
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Variable × DV × Gruppe",
           bbox_to_anchor=(1.05,1), loc='upper left')

plt.tight_layout()
plt.show()
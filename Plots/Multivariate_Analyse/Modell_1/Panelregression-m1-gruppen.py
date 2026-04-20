import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =========================
# 1. DATEN LADEN + KONTROLLVARIABLEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"
df = pd.read_excel(file_path, sheet_name="Zeitreihe-Tag")
df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")

df = df.fillna(0)
df["Date"] = pd.to_datetime(df["Date"])
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

# Kontrollvariablen mergen
df = df.merge(df_controls.rename(columns={"Ort":"Location"})[["Location","Einwohner","Wahl Opposition"]],
              on="Location", how="left")
df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

# =========================
# 2. LAG-VARIABLEN nach Location + Young_Account
# =========================
df = df.sort_values(["Location","Young_Account","Date"])
df["Protests_lag1"] = df.groupby(["Location","Young_Account"])["Protests"].shift(1).fillna(0)
df["participants_lag1"] = df.groupby(["Location","Young_Account"])["participants"].shift(1).fillna(0)

# =========================
# 3. FUNKTION ZUM NB-MODELLIEREN
# =========================
def run_model_nb(df_group, time_col, dv, label):
    # numerische Zeitvariable, um Kollinearität bei kleinen Gruppen zu vermeiden
    df_group = df_group.copy()
    df_group = df_group.reset_index(drop=True)
    df_group["t_numeric"] = np.arange(len(df_group))
    
    # NaN/Inf entfernen
    df_group = df_group.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[dv, "Protests_lag1","participants_lag1","Einwohner","Wahl Opposition"]
    )
    
    formula = f"{dv} ~ Protests_lag1 + participants_lag1 + Einwohner + Q('Wahl Opposition') + C(Location) + t_numeric"
    
    model = smf.glm(formula=formula, data=df_group,
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
# 4. FUNKTION FÜR ALLE ZEITEBENEN UND GROUPS
# =========================
def compute_results_by_group(df, time_col, label_prefix):
    results_all = []
    for group, df_group in df.groupby("Young_Account"):
        label = f"{label_prefix}_Y{group}"
        res_posts, _ = run_model_nb(df_group, time_col, "Posts", label)
        res_accounts, _ = run_model_nb(df_group, time_col, "Active_Accounts", label)
        res = pd.concat([res_posts, res_accounts])
        res["effect_pct"] = (np.exp(res["coef"])-1)*100
        res["ci_low"] = (np.exp(res["coef"]-1.96*res["std_err"])-1)*100
        res["ci_high"] = (np.exp(res["coef"]+1.96*res["std_err"])-1)*100
        res["Young_Account"] = group
        results_all.append(res)
    return pd.concat(results_all)

# =========================
# 5. AGGREGATIONEN
# =========================
# Wöchentlich
df_week = df.groupby(["Location","week","Young_Account"]).agg({
    "Posts":"sum",
    "Active_Accounts":"sum",
    "Protests":"sum",
    "participants":"sum",
    "Einwohner":"first",
    "Wahl Opposition":"first"
}).reset_index().sort_values(["Location","week"])
df_week["Protests_lag1"] = df_week.groupby(["Location","Young_Account"])["Protests"].shift(1).fillna(0)
df_week["participants_lag1"] = df_week.groupby(["Location","Young_Account"])["participants"].shift(1).fillna(0)

# Monatlich
df_month = df.groupby(["Location","month","Young_Account"]).agg({
    "Posts":"sum",
    "Active_Accounts":"sum",
    "Protests":"sum",
    "participants":"sum",
    "Einwohner":"first",
    "Wahl Opposition":"first"
}).reset_index().sort_values(["Location","month"])
df_month["Protests_lag1"] = df_month.groupby(["Location","Young_Account"])["Protests"].shift(1).fillna(0)
df_month["participants_lag1"] = df_month.groupby(["Location","Young_Account"])["participants"].shift(1).fillna(0)

# =========================
# 6. ERGEBNISSE BERECHNEN
# =========================
results_all = []
results_all.append(compute_results_by_group(df,"Date","Daily"))
results_all.append(compute_results_by_group(df_week,"week","Weekly"))
results_all.append(compute_results_by_group(df_month,"month","Monthly"))

results = pd.concat(results_all).reset_index(drop=True)

# =========================
# 7. ERGEBNISSE TERMINALPRINT
# =========================
print(results[["model","Young_Account","variable","dv","effect_pct","ci_low","ci_high"]])



# =========================
# 8. NB-Grafiken & Tabellen für Young vs. Nicht-Young Accounts
# =========================
time_levels = ["Daily","Weekly","Monthly"]
variables = ["Protests_lag1","participants_lag1"]
dvs = ["Posts","Active_Accounts"]

# Farben: DV × Young_Account
colors = {
    ("Posts",0): "tab:blue", ("Posts",1): "lightblue",
    ("Active_Accounts",0): "tab:orange", ("Active_Accounts",1): "moccasin"
}
markers = {"Protests_lag1":"o","participants_lag1":"s"}

plt.figure(figsize=(12,6))
time_order = ["Daily_Y0","Daily_Y1","Weekly_Y0","Weekly_Y1","Monthly_Y0","Monthly_Y1"]

for i, model in enumerate(time_order):
    df_plot = results[results["model"]==model]
    young = 1 if "_Y1" in model else 0
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
                             color=colors[(dv,young)],
                             capsize=5,
                             markersize=8,
                             label=f"{variable} × {dv} × Y{young}")

plt.xticks(ticks=range(len(time_order)),
           labels=["Daily 0","Daily 1","Weekly 0","Weekly 1","Monthly 0","Monthly 1"])
plt.axhline(0,color="grey",linestyle="--")
plt.ylabel("Effekt auf Instagram (%)")
plt.xlabel("Zeitebene × Young_Account")
plt.title("Negative-Binomial-Modelle: Protestaktivität (t-1) → Instagram nach Young_Account")

# Legende nur eindeutige Einträge
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Variable × DV × Gruppe",
           bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# =========================
# 9. Terminal-Tabellen
# =========================
def print_nb_results_table(df_results, title):
    table = df_results[["model","Young_Account","variable","dv","effect_pct","ci_low","ci_high"]].copy()
    table = table.round({
        "effect_pct":3,"ci_low":3,"ci_high":3
    })
    print(f"\n===== {title} =====")
    print(table.to_string(index=False))

for model in time_levels:
    for young in [0,1]:
        df_sub = results[(results["model"].str.contains(model)) & (results["Young_Account"]==young)]
        label = f"{model} Y{young}"
        print_nb_results_table(df_sub, label)

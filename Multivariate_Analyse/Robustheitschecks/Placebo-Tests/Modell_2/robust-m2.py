import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =========================
# 1. DATEN LADEN
# =========================
file_path = "Datensatz-Master-Final.xlsx"

df = pd.read_excel(file_path, sheet_name="Lag-Tag")
df = df.fillna(0)
df["Date"] = pd.to_datetime(df["Date"])

df_controls = pd.read_excel(file_path, sheet_name="Stadt_merged")
df_controls = df_controls.rename(columns={"Ort": "Location"})
df = df.merge(df_controls[["Location","Einwohner","Wahl Opposition"]],
              on="Location", how="left")
df["Einwohner"] = df["Einwohner"].fillna(0)
df["Wahl Opposition"] = df["Wahl Opposition"].fillna(0)

df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time
df = df.sort_values(["Location", "Date"])

# =========================
# 2. HAUPTMODELL-ERGEBNISSE LADEN
# =========================
results_nb = pd.read_csv("results_hauptmodell_m2.csv")
print("Hauptmodell M2 geladen:")
print(results_nb[["variable","dv","model","effect_pct","ci_low","ci_high"]].round(2).to_string(index=False))

# =========================
# 3. AGGREGATIONEN & LAGS VORBEREITEN
# =========================
lag_vars = ["Protests","participants","Posts","Active_Accounts"]
for var in lag_vars:
    df[f"{var}_lag1"] = df.groupby("Location")[var].shift(1).fillna(0)

df_week = df.groupby(["Location", "week"]).agg({
    "Posts": "sum", "Active_Accounts": "sum",
    "Protests": "sum", "participants": "sum",
    "Einwohner": "first", "Wahl Opposition": "first"
}).reset_index()
for var in lag_vars:
    df_week[f"{var}_lag1"] = df_week.groupby("Location")[var].shift(1).fillna(0)

df_month = df.groupby(["Location", "month"]).agg({
    "Posts": "sum", "Active_Accounts": "sum",
    "Protests": "sum", "participants": "sum",
    "Einwohner": "first", "Wahl Opposition": "first"
}).reset_index()
for var in lag_vars:
    df_month[f"{var}_lag1"] = df_month.groupby("Location")[var].shift(1).fillna(0)

df["Date_cat"]        = df["Date"].dt.strftime("%Y-%m-%d")
df_week["week_cat"]   = df_week["week"].dt.strftime("%Y-%m-%d")
df_month["month_cat"] = df_month["month"].dt.strftime("%Y-%m")

# =========================
# 4. FUTURE LAGS ERSTELLEN
# =========================
placebo_vars = ["Posts", "Active_Accounts"]

for var in placebo_vars:
    df[f"{var}_future1"]       = df.groupby("Location")[var].shift(-1).fillna(0)
    df_week[f"{var}_future1"]  = df_week.groupby("Location")[var].shift(-1).fillna(0)
    df_month[f"{var}_future1"] = df_month.groupby("Location")[var].shift(-1).fillna(0)

# =========================
# 5. PLACEBO-MODELLFUNKTION
# =========================
def run_placebo_model_m2(data, time_var, dv, label):
    cols = [dv, "Posts_future1", "Active_Accounts_future1", "Location", time_var]
    data_clean = data[cols].replace([np.inf, -np.inf], np.nan).dropna()

    formula = (f"{dv} ~ Posts_future1 + Active_Accounts_future1"
               f" + C(Location) + C({time_var})")
    model = smf.glm(formula=formula, data=data_clean,
                    family=sm.families.NegativeBinomial()).fit()
    results = pd.DataFrame({
        "variable": ["Posts_future1", "Active_Accounts_future1"],
        "coef": model.params[["Posts_future1", "Active_Accounts_future1"]],
        "std_err": model.bse[["Posts_future1", "Active_Accounts_future1"]],
        "dv": dv,
        "model": label
    })
    return results, model

# =========================
# 6. PLACEBO-MODELLE AUSFÜHREN
# =========================
plac_day_protests,     _ = run_placebo_model_m2(df,       "Date_cat",  "Protests",     "Daily")
plac_day_participants, _ = run_placebo_model_m2(df,       "Date_cat",  "participants", "Daily")
plac_week_protests,    _ = run_placebo_model_m2(df_week,  "week_cat",  "Protests",     "Weekly")
plac_month_protests,   _ = run_placebo_model_m2(df_month, "month_cat", "Protests",     "Monthly")

results_placebo = pd.concat([
    plac_day_protests, plac_day_participants,
    plac_week_protests,
    plac_month_protests,
]).reset_index(drop=True)

results_placebo["effect_pct"] = (np.exp(results_placebo["coef"]) - 1) * 100
results_placebo["ci_low"]     = (np.exp(results_placebo["coef"] - 1.96 * results_placebo["std_err"]) - 1) * 100
results_placebo["ci_high"]    = (np.exp(results_placebo["coef"] + 1.96 * results_placebo["std_err"]) - 1) * 100

print("\n===== Zeitplacebo M2-Ergebnisse =====")
print(results_placebo[["variable","dv","model","effect_pct","ci_low","ci_high"]].round(2).to_string(index=False))

# =========================
# 7. VISUALISIERUNG: HAUPTMODELL VS. PLACEBO (participants, Daily)
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_configs = [
    ("Posts_lag1",           "Posts_future1",           "Posts → participants"),
    ("Active_Accounts_lag1", "Active_Accounts_future1", "Active Accounts → participants"),
]

for row, (main_var, placebo_var, row_title) in enumerate(plot_configs):
    for col, (results, title, var_name) in enumerate([
        (results_nb,      "Hauptmodell (Lag t-1)", main_var),
        (results_placebo, "Placebo (Lag t+1)",     placebo_var),
    ]):
        ax = axes[row][col]
        subset = results[
            (results["variable"] == var_name) &
            (results["model"] == "Daily") &
            (results["dv"] == "participants")
        ].copy().reset_index(drop=True)

        if subset.empty:
            ax.set_visible(False)
            continue

        colors = [
            "steelblue" if lo > 0 or hi < 0 else "salmon"
            for lo, hi in zip(subset["ci_low"], subset["ci_high"])
        ]

        ax.barh(subset["dv"], subset["effect_pct"],
                xerr=[subset["effect_pct"] - subset["ci_low"],
                      subset["ci_high"] - subset["effect_pct"]],
                color=colors, capsize=4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{row_title}\n{title}", fontsize=10)
        ax.set_xlabel("Effekt in %")

plt.suptitle("Instagram → participants: Hauptmodell vs. Zeitplacebo (Daily)", fontsize=13)
plt.tight_layout()
plt.savefig("zeitplacebo_vergleich_m2_participants.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot gespeichert: zeitplacebo_vergleich_m2_participants.png")

# =========================
# 8. LAG-DECAY PLACEBO-TEST (Daily, reduzierter Datensatz)
# =========================
df_reduced = df[df["Protests"] > 0].copy()
results_decay = []

for lag in [1, 2, 3, 7]:
    print(f"Berechne Lag t+{lag}...")
    for var in ["Posts", "Active_Accounts"]:
        df_reduced[f"{var}_future{lag}"] = df_reduced.groupby("Location")[var].shift(-lag).fillna(0)

    print(f"  Modell: participants")
    try:
        cols = ["participants", f"Posts_future{lag}",
                f"Active_Accounts_future{lag}", "Location", "Date_cat"]
        data_clean = df_reduced[cols].replace([np.inf, -np.inf], np.nan).dropna()

        formula = (f"participants ~ Posts_future{lag} + Active_Accounts_future{lag}"
                   f" + C(Location) + C(Date_cat)")
        model = smf.glm(formula=formula, data=data_clean,
                        family=sm.families.NegativeBinomial()).fit()

        # Posts_future separat speichern
        for iv in [f"Posts_future{lag}", f"Active_Accounts_future{lag}"]:
            coef    = model.params[iv]
            std_err = model.bse[iv]
            results_decay.append({
                "lag":        lag,
                "lag_label":  f"t+{lag}",
                "variable":   iv.replace(str(lag), ""),  # z.B. "Posts_future"
                "dv":         "participants",
                "effect_pct": (np.exp(coef) - 1) * 100,
                "ci_low":     (np.exp(coef - 1.96 * std_err) - 1) * 100,
                "ci_high":    (np.exp(coef + 1.96 * std_err) - 1) * 100,
            })
        del model

    except Exception as e:
        print(f"  Fehler: {e}")
        continue

    pd.DataFrame(results_decay).to_csv("lag_decay_m2_participants.csv", index=False)
    print(f"  Lag t+{lag} gespeichert.")

df_decay = pd.DataFrame(results_decay)
print("\n===== Lag-Decay Placebo M2 (participants, Daily) =====")
print(df_decay.round(2).to_string(index=False))

# --- Visualisierung ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_configs = [
    ("Posts_future",           "Posts_lag1",           "Posts → participants"),
    ("Active_Accounts_future", "Active_Accounts_lag1", "Active Accounts → participants"),
]

for ax, (placebo_var, main_var, title) in zip(axes, plot_configs):
    subset = df_decay[df_decay["variable"] == placebo_var].sort_values("lag").reset_index(drop=True)

    ax.plot(subset["lag_label"], subset["effect_pct"],
            marker="o", color="steelblue", linewidth=2, label="Placebo-Effekt")
    ax.fill_between(range(len(subset)),
                    subset["ci_low"].values,
                    subset["ci_high"].values,
                    alpha=0.2, color="steelblue", label="95% KI")

    ref_rows = results_nb[
        (results_nb["variable"] == main_var) &
        (results_nb["model"] == "Daily") &
        (results_nb["dv"] == "participants")
    ]
    if not ref_rows.empty:
        ref = ref_rows["effect_pct"].values[0]
        ax.axhline(ref, color="coral", linewidth=1.5, linestyle="--",
                   label=f"Hauptmodell t-1 ({ref:.1f}%)")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Placebo-Lag")
    ax.set_ylabel("Effekt in %")
    ax.legend()

plt.suptitle("Lag-Decay Placebo: Instagram → participants (Daily)", fontsize=13)
plt.tight_layout()
plt.savefig("lag_decay_placebo_m2_participants.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot gespeichert: lag_decay_placebo_m2_participants.png")
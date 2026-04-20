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
# 2. HAUPTMODELL-ERGEBNISSE LADEN (nicht neu berechnen!)
# =========================
results_nb = pd.read_csv("results_hauptmodell.csv")
print("Hauptmodell-Ergebnisse geladen:")
print(results_nb[["variable","dv","model","effect_pct","ci_low","ci_high"]].round(2).to_string(index=False))

# =========================
# 3. AGGREGATIONEN VORBEREITEN
# =========================
df_week = df.groupby(["Location", "week"]).agg({
    "Posts": "sum", "Active_Accounts": "sum",
    "Protests": "sum", "participants": "sum",
    "Einwohner": "first", "Wahl Opposition": "first"
}).reset_index()

df_month = df.groupby(["Location", "month"]).agg({
    "Posts": "sum", "Active_Accounts": "sum",
    "Protests": "sum", "participants": "sum",
    "Einwohner": "first", "Wahl Opposition": "first"
}).reset_index()

df["Date_cat"]        = df["Date"].dt.strftime("%Y-%m-%d")
df_week["week_cat"]   = df_week["week"].dt.strftime("%Y-%m-%d")
df_month["month_cat"] = df_month["month"].dt.strftime("%Y-%m")

# =========================
# 4. FUTURE LAGS ERSTELLEN
# =========================
placebo_vars = ["Protests", "participants"]

for var in placebo_vars:
    df[f"{var}_future1"]       = df.groupby("Location")[var].shift(-1).fillna(0)
    df_week[f"{var}_future1"]  = df_week.groupby("Location")[var].shift(-1).fillna(0)
    df_month[f"{var}_future1"] = df_month.groupby("Location")[var].shift(-1).fillna(0)

# =========================
# 5. PLACEBO-MODELLE AUSFÜHREN
# =========================
def run_placebo_model(data, time_var, dv, label):
    formula = f"{dv} ~ Protests_future1 + participants_future1 + C(Location) + C({time_var})"
    model = smf.glm(formula=formula, data=data,
                    family=sm.families.NegativeBinomial()).fit()
    results = pd.DataFrame({
        "variable": ["Protests_future1", "participants_future1"],
        "coef": model.params[["Protests_future1", "participants_future1"]],
        "std_err": model.bse[["Protests_future1", "participants_future1"]],
        "dv": dv,
        "model": label
    })
    return results, model

plac_day_posts,      _ = run_placebo_model(df,       "Date_cat",  "Posts",            "Daily")
plac_day_accounts,   _ = run_placebo_model(df,       "Date_cat",  "Active_Accounts",  "Daily")
plac_week_posts,     _ = run_placebo_model(df_week,  "week_cat",  "Posts",            "Weekly")
plac_week_accounts,  _ = run_placebo_model(df_week,  "week_cat",  "Active_Accounts",  "Weekly")
plac_month_posts,    _ = run_placebo_model(df_month, "month_cat", "Posts",            "Monthly")
plac_month_accounts, _ = run_placebo_model(df_month, "month_cat", "Active_Accounts",  "Monthly")

results_placebo = pd.concat([
    plac_day_posts, plac_day_accounts,
    plac_week_posts, plac_week_accounts,
    plac_month_posts, plac_month_accounts
]).reset_index(drop=True)

results_placebo["effect_pct"] = (np.exp(results_placebo["coef"]) - 1) * 100
results_placebo["ci_low"]     = (np.exp(results_placebo["coef"] - 1.96 * results_placebo["std_err"]) - 1) * 100
results_placebo["ci_high"]    = (np.exp(results_placebo["coef"] + 1.96 * results_placebo["std_err"]) - 1) * 100

print("\n===== Zeitplacebo-Ergebnisse =====")
print(results_placebo[["variable","dv","model","effect_pct","ci_low","ci_high"]].round(2).to_string(index=False))

# =========================
# 6. VISUALISIERUNG
# =========================
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey="row")

plot_configs = [
    (results_nb,      "Hauptmodell (Lag t-1)", "Protests_lag1"),
    (results_placebo, "Placebo (Lag t+1)",     "Protests_future1"),
]

aggregations_plot = [
    ("Daily",   "Täglich"),
    ("Weekly",  "Wöchentlich"),
    ("Monthly", "Monatlich"),
]

for row, (agg_label, agg_title) in enumerate(aggregations_plot):
    for col, (results, title, var_name) in enumerate(plot_configs):
        ax = axes[row][col]
        subset = results[
            (results["variable"] == var_name) &
            (results["model"] == agg_label)
        ].copy().reset_index(drop=True)

        colors = [
            "steelblue" if lo > 0 or hi < 0 else "salmon"
            for lo, hi in zip(subset["ci_low"], subset["ci_high"])
        ]

        ax.barh(subset["dv"], subset["effect_pct"],
                xerr=[subset["effect_pct"] - subset["ci_low"],
                      subset["ci_high"] - subset["effect_pct"]],
                color=colors, capsize=4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{agg_title} – {title}", fontsize=10)
        ax.set_xlabel("Effekt in %")

plt.suptitle("Protests → Instagram: Hauptmodell vs. Zeitplacebo", fontsize=13)
plt.tight_layout()
plt.savefig("zeitplacebo_vergleich.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot gespeichert: zeitplacebo_vergleich.png")

# Autokorrelation der Protest-Variable prüfen
from statsmodels.stats.stattools import durbin_watson

# Beispiel für einen Ort
loc = df["Location"].unique()[0]
dw = durbin_watson(df[df["Location"] == loc]["Protests"].values)
print(f"Durbin-Watson: {dw:.3f}")  # ~2.0 = keine Autokorrelation


# =========================
# LAG-DECAY PLACEBO-TEST
# =========================

results_decay = []

for lag in [1, 2, 3, 7]:
    # Future Lags erstellen
    for var in ["Protests", "participants"]:
        df[f"{var}_future{lag}"] = df.groupby("Location")[var].shift(-lag).fillna(0)

    # Modell pro DV
    for dv in ["Posts", "Active_Accounts"]:
        formula = (f"{dv} ~ Protests_future{lag} + participants_future{lag}"
                   f" + C(Location) + C(Date_cat)")
        model = smf.glm(formula=formula, data=df,
                        family=sm.families.NegativeBinomial()).fit()

        coef    = model.params[f"Protests_future{lag}"]
        std_err = model.bse[f"Protests_future{lag}"]

        results_decay.append({
            "lag":        lag,
            "lag_label":  f"t+{lag}",
            "dv":         dv,
            "effect_pct": (np.exp(coef) - 1) * 100,
            "ci_low":     (np.exp(coef - 1.96 * std_err) - 1) * 100,
            "ci_high":    (np.exp(coef + 1.96 * std_err) - 1) * 100,
        })

df_decay = pd.DataFrame(results_decay)

print("\n===== Lag-Decay Placebo (Daily) =====")
print(df_decay.round(2).to_string(index=False))

# --- Visualisierung ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, dv in zip(axes, ["Posts", "Active_Accounts"]):
    subset = df_decay[df_decay["dv"] == dv].sort_values("lag")

    # Placebo-Effekte
    ax.plot(subset["lag_label"], subset["effect_pct"],
            marker="o", color="steelblue", linewidth=2, label="Placebo-Effekt")
    ax.fill_between(range(len(subset)),
                    subset["ci_low"].values,
                    subset["ci_high"].values,
                    alpha=0.2, color="steelblue", label="95% KI")

    # Hauptmodell-Referenzlinie
    ref = results_nb[
        (results_nb["variable"] == "Protests_lag1") &
        (results_nb["model"] == "Daily") &
        (results_nb["dv"] == dv)
    ]["effect_pct"].values[0]
    ax.axhline(ref, color="coral", linewidth=1.5, linestyle="--",
               label=f"Hauptmodell t-1 ({ref:.1f}%)")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"Lag-Decay: {dv}", fontsize=11)
    ax.set_xlabel("Placebo-Lag")
    ax.set_ylabel("Effekt in %")
    ax.legend()

plt.suptitle("Placebo-Effekt nach Lag-Distanz (Daily)", fontsize=13)
plt.tight_layout()
plt.savefig("lag_decay_placebo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot gespeichert: lag_decay_placebo.png")

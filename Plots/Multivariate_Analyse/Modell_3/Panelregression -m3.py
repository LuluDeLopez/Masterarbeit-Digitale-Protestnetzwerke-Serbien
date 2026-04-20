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
# =========================
lag_vars = ["Protests","Posts","participants","Active_Accounts"]
for var in lag_vars:
    df[f"{var}_lag1"] = df.groupby("Location")[var].shift(1)
    df[f"{var}_lag2"] = df.groupby("Location")[var].shift(2)

for var in lag_vars:
    df[f"{var}_log"] = np.log1p(df[var])
    df[f"{var}_log_lag1"] = np.log1p(df[f"{var}_lag1"])
    df[f"{var}_log_lag2"] = np.log1p(df[f"{var}_lag2"])

df = df.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# 3. ZEIT-AGGREGATION
# =========================
df["week"] = df["Date"].dt.to_period("W").dt.start_time
df["month"] = df["Date"].dt.to_period("M").dt.start_time

# wöchentliche Aggregation
df_week = df.groupby(["Location","week"]).agg({
    "Posts":"sum","Active_Accounts":"sum","Protests":"sum","participants":"sum",
    "Einwohner":"first","Wahl Opposition":"first"
}).reset_index()

# monatl. Aggregation
df_month = df.groupby(["Location","month"]).agg({
    "Posts":"sum","Active_Accounts":"sum","Protests":"sum","participants":"sum",
    "Einwohner":"first","Wahl Opposition":"first"
}).reset_index()

# Lags für aggregierte Daten
agg_dfs = {"Daily": df, "Weekly": df_week, "Monthly": df_month}
for name, df_agg in agg_dfs.items():
    for var in ["Posts","Active_Accounts","Protests","participants"]:
        df_agg[f"{var}_lag1"] = df_agg.groupby("Location")[var].shift(1)
        df_agg[f"{var}_lag2"] = df_agg.groupby("Location")[var].shift(2)
        df_agg[f"{var}_log"] = np.log1p(df_agg[var])
        df_agg[f"{var}_log_lag1"] = np.log1p(df_agg[f"{var}_lag1"])
        df_agg[f"{var}_log_lag2"] = np.log1p(df_agg[f"{var}_lag2"])
    df_agg.dropna(inplace=True)
agg_dfs["Daily"], agg_dfs["Weekly"], agg_dfs["Monthly"] = df, df_week, df_month

# =========================
# 4. SEQUENTIELLE RÜCKKOPPLUNG MIT FE
# =========================
def sequential_feedback_fe(df, step1_dv, step1_iv, step2_dv, step2_iv,
                           control_vars=None, time_var=None, label=""):
    if control_vars is None:
        control_vars = []

    # STEP1: t → t+1
    formula1 = f"{step1_dv} ~ {step1_iv} + " + " + ".join(control_vars)
    if time_var:
        formula1 += f" + C(Location) + C({time_var})"
    else:
        formula1 += " + C(Location)"

    step1_model = smf.glm(formula=formula1, data=df, family=sm.families.NegativeBinomial()).fit()

    # STEP2: t+1 → t+2
    formula2 = f"{step2_dv} ~ {step2_iv} + " + " + ".join(control_vars)
    if time_var:
        formula2 += f" + C(Location) + C({time_var})"
    else:
        formula2 += " + C(Location)"

    step2_model = smf.glm(formula=formula2, data=df, family=sm.families.NegativeBinomial()).fit()

    # Effekte in %
    step1_effect = (np.exp(step1_model.params[step1_iv]) - 1) * 100
    step2_effect = (np.exp(step2_model.params[step2_iv]) - 1) * 100
    indirect_effect = (np.exp(step1_model.params[step1_iv] * step2_model.params[step2_iv]) - 1) * 100

    # Standardfehler
    se1 = step1_model.bse[step1_iv]
    se2 = step2_model.bse[step2_iv]
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
# 5. RUN ALL LEVELS
# =========================
controls = ["Protests_log_lag2", "participants_log_lag2", "Posts_log_lag2", "Active_Accounts_log_lag2"]


# =========================
# Angepasste Labels für Tabellen
# =========================
h3a_hypotheses = [
    {"label":"H3a1: Protests → Posts → Protests", "step1_dv":"Posts_log_lag1", "step1_iv":"Protests_log_lag2",
     "step2_dv":"Protests_log", "step2_iv":"Posts_log_lag1"},
    {"label":"H3a2: Protests → ActiveAccounts → Protests", "step1_dv":"Active_Accounts_log_lag1", "step1_iv":"Protests_log_lag2",
     "step2_dv":"Protests_log", "step2_iv":"Active_Accounts_log_lag1"},
    {"label":"H3a3: Participants → Posts → Participants", "step1_dv":"Posts_log_lag1", "step1_iv":"participants_log_lag2",
     "step2_dv":"participants_log", "step2_iv":"Posts_log_lag1"},
]

h3b_hypotheses = [
    {"label":"H3b1: Posts → Protests → Posts", "step1_dv":"Protests_log_lag1", "step1_iv":"Posts_log_lag2",
     "step2_dv":"Posts_log", "step2_iv":"Protests_log_lag1"},
    {"label":"H3b2:ActiveAccounts → Protests → ActiveAccounts", "step1_dv":"Protests_log_lag1", "step1_iv":"Active_Accounts_log_lag2",
     "step2_dv":"Active_Accounts_log", "step2_iv":"Protests_log_lag1"},
    {"label":"H3b3: Posts → Participants → Posts", "step1_dv":"participants_log_lag1", "step1_iv":"Posts_log_lag2",
     "step2_dv":"Posts_log", "step2_iv":"participants_log_lag1"},
]

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
        # Daily: keine Tages-FE, nur Location-FE

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
# 6. RUN MODELS
# =========================
results_df = run_h3_all_levels()
print(results_df.round(1))


# =========================
# 7. Ergebnisse speichern
# =========================
output_file = "H3_Actual_Results.xlsx"
results_df.to_excel(output_file, index=False)
print(f"✅ H3 Actual-Results gespeichert in {output_file}")


def plot_and_print_feedback_tables_per_level(df_results, level):
    """
    Plotte und print Tabellen pro Zeitebene (Daily, Weekly, Monthly)
    inklusive Sternchen für signifikante Effekte und indirekter CI-Spalte.
    Farbe: grün = positiver Effekt, rot = negativer Effekt
    """
    df_level = df_results[df_results["label"].str.startswith(level)].copy()

    # =========================
    # Effekte-Tabelle mit Sternchen
    # =========================
    df_effects = df_level.copy()
    for effect, ci_low, ci_high in [
        ("step1_effect", "step1_ci_low", "step1_ci_high"),
        ("step2_effect", "step2_ci_low", "step2_ci_high"),
        ("indirect_effect", "indirect_ci_low", "indirect_ci_high")
    ]:
        df_effects[effect] = df_effects.apply(
            lambda row: f"{row[effect]:.1f}{'*' if row[ci_low] > 0 or row[ci_high] < 0 else ''}",
            axis=1
        )

    # zusätzliche Spalte für indirektes CI
    df_effects["indirect_ci"] = df_level.apply(
        lambda row: f"[{row['indirect_ci_low']:.1f}; {row['indirect_ci_high']:.1f}]", axis=1
    )

    df_effects_to_print = df_effects[["label","step1_effect","step2_effect","indirect_effect","indirect_ci"]]

    # Print im Terminal
    print(f"\n===== {level} - Effekte (%) =====")
    print(df_effects_to_print.to_string(index=False))

    # Farblogik: grün = positiver indirekter Effekt, rot = negativer indirekter Effekt, neutral = 0
    row_colors = []
    for eff in df_level["indirect_effect"]:
        if eff > 0:
            row_colors.append([0.8,1.0,0.8])
        elif eff < 0:
            row_colors.append([1.0,0.8,0.8])
        else:
            row_colors.append([1,1,1])
    cell_colors = [[c for _ in range(df_effects_to_print.shape[1]-1)] + [[1,1,1]]  # CI-Spalte immer weiß
                   for c in row_colors]

    # Plot der Effekte-Tabelle
    fig, ax = plt.subplots(figsize=(10, max(2, df_effects_to_print.shape[0]*0.7)))
    ax.axis("off")
    table = ax.table(
        cellText=df_effects_to_print.values,
        colLabels=df_effects_to_print.columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(list(range(df_effects_to_print.shape[1])))
    for (i,j), cell in table.get_celld().items():
        cell.set_height(0.16 if i==0 else 0.14)
    plt.title(f"{level} - Effekte (%)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

    # =========================
    # Konfidenzintervalle-Tabelle
    # =========================
    df_cis = df_level[[
        "label",
        "step1_ci_low","step1_ci_high",
        "step2_ci_low","step2_ci_high",
        "indirect_ci_low","indirect_ci_high"
    ]].round(1)

    # Print im Terminal
    print(f"\n===== {level} - 95%-Konfidenzintervalle (%) =====")
    print(df_cis.to_string(index=False))

    # Plot der CI-Tabelle (neutral weiß)
    fig, ax = plt.subplots(figsize=(10, max(2, df_cis.shape[0]*0.7)))
    ax.axis("off")
    table = ax.table(
        cellText=df_cis.values,
        colLabels=df_cis.columns,
        cellLoc="center",
        loc="center",
        cellColours=[[ [1,1,1] for _ in range(df_cis.shape[1])] for _ in range(df_cis.shape[0])]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(list(range(df_cis.shape[1])))
    for (i,j), cell in table.get_celld().items():
        cell.set_height(0.16 if i==0 else 0.14)
    plt.title(f"{level} - 95%-Konfidenzintervalle (%)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()



# =========================
# Plotte + print pro Zeitebene
# =========================
for level in ["Daily","Weekly","Monthly"]:
    print(f"\n===== ZEITEBENE: {level} =====\n")
    
   # H3a: Offline initiiert
    df_h3a = results_df[results_df["label"].str.contains("H3a")]
    for level in ["Daily","Weekly","Monthly"]:
        df_level = df_h3a[df_h3a["label"].str.startswith(level)]
        plot_and_print_feedback_tables_per_level(df_level, level)

    # H3b: Online initiiert
    df_h3b = results_df[results_df["label"].str.contains("H3b")]
    for level in ["Daily","Weekly","Monthly"]:
        df_level = df_h3b[df_h3b["label"].str.startswith(level)]
        plot_and_print_feedback_tables_per_level(df_level, level)




import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_feedback_by_hypothesis(results_df, hypothesis_label):
    """
    Plotte partielle Effekte mit 95%-Konfidenzintervallen getrennt für H3a oder H3b.
    """
    df_hyp = results_df[results_df["label"].str.contains(hypothesis_label)]
    time_levels = ["Daily","Weekly","Monthly"]
    effects_to_plot = ["step1_effect", "step2_effect", "indirect_effect"]
    colors = {"step1_effect":"tab:blue","step2_effect":"tab:orange","indirect_effect":"tab:green"}
    markers = {"step1_effect":"o","step2_effect":"s","indirect_effect":"^"}

    plt.figure(figsize=(12,6))

    for i, level in enumerate(time_levels):
        df_level = df_hyp[df_hyp["label"].str.startswith(level)]
        for j, effect in enumerate(effects_to_plot):
            for idx, row in df_level.iterrows():
                x = i + (j - 1)*0.15  # kleine Verschiebung pro Effekt
                y = row[effect]
                ci_low = row[effect] - row[effect.replace("effect","ci_low")]
                ci_high = row[effect.replace("effect","ci_high")] - row[effect]
                plt.errorbar(
                    x, y,
                    yerr=[[ci_low],[ci_high]],
                    fmt=markers[effect],
                    color=colors[effect],
                    capsize=5,
                    markersize=8,
                    label=f"{effect} × {row['label']}"
                )

    plt.xticks(ticks=range(len(time_levels)), labels=["Tag","Woche","Monat"])
    plt.axhline(0, color="grey", linestyle="--")
    plt.ylabel("Partielle Effekte (%)")
    plt.xlabel("Zeitebene")
    plt.title(f"Sequenzielles Feedback: {hypothesis_label}")
    
    # Legende bereinigen (duplikate entfernen)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Effekt × Hypothese", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()

# =========================
# H3a: Offline initiiert
# =========================
plot_feedback_by_hypothesis(results_df, "H3a")

# =========================
# H3b: Online initiiert
# =========================
plot_feedback_by_hypothesis(results_df, "H3b")










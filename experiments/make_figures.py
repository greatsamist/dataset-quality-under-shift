"""Generate the thesis figures from results/ into thesis_project/figures/ (PNG, 200 dpi).
Fonts are sized for a 6.3 in text width at 12 pt body text."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results", help="results root holding the run directories")
ap.add_argument("--out", default="thesis_project/figures", help="destination for the PNGs")
args = ap.parse_args()

R = args.results; OUT = args.out; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10, "figure.dpi": 200})
MODELS = C.MODEL_NAMES; SHORT = {"Logistic Regression": "LR", "LightGBM": "LightGBM"}
ID, SH, GAP = "In-domain Test", "Shifted Test", "Generalization Gap (Shifted - In-domain)"
COL = {ID: "#1f77b4", SH: "#d62728"}

def load(run):
    p = f"{R}/{run}/raw_results.csv"; return pd.read_csv(p, keep_default_na=False, na_values=[""]) if os.path.exists(p) else None
def ms(df, exp, cond, model, ts, metric, **kw):
    q = (df.Experiment == exp) & (df.Condition == cond) & (df.Model == model) & (df["Test Set"] == ts)
    for k, v in kw.items(): q &= df[k] == v
    s = df[q][metric]; return s.mean(), s.std(ddof=1)
def paired(df, exp, cond, model, ts, metric, **kw):
    q = (df.Experiment == exp) & (df.Condition == cond) & (df.Model == model) & (df["Test Set"] == ts)
    for k, v in kw.items(): q &= df[k] == v
    s = df[q].set_index("Random Seed")[metric]
    b = df[(df.Experiment == "Baseline") & (df.Model == model) & (df["Test Set"] == ts)].set_index("Random Seed")[metric]
    d = (s - b).dropna(); return d.mean(), d.std(ddof=1)
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close(fig); print("wrote", name)

dg, an = load("diabetes"), load("acs")

# ---- Fig: reliability diagrams (baseline) ----
def reliability(run, name, title):
    p = f"{R}/{run}/reliability_bins.csv"
    if not os.path.exists(p): return
    b = pd.read_csv(p); b = b[(b.Experiment == "Baseline")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, m in zip(axes, MODELS):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for ts in [ID, SH]:
            g = b[(b.Model == m) & (b["Test Set"] == ts)].groupby("bin").agg(mean_pred=("mean_pred", "mean"), obs=("obs_rate", "mean"), n=("n", "mean")).dropna()
            ax.plot(g.mean_pred, g.obs, "o-", color=COL[ts], label="In-domain" if ts == ID else "Shifted", ms=5)
            for r in g.itertuples():
                if r.n >= 1: ax.annotate(f"{int(r.n):,}", (r.mean_pred, r.obs), textcoords="offset points", xytext=(0, 6 if ts == ID else -12), ha="center", fontsize=7, color=COL[ts])
        ax.set_title(SHORT[m]); ax.set_xlabel("Mean predicted probability (10 equal-width bins)"); ax.grid(ls="--", alpha=.4)
    axes[0].set_ylabel("Observed positive rate"); axes[0].legend(loc="upper left"); fig.suptitle(title, fontsize=12)
    save(fig, name)
reliability("diabetes", "fig_reliability_diabetes", "Diabetes, clean models; labels = mean bin count per seed")
reliability("acs", "fig_reliability_acs", "ACSEmployment, clean models; labels = mean bin count per seed")

# ---- Fig: label noise (2x2) ----
if dg is not None and an is not None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    rates = [0.0, 0.05, 0.10, 0.20]
    for row, (d, dname) in enumerate([(dg, "Diabetes"), (an, "ACSEmployment")]):
        for col, met in enumerate(["AUROC", "Brier Score"]):
            ax = axes[row, col]
            for m, ls in zip(MODELS, ["-", "--"]):
                for ts in [ID, SH]:
                    mu = [ms(d, "Label Noise", f"{r*100:.0f}% Noise", m, ts, met)[0] for r in rates]
                    sd = [ms(d, "Label Noise", f"{r*100:.0f}% Noise", m, ts, met)[1] for r in rates]
                    ax.errorbar([r * 100 for r in rates], mu, yerr=sd, ls=ls, marker="o", ms=4, capsize=3, color=COL[ts],
                                label=f"{SHORT[m]}, {'in-domain' if ts == ID else 'shifted'}")
            ax.set_title(f"{dname}: {met.replace(' Score','')}"); ax.set_xlabel("Training labels flipped (%)"); ax.grid(ls="--", alpha=.4)
    axes[0, 0].legend(fontsize=8); save(fig, "fig_label_noise")

# ---- Fig: feature shift deltas (paired, both domains) ----
if dg is not None and an is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, (d, groups, dname) in zip(axes, [(dg, ["Diagnoses", "Medications"], "Diabetes"), (an, ["Education", "Disability", "Citizenship_Nativity_Ancestry"], "ACSEmployment")]):
        x = np.arange(len(groups)); w = 0.2
        for i, (m, ts) in enumerate([(m, ts) for m in MODELS for ts in (ID, SH)]):
            mu = [paired(d, "Feature Shift", g, m, ts, "AUROC")[0] for g in groups]; sd = [paired(d, "Feature Shift", g, m, ts, "AUROC")[1] for g in groups]
            ax.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=2, color=COL[ts], hatch="" if m == MODELS[0] else "//", edgecolor="k", lw=.5,
                   label=f"{SHORT[m]}, {'in-domain' if ts == ID else 'shifted'}")
        ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([g.replace("_", "/\n") for g in groups], fontsize=9)
        ax.set_title(dname); ax.grid(axis="y", ls="--", alpha=.4)
    axes[0].set_ylabel("$\\Delta$AUROC (stressed $-$ clean)"); axes[0].legend(fontsize=8, loc="upper right"); save(fig, "fig_feature_shift_delta")

# ---- Fig: prediction distributions before/after blanking (Diabetes, shifted set, seed 42) ----
for run in ["diabetes"]:
    p = f"{R}/{run}/predictions.npz"
    if not os.path.exists(p): continue
    P = np.load(p)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), sharex=True)
    for row, m in enumerate(MODELS):
        for col, grp in enumerate(["Diagnoses", "Medications"]):
            ax = axes[row, col]
            p0 = P[f"Baseline|None|{m}|42|{SH}|p"]; p1 = P[f"Feature Shift|{grp}|{m}|42|{SH}|p"]
            y = P[f"Baseline|None|{m}|42|{SH}|y"]
            ax.hist(p0, bins=40, range=(0, 1), alpha=.55, color="#1f77b4", label=f"clean (mean {p0.mean():.2f})")
            ax.hist(p1, bins=40, range=(0, 1), alpha=.55, color="#ff7f0e", label=f"{grp} blanked (mean {p1.mean():.2f})")
            ax.axvline(y.mean(), color="k", ls="--", lw=1, label=f"prevalence {y.mean():.2f}")
            ax.set_title(f"{SHORT[m]}, {grp} blanked"); ax.legend(fontsize=8); ax.grid(ls="--", alpha=.3)
    for ax in axes[1]: ax.set_xlabel("Predicted probability (shifted test set, seed 42)")
    save(fig, "fig_blanking_predictions")

# ---- Fig: cross-stressor (paired deltas on the shifted set; identical quantity for both datasets) ----
if dg is not None and an is not None:
    S = {"Diabetes": [("Label noise 20%", "Label Noise", "20% Noise", {}), ("Missing 20% MCAR", "Missingness", "MCAR", {"Missing Rate": .2}), ("Feature shift\n(Diagnoses)", "Feature Shift", "Diagnoses", {})],
         "ACSEmployment": [("Label noise 20%", "Label Noise", "20% Noise", {}), ("Missing 20% MCAR", "Missingness", "MCAR", {"Missing Rate": .2}), ("Feature shift\n(Disability)", "Feature Shift", "Disability", {})]}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    for row, (dname, d) in enumerate([("Diabetes", dg), ("ACSEmployment", an)]):
        for col, met in enumerate(["AUROC", "Brier Score"]):
            ax = axes[row, col]; x = np.arange(3); w = .35
            for i, m in enumerate(MODELS):
                mu = [paired(d, e, c, m, SH, met, **kw)[0] for _, e, c, kw in S[dname]]; sd = [paired(d, e, c, m, SH, met, **kw)[1] for _, e, c, kw in S[dname]]
                ax.bar(x + (i - .5) * w, mu, w, yerr=sd, capsize=3, label=SHORT[m], color=["#4c72b0", "#dd8452"][i])
            ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([s[0] for s in S[dname]], fontsize=9)
            ax.set_title(f"{dname}: $\\Delta${met.replace(' Score','')}, shifted set"); ax.grid(axis="y", ls="--", alpha=.4)
    axes[0, 0].legend(); save(fig, "fig_cross_stressor")

# ---- Fig: missingness deltas at 20% ----
if dg is not None and an is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, (d, scen, dname) in zip(axes, [(dg, ["MCAR", "MAR_Female", "MAR_Age_Older"], "Diabetes"), (an, ["MCAR", "MAR_Age_Older"], "ACSEmployment")]):
        x = np.arange(len(scen)); w = .2
        for i, (m, ts) in enumerate([(m, ts) for m in MODELS for ts in (ID, SH)]):
            mu = [paired(d, "Missingness", s, m, ts, "AUROC", **{"Missing Rate": .2})[0] for s in scen]; sd = [paired(d, "Missingness", s, m, ts, "AUROC", **{"Missing Rate": .2})[1] for s in scen]
            ax.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=2, color=COL[ts], hatch="" if m == MODELS[0] else "//", edgecolor="k", lw=.5, label=f"{SHORT[m]}, {'in-domain' if ts == ID else 'shifted'}")
        ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in scen], fontsize=9)
        ax.set_title(dname); ax.grid(axis="y", ls="--", alpha=.4)
    axes[0].set_ylabel("$\\Delta$AUROC (stressed $-$ clean)"); axes[0].legend(fontsize=8, loc="lower left"); save(fig, "fig_missingness_delta")
print("figures done")

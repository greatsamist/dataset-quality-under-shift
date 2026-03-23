"""
Post-hoc analysis on saved predictions (results/<run>/predictions.npz):

  1. Calibration: 10 equal-width bins -> ECE, binned Brier decomposition
     (reliability / resolution / uncertainty, Murphy 1973), mean predicted
     probability vs observed prevalence, and the constant-probability baseline
     (predict the TRAINING prevalence for everyone).
  2. Feature-blanking investigation: what happens to the prediction distribution
     when a feature group is blanked (mean, spread, share of near-0/near-1).
  3. Patient-clustered bootstrap (Diabetes grouped protocol only): percentile
     intervals for the AUROC/AP/Brier domain gap and for the LightGBM - LR
     difference, resampling PATIENTS with replacement, conditional on the
     fitted models. Seed-to-seed SD (3 refits) is reported separately.

Outputs CSVs next to the predictions file.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import spearmanr
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="results/<run dir>")
ap.add_argument("--dataset", choices=["diabetes", "acs"], required=True)
ap.add_argument("--protocol", default="patient")
ap.add_argument("--bootstrap", type=int, default=0, help="patient-clustered bootstrap replicates (diabetes only)")
ap.add_argument("--bins", type=int, default=10)
args = ap.parse_args()

P = np.load(os.path.join(args.run, "predictions.npz"))
keys = sorted({k.rsplit("|", 1)[0] for k in P.files})

def calib(y, p, nb):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, nb + 1); b = np.clip(np.digitize(p, edges[1:-1]), 0, nb - 1)
    N, ybar = len(y), y.mean()
    rel = res = ece = 0.0; rows = []
    for i in range(nb):
        m = b == i; n = m.sum()
        if n == 0:
            rows.append((i, 0, np.nan, np.nan)); continue
        conf, obs = p[m].mean(), y[m].mean()
        rel += n / N * (conf - obs) ** 2; res += n / N * (obs - ybar) ** 2; ece += n / N * abs(conf - obs)
        rows.append((i, int(n), conf, obs))
    return {"ECE": ece, "reliability": rel, "resolution": res, "uncertainty": ybar * (1 - ybar),
            "mean_pred": p.mean(), "prevalence": ybar, "Brier": brier_score_loss(y, p)}, rows

# training prevalence per seed (for the constant baseline)
train_prev = {}
if args.dataset == "diabetes":
    df = C.load_diabetes("raw_data/diabetic_data.csv")
    split_fn = C.split_diabetes_record if args.protocol == "record" else C.split_diabetes_patient_disjoint
    parts = {s: split_fn(df, s) for s in C.SEEDS}
    train_prev = {s: float(parts[s]["train"]["y"].mean()) for s in C.SEEDS}
else:
    lp = pd.read_csv(os.path.join(args.run, "label_noise_prevalence.csv"))
    train_prev = {int(r.seed): float(r.train_prevalence_clean) for r in lp[lp.noise_rate == 0].itertuples()}

cal_rows, bin_rows = [], []
for k in keys:
    exp, cond, model, seed, ts = k.split("|"); seed = int(seed)
    y, p = P[k + "|y"], P[k + "|p"]
    c, bins = calib(y, p, args.bins)
    pi = train_prev[seed]
    c["constant_baseline_Brier"] = float(np.mean((y - pi) ** 2))
    c["train_prevalence"] = pi
    c["share_p_lt_0.1"] = float((p < 0.1).mean()); c["share_p_gt_0.9"] = float((p > 0.9).mean())
    c["p_sd"] = float(p.std())
    cal_rows.append({"Experiment": exp, "Condition": cond, "Model": model, "Random Seed": seed, "Test Set": ts, **c})
    for i, n, conf, obs in bins:
        bin_rows.append({"Experiment": exp, "Condition": cond, "Model": model, "Random Seed": seed,
                         "Test Set": ts, "bin": i, "n": n, "mean_pred": conf, "obs_rate": obs})
cal = pd.DataFrame(cal_rows); cal.to_csv(os.path.join(args.run, "calibration_per_seed.csv"), index=False)
pd.DataFrame(bin_rows).to_csv(os.path.join(args.run, "reliability_bins.csv"), index=False)
num = [c for c in cal.columns if c not in ("Experiment", "Condition", "Model", "Random Seed", "Test Set")]
cal.groupby(["Experiment", "Condition", "Model", "Test Set"])[num].agg(["mean", "std"]).to_csv(
    os.path.join(args.run, "calibration_summary.csv"))
print("calibration written:", len(cal), "rows")

# ---- blanking investigation: pair each Feature Shift condition with its Baseline ----
inv = []
for k in keys:
    exp, cond, model, seed, ts = k.split("|")
    if exp != "Feature Shift":
        continue
    kb = f"Baseline|None|{model}|{seed}|{ts}"
    y, p1, p0 = P[k + "|y"], P[k + "|p"], P[kb + "|p"]
    inv.append({"Model": model, "Random Seed": int(seed), "Test Set": ts, "Feature Group Blanked": cond,
                "prevalence": float(y.mean()),
                "mean_pred_clean": float(p0.mean()), "mean_pred_blanked": float(p1.mean()),
                "sd_pred_clean": float(p0.std()), "sd_pred_blanked": float(p1.std()),
                "share_gt0.5_clean": float((p0 > .5).mean()), "share_gt0.5_blanked": float((p1 > .5).mean()),
                "corr_clean_blanked": float(np.corrcoef(p0, p1)[0, 1]),
                "spearman_clean_blanked": float(spearmanr(p0, p1)[0]),
                "Brier_clean": brier_score_loss(y, p0), "Brier_blanked": brier_score_loss(y, p1),
                "AUROC_clean": roc_auc_score(y, p0), "AUROC_blanked": roc_auc_score(y, p1)})
if inv:
    pd.DataFrame(inv).to_csv(os.path.join(args.run, "blanking_investigation.csv"), index=False)
    print("blanking investigation written:", len(inv), "rows")

# ---- patient-clustered bootstrap (diabetes) ----
if args.bootstrap and args.dataset == "diabetes":
    rng = np.random.default_rng(0)
    out = []
    for seed in C.SEEDS:
        pid = {"In-domain Test": parts[seed]["test"]["patient_nbr"].to_numpy(),
               "Shifted Test": parts[seed]["shifted"]["patient_nbr"].to_numpy()}
        preds = {}
        for model in C.MODEL_NAMES:
            for ts in pid:
                k = f"Baseline|None|{model}|{seed}|{ts}"
                preds[(model, ts)] = (P[k + "|y"].astype(float), P[k + "|p"].astype(float))
        # index rows by patient for fast clustered resampling
        groups = {ts: pd.Series(np.arange(len(pid[ts]))).groupby(pid[ts]).apply(lambda s: s.to_numpy()).to_dict() for ts in pid}
        pat = {ts: np.array(list(groups[ts])) for ts in pid}
        stats = {m: [] for m in ["gap_AUROC", "gap_AUPRC", "gap_Brier", "diff_shifted_AUROC_LGBM_minus_LR",
                                 "diff_gap_AUROC_LGBM_minus_LR"]}
        for _ in range(args.bootstrap):
            idx = {}
            for ts in pid:
                samp = rng.choice(pat[ts], size=len(pat[ts]), replace=True)
                idx[ts] = np.concatenate([groups[ts][s] for s in samp])
            res = {}
            for model in C.MODEL_NAMES:
                for ts in pid:
                    y, p = preds[(model, ts)]; yi, pi_ = y[idx[ts]], p[idx[ts]]
                    res[(model, ts)] = (roc_auc_score(yi, pi_), average_precision_score(yi, pi_), brier_score_loss(yi, pi_))
            g = {m: tuple(res[(m, "Shifted Test")][i] - res[(m, "In-domain Test")][i] for i in range(3)) for m in C.MODEL_NAMES}
            stats["gap_AUROC"].append(g["LightGBM"][0]); stats["gap_AUPRC"].append(g["LightGBM"][1]); stats["gap_Brier"].append(g["LightGBM"][2])
            stats["diff_shifted_AUROC_LGBM_minus_LR"].append(res[("LightGBM", "Shifted Test")][0] - res[("Logistic Regression", "Shifted Test")][0])
            stats["diff_gap_AUROC_LGBM_minus_LR"].append(g["LightGBM"][0] - g["Logistic Regression"][0])
        # also LR gap
        for name, arr in stats.items():
            a = np.array(arr)
            out.append({"Random Seed": seed, "statistic": name, "model": "LightGBM" if name.startswith("gap") else "both",
                        "boot_mean": a.mean(), "ci_lo": np.percentile(a, 2.5), "ci_hi": np.percentile(a, 97.5), "B": args.bootstrap})
    # LR gaps as well
    for seed in C.SEEDS:
        pid = {"In-domain Test": parts[seed]["test"]["patient_nbr"].to_numpy(), "Shifted Test": parts[seed]["shifted"]["patient_nbr"].to_numpy()}
        groups = {ts: pd.Series(np.arange(len(pid[ts]))).groupby(pid[ts]).apply(lambda s: s.to_numpy()).to_dict() for ts in pid}
        pat = {ts: np.array(list(groups[ts])) for ts in pid}
        y_p = {ts: (P[f"Baseline|None|Logistic Regression|{seed}|{ts}|y"].astype(float), P[f"Baseline|None|Logistic Regression|{seed}|{ts}|p"].astype(float)) for ts in pid}
        arr = []
        for _ in range(args.bootstrap):
            r = {}
            for ts in pid:
                samp = rng.choice(pat[ts], size=len(pat[ts]), replace=True); ii = np.concatenate([groups[ts][s] for s in samp])
                r[ts] = roc_auc_score(y_p[ts][0][ii], y_p[ts][1][ii])
            arr.append(r["Shifted Test"] - r["In-domain Test"])
        a = np.array(arr)
        out.append({"Random Seed": seed, "statistic": "gap_AUROC", "model": "Logistic Regression",
                    "boot_mean": a.mean(), "ci_lo": np.percentile(a, 2.5), "ci_hi": np.percentile(a, 97.5), "B": args.bootstrap})
    pd.DataFrame(out).to_csv(os.path.join(args.run, "bootstrap_clustered.csv"), index=False)
    print("bootstrap written")

"""
ACSEmployment 2018 (CA -> TX): full experiment suite.

  --pipeline numeric   all 16 features numeric, exactly as ACSEmployment.ipynb
  --pipeline onehot    nominal codes one-hot encoded (sensitivity analysis)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", choices=["numeric", "onehot"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data-root", default="data")
ap.add_argument("--experiments", default="baseline,label_noise,missingness,feature_shift")
ap.add_argument("--seeds", default="42,43,44")
args = ap.parse_args()
EXPS = set(args.experiments.split(","))
SEEDS = [int(s) for s in args.seeds.split(",")]

df_ca, df_tx = C.load_acs(args.data_root)
print(f"CA rows={len(df_ca)} prevalence={df_ca.y.mean():.4f} | TX rows={len(df_tx)} prevalence={df_tx.y.mean():.4f}", flush=True)

# Dataset facts quoted in the thesis (population sizes, prevalences, share of
# respondents under 16, presence of the -1 "not applicable" code).
feat = [c for c in df_ca.columns if c != "y"]
stats = {"CA_rows": int(len(df_ca)), "TX_rows": int(len(df_tx)),
         "CA_prev": float(df_ca.y.mean()), "TX_prev": float(df_tx.y.mean()),
         "CA_under16_share": float((df_ca["AGEP"] < 16).mean()),
         "TX_under16_share": float((df_tx["AGEP"] < 16).mean()),
         "CA_any_minus1_share": float((df_ca[feat] == -1).any(axis=1).mean()),
         "cols_with_minus1_CA": {c: int((df_ca[c] == -1).sum()) for c in feat if (df_ca[c] == -1).any()}}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(os.path.join(os.path.dirname(args.out) or ".", "acs_dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
store = C.Store(args.out, save_preds_for={"Baseline", "Feature Shift", "Label Noise"})
audits, noise_prev = {}, []
t0 = time.time()

def fit(Xtr, ytr, seed):
    models = C.build_acs_models(seed, args.pipeline)
    for m in models.values():
        m.fit(Xtr, ytr)
    return models

def evaluate_both(models, meta_fn, Xte, yte, Xsh, ysh):
    for name, m in models.items():
        meta = meta_fn(name)
        store.add(meta, "In-domain Test", yte, m.predict_proba(Xte)[:, 1])
        store.add(meta, "Shifted Test",   ysh, m.predict_proba(Xsh)[:, 1])

for seed in SEEDS:
    P = C.split_acs(df_ca, df_tx, seed)
    (Xtr, ytr), (Xte, yte), (Xsh, ysh) = P["train"], P["test"], P["shifted"]
    audits[seed] = {k: {"rows": int(len(v[0])), "prevalence": float(v[1].mean())} for k, v in P.items()}
    print(f"[{args.pipeline}] seed {seed}: train={len(Xtr)} test={len(Xte)} shifted={len(Xsh)}", flush=True)

    base_models = fit(Xtr, ytr, seed)
    if "baseline" in EXPS:
        evaluate_both(base_models, lambda n: C.base_meta("Baseline", "None", n, seed), Xte, yte, Xsh, ysh)
    print(f"   baseline done ({time.time()-t0:.0f}s)", flush=True)

    if "label_noise" in EXPS:
        for rate in [0.0, 0.05, 0.10, 0.20]:
            yn = C.label_noise(ytr, rate, seed, style="acs")
            noise_prev.append({"seed": seed, "noise_rate": rate, "train_prevalence_clean": float(ytr.mean()),
                               "train_prevalence_noisy": float(yn.mean()), "n_flipped": int((yn.to_numpy() != ytr.to_numpy()).sum())})
            models = fit(Xtr, yn, seed)
            evaluate_both(models, lambda n: C.base_meta("Label Noise", f"{rate*100:.0f}% Noise", n, seed, noise=rate),
                          Xte, yte, Xsh, ysh)
            print(f"   label noise {rate:.2f} done ({time.time()-t0:.0f}s)", flush=True)

    if "missingness" in EXPS:
        for label in ["MCAR", "MAR_Age_Older"]:
            for rate in [0.0, 0.10, 0.20]:
                if label == "MCAR":
                    Xn = C.mcar(Xtr, C.ACS_MISSING_COLS, rate, seed, style="acs")
                elif rate == 0:
                    Xn = Xtr.copy()
                else:
                    # The conditioning variable (AGEP) is never masked: masking it would make
                    # the mechanism MNAR rather than MAR.
                    mar_cols = [c for c in C.ACS_MISSING_COLS if c != "AGEP"]
                    Xn = C.mar(Xtr, mar_cols, rate, Xtr["AGEP"] >= 65, seed, style="acs")
                models = fit(Xn, ytr, seed)
                evaluate_both(models, lambda n: C.base_meta("Missingness", label, n, seed, mtype=label, mrate=rate),
                              Xte, yte, Xsh, ysh)
            print(f"   missingness {label} done ({time.time()-t0:.0f}s)", flush=True)

    if "feature_shift" in EXPS:
        for gname, gcols in C.ACS_FEATURE_GROUPS.items():
            evaluate_both(base_models, lambda n: C.base_meta("Feature Shift", gname, n, seed, group=gname),
                          C.blank_out(Xte, gcols), yte, C.blank_out(Xsh, gcols), ysh)

raw, summ = store.finalize({"dataset": "acs_employment", "pipeline": args.pipeline, "experiments": sorted(EXPS),
                            "survey_year": 2018, "horizon": "1-Year", "source": "CA", "target": "TX",
                            "elapsed_s": round(time.time() - t0)})
with open(os.path.join(args.out, "split_audit.json"), "w") as f:
    json.dump(audits, f, indent=2)
pd.DataFrame(noise_prev).to_csv(os.path.join(args.out, "label_noise_prevalence.csv"), index=False)
print(f"DONE {args.out} in {time.time()-t0:.0f}s; rows={len(raw)}")

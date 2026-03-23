"""
Diabetes 130-US hospitals: full experiment suite.

  --protocol patient  patient-disjoint split (the evaluation protocol)
  --protocol record   record-level split, for the overlap diagnostic only;
                      combine with --audit-only to skip model fitting

Sensitivity switches (grouped protocol):
  --drop-admission-source   remove the cohort-defining column from the predictors
  --ids-categorical         treat the three *_id code columns as categorical, not numeric
  --experiments             comma list of baseline,label_noise,missingness,feature_shift
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", choices=["patient", "record"], required=True)
ap.add_argument("--audit-only", action="store_true",
                help="write the split audit and stop; no models are fitted")
ap.add_argument("--out", required=True)
ap.add_argument("--raw", default="raw_data/diabetic_data.csv")
ap.add_argument("--drop-admission-source", action="store_true")
ap.add_argument("--ids-categorical", action="store_true")
ap.add_argument("--no-class-weight", action="store_true", help="sensitivity: fit without balanced class weights")
ap.add_argument("--experiments", default="baseline,label_noise,missingness,feature_shift")
ap.add_argument("--seeds", default="42,43,44")
args = ap.parse_args()
EXPS = set(args.experiments.split(","))
SEEDS = [int(s) for s in args.seeds.split(",")]
drop_cols = [C.SHIFT_COHORT_COL] if args.drop_admission_source else []

df = C.load_diabetes(args.raw)
split_fn = C.split_diabetes_record if args.protocol == "record" else C.split_diabetes_patient_disjoint

if args.audit_only:
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split_audit.json"), "w") as f:
        json.dump({s: C.split_audit(split_fn(df, s)) for s in SEEDS}, f, indent=2)
    print(f"DONE {args.out} (split audit only, no models fitted)")
    sys.exit(0)

store = C.Store(args.out, save_preds_for={"Baseline", "Feature Shift", "Label Noise"})
audits, noise_prev, feature_info = {}, [], None
t0 = time.time()

def fit(Xtr, ytr, seed):
    models, cols = C.build_diabetes_models(C.cast_ids(Xtr, args.ids_categorical), seed, args.ids_categorical,
                                          class_weight=None if args.no_class_weight else "balanced")
    for m in models.values():
        m.fit(C.cast_ids(Xtr, args.ids_categorical), ytr)
    return models, cols

def evaluate_both(models, meta_fn, Xte, yte, Xsh, ysh):
    for name, m in models.items():
        meta = meta_fn(name)
        store.add(meta, "In-domain Test", yte, m.predict_proba(C.cast_ids(Xte, args.ids_categorical))[:, 1])
        store.add(meta, "Shifted Test",   ysh, m.predict_proba(C.cast_ids(Xsh, args.ids_categorical))[:, 1])

for seed in SEEDS:
    parts = split_fn(df, seed)
    audits[seed] = C.split_audit(parts)
    Xtr, ytr = C.diabetes_xy(parts["train"], drop_cols)
    Xte, yte = C.diabetes_xy(parts["test"], drop_cols)
    Xsh, ysh = C.diabetes_xy(parts["shifted"], drop_cols)
    print(f"[{args.protocol}] seed {seed}: train={len(Xtr)} test={len(Xte)} shifted={len(Xsh)} "
          f"predictors={Xtr.shape[1]}", flush=True)

    # ---- Baseline (models reused for feature shift) ----
    base_models, cols = fit(Xtr, ytr, seed)
    feature_info = {"n_predictors": int(Xtr.shape[1]), **cols}
    if "baseline" in EXPS:
        evaluate_both(base_models, lambda n: C.base_meta("Baseline", "None", n, seed), Xte, yte, Xsh, ysh)

    # ---- Label noise: training labels flipped, models refit, clean evaluation ----
    if "label_noise" in EXPS:
        for rate in [0.0, 0.05, 0.10, 0.20]:
            yn = C.label_noise(ytr, rate, seed, style="diabetes")
            noise_prev.append({"seed": seed, "noise_rate": rate, "train_prevalence_clean": float(ytr.mean()),
                               "train_prevalence_noisy": float(yn.mean()), "n_flipped": int((yn != ytr).sum())})
            models, _ = fit(Xtr, yn, seed)
            evaluate_both(models, lambda n: C.base_meta("Label Noise", f"{rate*100:.0f}% Noise", n, seed, noise=rate),
                          Xte, yte, Xsh, ysh)
            print(f"   label noise {rate:.2f} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- Missingness: training features masked, models refit, clean evaluation ----
    if "missingness" in EXPS:
        for sc in C.DIAB_MISSING_SCENARIOS:
            for rate in [0.0, 0.10, 0.20]:
                if sc["type"] == "MCAR":
                    Xn = C.mcar(Xtr, C.DIAB_MISSING_COLS, rate, seed, style="diabetes")
                elif rate == 0:
                    Xn = Xtr.copy()
                else:
                    cond = sc["condition"]
                    Xn = C.mar(Xtr, C.DIAB_MISSING_COLS, rate, Xtr[cond["column"]].isin(cond["values"]), seed, style="diabetes")
                models, _ = fit(Xn, ytr, seed)
                evaluate_both(models, lambda n: C.base_meta("Missingness", sc["label"], n, seed, mtype=sc["label"], mrate=rate),
                              Xte, yte, Xsh, ysh)
            print(f"   missingness {sc['label']} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- Feature availability: evaluation-time blanking, baseline models reused ----
    if "feature_shift" in EXPS:
        for gname, gcols in C.DIAB_FEATURE_GROUPS.items():
            evaluate_both(base_models, lambda n: C.base_meta("Feature Shift", gname, n, seed, group=gname),
                          C.blank_out(Xte, gcols), yte, C.blank_out(Xsh, gcols), ysh)

raw, summ = store.finalize({"dataset": "diabetes", "protocol": args.protocol,
                            "drop_admission_source": args.drop_admission_source,
                            "ids_categorical": args.ids_categorical, "class_weight": None if args.no_class_weight else "balanced", "experiments": sorted(EXPS),
                            "features": feature_info, "elapsed_s": round(time.time() - t0)})
with open(os.path.join(args.out, "split_audit.json"), "w") as f:
    json.dump(audits, f, indent=2)
pd.DataFrame(noise_prev).to_csv(os.path.join(args.out, "label_noise_prevalence.csv"), index=False)
print(f"DONE {args.out} in {time.time()-t0:.0f}s; rows={len(raw)}")

"""
Generate every numerical table in the thesis (thesis_project/tables/*.tex) and
a macro file (thesis_project/tables/numbers.tex) from results/, so that no
number in the document is typed by hand.

Conventions
  * "ID"  = in-domain test set, "SH" = shifted test set.
  * gap   = SH - ID within the same seed (already a paired quantity).
  * delta = stressed - clean on the SAME domain within the same seed (paired).
  * cells = mean (SD) across the n seeds; n is printed in every caption.
  * Higher AUROC / AP better; lower Brier better.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results", help="results root holding the run directories")
ap.add_argument("--tables", default="thesis_project/tables", help="destination for the .tex tables and numbers.tex")
args = ap.parse_args()

R = args.results; T = args.tables; os.makedirs(T, exist_ok=True); os.makedirs(f"{R}/tables", exist_ok=True)
MODELS = ["Logistic Regression", "LightGBM"]; SHORT = {"Logistic Regression": "LR", "LightGBM": "LightGBM"}
METRICS = ["AUROC", "AUPRC", "Brier Score"]; MLAB = {"AUROC": "AUROC", "AUPRC": "AP", "Brier Score": "Brier"}
GAP = "Generalization Gap (Shifted - In-domain)"; ID, SH = "In-domain Test", "Shifted Test"
macros = {}

def load(run):
    p = f"{R}/{run}/raw_results.csv"
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if os.path.exists(p) else None

def f3(x, sd=None, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    s = s.replace("-", "$-$").replace("+", "$+$")
    return s if sd is None or np.isnan(sd) else f"{s} ({sd:.3f})"

def ms(series): return series.mean(), series.std(ddof=1)

def write(name, body, csv=None):
    open(f"{T}/{name}.tex", "w").write(body)
    if csv is not None: csv.to_csv(f"{R}/tables/{name}.csv", index=False)
    print("wrote", name)

def rows_of(df, exp, cond=None, model=None, ts=None, **kw):
    q = df["Experiment"] == exp
    if cond is not None: q &= df["Condition"] == cond
    if model is not None: q &= df["Model"] == model
    if ts is not None: q &= df["Test Set"] == ts
    for k, v in kw.items(): q &= df[k] == v
    return df[q]

def paired_delta(df, exp, cond, model, ts, metric, **kw):
    """stressed - clean (Baseline) on the same domain, matched by seed."""
    s = rows_of(df, exp, cond, model, ts, **kw).set_index("Random Seed")[metric]
    b = rows_of(df, "Baseline", "None", model, ts).set_index("Random Seed")[metric]
    d = (s - b).dropna(); return d.mean(), d.std(ddof=1)

def stat(df, exp, cond, model, ts, metric, **kw):
    return ms(rows_of(df, exp, cond, model, ts, **kw)[metric])

# ============================================================ cohort description
df_raw = pd.read_csv("raw_data/diabetic_data.csv", low_memory=False).replace("?", np.nan)
df_raw["y"] = (df_raw.readmitted == "<30").astype(int)
sh, src = df_raw[df_raw.admission_source_id == 17], df_raw[df_raw.admission_source_id != 17]
def cohort_row(label, f): return f"{label} & {f(sh)} & {f(src)} \\\\"
lines = [r"\begin{tabular}{lrr}", r"\toprule", r"\textbf{Characteristic} & \textbf{Shifted cohort} & \textbf{Source population} \\", r"\midrule",
 cohort_row("Encounters", lambda d: f"{len(d):,}"),
 cohort_row("Unique patients", lambda d: f"{d.patient_nbr.nunique():,}"),
 cohort_row("Encounters per patient", lambda d: f"{len(d)/d.patient_nbr.nunique():.2f}"),
 cohort_row(r"Readmitted $<$30 days (\%)", lambda d: f"{d.y.mean()*100:.2f}"),
 cohort_row(r"Admission type coded NULL or Not Available (\%)", lambda d: f"{d.admission_type_id.isin([5,6]).mean()*100:.1f}"),
 cohort_row(r"Discharge disposition coded 1 (home) (\%)", lambda d: f"{(d.discharge_disposition_id==1).mean()*100:.1f}"),
 cohort_row(r"Payer code missing (\%)", lambda d: f"{d.payer_code.isna().mean()*100:.1f}"),
 cohort_row(r"Medical specialty missing (\%)", lambda d: f"{d.medical_specialty.isna().mean()*100:.1f}"),
 cohort_row(r"Race missing (\%)", lambda d: f"{d.race.isna().mean()*100:.1f}"),
 cohort_row(r"Race Caucasian (\%)", lambda d: f"{(d.race=='Caucasian').mean()*100:.1f}"),
 cohort_row(r"Race African American (\%)", lambda d: f"{(d.race=='AfricanAmerican').mean()*100:.1f}"),
 cohort_row(r"Age 70 or older (\%)", lambda d: f"{d.age.isin(['[70-80)','[80-90)','[90-100)']).mean()*100:.1f}"),
 cohort_row("Mean lab procedures", lambda d: f"{d.num_lab_procedures.mean():.1f}"),
 cohort_row("Mean time in hospital (days)", lambda d: f"{d.time_in_hospital.mean():.2f}"),
 cohort_row("Mean prior emergency visits", lambda d: f"{d.number_emergency.mean():.2f}"),
 cohort_row("Mean prior inpatient visits", lambda d: f"{d.number_inpatient.mean():.2f}"),
 r"\bottomrule", r"\end{tabular}"]
write("tab_cohort_description", "\n".join(lines))
macros.update({"nShiftedEnc": f"{len(sh):,}", "nShiftedPat": f"{sh.patient_nbr.nunique():,}", "nSourceEnc": f"{len(src):,}",
               "nSourcePat": f"{src.patient_nbr.nunique():,}", "nSharedPat": f"{len(set(sh.patient_nbr)&set(src.patient_nbr))}",
               "prevShifted": f"{sh.y.mean()*100:.2f}", "prevSource": f"{src.y.mean()*100:.2f}",
               "shiftedAdmTypeNull": f"{sh.admission_type_id.isin([5,6]).mean()*100:.1f}", "sourceAdmTypeNull": f"{src.admission_type_id.isin([5,6]).mean()*100:.1f}",
               "nEncTotal": f"{len(df_raw):,}", "nPatTotal": f"{df_raw.patient_nbr.nunique():,}"})

# ================================================ split audit + record-level overlap diagnostic
def audit(run): 
    p = f"{R}/{run}/split_audit.json"; return json.load(open(p)) if os.path.exists(p) else None
ag, al = audit("diabetes"), audit("diabetes_split_audit")
if ag:
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"\textbf{Seed} & \textbf{Partition} & \textbf{Encounters} & \textbf{Patients} & \textbf{Prevalence (\%)} \\", r"\midrule"]
    for s in ["42", "43", "44"]:
        for i, part in enumerate(["train", "val", "test", "shifted"]):
            a = ag[s][part]; L.append(f"{s if i==0 else ''} & {part} & {a['encounters']:,} & {a['patients']:,} & {a['prevalence']*100:.2f} \\\\")
        if s != "44": L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_split_audit", "\n".join(L))
    macros.update({"splitTrainEnc": f"{ag['42']['train']['encounters']:,}", "splitTestEnc": f"{ag['42']['test']['encounters']:,}",
                   "splitTrainPat": f"{ag['42']['train']['patients']:,}", "splitTestPat": f"{ag['42']['test']['patients']:,}",
                   "splitValEnc": f"{ag['42']['val']['encounters']:,}"})
# How much patient overlap a record-level split would produce (Section 3.3).
if al:
    o42 = al["42"]["overlap"]["train|test"]
    macros.update({"recordOverlapEnc": f"{o42['test_encounters_from_shared']:,}", "recordTestEnc": f"{al['42']['test']['encounters']:,}",
                   "recordOverlapPct": f"{o42['test_encounters_from_shared']/al['42']['test']['encounters']*100:.1f}",
                   "recordShiftedOverlapEnc": f"{al['42']['overlap']['train|shifted']['shifted_encounters_from_shared']}"})

# ============================================================ baseline tables
def baseline_table(df, name, n):
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"\textbf{Model} & \textbf{Metric} & \textbf{In-domain} & \textbf{Shifted} & \textbf{Gap (SH $-$ ID)} \\", r"\midrule"]
    rows = []
    for m in MODELS:
        for i, met in enumerate(METRICS):
            a, b, g = stat(df, "Baseline", "None", m, ID, met), stat(df, "Baseline", "None", m, SH, met), stat(df, "Baseline", "None", m, GAP, met)
            L.append(f"{SHORT[m] if i==0 else ''} & {MLAB[met]} & {f3(*a)} & {f3(*b)} & {f3(*g, sign=True)} \\\\")
            rows.append({"model": m, "metric": met, "ID_mean": a[0], "ID_sd": a[1], "SH_mean": b[0], "SH_sd": b[1], "gap_mean": g[0], "gap_sd": g[1]})
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write(name, "\n".join(L), pd.DataFrame(rows)); return rows

dg, an = load("diabetes"), load("acs")
if dg is not None:
    rows = baseline_table(dg, "tab_baseline_diabetes", 3)
    for r in rows:
        k = f"{'lr' if r['model'].startswith('Log') else 'gb'}{MLAB[r['metric']].replace(' ','')}"
        macros[f"diab{k}ID"] = f"{r['ID_mean']:.3f}"; macros[f"diab{k}SH"] = f"{r['SH_mean']:.3f}"
        macros[f"diab{k}Gap"] = f"{r['gap_mean']:+.3f}".replace("-", "$-$").replace("+", "$+$"); macros[f"diab{k}GapSD"] = f"{r['gap_sd']:.3f}"
if an is not None:
    rows = baseline_table(an, "tab_baseline_acs", 3)
    for r in rows:
        k = f"{'lr' if r['model'].startswith('Log') else 'gb'}{MLAB[r['metric']].replace(' ','')}"
        macros[f"acs{k}ID"] = f"{r['ID_mean']:.3f}"; macros[f"acs{k}SH"] = f"{r['SH_mean']:.3f}"
        macros[f"acs{k}Gap"] = f"{r['gap_mean']:+.3f}".replace("-", "$-$").replace("+", "$+$")

# ============================================================ label noise
def label_noise_table(df, run, name):
    prev = pd.read_csv(f"{R}/{run}/label_noise_prevalence.csv").groupby("noise_rate")["train_prevalence_noisy"].mean()
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Noise} & \textbf{Train prev.\ \%} & \textbf{AUROC ID} & \textbf{AUROC SH} & \textbf{Gap (SD)} & \textbf{Brier ID} & \textbf{Brier SH} \\", r"\midrule"]
    rows = []
    for m in MODELS:
        for i, rate in enumerate([0.0, 0.05, 0.10, 0.20]):
            c = f"{rate*100:.0f}% Noise"
            a, b, g = stat(df, "Label Noise", c, m, ID, "AUROC"), stat(df, "Label Noise", c, m, SH, "AUROC"), stat(df, "Label Noise", c, m, GAP, "AUROC")
            bi, bs = stat(df, "Label Noise", c, m, ID, "Brier Score"), stat(df, "Label Noise", c, m, SH, "Brier Score")
            L.append(f"{SHORT[m] if i==0 else ''} & {rate*100:.0f}\\% & {prev[rate]*100:.1f} & {f3(a[0])} & {f3(b[0])} & {f3(*g, sign=True)} & {f3(bi[0])} & {f3(bs[0])} \\\\")
            rows.append({"model": m, "rate": rate, "train_prev_noisy": prev[rate], "AUROC_ID": a[0], "AUROC_SH": b[0], "gap": g[0], "gap_sd": g[1], "Brier_ID": bi[0], "Brier_SH": bs[0]})
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write(name, "\n".join(L), pd.DataFrame(rows)); return prev
if dg is not None:
    pv = label_noise_table(dg, "diabetes", "tab_label_noise_diabetes"); macros["diabPrevNoise20"] = f"{pv[0.2]*100:.1f}"; macros["diabPrevClean"] = f"{pv[0.0]*100:.1f}"
if an is not None:
    pv = label_noise_table(an, "acs", "tab_label_noise_acs"); macros["acsPrevNoise20"] = f"{pv[0.2]*100:.1f}"; macros["acsPrevClean"] = f"{pv[0.0]*100:.1f}"

# ============================================================ missingness (20 %) — paired deltas vs clean
def missingness_table(df, scenarios, name, rates=(0.10, 0.20)):
    """Appendix version: every severity, plus the domain gap."""
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Mechanism} & \textbf{Rate} & \textbf{$\Delta$AUROC ID} & \textbf{$\Delta$AUROC SH} & \textbf{AUROC gap} & \textbf{$\Delta$Brier ID} & \textbf{$\Delta$Brier SH} \\", r"\midrule"]
    rows = []
    for m in MODELS:
        for i, sc in enumerate(scenarios):
            for j, rate in enumerate(rates):
                kw = {"Missing Rate": rate}
                di, ds = paired_delta(df, "Missingness", sc, m, ID, "AUROC", **kw), paired_delta(df, "Missingness", sc, m, SH, "AUROC", **kw)
                g = stat(df, "Missingness", sc, m, GAP, "AUROC", **kw)
                bi, bs = paired_delta(df, "Missingness", sc, m, ID, "Brier Score", **kw), paired_delta(df, "Missingness", sc, m, SH, "Brier Score", **kw)
                L.append(f"{SHORT[m] if (i==0 and j==0) else ''} & {sc.replace('_', chr(92)+'_') if j==0 else ''} & {rate*100:.0f}\\% & {f3(*di, sign=True)} & {f3(*ds, sign=True)} & {f3(*g, sign=True)} & {f3(*bi, sign=True)} & {f3(*bs, sign=True)} \\\\")
                rows.append({"model": m, "scenario": sc, "rate": rate, "dAUROC_ID": di[0], "dAUROC_SH": ds[0], "gap": g[0], "dBrier_ID": bi[0], "dBrier_SH": bs[0]})
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write(name, "\n".join(L), pd.DataFrame(rows))
if dg is not None: missingness_table(dg, ["MCAR", "MAR_Female", "MAR_Age_Older"], "tab_missingness_diabetes")
if an is not None: missingness_table(an, ["MCAR", "MAR_Age_Older"], "tab_missingness_acs")

# ============================================================ feature shift — absolute + paired deltas
def feature_table(df, groups, name):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Group blanked} & \textbf{AUROC ID} & \textbf{$\Delta$ ID} & \textbf{AUROC SH} & \textbf{$\Delta$ SH} & \textbf{Brier ID ($\Delta$)} & \textbf{Brier SH ($\Delta$)} \\", r"\midrule"]
    rows = []
    for m in MODELS:
        for i, g in enumerate(groups):
            ai, asx = stat(df, "Feature Shift", g, m, ID, "AUROC"), stat(df, "Feature Shift", g, m, SH, "AUROC")
            di, ds = paired_delta(df, "Feature Shift", g, m, ID, "AUROC"), paired_delta(df, "Feature Shift", g, m, SH, "AUROC")
            bi, bs = stat(df, "Feature Shift", g, m, ID, "Brier Score"), stat(df, "Feature Shift", g, m, SH, "Brier Score")
            dbi, dbs = paired_delta(df, "Feature Shift", g, m, ID, "Brier Score"), paired_delta(df, "Feature Shift", g, m, SH, "Brier Score")
            L.append(f"{SHORT[m] if i==0 else ''} & {g.replace('_', ' ')} & {f3(ai[0])} & {f3(*di, sign=True)} & {f3(asx[0])} & {f3(*ds, sign=True)} & {f3(bi[0])} ({f3(dbi[0], sign=True)}) & {f3(bs[0])} ({f3(dbs[0], sign=True)}) \\\\")
            rows.append({"model": m, "group": g, "AUROC_ID": ai[0], "dAUROC_ID": di[0], "dAUROC_ID_sd": di[1], "AUROC_SH": asx[0], "dAUROC_SH": ds[0], "dAUROC_SH_sd": ds[1],
                         "Brier_ID": bi[0], "dBrier_ID": dbi[0], "Brier_SH": bs[0], "dBrier_SH": dbs[0]})
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write(name, "\n".join(L), pd.DataFrame(rows)); return rows
if dg is not None:
    rows = feature_table(dg, ["Diagnoses", "Medications"], "tab_feature_shift_diabetes")
    for r in rows:
        k = ("lr" if r["model"].startswith("Log") else "gb") + r["group"][:4]
        macros[f"fsDiab{k}dAUROCSH"] = f"{r['dAUROC_SH']:+.3f}".replace("-", "$-$").replace("+", "$+$")
        macros[f"fsDiab{k}BrierSH"] = f"{r['Brier_SH']:.3f}"; macros[f"fsDiab{k}dBrierSH"] = f"{r['dBrier_SH']:+.3f}".replace("-", "$-$").replace("+", "$+$")
        macros[f"fsDiab{k}BrierID"] = f"{r['Brier_ID']:.3f}"
if an is not None: feature_table(an, ["Education", "Disability", "Citizenship_Nativity_Ancestry"], "tab_feature_shift_acs")

# ============================================================ cross-stressor — identical quantities for both datasets
def cross_rows(df, dataset, stressors):
    out = []
    for m in MODELS:
        for label, exp, cond, kw in stressors:
            di = paired_delta(df, exp, cond, m, ID, "AUROC", **kw); ds = paired_delta(df, exp, cond, m, SH, "AUROC", **kw)
            g = stat(df, exp, cond, m, GAP, "AUROC", **kw); g0 = stat(df, "Baseline", "None", m, GAP, "AUROC")
            dg = paired_delta(df, exp, cond, m, GAP, "AUROC", **kw)   # within-seed change of the gap
            dbi = paired_delta(df, exp, cond, m, ID, "Brier Score", **kw); dbs = paired_delta(df, exp, cond, m, SH, "Brier Score", **kw)
            out.append({"dataset": dataset, "model": m, "stressor": label, "dAUROC_ID": di, "dAUROC_SH": ds, "gap": g, "dgap": dg, "dBrier_ID": dbi, "dBrier_SH": dbs})
    return out
if dg is not None and an is not None:
    S_D = [("Label noise 20\\%", "Label Noise", "20% Noise", {}), ("Missingness 20\\% MCAR", "Missingness", "MCAR", {"Missing Rate": 0.2}), ("Feature shift (Diagnoses)", "Feature Shift", "Diagnoses", {})]
    S_A = [("Label noise 20\\%", "Label Noise", "20% Noise", {}), ("Missingness 20\\% MCAR", "Missingness", "MCAR", {"Missing Rate": 0.2}), ("Feature shift (Disability)", "Feature Shift", "Disability", {})]
    rows = cross_rows(dg, "Diabetes", S_D) + cross_rows(an, "ACSEmployment", S_A)
    L = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Model} & \textbf{Stressor (strongest tested)} & \textbf{$\Delta$AUROC ID} & \textbf{$\Delta$AUROC SH} & \textbf{AUROC gap} & \textbf{$\Delta$gap} & \textbf{$\Delta$Brier ID} & \textbf{$\Delta$Brier SH} \\", r"\midrule"]
    last = None
    for r in rows:
        key = (r["dataset"], r["model"])
        L.append(f"{r['dataset'] if key[0]!=(last or ('',''))[0] else ''} & {SHORT[r['model']] if key!=last else ''} & {r['stressor']} & {f3(*r['dAUROC_ID'], sign=True)} & {f3(*r['dAUROC_SH'], sign=True)} & {f3(*r['gap'], sign=True)} & {f3(*r['dgap'], sign=True)} & {f3(*r['dBrier_ID'], sign=True)} & {f3(*r['dBrier_SH'], sign=True)} \\\\")
        last = key
    L += [r"\bottomrule", r"\end{tabular}"]
    flat = pd.DataFrame([{"dataset": r["dataset"], "model": r["model"], "stressor": r["stressor"], "dAUROC_ID": r["dAUROC_ID"][0], "dAUROC_SH": r["dAUROC_SH"][0], "gap": r["gap"][0], "dgap": r["dgap"][0], "dgap_sd": r["dgap"][1], "dBrier_ID": r["dBrier_ID"][0], "dBrier_SH": r["dBrier_SH"][0]} for r in rows])
    write("tab_cross_stressor", "\n".join(L), flat)

# ============================================================ calibration
def cal_table(run, name, conds):
    p = f"{R}/{run}/calibration_per_seed.csv"
    if not os.path.exists(p): return None
    c = pd.read_csv(p, keep_default_na=False, na_values=[""])
    L = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
         r"\textbf{Condition} & \textbf{Model} & \textbf{Set} & \textbf{Brier} & \textbf{Const.\ Brier} & \textbf{ECE} & \textbf{Reliab.} & \textbf{Resol.} & \textbf{Mean $\hat p$ / prev.} \\", r"\midrule"]
    rows = []
    for (exp, cond, lab) in conds:
        for m in MODELS:
            for ts, tl in [(ID, "ID"), (SH, "SH")]:
                q = c[(c.Experiment == exp) & (c.Condition == cond) & (c.Model == m) & (c["Test Set"] == ts)]
                if q.empty: continue
                g = q.mean(numeric_only=True)
                L.append(f"{lab if (m==MODELS[0] and tl=='ID') else ''} & {SHORT[m] if tl=='ID' else ''} & {tl} & {g.Brier:.3f} & {g.constant_baseline_Brier:.3f} & {g.ECE:.3f} & {g.reliability:.3f} & {g.resolution:.3f} & {g.mean_pred:.3f} / {g.prevalence:.3f} \\\\")
                rows.append({"condition": lab, "model": m, "set": tl, **{k: g[k] for k in ["Brier", "constant_baseline_Brier", "ECE", "reliability", "resolution", "uncertainty", "mean_pred", "prevalence"]}})
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]; write(name, "\n".join(L), pd.DataFrame(rows)); return pd.DataFrame(rows)
noise_conds = [("Baseline", "None", "Clean"), ("Label Noise", "5% Noise", "Noise 5\\%"), ("Label Noise", "10% Noise", "Noise 10\\%"), ("Label Noise", "20% Noise", "Noise 20\\%")]
cd = cal_table("diabetes", "tab_calibration_diabetes", noise_conds)
ca = cal_table("acs", "tab_calibration_acs", noise_conds)
for tag, cdf in [("diab", cd), ("acs", ca)]:
    if cdf is None: continue
    for r in cdf[cdf.condition == "Clean"].itertuples():
        k = f"{tag}{'lr' if r.model.startswith('Log') else 'gb'}{r.set}"
        macros[f"cal{k}Brier"] = f"{r.Brier:.3f}"; macros[f"cal{k}Const"] = f"{r.constant_baseline_Brier:.3f}"; macros[f"cal{k}ECE"] = f"{r.ECE:.3f}"
        macros[f"cal{k}MeanP"] = f"{r.mean_pred:.3f}"; macros[f"cal{k}Prev"] = f"{r.prevalence:.3f}"; macros[f"cal{k}Rel"] = f"{r.reliability:.3f}"
    for r in cdf[cdf.condition == "Noise 20\\%"].itertuples():
        k = f"{tag}{'lr' if r.model.startswith('Log') else 'gb'}{r.set}"
        macros[f"calN{k}Brier"] = f"{r.Brier:.3f}"; macros[f"calN{k}ECE"] = f"{r.ECE:.3f}"; macros[f"calN{k}Rel"] = f"{r.reliability:.3f}"; macros[f"calN{k}MeanP"] = f"{r.mean_pred:.3f}"

# ============================================================ blanking investigation
for run, name in [("diabetes", "tab_blanking")]:
    p = f"{R}/{run}/blanking_investigation.csv"
    if not os.path.exists(p): continue
    b = pd.read_csv(p, keep_default_na=False, na_values=[""]); g = b.groupby(["Model", "Feature Group Blanked", "Test Set"]).mean(numeric_only=True).reset_index()
    L = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Group} & \textbf{Set} & \textbf{Mean $\hat p$ clean} & \textbf{blanked} & \textbf{Spearman $\rho$} & \textbf{AUROC clean} & \textbf{blanked} & \textbf{Brier clean} & \textbf{blanked} \\", r"\midrule"]
    for m in MODELS:
        for grp in ["Diagnoses", "Medications"]:
            for ts, tl in [(ID, "ID"), (SH, "SH")]:
                r = g[(g.Model == m) & (g["Feature Group Blanked"] == grp) & (g["Test Set"] == ts)].iloc[0]
                L.append(f"{SHORT[m] if (grp=='Diagnoses' and tl=='ID') else ''} & {grp if tl=='ID' else ''} & {tl} & {r['mean_pred_clean']:.2f} & {r['mean_pred_blanked']:.2f} & {r.get('spearman_clean_blanked', float('nan')):.2f} & {r['AUROC_clean']:.3f} & {r['AUROC_blanked']:.3f} & {r['Brier_clean']:.3f} & {r['Brier_blanked']:.3f} \\\\")
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write(name, "\n".join(L), g)
    if run == "diabetes":
        r = g[(g.Model == "Logistic Regression") & (g["Feature Group Blanked"] == "Diagnoses") & (g["Test Set"] == SH)].iloc[0]
        macros.update({"blankLRDiagSpearman": f"{r.get('spearman_clean_blanked', float('nan')):.2f}", "blankLRDiagPearson": f"{r['corr_clean_blanked']:.2f}", "blankLRDiagMeanPClean": f"{r['mean_pred_clean']:.3f}", "blankLRDiagMeanPBlank": f"{r['mean_pred_blanked']:.3f}", "blankLRDiagShareClean": f"{r['share_gt0.5_clean']:.2f}", "blankLRDiagShareBlank": f"{r['share_gt0.5_blanked']:.2f}", "blankLRDiagBrierClean": f"{r['Brier_clean']:.3f}", "blankLRDiagBrierBlank": f"{r['Brier_blanked']:.3f}"})
        r = g[(g.Model == "LightGBM") & (g["Feature Group Blanked"] == "Medications") & (g["Test Set"] == ID)].iloc[0]
        r2 = g[(g.Model == "Logistic Regression") & (g["Feature Group Blanked"] == "Medications") & (g["Test Set"] == SH)].iloc[0]
        macros.update({"blankLRMedSpearman": f"{r2.get('spearman_clean_blanked', float('nan')):.2f}"})
        macros.update({"blankGBMedSpearman": f"{r.get('spearman_clean_blanked', float('nan')):.2f}", "blankGBMedMeanPClean": f"{r['mean_pred_clean']:.3f}", "blankGBMedMeanPBlank": f"{r['mean_pred_blanked']:.3f}", "blankGBMedBrierClean": f"{r['Brier_clean']:.3f}", "blankGBMedBrierBlank": f"{r['Brier_blanked']:.3f}", "blankGBMedAUROCClean": f"{r['AUROC_clean']:.3f}", "blankGBMedAUROCBlank": f"{r['AUROC_blanked']:.3f}"})

# ============================================================ RQ3 exploratory correlations (both datasets), unit of observation stated
if dg is not None and an is not None:
    for dname, d, fname in [("Diabetes", dg, "tab_rq3_diabetes"), ("ACSEmployment", an, "tab_rq3_acs")]:
        q = C.rq3_correspondence(d, dname)
        L = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule", r"\textbf{Stressor} & \textbf{Model / metric} & \textbf{distinct cond.} & \multicolumn{3}{c}{\textbf{absolute scores}} & \multicolumn{3}{c}{\textbf{within-seed changes}} & \textbf{ID range} \\", r" & & & $n$ & $\rho$ & $r$ & $n$ & $\rho$ & $r$ & \\", r"\midrule"]
        last = None
        for r in q.itertuples():
            if last is not None and r.stressor != last: L.append(r"\addlinespace")
            L.append(f"{r.stressor if r.stressor != last else ''} & {SHORT[r.model]} / {MLAB[r.metric]} & {r.n_distinct_conditions} & {r.n_obs} & {r.spearman:.2f} & {r.pearson:.2f} & {r.n_obs_delta} & {r.spearman_delta:.2f} & {r.pearson_delta:.2f} & {r.range_ID:.3f} \\\\"); last = r.stressor
        L += [r"\bottomrule", r"\end{tabular}"]; write(fname, "\n".join(L), q)
    pd.concat([C.rq3_correspondence(dg, "Diabetes"), C.rq3_correspondence(an, "ACSEmployment")]).to_csv(f"{R}/tables/tab_rq3_correlations.csv", index=False)

# ============================================================ sensitivity: diabetes variants, acs onehot
variants = [("diabetes", "Main protocol (42 predictors)"), ("diabetes_no_admission_source", "$-$ admission\\_source\\_id"), ("diabetes_ids_categorical", "ID codes as categorical")]
dfs = [(load(r), lab) for r, lab in variants]
if all(d is not None for d, _ in dfs):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule", r"\textbf{Model} & \textbf{Variant} & \textbf{AUROC ID} & \textbf{AUROC SH} & \textbf{Gap (SD)} & \textbf{AP SH} & \textbf{Brier ID} & \textbf{Brier SH} \\", r"\midrule"]
    rows = []
    for m in MODELS:
        for i, (d, lab) in enumerate(dfs):
            a, b, g = stat(d, "Baseline", "None", m, ID, "AUROC"), stat(d, "Baseline", "None", m, SH, "AUROC"), stat(d, "Baseline", "None", m, GAP, "AUROC")
            ap, bi, bs = stat(d, "Baseline", "None", m, SH, "AUPRC"), stat(d, "Baseline", "None", m, ID, "Brier Score"), stat(d, "Baseline", "None", m, SH, "Brier Score")
            L.append(f"{SHORT[m] if i==0 else ''} & {lab} & {f3(a[0])} & {f3(b[0])} & {f3(*g, sign=True)} & {f3(ap[0])} & {f3(bi[0])} & {f3(bs[0])} \\\\")
            rows.append({"model": m, "variant": lab, "AUROC_ID": a[0], "AUROC_SH": b[0], "gap": g[0], "AP_SH": ap[0], "Brier_ID": bi[0], "Brier_SH": bs[0]})
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_sensitivity_diabetes", "\n".join(L), pd.DataFrame(rows))
ao = load("acs_onehot")
if an is not None and ao is not None:
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule", r"\textbf{Model} & \textbf{Pipeline} & \textbf{AUROC ID} & \textbf{AUROC SH} & \textbf{Gap (SD)} & \textbf{Brier ID} & \textbf{Brier SH} & \textbf{Brier SH, 20\% noise} \\", r"\midrule"]
    rows = []
    for m in MODELS:
        for i, (d, lab) in enumerate([(an, "All features numeric (as implemented)"), (ao, "Nominal codes one-hot encoded")]):
            a, b, g = stat(d, "Baseline", "None", m, ID, "AUROC"), stat(d, "Baseline", "None", m, SH, "AUROC"), stat(d, "Baseline", "None", m, GAP, "AUROC")
            bi, bs, bn = stat(d, "Baseline", "None", m, ID, "Brier Score"), stat(d, "Baseline", "None", m, SH, "Brier Score"), stat(d, "Label Noise", "20% Noise", m, SH, "Brier Score")
            L.append(f"{SHORT[m] if i==0 else ''} & {lab} & {f3(a[0])} & {f3(b[0])} & {f3(*g, sign=True)} & {f3(bi[0])} & {f3(bs[0])} & {f3(bn[0])} \\\\")
            rows.append({"model": m, "pipeline": lab, "AUROC_ID": a[0], "AUROC_SH": b[0], "gap": g[0], "Brier_ID": bi[0], "Brier_SH": bs[0], "Brier_SH_noise20": bn[0]})
        if m == MODELS[0]: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_sensitivity_acs", "\n".join(L), pd.DataFrame(rows))
    for r in rows:
        k = ("lr" if r["model"].startswith("Log") else "gb") + ("Num" if r["pipeline"].startswith("All") else "OH")
        macros[f"acsSens{k}AUROCID"] = f"{r['AUROC_ID']:.3f}"; macros[f"acsSens{k}AUROCSH"] = f"{r['AUROC_SH']:.3f}"

# ============================================================ clustered bootstrap
p = f"{R}/diabetes/bootstrap_clustered.csv"
if os.path.exists(p):
    bt = pd.read_csv(p, keep_default_na=False, na_values=[""])
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"\textbf{Statistic} & \textbf{Seed} & \textbf{Bootstrap mean} & \textbf{2.5\%} & \textbf{97.5\%} \\", r"\midrule"]
    labels = {"gap_AUROC": "AUROC gap", "gap_AUPRC": "AP gap (LightGBM)", "gap_Brier": "Brier gap (LightGBM)", "diff_shifted_AUROC_LGBM_minus_LR": "Shifted AUROC, LightGBM $-$ LR", "diff_gap_AUROC_LGBM_minus_LR": "AUROC gap, LightGBM $-$ LR"}
    for st in ["gap_AUROC", "diff_shifted_AUROC_LGBM_minus_LR", "diff_gap_AUROC_LGBM_minus_LR", "gap_Brier"]:
        sub = bt[bt.statistic == st].sort_values(["model", "Random Seed"])
        for i, r in enumerate(sub.itertuples()):
            lab = labels[st] + (f" ({SHORT[r.model]})" if st == "gap_AUROC" else "")
            L.append(f"{lab if (i==0 or (st=='gap_AUROC' and r._1==42)) else ''} & {r._1} & {f3(r.boot_mean, sign=True)} & {f3(r.ci_lo, sign=True)} & {f3(r.ci_hi, sign=True)} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]; write("tab_bootstrap", "\n".join(L), bt)
    B = int(bt.B.iloc[0]); macros["bootB"] = f"{B:,}"
    for r in bt[(bt.statistic == "gap_AUROC")].itertuples():
        k = f"boot{'lr' if r.model.startswith('Log') else 'gb'}Gap{r._1}"
        macros[k + "Lo"] = f"{r.ci_lo:+.3f}".replace("-", "$-$").replace("+", "$+$"); macros[k + "Hi"] = f"{r.ci_hi:+.3f}".replace("-", "$-$").replace("+", "$+$")
    for r in bt[bt.statistic == "diff_gap_AUROC_LGBM_minus_LR"].itertuples():
        macros[f"bootDiffGap{r._1}Lo"] = f"{r.ci_lo:+.3f}".replace("-", "$-$").replace("+", "$+$"); macros[f"bootDiffGap{r._1}Hi"] = f"{r.ci_hi:+.3f}".replace("-", "$-$").replace("+", "$+$")

# ============================================================ class-weight sensitivity (Diabetes clean models)
pw, pu = f"{R}/diabetes/calibration_per_seed.csv", f"{R}/diabetes_unweighted/calibration_per_seed.csv"
if os.path.exists(pw) and os.path.exists(pu):
    cw = pd.read_csv(pw, keep_default_na=False, na_values=[""]); cu = pd.read_csv(pu, keep_default_na=False, na_values=[""])
    rw = load("diabetes"); ru = load("diabetes_unweighted")
    L = [r"\begin{tabular}{lllrrrrrr}", r"\toprule", r"\textbf{Weighting} & \textbf{Model} & \textbf{Set} & \textbf{AUROC} & \textbf{AP} & \textbf{Brier} & \textbf{ECE} & \textbf{Reliab.} & \textbf{Mean $\hat p$} \\", r"\midrule"]
    rows = []
    for wlab, cdf, rdf in [("Balanced", cw, rw), ("None", cu, ru)]:
        for m in MODELS:
            for ts, tl in [(ID, "ID"), (SH, "SH")]:
                q = cdf[(cdf.Experiment == "Baseline") & (cdf.Model == m) & (cdf["Test Set"] == ts)].mean(numeric_only=True)
                a, ap = stat(rdf, "Baseline", "None", m, ts, "AUROC")[0], stat(rdf, "Baseline", "None", m, ts, "AUPRC")[0]
                L.append(f"{wlab if (m == MODELS[0] and tl == 'ID') else ''} & {SHORT[m] if tl == 'ID' else ''} & {tl} & {a:.3f} & {ap:.3f} & {q.Brier:.3f} & {q.ECE:.3f} & {q.reliability:.3f} & {q.mean_pred:.3f} \\\\")
                rows.append({"weighting": wlab, "model": m, "set": tl, "AUROC": a, "AP": ap, "Brier": q.Brier, "ECE": q.ECE, "reliability": q.reliability, "mean_pred": q.mean_pred})
                k = f"cw{'Bal' if wlab == 'Balanced' else 'None'}{'lr' if m.startswith('Log') else 'gb'}{tl}"
                macros[k + "Brier"] = f"{q.Brier:.3f}"; macros[k + "ECE"] = f"{q.ECE:.3f}"; macros[k + "MeanP"] = f"{q.mean_pred:.3f}"; macros[k + "AUROC"] = f"{a:.3f}"; macros[k + "Rel"] = f"{q.reliability:.3f}"
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]; write("tab_class_weight", "\n".join(L), pd.DataFrame(rows))

# ============================================================ ACS dataset facts
p = f"{R}/acs_dataset_stats.json"
if os.path.exists(p):
    a = json.load(open(p))
    macros.update({"acsCArows": f"{a['CA_rows']:,}", "acsTXrows": f"{a['TX_rows']:,}", "acsCAprev": f"{a['CA_prev']*100:.1f}", "acsTXprev": f"{a['TX_prev']*100:.1f}",
                   "acsCAunderSixteen": f"{a['CA_under16_share']*100:.1f}", "acsTXunderSixteen": f"{a['TX_under16_share']*100:.1f}"})


# ============================================================ compact displays for the shortened main text
# (a) baseline, both datasets in one table
if dg is not None and an is not None:
    L = [r"\begin{tabular}{lllrrr}", r"\toprule", r"\textbf{Dataset} & \textbf{Model} & \textbf{Metric} & \textbf{In-domain} & \textbf{Shifted} & \textbf{Gap (SH $-$ ID)} \\", r"\midrule"]
    for dname, d in [("Diabetes", dg), ("ACSEmployment", an)]:
        for m in MODELS:
            for i, met in enumerate(METRICS):
                a, b, g = stat(d, "Baseline", "None", m, ID, met), stat(d, "Baseline", "None", m, SH, met), stat(d, "Baseline", "None", m, GAP, met)
                L.append(f"{dname if (m == MODELS[0] and i == 0) else ''} & {SHORT[m] if i == 0 else ''} & {MLAB[met]} & {f3(*a)} & {f3(*b)} & {f3(*g, sign=True)} \\\\")
            if not (dname == "ACSEmployment" and m == MODELS[1]): L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_baseline_both", "\n".join(L))
# (b) missingness at 20 %, both datasets, paired deltas
if dg is not None and an is not None:
    L = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"\textbf{Dataset} & \textbf{Model} & \textbf{Mechanism} & \textbf{$\Delta$AUROC ID} & \textbf{$\Delta$AUROC SH} & \textbf{$\Delta$Brier ID} & \textbf{$\Delta$Brier SH} \\", r"\midrule"]
    for dname, d, scen in [("Diabetes", dg, ["MCAR", "MAR_Female", "MAR_Age_Older"]), ("ACSEmployment", an, ["MCAR", "MAR_Age_Older"])]:
        for m in MODELS:
            for i, sc in enumerate(scen):
                kw = {"Missing Rate": 0.2}
                di, ds = paired_delta(d, "Missingness", sc, m, ID, "AUROC", **kw), paired_delta(d, "Missingness", sc, m, SH, "AUROC", **kw)
                bi, bs = paired_delta(d, "Missingness", sc, m, ID, "Brier Score", **kw), paired_delta(d, "Missingness", sc, m, SH, "Brier Score", **kw)
                L.append(f"{dname if (m == MODELS[0] and i == 0) else ''} & {SHORT[m] if i == 0 else ''} & {sc.replace('_', chr(92)+'_')} & {f3(*di, sign=True)} & {f3(*ds, sign=True)} & {f3(*bi, sign=True)} & {f3(*bs, sign=True)} \\\\")
            if not (dname == "ACSEmployment" and m == MODELS[1]): L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_missingness_both", "\n".join(L))
# (c) compact RQ3: within-seed-change Spearman correlations, dataset x stressor x model
if dg is not None and an is not None:
    q = pd.concat([C.rq3_correspondence(dg, "Diabetes"), C.rq3_correspondence(an, "ACSEmployment")])
    L = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"\textbf{Dataset} & \textbf{Stressor} & \textbf{Model} & \textbf{$n$ (cond.)} & \textbf{AUROC $\rho$} & \textbf{AP $\rho$} & \textbf{Brier $\rho$} \\", r"\midrule"]
    last = None
    for dname in ["Diabetes", "ACSEmployment"]:
        for st in ["Label Noise", "Missingness", "Feature Shift"]:
            for m in MODELS:
                sub = q[(q.dataset == dname) & (q.stressor == st) & (q.model == m)].set_index("metric")
                if sub.empty: continue
                r0 = sub.iloc[0]
                L.append(f"{dname if (dname != last) else ''} & {st if m == MODELS[0] else ''} & {SHORT[m]} & {int(r0.n_obs_delta)} ({int(r0.n_distinct_nonclean)}) & {sub.loc['AUROC','spearman_delta']:.2f} & {sub.loc['AUPRC','spearman_delta']:.2f} & {sub.loc['Brier Score','spearman_delta']:.2f} \\\\")
                last = dname
            L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]; write("tab_rq3_compact", "\n".join(L), q)
# (d) principal bootstrap intervals only (AUROC gaps and their difference), seeds as columns
p = f"{R}/diabetes/bootstrap_clustered.csv"
if os.path.exists(p):
    bt = pd.read_csv(p, keep_default_na=False, na_values=[""])
    def cell(st, model, seed):
        r = bt[(bt.statistic == st) & (bt.model == model) & (bt["Random Seed"] == seed)].iloc[0]
        return f"[{f3(r.ci_lo, sign=True)}, {f3(r.ci_hi, sign=True)}]"
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"\textbf{95\% interval (AUROC)} & \textbf{Seed 42} & \textbf{Seed 43} & \textbf{Seed 44} \\", r"\midrule",
         "Gap, LightGBM & " + " & ".join(cell("gap_AUROC", "LightGBM", s) for s in (42, 43, 44)) + " \\\\",
         "Gap, LR & " + " & ".join(cell("gap_AUROC", "Logistic Regression", s) for s in (42, 43, 44)) + " \\\\",
         "Gap difference, LightGBM $-$ LR & " + " & ".join(cell("diff_gap_AUROC_LGBM_minus_LR", "both", s) for s in (42, 43, 44)) + " \\\\",
         "Shifted AUROC, LightGBM $-$ LR & " + " & ".join(cell("diff_shifted_AUROC_LGBM_minus_LR", "both", s) for s in (42, 43, 44)) + " \\\\",
         r"\bottomrule", r"\end{tabular}"]
    write("tab_bootstrap_compact", "\n".join(L))

# (e) calibration: clean vs 20 % noise, both datasets, in one table
rows_c = []
for run, dname in [("diabetes", "Diabetes"), ("acs", "ACSEmployment")]:
    pth = f"{R}/{run}/calibration_per_seed.csv"
    if not os.path.exists(pth): continue
    c = pd.read_csv(pth, keep_default_na=False, na_values=[""])
    for (exp, cond, lab) in [("Baseline", "None", "Clean"), ("Label Noise", "20% Noise", "Noise 20\\%")]:
        for m in MODELS:
            for ts, tl in [(ID, "ID"), (SH, "SH")]:
                q = c[(c.Experiment == exp) & (c.Condition == cond) & (c.Model == m) & (c["Test Set"] == ts)].mean(numeric_only=True)
                rows_c.append((dname, lab, m, tl, q))
if rows_c:
    L = [r"\begin{tabular}{llllrrrrrr}", r"\toprule", r"\textbf{Dataset} & \textbf{Condition} & \textbf{Model} & \textbf{Set} & \textbf{Brier} & \textbf{Const.} & \textbf{ECE} & \textbf{Reliab.} & \textbf{Resol.} & \textbf{Mean $\hat p$ / prev.} \\", r"\midrule"]
    prev = None
    for dname, lab, m, tl, q in rows_c:
        key = (dname, lab)
        if prev is not None and key != prev: L.append(r"\addlinespace")
        L.append(f"{dname if (prev is None or dname != prev[0]) else ''} & {lab if key != prev else ''} & {SHORT[m] if tl == 'ID' else ''} & {tl} & {q.Brier:.3f} & {q.constant_baseline_Brier:.3f} & {q.ECE:.3f} & {q.reliability:.3f} & {q.resolution:.3f} & {q.mean_pred:.3f} / {q.prevalence:.3f} \\\\")
        prev = key
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_calibration_compact", "\n".join(L))
# (f) feature blanking, both datasets (compact: absolute AUROC/Brier plus AUROC changes)
if dg is not None and an is not None:
    L = [r"\begin{tabular}{lllrrrrrr}", r"\toprule", r"\textbf{Dataset} & \textbf{Model} & \textbf{Group} & \textbf{AUROC ID} & \textbf{$\Delta$ ID} & \textbf{AUROC SH} & \textbf{$\Delta$ SH} & \textbf{Brier ID} & \textbf{Brier SH} \\", r"\midrule"]
    short = {"Citizenship_Nativity_Ancestry": "Citizenship"}
    for dname, d, groups in [("Diabetes", dg, ["Diagnoses", "Medications"]), ("ACSEmployment", an, ["Education", "Disability", "Citizenship_Nativity_Ancestry"])]:
        for m in MODELS:
            for i, g in enumerate(groups):
                ai, asx = stat(d, "Feature Shift", g, m, ID, "AUROC"), stat(d, "Feature Shift", g, m, SH, "AUROC")
                di, ds = paired_delta(d, "Feature Shift", g, m, ID, "AUROC"), paired_delta(d, "Feature Shift", g, m, SH, "AUROC")
                bi, bs = stat(d, "Feature Shift", g, m, ID, "Brier Score"), stat(d, "Feature Shift", g, m, SH, "Brier Score")
                L.append(f"{dname if (m == MODELS[0] and i == 0) else ''} & {SHORT[m] if i == 0 else ''} & {short.get(g, g)} & {f3(ai[0])} & {f3(*di, sign=True)} & {f3(asx[0])} & {f3(*ds, sign=True)} & {f3(bi[0])} & {f3(bs[0])} \\\\")
            if not (dname == "ACSEmployment" and m == MODELS[1]): L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]; write("tab_feature_shift_both", "\n".join(L))

# ============================================================ macros
with open(f"{T}/numbers.tex", "w") as f:
    f.write("% Auto-generated by experiments/make_tables.py -- do not edit by hand.\n")
    def key(k):  # LaTeX macro names cannot contain digits
        return k.replace("20", "Twenty").replace("42", "SeedA").replace("43", "SeedB").replace("44", "SeedC").replace("0.1", "").replace("5", "Five").replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{key(k)}}}{{{v}}}\n")
print("macros:", len(macros))

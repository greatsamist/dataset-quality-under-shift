"""
Shared experiment code for the thesis.

This module holds every method decision that the results depend on: loading and
cleaning the two datasets, defining the shifted populations, splitting, the
preprocessing pipelines and models, the three interventions, the metrics and the
result store.  The run scripts and the two notebooks call this code, so there is
one implementation of the methodology and the documented results cannot drift
from it.

Where the two datasets need different helper behaviour (for example different
random-number styles for label noise), both variants are kept and selected
explicitly by the caller, so nothing changes silently.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

SEEDS = [42, 43, 44]
MODEL_NAMES = ["Logistic Regression", "LightGBM"]

# --------------------------------------------------------------------------- #
# Environment record
# --------------------------------------------------------------------------- #
def environment_record(extra: dict | None = None) -> dict:
    import sklearn, lightgbm, scipy
    rec = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "scipy": scipy.__version__,
        "seeds": SEEDS,
        "argv": sys.argv,
    }
    try:
        import folktables
        rec["folktables"] = getattr(folktables, "__version__", "0.0.12 (pip)")
    except Exception:
        pass
    if extra:
        rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
# Diabetes 130-US hospitals
# --------------------------------------------------------------------------- #
DIAB_DROP = ["encounter_id", "weight", "max_glu_serum", "A1Cresult",
             "payer_code", "medical_specialty", "readmitted"]
# These three are integer *codes* (nominal) that the notebooks treated as numeric
# because select_dtypes(include=np.number) picked them up.
DIAB_ID_CODES = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
SHIFT_COHORT_COL, SHIFT_COHORT_VALUE = "admission_source_id", 17  # 17 = "NULL" in IDS_mapping.csv

MED_COLS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone']
DIAB_FEATURE_GROUPS = {"Diagnoses": ["diag_1", "diag_2", "diag_3"], "Medications": MED_COLS}
DIAB_MISSING_COLS = ['time_in_hospital', 'num_lab_procedures', 'race', 'diag_1']
DIAB_MISSING_SCENARIOS = [
    {"label": "MCAR",          "type": "MCAR", "condition": None},
    {"label": "MAR_Female",    "type": "MAR",  "condition": {"column": "gender", "values": ["Female"]}},
    {"label": "MAR_Age_Older", "type": "MAR",  "condition": {"column": "age",    "values": ["[70-80)", "[80-90)"]}},
]


def load_diabetes(path: str) -> pd.DataFrame:
    """Replicates the notebook cleaning cell, but keeps patient_nbr for grouping."""
    df = pd.read_csv(path, low_memory=False)
    df = df.replace("?", None)                      # '?' -> missing (verified: pandas 3.x sets NaN)
    df["y"] = df["readmitted"].map({"<30": 1, ">30": 0, "NO": 0}).astype(int)
    df = df.drop(columns=DIAB_DROP)
    return df


def diabetes_xy(d: pd.DataFrame, drop_cols=()):
    X = d.drop(columns=["y", "patient_nbr", *drop_cols])
    return X, d["y"]


def split_diabetes_record(df: pd.DataFrame, seed: int):
    """Record-level split: partitions are drawn over encounters, so a patient with
    several encounters can appear in more than one partition. Used only to document
    how much patient overlap a standard split would produce (thesis Section 3.3).
    The evaluation protocol uses split_diabetes_patient_disjoint."""
    sh = df[df[SHIFT_COHORT_COL] == SHIFT_COHORT_VALUE].copy()
    rem = df[df[SHIFT_COHORT_COL] != SHIFT_COHORT_VALUE].copy()
    tr, tmp = train_test_split(rem, test_size=0.30, stratify=rem["y"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["y"], random_state=seed)
    return {"train": tr, "val": va, "test": te, "shifted": sh}


def split_diabetes_patient_disjoint(df: pd.DataFrame, seed: int):
    """
    Patient-disjoint split.
      1. Shifted cohort = every encounter with admission_source_id == 17.
      2. Patients who appear in the shifted cohort are reserved: their *other*
         encounters are removed from the source pool entirely.
      3. Remaining source PATIENTS are shuffled (seeded) and cut 70/15/15 by
         patient; every encounter follows its patient.
    Encounter proportions are therefore only approximately 70/15/15.
    """
    sh = df[df[SHIFT_COHORT_COL] == SHIFT_COHORT_VALUE].copy()
    reserved = set(sh["patient_nbr"])
    rem = df[(df[SHIFT_COHORT_COL] != SHIFT_COHORT_VALUE) & (~df["patient_nbr"].isin(reserved))].copy()
    patients = np.array(sorted(rem["patient_nbr"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
    p_tr, p_va, p_te = patients[:n_tr], patients[n_tr:n_tr + n_va], patients[n_tr + n_va:]
    parts = {
        "train": rem[rem["patient_nbr"].isin(p_tr)].copy(),
        "val":   rem[rem["patient_nbr"].isin(p_va)].copy(),
        "test":  rem[rem["patient_nbr"].isin(p_te)].copy(),
        "shifted": sh,
    }
    assert_patient_disjoint(parts)
    return parts


def assert_patient_disjoint(parts: dict):
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = set(parts[a]["patient_nbr"]) & set(parts[b]["patient_nbr"])
            assert not ov, f"patient overlap {a}/{b}: {len(ov)}"


def split_audit(parts: dict) -> dict:
    """Counts, unique patients, prevalence and pairwise patient overlap."""
    out = {}
    for k, d in parts.items():
        out[k] = {"encounters": int(len(d)), "patients": int(d["patient_nbr"].nunique()),
                  "prevalence": float(d["y"].mean())}
    names = list(parts)
    ov = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = set(parts[a]["patient_nbr"]), set(parts[b]["patient_nbr"])
            shared = pa & pb
            ov[f"{a}|{b}"] = {"shared_patients": len(shared),
                              f"{b}_encounters_from_shared": int(parts[b]["patient_nbr"].isin(shared).sum())}
    out["overlap"] = ov
    return out


def build_diabetes_models(X_train: pd.DataFrame, seed: int, ids_categorical: bool = False, class_weight="balanced"):
    """Identical to the notebook's train_models(); ids_categorical is the sensitivity variant."""
    X_train = X_train.copy()
    if ids_categorical:
        for c in DIAB_ID_CODES:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype(str)
    num = X_train.select_dtypes(include=np.number).columns.tolist()
    cat = X_train.select_dtypes(exclude=np.number).columns.tolist()
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    pre_lr = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler(with_mean=False))]), num),
        ("cat", cat_pipe, cat)], remainder="drop")
    pre_gb = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", clone(cat_pipe), cat)], remainder="drop")
    lr = Pipeline([("preprocessor", pre_lr),
                   ("classifier", LogisticRegression(solver="liblinear", random_state=seed,
                                                     class_weight=class_weight))])
    gb = Pipeline([("preprocessor", pre_gb),
                   ("classifier", LGBMClassifier(random_state=seed, verbose=-1, class_weight=class_weight))])
    return {"Logistic Regression": lr, "LightGBM": gb}, {"numeric": num, "categorical": cat}


def cast_ids(X: pd.DataFrame, ids_categorical: bool) -> pd.DataFrame:
    if not ids_categorical:
        return X
    X = X.copy()
    for c in DIAB_ID_CODES:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X


# --------------------------------------------------------------------------- #
# ACSEmployment (folktables)
# --------------------------------------------------------------------------- #
ACS_MISSING_COLS = ['AGEP', 'SCHL', 'MAR', 'DIS', 'CIT', 'NATIVITY', 'SEX', 'RAC1P']
ACS_FEATURE_GROUPS = {
    "Education": ["SCHL"],
    "Disability": ["DIS", "DEAR", "DEYE", "DREM"],
    "Citizenship_Nativity_Ancestry": ["CIT", "NATIVITY", "ANC"],
}
# Nominal codes among the 16 ACSEmployment features (AGEP is a count; SCHL is ordinal).
ACS_NOMINAL = ['MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY',
               'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']


def load_acs(root_dir: str):
    from folktables import ACSDataSource, ACSEmployment
    ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=root_dir)
    out = {}
    for st in ["CA", "TX"]:
        raw = ds.get_data(states=[st], download=True)
        X, y, _ = ACSEmployment.df_to_numpy(raw)
        d = pd.DataFrame(X, columns=ACSEmployment.features)
        d["y"] = y.astype(int)
        out[st] = d
    return out["CA"], out["TX"]


def split_acs(df_ca: pd.DataFrame, df_tx: pd.DataFrame, seed: int):
    X_ca, y_ca = df_ca.drop(columns=["y"]), df_ca["y"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_ca, y_ca, test_size=0.30, random_state=seed, stratify=y_ca)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    X_sh, y_sh = df_tx.drop(columns=["y"]).reset_index(drop=True), df_tx["y"].reset_index(drop=True)
    return {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te), "shifted": (X_sh, y_sh)}


def build_acs_models(seed: int, pipeline: str = "numeric"):
    """'numeric' reproduces the notebook; 'onehot' one-hot encodes the nominal codes."""
    lr_clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=seed)
    gb_clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64, subsample=0.9,
                            colsample_bytree=0.9, class_weight="balanced", random_state=seed, verbose=-1)
    if pipeline == "numeric":
        lr = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler(with_mean=False)), ("classifier", lr_clf)])
        gb = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", gb_clf)])
    elif pipeline == "onehot":
        numeric = ["AGEP", "SCHL"]
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                             ("encoder", OneHotEncoder(handle_unknown="ignore"))])
        pre_lr = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler(with_mean=False))]), numeric),
            ("cat", cat_pipe, ACS_NOMINAL)])
        pre_gb = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric),
                                    ("cat", clone(cat_pipe), ACS_NOMINAL)])
        lr = Pipeline([("preprocessor", pre_lr), ("classifier", lr_clf)])
        gb = Pipeline([("preprocessor", pre_gb), ("classifier", gb_clf)])
    else:
        raise ValueError(pipeline)
    return {"Logistic Regression": lr, "LightGBM": gb}


# --------------------------------------------------------------------------- #
# Interventions (both notebook variants preserved)
# --------------------------------------------------------------------------- #
def label_noise(y: pd.Series, rate: float, seed: int, style: str) -> pd.Series:
    """Uniform random flips of a fixed number of TRAINING labels.
    style='diabetes': int(n*rate) flips, np.random.seed (as Diabetic.ipynb)
    style='acs':      int(round(n*rate)) flips, default_rng (as ACSEmployment.ipynb)"""
    y_noisy = pd.Series(np.asarray(y, dtype=int).copy(), index=y.index)
    if rate == 0:
        return y_noisy
    n = len(y_noisy)
    if style == "diabetes":
        k = int(n * rate)
        np.random.seed(seed)
        idx = np.random.choice(y_noisy.index, size=k, replace=False)
        y_noisy.loc[idx] = 1 - y_noisy.loc[idx]
    else:
        k = int(round(n * rate))
        rng = np.random.default_rng(seed)
        pos = rng.choice(n, size=k, replace=False)
        vals = y_noisy.to_numpy().copy(); vals[pos] = 1 - vals[pos]
        y_noisy = pd.Series(vals, index=y.index)
    return y_noisy


def mcar(df: pd.DataFrame, cols, rate: float, seed: int, style: str) -> pd.DataFrame:
    """style='diabetes': k = int(len(df)*rate) capped at #non-missing, np.random.seed
       style='acs':      k = int(round(rate*#non-missing)), default_rng"""
    d = df.copy()
    if style == "diabetes":
        np.random.seed(seed)
    else:
        rng = np.random.default_rng(seed)
    for c in cols:
        if c not in d.columns:
            continue
        non_nan = d[c].dropna().index.to_numpy()
        k = min(int(len(d[c]) * rate), len(non_nan)) if style == "diabetes" else int(round(rate * len(non_nan)))
        if k > 0:
            fill = None if d[c].dtype == object else np.nan
            pick = np.random.choice(non_nan, size=k, replace=False) if style == "diabetes" else rng.choice(non_nan, size=k, replace=False)
            d.loc[pick, c] = fill
    return d


def mar(df: pd.DataFrame, cols, rate: float, cond_mask: pd.Series, seed: int, style: str) -> pd.DataFrame:
    """Missingness only among rows where cond_mask is True; the *denominator is the
    eligible subgroup*, not the whole partition. The conditioning column is not masked."""
    d = df.copy()
    cond_idx = d[cond_mask].index
    if style == "diabetes":
        np.random.seed(seed)
    else:
        rng = np.random.default_rng(seed)
    for c in cols:
        if c not in d.columns:
            continue
        target = d.loc[cond_idx, c].dropna().index.to_numpy()
        k = int(len(target) * rate) if style == "diabetes" else int(round(rate * len(target)))
        if k > 0:
            fill = None if d[c].dtype == object else np.nan
            pick = np.random.choice(target, size=k, replace=False) if style == "diabetes" else rng.choice(target, size=k, replace=False)
            d.loc[pick, c] = fill
    return d


def blank_out(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Evaluation-time 'blanking': the entire column becomes missing (None for object
    columns, NaN otherwise). Downstream: numeric -> training median; categorical ->
    the constant 'missing' token -> one-hot (all-zero if 'missing' was never seen in training)."""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = None if d[c].dtype == object else np.nan
    return d


# --------------------------------------------------------------------------- #
# Evaluation + result / prediction accumulation
# --------------------------------------------------------------------------- #
def metrics(y, p) -> dict:
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),  # AUPRC == sklearn average precision
            "Brier Score": brier_score_loss(y, p)}


# --------------------------------------------------------------------------- #
# Evaluation vocabulary and the RQ3 correspondence measure
# --------------------------------------------------------------------------- #
ID_TEST = "In-domain Test"
SH_TEST = "Shifted Test"
DOMAIN_GAP = "Generalization Gap (Shifted - In-domain)"
METRICS = ["AUROC", "AUPRC", "Brier Score"]


def rq3_correspondence(df, dataset):
    """RQ3: how far the response to each stressor corresponds between the in-domain and
    the shifted evaluation domain, within one dataset. Correlates in-domain and shifted values across (condition, seed) observations, on absolute
    scores and on within-seed changes from the clean baseline (stressed - clean, per domain).
    'n_distinct_conditions' counts substantively different conditions: the zero-severity rows of
    the label-noise and missingness experiments are all the clean baseline and count once."""
    from scipy.stats import spearmanr, pearsonr
    out = []
    for exp, ccol in [("Label Noise", "Noise Rate"), ("Missingness", "Missingness Type"), ("Feature Shift", "Feature Group Blanked")]:
        e = df[(df.Experiment == exp) & (df["Test Set"].isin([ID_TEST, SH_TEST]))]
        base = df[(df.Experiment == "Baseline") & (df["Test Set"].isin([ID_TEST, SH_TEST]))]
        for m in MODEL_NAMES:
            for met in METRICS:
                w = e[e.Model == m].pivot_table(index=["Condition", "Missing Rate", "Noise Rate", "Random Seed"], columns="Test Set", values=met).dropna().reset_index()
                if len(w) < 3: continue
                sp, pe = spearmanr(w[ID_TEST], w[SH_TEST])[0], pearsonr(w[ID_TEST], w[SH_TEST])[0]
                b = base[base.Model == m].pivot_table(index="Random Seed", columns="Test Set", values=met)
                w["dID"] = w[ID_TEST] - w["Random Seed"].map(b[ID_TEST]); w["dSH"] = w[SH_TEST] - w["Random Seed"].map(b[SH_TEST])
                clean = (w["Missing Rate"].astype(float) == 0) & (w["Noise Rate"].astype(float) == 0) & (w["Condition"] != "Diagnoses") & (w["Condition"] != "Medications") & (~w["Condition"].isin(["Education", "Disability", "Citizenship_Nativity_Ancestry"]))
                d = w[~clean]
                spd, ped = (spearmanr(d["dID"], d["dSH"])[0], pearsonr(d["dID"], d["dSH"])[0]) if len(d) >= 3 else (np.nan, np.nan)
                n_nonclean = d[["Condition", "Missing Rate", "Noise Rate"]].drop_duplicates().shape[0]
                n_cond = n_nonclean + (1 if clean.any() else 0)
                out.append({"dataset": dataset, "stressor": exp, "model": m, "metric": met, "n_obs": len(w), "n_distinct_conditions": n_cond, "n_distinct_nonclean": n_nonclean,
                            "n_seeds": w["Random Seed"].nunique(), "spearman": sp, "pearson": pe, "n_obs_delta": len(d), "spearman_delta": spd, "pearson_delta": ped,
                            "range_ID": w[ID_TEST].max() - w[ID_TEST].min()})
    return pd.DataFrame(out)


@dataclass
class Store:
    out_dir: str
    save_preds_for: set = field(default_factory=set)   # experiment names whose predictions are kept
    rows: list = field(default_factory=list)
    preds: dict = field(default_factory=dict)

    def add(self, meta: dict, test_set: str, y, p):
        m = metrics(y, p)
        self.rows.append({**meta, "Test Set": test_set, **m})
        if meta["Experiment"] in self.save_preds_for:
            key = f"{meta['Experiment']}|{meta['Condition']}|{meta['Model']}|{meta['Random Seed']}|{test_set}"
            self.preds[key + "|y"] = np.asarray(y, dtype=np.int8)
            self.preds[key + "|p"] = np.asarray(p, dtype=np.float32)

    def finalize(self, meta: dict):
        os.makedirs(self.out_dir, exist_ok=True)
        df = pd.DataFrame(self.rows)
        keys = ['Experiment', 'Condition', 'Noise Rate', 'Missingness Type', 'Missing Rate',
                'Feature Group Blanked', 'Model', 'Random Seed']
        gaps = []
        for _, g in df.groupby(keys, dropna=False):
            i, s = g[g["Test Set"] == "In-domain Test"], g[g["Test Set"] == "Shifted Test"]
            if len(i) and len(s):
                r = i.iloc[0].to_dict(); r["Test Set"] = "Generalization Gap (Shifted - In-domain)"
                for c in ["AUROC", "AUPRC", "Brier Score"]:
                    r[c] = s.iloc[0][c] - i.iloc[0][c]
                gaps.append(r)
        df = pd.concat([df, pd.DataFrame(gaps)], ignore_index=True)
        df.to_csv(os.path.join(self.out_dir, "raw_results.csv"), index=False)
        summ = df.groupby(keys[:-1] + ["Test Set"], dropna=False).agg(
            mean_AUROC=("AUROC", "mean"), std_AUROC=("AUROC", "std"),
            mean_AUPRC=("AUPRC", "mean"), std_AUPRC=("AUPRC", "std"),
            mean_Brier_Score=("Brier Score", "mean"), std_Brier_Score=("Brier Score", "std"),
            n_seeds=("AUROC", "size")).reset_index()
        summ.to_csv(os.path.join(self.out_dir, "summary.csv"), index=False)
        if self.preds:
            np.savez_compressed(os.path.join(self.out_dir, "predictions.npz"), **self.preds)
        with open(os.path.join(self.out_dir, "run_meta.json"), "w") as f:
            json.dump(environment_record(meta), f, indent=2)
        return df, summ


def base_meta(experiment, condition, model, seed, noise=0.0, mtype="None", mrate=0.0, group="None"):
    return {"Experiment": experiment, "Condition": condition, "Noise Rate": noise,
            "Missingness Type": mtype, "Missing Rate": mrate, "Feature Group Blanked": group,
            "Model": model, "Random Seed": seed}

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce every result, table and figure in the thesis from the raw data.
#
#   ./reproduce.sh                        # writes results/, thesis_project/tables/
#                                         and thesis_project/figures/
#   ./reproduce.sh results_check results_check/tables    # independent verification run
#
# Requires raw_data/diabetic_data.csv (UCI, DOI 10.24432/C5230J). The ACS files
# are downloaded by Folktables into data/ on first use.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
R="${1:-results}"
T="${2:-thesis_project/tables}"
mkdir -p "$R/logs"
say() { printf '\n=== %s ===\n' "$1"; }

say "Diabetes, patient-disjoint protocol (main)"
python3 experiments/run_diabetes.py --protocol patient \
    --out "$R/diabetes" 2>&1 | tee "$R/logs/diabetes.log"

say "Record-level split audit (overlap diagnostic for Section 3.3; no models fitted)"
python3 experiments/run_diabetes.py --protocol record --audit-only \
    --out "$R/diabetes_split_audit" 2>&1 | tee "$R/logs/diabetes_split_audit.log"

say "Diabetes sensitivity variants"
python3 experiments/run_diabetes.py --protocol patient --drop-admission-source \
    --experiments baseline,feature_shift \
    --out "$R/diabetes_no_admission_source" 2>&1 | tee "$R/logs/diabetes_no_admission_source.log"
python3 experiments/run_diabetes.py --protocol patient --ids-categorical \
    --experiments baseline,feature_shift \
    --out "$R/diabetes_ids_categorical" 2>&1 | tee "$R/logs/diabetes_ids_categorical.log"
python3 experiments/run_diabetes.py --protocol patient --no-class-weight \
    --experiments baseline \
    --out "$R/diabetes_unweighted" 2>&1 | tee "$R/logs/diabetes_unweighted.log"

say "ACSEmployment, numeric pipeline (main) and one-hot sensitivity"
python3 experiments/run_acs.py --pipeline numeric \
    --out "$R/acs" 2>&1 | tee "$R/logs/acs.log"
python3 experiments/run_acs.py --pipeline onehot \
    --experiments baseline,label_noise,feature_shift \
    --out "$R/acs_onehot" 2>&1 | tee "$R/logs/acs_onehot.log"

say "Probability quality, blanking investigation, patient-clustered bootstrap"
python3 experiments/analyze_predictions.py --run "$R/diabetes" \
    --dataset diabetes --protocol patient --bootstrap 1000 2>&1 | tee "$R/logs/analyze_diabetes.log"
python3 experiments/analyze_predictions.py --run "$R/diabetes_unweighted" \
    --dataset diabetes --protocol patient 2>&1 | tee "$R/logs/analyze_diabetes_unweighted.log"
python3 experiments/analyze_predictions.py --run "$R/acs" \
    --dataset acs 2>&1 | tee "$R/logs/analyze_acs.log"

say "Tables, macros and figures"
python3 experiments/make_tables.py --results "$R" --tables "$T" 2>&1 | tee "$R/logs/make_tables.log"
python3 experiments/make_figures.py --results "$R" --out "$(dirname "$T")/figures" 2>&1 | tee "$R/logs/make_figures.log"

say "Done: results in $R, tables in $T, figures in $(dirname "$T")/figures"

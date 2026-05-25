"""Stepwise logistic regression for rehospitalisation prediction.

Variables follow the construction table provided for the thesis:
    - Admission SBP (continuous, per 10-mmHg decrease)
    - Admission SCr (continuous, mg/dL)
    - Admission haemoglobin (continuous, g/dL)
    - ACE inhibitor at discharge       (binary, MEDS)
    - ARB at discharge                 (binary, MEDS)
    - Statin at discharge              (binary, MEDS)
    - Digoxin on admission             (binary, MEDS)
    - Diuretic on admission            (binary, MEDS)
    - Nitrates on admission            (binary, MEDS)
    - COPD                             (binary, MAIN: J44.x)
    - Prior CVA/TIA                    (binary, MAIN: I63/I64/I69/G45)
    - Prior HF admission               (binary, MAIN: I50 in any other admission of same personID)
    - Coronary angiography / PCI hist. (binary, MAIN: Z95.5)
    - ICD/pacemaker placement          (binary, MAIN: Z95.0)

Target: REHOSPITAL (TRUE if the cell is non-empty, else FALSE) following
data_manipulation.py.

Approach:
    1. Build the feature matrix from working.xlsx.
    2. Fit a *full* multivariable logistic regression (statsmodels).
    3. Backward elimination, one variable at a time, removing the predictor with
       the highest p-value as long as it is > 0.05 — record AIC/BIC/LL/df at
       every step.
    4. Likelihood ratio tests between adjacent models.
    5. 5-fold stratified cross-validation on the final reduced model
       (AUC, accuracy, Brier, log-loss).

Outputs are written to model_outputs/ as CSVs and a plain-text report.
"""

from __future__ import annotations

import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "working.xlsx"
OUT_DIR = ROOT / "model_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generic-name vocabularies (lowercased, substring-matched against Generic).
ACE_INHIBITOR_TERMS = (
    "ramipril", "perindopril", "enalapril", "lisinopril",
    "captopril", "fosinopril", "trandolapril", "quinapril",
)
ARB_TERMS = (
    "losartan", "candesartan", "valsartan", "telmisartan",
    "azilsartan", "olmesartan", "irbesartan", "eprosartan",
)
STATIN_TERMS = (
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
    "fluvastatin", "lovastatin", "pitavastatin",
)
DIGOXIN_TERMS = ("digoxin", "digoxinum")
DIURETIC_TERMS = (
    "furosemide", "torasemide", "torsemide", "bumetanide",
    "spironolactone", "eplerenone",
    "hydrochlorothiazide", "indapamide", "chlorthalidone",
)
NITRATE_TERMS = (
    "isosorbide mononitrate", "isosorbide dinitrate",
    "isosorbid  mononitrate",  # spelling seen in the data
    "nitroglycerin", "pentaerithrityl tetranitrate", "molsidomine",
)


def to_float(value) -> float:
    """Parse Georgian-locale decimals (comma as separator)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _has_any_term(series_lower: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    pattern = "|".join(re.escape(t) for t in terms)
    return series_lower.str.contains(pattern, regex=True, na=False)


def build_features() -> pd.DataFrame:
    sheets = pd.read_excel(EXCEL, sheet_name=None)
    main = sheets["MAIN"].copy()
    labs = sheets["LABS"].copy()
    meds = sheets["MEDS"].copy()

    # Normalise column names (SBP has trailing space in MAIN).
    main.columns = main.columns.astype(str).str.strip()

    # Target.
    main["REHOSPITAL"] = main["REHOSPITAL"].notna().astype(int)

    # ---- Numeric admission predictors ----
    # SBP is already integer in MAIN.
    main["SBP"] = pd.to_numeric(main["SBP"], errors="coerce")

    # SCr & Hb come from LABS — take median per historyID (multiple draws/admission).
    labs["LABANS_NUM"] = labs["LABANS"].apply(to_float)
    scr = (
        labs.loc[labs["OUR_VARIABLE_NAME"] == "SERUM_CREATININE"]
        .groupby("historyID")["LABANS_NUM"].median().rename("SCR")
    )
    hb = (
        labs.loc[labs["OUR_VARIABLE_NAME"] == "HEMOGLOBIN"]
        .groupby("historyID")["LABANS_NUM"].median().rename("HB")
    )
    main = main.merge(scr, left_on="historyID", right_index=True, how="left")
    main = main.merge(hb, left_on="historyID", right_index=True, how="left")

    # ---- Medication flags (per historyID, from MEDS) ----
    meds["Generic"] = meds["Generic"].astype(str).str.lower().str.strip()
    g = meds["Generic"]
    flags = pd.DataFrame({"historyID": meds["historyID"]})
    flags["ACE_INHIBITOR"] = _has_any_term(g, ACE_INHIBITOR_TERMS).astype(int)
    flags["ARB"]           = _has_any_term(g, ARB_TERMS).astype(int)
    flags["STATIN"]        = _has_any_term(g, STATIN_TERMS).astype(int)
    flags["DIGOXIN"]       = _has_any_term(g, DIGOXIN_TERMS).astype(int)
    flags["DIURETIC"]      = _has_any_term(g, DIURETIC_TERMS).astype(int)
    flags["NITRATES"]      = _has_any_term(g, NITRATE_TERMS).astype(int)
    med_flags = flags.groupby("historyID").max()
    main = main.merge(med_flags, left_on="historyID", right_index=True, how="left")
    for c in ["ACE_INHIBITOR", "ARB", "STATIN", "DIGOXIN", "DIURETIC", "NITRATES"]:
        main[c] = main[c].fillna(0).astype(int)

    # ---- ICD-code-based diagnostic flags (any of MAIN/COMPLICATION/FOLLOWING) ----
    diag_text = (
        main["MAIN"].fillna("").astype(str)
        + " | " + main["COMPLICATION"].fillna("").astype(str)
        + " | " + main["FOLLOWING"].fillna("").astype(str)
    )
    main["COPD"]          = diag_text.str.contains(r"J44", regex=True).astype(int)
    main["CVA_TIA"]       = diag_text.str.contains(r"I63|I64|I69|G45", regex=True).astype(int)
    main["PCI_HISTORY"]   = diag_text.str.contains(r"Z95\.5", regex=True).astype(int)
    main["ICD_PACEMAKER"] = diag_text.str.contains(r"Z95\.0", regex=True).astype(int)

    # Prior HF admission: same personID has an I50 code in ANOTHER admission row.
    main["_has_I50"] = diag_text.str.contains(r"I50", regex=True).astype(int)
    person_total_I50 = main.groupby("personID")["_has_I50"].transform("sum")
    main["PRIOR_HF_ADMISSION"] = ((person_total_I50 - main["_has_I50"]) > 0).astype(int)
    main = main.drop(columns="_has_I50")

    # Rescale SBP so a unit increase = 10 mmHg DECREASE (matches the picture).
    main["SBP_PER10_DECREASE"] = -main["SBP"] / 10.0

    keep = [
        "REHOSPITAL",
        "SBP_PER10_DECREASE", "SCR", "HB",
        "ACE_INHIBITOR", "ARB", "STATIN",
        "DIGOXIN", "DIURETIC", "NITRATES",
        "COPD", "CVA_TIA", "PRIOR_HF_ADMISSION",
        "PCI_HISTORY", "ICD_PACEMAKER",
    ]
    feat = main[keep].copy()
    # Complete-case analysis on the three continuous admission measures.
    feat = feat.dropna(subset=["SBP_PER10_DECREASE", "SCR", "HB"]).reset_index(drop=True)
    return feat


# ---------- Stepwise (backward elimination) ----------
def fit_logit(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResults:
    Xc = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, Xc).fit(disp=0, method="newton", maxiter=200)
    return model


def backward_stepwise(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
) -> tuple[list[sm.regression.linear_model.RegressionResults], list[str | None]]:
    """Iteratively remove the predictor with the largest p-value > alpha.

    Returns the sequence of fitted models and the variable dropped at each step
    (None for the initial full model).
    """
    predictors = list(X.columns)
    models: list[sm.regression.linear_model.RegressionResults] = []
    dropped: list[str | None] = []

    current = fit_logit(X[predictors], y)
    models.append(current)
    dropped.append(None)

    while True:
        pvals = current.pvalues.drop("const", errors="ignore")
        worst_p = pvals.max()
        if pd.isna(worst_p) or worst_p <= alpha:
            break
        worst_var = pvals.idxmax()
        predictors = [p for p in predictors if p != worst_var]
        if not predictors:
            break
        current = fit_logit(X[predictors], y)
        models.append(current)
        dropped.append(worst_var)
    return models, dropped


def lr_test(model_full, model_reduced) -> tuple[float, int, float]:
    """Likelihood ratio test: full nests reduced. Returns (statistic, df, p)."""
    stat = 2.0 * (model_full.llf - model_reduced.llf)
    df = int(model_reduced.df_resid - model_full.df_resid)
    p = stats.chi2.sf(stat, df) if df > 0 else np.nan
    return stat, df, p


def model_summary_row(name: str, model) -> dict:
    return {
        "model": name,
        "k_predictors": int(model.df_model),
        "log_likelihood": float(model.llf),
        "AIC": float(model.aic),
        "BIC": float(model.bic),
        "pseudo_R2_McFadden": float(model.prsquared),
    }


def coef_table(model) -> pd.DataFrame:
    params = model.params
    conf = model.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    out = pd.DataFrame({
        "coef": params,
        "std_err": model.bse,
        "z": model.tvalues,
        "p_value": model.pvalues,
        "odds_ratio": np.exp(params),
        "OR_ci_low": np.exp(conf["ci_low"]),
        "OR_ci_high": np.exp(conf["ci_high"]),
    })
    return out.reset_index().rename(columns={"index": "term"})


def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        Xtr_c = sm.add_constant(Xtr, has_constant="add")
        Xte_c = sm.add_constant(Xte, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(ytr, Xtr_c).fit(disp=0, method="newton", maxiter=200)
        prob = m.predict(Xte_c)
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "AUC": roc_auc_score(yte, prob),
            "accuracy": accuracy_score(yte, pred),
            "log_loss": log_loss(yte, prob, labels=[0, 1]),
            "brier": brier_score_loss(yte, prob),
        })
    return pd.DataFrame(rows)


def main() -> None:
    feat = build_features()
    y = feat["REHOSPITAL"].astype(int)
    X = feat.drop(columns="REHOSPITAL")

    print(f"Sample size after complete-case filter on SBP/SCr/Hb: n = {len(feat):,}")
    print(f"Rehospitalisation prevalence: {y.mean():.3%}")
    print(f"Predictors entered (full model): {list(X.columns)}\n")

    # 1) Backward stepwise.
    models, dropped = backward_stepwise(X, y, alpha=0.05)

    summary_rows = []
    coef_frames = []
    for i, (mdl, drop) in enumerate(zip(models, dropped)):
        name = "M0_full" if i == 0 else f"M{i}_drop_{drop}"
        summary_rows.append({**model_summary_row(name, mdl), "dropped_this_step": drop})
        ct = coef_table(mdl)
        ct.insert(0, "model", name)
        coef_frames.append(ct)
    summary = pd.DataFrame(summary_rows)
    coefs = pd.concat(coef_frames, ignore_index=True)

    # 2) Likelihood ratio tests between adjacent models (each step nests inside the previous).
    lr_rows = []
    for i in range(1, len(models)):
        stat, df, p = lr_test(models[i - 1], models[i])
        lr_rows.append({
            "step": i,
            "compared": f"M{i-1} vs M{i}",
            "dropped": dropped[i],
            "LR_stat": stat,
            "df": df,
            "p_value": p,
        })
    lr = pd.DataFrame(lr_rows)

    # 3) Cross-validate the chosen (final) reduced model.
    final = models[-1]
    final_predictors = [p for p in final.params.index if p != "const"]
    cv = cross_validate(X[final_predictors], y, n_splits=5, seed=42)

    # ---- Persist artefacts ----
    summary.to_csv(OUT_DIR / "stepwise_model_summary.csv", index=False)
    coefs.to_csv(OUT_DIR / "stepwise_coefficients.csv", index=False)
    lr.to_csv(OUT_DIR / "likelihood_ratio_tests.csv", index=False)
    cv.to_csv(OUT_DIR / "cv_5fold.csv", index=False)

    # ---- Plain-text report ----
    buf = io.StringIO()
    print("Stepwise (backward) logistic regression — REHOSPITAL", file=buf)
    print("=" * 70, file=buf)
    print(f"n = {len(feat):,}   events = {int(y.sum()):,} ({y.mean():.2%})\n", file=buf)
    print("Predictors entered:", ", ".join(X.columns), file=buf)
    print(f"\nFull model summary:\n{models[0].summary().as_text()}", file=buf)
    print("\n\nStep-by-step model comparison", file=buf)
    print("-" * 70, file=buf)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    print("\n\nLikelihood ratio tests (step i vs step i-1)", file=buf)
    print("-" * 70, file=buf)
    if lr.empty:
        print("No variables removed — full model already minimal at alpha=0.05.", file=buf)
    else:
        print(lr.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    print("\n\nFinal reduced model coefficients (with OR & 95% CI)", file=buf)
    print("-" * 70, file=buf)
    last = coef_frames[-1]
    print(last.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    print("\n\n5-fold stratified cross-validation (final model)", file=buf)
    print("-" * 70, file=buf)
    print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    means = cv[["AUC", "accuracy", "log_loss", "brier"]].mean()
    sds = cv[["AUC", "accuracy", "log_loss", "brier"]].std()
    print("\nMean +/- SD across folds:", file=buf)
    for m in means.index:
        print(f"  {m:<10s}: {means[m]:.4f} +/- {sds[m]:.4f}", file=buf)
    report = buf.getvalue()
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

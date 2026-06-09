"""Stepwise logistic regression for rehospitalisation prediction.

Outputs are written to model_outputs/logreg/ as CSVs and a plain-text report.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hf_ml_features import build_features, build_features_with_labs_wide

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "model_outputs" / "logreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    out = pd.DataFrame(
        {
            "coef": params,
            "std_err": model.bse,
            "z": model.tvalues,
            "p_value": model.pvalues,
            "odds_ratio": np.exp(params),
            "OR_ci_low": np.exp(conf["ci_low"]),
            "OR_ci_high": np.exp(conf["ci_high"]),
        }
    )
    return out.reset_index().rename(columns={"index": "term"})


def cross_validate_statsmodels_logit(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
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
        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "AUC": roc_auc_score(yte, prob),
                "accuracy": accuracy_score(yte, pred),
                "log_loss": log_loss(yte, prob, labels=[0, 1]),
                "brier": brier_score_loss(yte, prob),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_sklearn_logreg(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, solver="liblinear")),
        ]
    )
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "AUC": roc_auc_score(yte, prob),
                "accuracy": accuracy_score(yte, pred),
                "log_loss": log_loss(yte, prob, labels=[0, 1]),
                "brier": brier_score_loss(yte, prob),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    feat = build_features()
    y = feat["REHOSPITAL"].astype(int)
    X = feat.drop(columns="REHOSPITAL")

    print(f"Sample size after complete-case filter on SBP/SCr/Hb: n = {len(feat):,}")
    print(f"Rehospitalisation prevalence: {y.mean():.3%}")
    print(f"Predictors entered (full model): {list(X.columns)}\n")

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

    lr_rows = []
    for i in range(1, len(models)):
        stat, df, p = lr_test(models[i - 1], models[i])
        lr_rows.append(
            {
                "step": i,
                "compared": f"M{i-1} vs M{i}",
                "dropped": dropped[i],
                "LR_stat": stat,
                "df": df,
                "p_value": p,
            }
        )
    lr = pd.DataFrame(lr_rows)

    final = models[-1]
    final_predictors = [p for p in final.params.index if p != "const"]
    (OUT_DIR / "final_predictors.txt").write_text(
        "\n".join(final_predictors) + "\n", encoding="utf-8"
    )

    cv = cross_validate_statsmodels_logit(X[final_predictors], y, n_splits=5, seed=42)

    summary.to_csv(OUT_DIR / "stepwise_model_summary.csv", index=False)
    coefs.to_csv(OUT_DIR / "stepwise_coefficients.csv", index=False)
    lr.to_csv(OUT_DIR / "likelihood_ratio_tests.csv", index=False)
    cv.to_csv(OUT_DIR / "cv_5fold.csv", index=False)

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

    # Optional labs-augmented logistic regression (separate result files).
    feat_labs, lab_name_map = build_features_with_labs_wide(min_lab_non_null=200)
    y2 = feat_labs["REHOSPITAL"].astype(int)
    X2 = feat_labs.drop(columns="REHOSPITAL")
    cv2 = cross_validate_sklearn_logreg(X2, y2, n_splits=5, seed=42)

    pipe2 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, solver="liblinear")),
        ]
    )
    pipe2.fit(X2, y2)
    lr2: LogisticRegression = pipe2.named_steps["logreg"]
    coef2 = (
        pd.DataFrame(
            {
                "feature": list(X2.columns),
                "coef_scaled": lr2.coef_.ravel(),
                "odds_ratio_per_1SD": np.exp(lr2.coef_.ravel()),
            }
        )
        .sort_values("odds_ratio_per_1SD", ascending=False)
        .reset_index(drop=True)
    )

    cv2.to_csv(OUT_DIR / "labs_augmented_cv_5fold.csv", index=False)
    coef2.to_csv(OUT_DIR / "labs_augmented_coefficients.csv", index=False)
    lab_name_map.to_csv(OUT_DIR / "labs_feature_map.csv", index=False)


if __name__ == "__main__":
    main()


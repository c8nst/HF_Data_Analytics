"""LASSO (L1) and Ridge (L2) logistic regression for rehospitalisation prediction.

Uses the *same* 14 predictors and the same complete-case filter as
``stepwise_logreg.py`` (imported via :func:`build_features`) so results are
directly comparable with the unpenalised stepwise model.

Pipeline (per penalty):
    1. Standardise predictors (mean 0, std 1) inside a Pipeline so each
       cross-validation fold uses train-fold statistics only (no leakage).
    2. Tune the regularisation strength C via 5-fold CV on the *training*
       data of an outer 5-fold split (nested CV) — honest held-out metrics.
    3. Refit on the full dataset at the cross-validated C and report
       back-transformed coefficients & odds ratios on the original feature scale.

Outputs are written under model_outputs/penalized/.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stepwise_logreg import build_features

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "model_outputs" / "penalized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_GRID = np.logspace(-4, 4, 50)
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
SEED = 42


def _make_cv_estimator(penalty: str) -> LogisticRegressionCV:
    """LogisticRegressionCV with the right solver for the chosen penalty."""
    return LogisticRegressionCV(
        Cs=C_GRID,
        cv=StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED),
        penalty=penalty,
        solver="saga",
        scoring="neg_log_loss",
        max_iter=10_000,
        n_jobs=-1,
        refit=True,
        random_state=SEED,
    )


def _make_pipeline(penalty: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", _make_cv_estimator(penalty)),
    ])


def nested_cv(X: pd.DataFrame, y: pd.Series, penalty: str) -> tuple[pd.DataFrame, list[float]]:
    """Outer 5-fold CV; each training fold re-tunes C via inner 5-fold CV."""
    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    chosen_Cs: list[float] = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe = _make_pipeline(penalty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, ytr)
        clf: LogisticRegressionCV = pipe.named_steps["clf"]
        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "chosen_C": float(clf.C_[0]),
            "AUC": roc_auc_score(yte, prob),
            "accuracy": accuracy_score(yte, pred),
            "log_loss": log_loss(yte, prob, labels=[0, 1]),
            "brier": brier_score_loss(yte, prob),
        })
        chosen_Cs.append(float(clf.C_[0]))
    return pd.DataFrame(rows), chosen_Cs


def fit_final(X: pd.DataFrame, y: pd.Series, penalty: str) -> tuple[LogisticRegression, StandardScaler, float]:
    """Tune C on the full data, then refit one final model at that C."""
    pipe = _make_pipeline(penalty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    cv_clf: LogisticRegressionCV = pipe.named_steps["clf"]
    chosen_C = float(cv_clf.C_[0])
    scaler: StandardScaler = pipe.named_steps["scaler"]
    final = LogisticRegression(
        C=chosen_C,
        penalty=penalty,
        solver="saga",
        max_iter=20_000,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(scaler.transform(X), y)
    return final, scaler, chosen_C


def coefficient_table(
    clf: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    """Back-transform standardised coefficients to the original feature scale."""
    beta_std = clf.coef_.ravel()
    intercept_std = float(clf.intercept_[0])
    means = scaler.mean_
    sds = scaler.scale_

    beta_raw = beta_std / sds
    intercept_raw = intercept_std - float(np.sum(beta_std * means / sds))

    rows = [{
        "term": "const",
        "coef_standardised": intercept_std,
        "coef_raw": intercept_raw,
        "odds_ratio_raw": float(np.exp(intercept_raw)),
        "kept": True,
    }]
    for name, b_std, b_raw in zip(feature_names, beta_std, beta_raw):
        rows.append({
            "term": name,
            "coef_standardised": float(b_std),
            "coef_raw": float(b_raw),
            "odds_ratio_raw": float(np.exp(b_raw)),
            "kept": bool(abs(b_std) > 1e-12),
        })
    return pd.DataFrame(rows)


def run(penalty: str, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    print(f"\n=== {label} (penalty='{penalty}') ===")
    cv_df, chosen_Cs = nested_cv(X, y, penalty)
    final, scaler, full_C = fit_final(X, y, penalty)
    coef_df = coefficient_table(final, scaler, list(X.columns))

    cv_df.to_csv(OUT_DIR / f"cv_5fold_{label.lower()}.csv", index=False)
    coef_df.to_csv(OUT_DIR / f"coefficients_{label.lower()}.csv", index=False)

    print(f"Chosen C (full-data refit): {full_C:.6g}")
    print(f"Mean CV AUC: {cv_df['AUC'].mean():.4f}  (+/- {cv_df['AUC'].std():.4f})")
    print(f"Outer-fold C values: {[f'{c:.4g}' for c in chosen_Cs]}")
    return {
        "label": label,
        "penalty": penalty,
        "cv_df": cv_df,
        "coef_df": coef_df,
        "full_C": full_C,
        "outer_Cs": chosen_Cs,
    }


def write_report(results: list[dict], n: int, prevalence: float) -> None:
    buf = io.StringIO()
    print("Penalised logistic regression — REHOSPITAL", file=buf)
    print("=" * 70, file=buf)
    print(f"n = {n:,}   prevalence = {prevalence:.2%}", file=buf)
    print(f"Predictors (same as stepwise model): {len(results[0]['coef_df']) - 1}", file=buf)
    print(
        "Tuning: 5-fold inner CV across "
        f"{len(C_GRID)} log-spaced C values (1e-4..1e4); scoring = neg log-loss.",
        file=buf,
    )
    print("Evaluation: nested 5-fold stratified outer CV.", file=buf)

    # Per-model sections.
    for res in results:
        print("\n", file=buf)
        print(f"{res['label']}  (penalty = {res['penalty']})", file=buf)
        print("-" * 70, file=buf)
        print(f"Full-data refit C = {res['full_C']:.6g}", file=buf)
        print(f"Outer-fold chosen C values: {[round(c, 4) for c in res['outer_Cs']]}", file=buf)
        print("\n5-fold cross-validation:", file=buf)
        print(res["cv_df"].to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
        means = res["cv_df"][["AUC", "accuracy", "log_loss", "brier"]].mean()
        sds = res["cv_df"][["AUC", "accuracy", "log_loss", "brier"]].std()
        print("\nMean +/- SD across folds:", file=buf)
        for m in means.index:
            print(f"  {m:<10s}: {means[m]:.4f} +/- {sds[m]:.4f}", file=buf)
        print("\nCoefficients (standardised and back-transformed to raw scale):", file=buf)
        print(res["coef_df"].to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
        kept = res["coef_df"][res["coef_df"]["kept"] & (res["coef_df"]["term"] != "const")]
        n_kept = len(kept)
        n_total = len(res["coef_df"]) - 1
        print(f"\nNon-zero predictors: {n_kept} / {n_total}", file=buf)

    # Side-by-side AUC comparison.
    print("\n", file=buf)
    print("Side-by-side comparison (mean across 5 folds)", file=buf)
    print("-" * 70, file=buf)
    rows = []
    for res in results:
        m = res["cv_df"][["AUC", "accuracy", "log_loss", "brier"]].mean()
        s = res["cv_df"][["AUC", "accuracy", "log_loss", "brier"]].std()
        rows.append({
            "model": res["label"],
            "AUC_mean": m["AUC"], "AUC_sd": s["AUC"],
            "acc_mean": m["accuracy"], "acc_sd": s["accuracy"],
            "logloss_mean": m["log_loss"], "logloss_sd": s["log_loss"],
            "brier_mean": m["brier"], "brier_sd": s["brier"],
        })
    comp = pd.DataFrame(rows)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    comp.to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    text = buf.getvalue()
    (OUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    feat = build_features()
    y = feat["REHOSPITAL"].astype(int)
    X = feat.drop(columns="REHOSPITAL")
    print(f"n = {len(feat):,}   events = {int(y.sum()):,} ({y.mean():.2%})")
    print(f"Predictors: {list(X.columns)}")

    lasso = run("l1", X, y, label="LASSO")
    ridge = run("l2", X, y, label="Ridge")
    write_report([lasso, ridge], n=len(feat), prevalence=float(y.mean()))


if __name__ == "__main__":
    main()

"""Experimental LASSO and ridge logistic regression.

Consumes the feature list saved by ``experimental_pvalues.py`` and follows the
same output pattern as the experimental RF script:
  - model_summary.csv
  - coefficients.csv
  - cv_5fold.csv
  - selected_features.txt
  - report.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hf_experimental_features import (
    ROOT,
    build_experimental_frames,
    load_selected_feature_names,
    select_feature_frame,
)

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

OUT_DIR = ROOT / "model_outputs" / "experimental_penalized"
PVAL_DIR = ROOT / "model_outputs" / "experimental_pvalues"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PVAL_DIR.mkdir(parents=True, exist_ok=True)

FIXED_C = 1.0
N_OUTER_FOLDS = 5
SEED = 42


def cross_validate_fixed_c(
    X: pd.DataFrame,
    y: pd.Series,
    penalty: str,
    model_label: str,
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=FIXED_C,
                    penalty=penalty,
                    solver="liblinear",
                    max_iter=5_000,
                    random_state=SEED,
                ),
            ),
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
                "model": model_label,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "chosen_C": float(FIXED_C),
                "AUC": roc_auc_score(yte, prob),
                "accuracy": accuracy_score(yte, pred),
                "log_loss": log_loss(yte, prob, labels=[0, 1]),
                "brier": brier_score_loss(yte, prob),
            }
        )
    return pd.DataFrame(rows)


def fit_final(X: pd.DataFrame, y: pd.Series, penalty: str) -> tuple[LogisticRegression, StandardScaler]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=FIXED_C,
                    penalty=penalty,
                    solver="liblinear",
                    max_iter=5_000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    clf: LogisticRegression = pipe.named_steps["clf"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    return clf, scaler


def coefficient_table(
    model_label: str,
    clf: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    beta_std = clf.coef_.ravel()
    intercept_std = float(clf.intercept_[0])
    means = scaler.mean_
    sds = scaler.scale_

    beta_raw = beta_std / sds
    intercept_raw = intercept_std - float(np.sum(beta_std * means / sds))

    rows = [
        {
            "model": model_label,
            "term": "const",
            "coef_standardised": intercept_std,
            "coef_raw": intercept_raw,
            "odds_ratio_raw": float(np.exp(intercept_raw)),
            "kept": True,
        }
    ]
    for name, b_std, b_raw in zip(feature_names, beta_std, beta_raw):
        rows.append(
            {
                "model": model_label,
                "term": name,
                "coef_standardised": float(b_std),
                "coef_raw": float(b_raw),
                "odds_ratio_raw": float(np.exp(b_raw)),
                "kept": bool(abs(b_std) > 1e-12),
            }
        )
    return pd.DataFrame(rows)


def run_model(penalty: str, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    print(f"\n=== {label} (penalty='{penalty}') ===")
    cv_df = cross_validate_fixed_c(X, y, penalty, label)
    final, scaler = fit_final(X, y, penalty)
    coef_df = coefficient_table(label, final, scaler, list(X.columns))

    print(f"Chosen C (fixed): {FIXED_C:.6g}")
    print(f"Mean CV AUC: {cv_df['AUC'].mean():.4f}  (+/- {cv_df['AUC'].std():.4f})")
    return {
        "label": label,
        "penalty": penalty,
        "cv_df": cv_df,
        "coef_df": coef_df,
        "full_C": FIXED_C,
        "outer_Cs": [FIXED_C] * N_OUTER_FOLDS,
    }


def write_report(results: list[dict], n: int, prevalence: float, feature_mode: str, feature_names: list[str]) -> None:
    buf = io.StringIO()
    print("Experimental penalized logistic regression - Main_Data + LABS + ECHO", file=buf)
    print("=" * 70, file=buf)
    print(f"n = {n:,}   prevalence = {prevalence:.2%}", file=buf)
    print(f"Feature mode: {feature_mode}", file=buf)
    print(f"Predictors in model: {len(feature_names)}", file=buf)
    print(f"Regularization strength fixed at C = {FIXED_C:.4f}.", file=buf)
    print("Evaluation: 5-fold stratified outer CV.", file=buf)

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
        for metric in means.index:
            print(f"  {metric:<10s}: {means[metric]:.4f} +/- {sds[metric]:.4f}", file=buf)
        print("\nCoefficients (standardised and back-transformed to raw scale):", file=buf)
        print(res["coef_df"].to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
        kept = res["coef_df"][res["coef_df"]["kept"] & (res["coef_df"]["term"] != "const")]
        print(f"\nNon-zero predictors: {len(kept)} / {len(res['coef_df']) - 1}", file=buf)

    comp_rows = []
    for res in results:
        m = res["cv_df"][["AUC", "accuracy", "log_loss", "brier"]].mean()
        s = res["cv_df"][["AUC", "accuracy", "log_loss", "brier"]].std()
        comp_rows.append(
            {
                "model": res["label"],
                "AUC_mean": m["AUC"],
                "AUC_sd": s["AUC"],
                "acc_mean": m["accuracy"],
                "acc_sd": s["accuracy"],
                "logloss_mean": m["log_loss"],
                "logloss_sd": s["log_loss"],
                "brier_mean": m["brier"],
                "brier_sd": s["brier"],
            }
        )
    comparison = pd.DataFrame(comp_rows)
    print("\nSide-by-side comparison (mean across 5 folds)", file=buf)
    print("-" * 70, file=buf)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    comparison.to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    text = buf.getvalue()
    (OUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental penalized logistic regression.")
    parser.add_argument(
        "--use-all-features",
        action="store_true",
        help="Fit on the full imputed feature matrix instead of the screened subset.",
    )
    parser.add_argument(
        "--selected-features-file",
        type=Path,
        default=PVAL_DIR / "selected_features.csv",
        help="CSV generated by experimental_pvalues.py with a feature_name column.",
    )
    args = parser.parse_args()

    raw_frame, imputed_frame, y, _meta = build_experimental_frames()

    if args.use_all_features:
        feature_names = [c for c in imputed_frame.columns if c != "historyID"]
        feature_mode = "all imputed features"
    else:
        feature_names = load_selected_feature_names(args.selected_features_file, imputed_frame, use_all=False)
        feature_mode = "screened features"

    X = select_feature_frame(imputed_frame, feature_names).drop(columns="historyID")
    print(f"n = {len(raw_frame):,}   events = {int(y.sum()):,} ({y.mean():.2%})")
    print(f"Feature mode: {feature_mode}")
    print(f"Predictors: {list(X.columns)}")

    lasso = run_model("l1", X, y, label="LASSO")
    ridge = run_model("l2", X, y, label="Ridge")

    for res in (lasso, ridge):
        slug = res["label"].lower()
        summary = pd.DataFrame(
            [
                {
                    "model": res["label"],
                    "penalty": res["penalty"],
                    "n_features": len(feature_names),
                    "full_C": res["full_C"],
                }
            ]
        )
        summary.to_csv(OUT_DIR / f"model_summary_{slug}.csv", index=False)
        res["cv_df"].to_csv(OUT_DIR / f"cv_5fold_{slug}.csv", index=False)
        res["coef_df"].to_csv(OUT_DIR / f"coefficients_{slug}.csv", index=False)

    (OUT_DIR / "selected_features.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    (OUT_DIR / "selected_features_lasso.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    (OUT_DIR / "selected_features_ridge.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")

    write_report([lasso, ridge], n=len(raw_frame), prevalence=float(y.mean()), feature_mode=feature_mode, feature_names=feature_names)


if __name__ == "__main__":
    main()

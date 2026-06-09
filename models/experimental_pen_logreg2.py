"""Experimental LASSO and ridge logistic regression using Main_Data + LABS + ECHO."""

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
SEED = 42


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _make_pipeline(penalty: str) -> Pipeline:
    return Pipeline(
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


def cross_validate_penalized(
    X: pd.DataFrame,
    y: pd.Series,
    penalty: str,
    n_splits: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        pipe = _make_pipeline(penalty)
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


def model_summary_row(name: str, clf: LogisticRegression, penalty: str, n_features: int) -> dict:
    return {
        "model": name,
        "penalty": penalty,
        "n_features": int(n_features),
        "C": float(clf.C),
        "solver": clf.solver,
        "max_iter": int(clf.max_iter),
        "n_iter": int(clf.n_iter_[0]),
    }


def coefficient_table(
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
                "term": name,
                "coef_standardised": float(b_std),
                "coef_raw": float(b_raw),
                "odds_ratio_raw": float(np.exp(b_raw)),
                "kept": bool(abs(b_std) > 1e-12),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-model fit
# ---------------------------------------------------------------------------

def fit_model(
    name: str,
    penalty: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, coef_df, cv_df) for one penalty type."""
    cv = cross_validate_penalized(X, y, penalty, n_splits=5, seed=SEED)

    pipe = _make_pipeline(penalty)
    pipe.fit(X, y)
    clf: LogisticRegression = pipe.named_steps["clf"]
    scaler: StandardScaler = pipe.named_steps["scaler"]

    summary = pd.DataFrame([model_summary_row(name, clf, penalty, len(X.columns))])
    coef = coefficient_table(clf, scaler, list(X.columns))

    return summary, coef, cv


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    use_all: bool,
    selected: pd.DataFrame,
    results: list[dict],
) -> str:
    buf = io.StringIO()
    print("Experimental penalized logistic regression - Main_Data + LABS + ECHO", file=buf)
    print("=" * 70, file=buf)
    print(f"Feature mode: {'all imputed features' if use_all else 'screened features'}", file=buf)
    n_features = results[0]["summary"]["n_features"].iloc[0]
    print(f"Selected features: {n_features}", file=buf)
    print(f"Regularization strength fixed at C = {FIXED_C:.4f}", file=buf)
    print("Evaluation: 5-fold stratified CV", file=buf)

    print("\nSelected features", file=buf)
    print("-" * 70, file=buf)
    if selected.empty:
        print("No features passed screening.", file=buf)
    else:
        print(selected.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)

    for res in results:
        name = res["name"]
        cv = res["cv"]
        coef = res["coef"]
        summary = res["summary"]

        print(f"\nModel summary — {name}", file=buf)
        print("-" * 70, file=buf)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)

        print(f"\nCoefficients — {name}", file=buf)
        print("-" * 70, file=buf)
        print(coef.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
        kept = coef[coef["kept"] & (coef["term"] != "const")]
        print(f"\nNon-zero predictors: {len(kept)} / {len(coef) - 1}", file=buf)

        print(f"\n5-fold stratified cross-validation — {name}", file=buf)
        print("-" * 70, file=buf)
        print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
        means = cv[["AUC", "accuracy", "log_loss", "brier"]].mean()
        sds = cv[["AUC", "accuracy", "log_loss", "brier"]].std()
        print("\nMean +/- SD across folds:", file=buf)
        for metric in means.index:
            print(f"  {metric:<10s}: {means[metric]:.4f} +/- {sds[metric]:.4f}", file=buf)

    # Side-by-side comparison
    comp_rows = []
    for res in results:
        m = res["cv"][["AUC", "accuracy", "log_loss", "brier"]].mean()
        s = res["cv"][["AUC", "accuracy", "log_loss", "brier"]].std()
        comp_rows.append(
            {
                "model": res["name"],
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

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    raw_frame, imputed_frame, y, meta = build_experimental_frames()

    if args.use_all_features:
        feature_names = load_selected_feature_names(args.selected_features_file, imputed_frame, use_all=True)
        selected_for_report = pd.DataFrame(
            {
                "sheet": ["ALL"] * len(feature_names),
                "base_name": feature_names,
                "feature_name": feature_names,
                "representation": ["all"] * len(feature_names),
                "test_type": ["n/a"] * len(feature_names),
                "p_value": [np.nan] * len(feature_names),
            }
        )
    else:
        selected = pd.read_csv(args.selected_features_file) if args.selected_features_file.exists() else pd.DataFrame()
        feature_names = load_selected_feature_names(args.selected_features_file, imputed_frame, use_all=False)
        selected_for_report = selected

    X = select_feature_frame(imputed_frame, feature_names).drop(columns="historyID")

    results = []
    for name, penalty in [("LASSO", "l1"), ("Ridge", "l2")]:
        summary, coef, cv = fit_model(name, penalty, X, y)
        results.append({"name": name, "penalty": penalty, "summary": summary, "coef": coef, "cv": cv})

        slug = name.lower()
        summary.to_csv(OUT_DIR / f"model_summary_{slug}.csv", index=False)
        coef.to_csv(OUT_DIR / f"coefficients_{slug}.csv", index=False)
        cv.to_csv(OUT_DIR / f"cv_5fold_{slug}.csv", index=False)

    # Comparison CSV
    comp_rows = []
    for res in results:
        m = res["cv"][["AUC", "accuracy", "log_loss", "brier"]].mean()
        s = res["cv"][["AUC", "accuracy", "log_loss", "brier"]].std()
        comp_rows.append({"model": res["name"], "AUC_mean": m["AUC"], "AUC_sd": s["AUC"],
                          "acc_mean": m["accuracy"], "acc_sd": s["accuracy"],
                          "logloss_mean": m["log_loss"], "logloss_sd": s["log_loss"],
                          "brier_mean": m["brier"], "brier_sd": s["brier"]})
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    (OUT_DIR / "selected_features.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")

    report = build_report(args.use_all_features, selected_for_report, results)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
"""Run the full experimental HF modelling pipeline in one pass.

The pipeline is ordered as:
1. p-value screening
2. penalized logistic regression
3. standard logistic regression
4. random forest
5. gradient boosting

Each stage writes its own artifacts under ``model_outputs/``. This script then
builds a single aggregate report that summarizes the screening results and the
model comparisons.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
OUTPUT_DIR = ROOT / "model_outputs"
PVAL_DIR = OUTPUT_DIR / "experimental_pvalues"
PENALIZED_DIR = OUTPUT_DIR / "experimental_penalized"
LR_DIR = OUTPUT_DIR / "experimental_lr"
RF_DIR = OUTPUT_DIR / "experimental_rf"
GBM_DIR = OUTPUT_DIR / "experimental_gbm"
PIPELINE_DIR = OUTPUT_DIR / "experimental_pipeline"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


def run_step(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=SCRIPTS_DIR)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected output: {path}")
    return pd.read_csv(path)


def _mean_sd_row(name: str, cv: pd.DataFrame) -> dict[str, float | str]:
    stats = cv[["AUC", "accuracy", "log_loss", "brier"]]
    means = stats.mean()
    sds = stats.std()
    return {
        "model": name,
        "AUC_mean": float(means["AUC"]),
        "AUC_sd": float(sds["AUC"]),
        "accuracy_mean": float(means["accuracy"]),
        "accuracy_sd": float(sds["accuracy"]),
        "log_loss_mean": float(means["log_loss"]),
        "log_loss_sd": float(sds["log_loss"]),
        "brier_mean": float(means["brier"]),
        "brier_sd": float(sds["brier"]),
    }


def _parse_screening_report(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    out: dict[str, str] = {}
    for key, pattern in {
        "n": r"n = ([\d,]+)",
        "events": r"events = ([\d,]+)",
        "prevalence": r"events = [\d,]+ \(([^)]+)\)",
        "alpha": r"Screening alpha = ([0-9.]+)",
        "tested": r"Total tested features = ([\d,]+)",
        "selected": r"Selected features = ([\d,]+)",
    }.items():
        m = re.search(pattern, text)
        if m:
            out[key] = m.group(1)
    return out


def _top_screened_features(path: Path, n: int = 10) -> pd.DataFrame:
    df = _load_csv(path)
    cols = [c for c in ["sheet", "base_name", "feature_name", "representation", "test_type", "p_value"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df.loc[:, cols].sort_values("p_value", na_position="last").head(n)


def _comparison_rows() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, path in [
        ("experimental_lr", LR_DIR / "cv_5fold.csv"),
        ("experimental_rf", RF_DIR / "cv_5fold.csv"),
        ("experimental_gbm", GBM_DIR / "cv_5fold.csv"),
        ("penalized_LASSO", PENALIZED_DIR / "cv_5fold_lasso.csv"),
        ("penalized_Ridge", PENALIZED_DIR / "cv_5fold_ridge.csv"),
    ]:
        if path.exists():
            rows.append(_mean_sd_row(name, _load_csv(path)))
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values("AUC_mean", ascending=False).reset_index(drop=True)
    return comparison


def _selected_counts(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty or "sheet" not in selected.columns:
        return pd.DataFrame()
    return selected.groupby("sheet").size().rename("count").reset_index().sort_values("count", ascending=False)


def build_aggregate_report(alpha: float, use_all_features: bool) -> str:
    screening = _parse_screening_report(PVAL_DIR / "report.txt")
    selected = _load_csv(PVAL_DIR / "selected_features.csv") if (PVAL_DIR / "selected_features.csv").exists() else pd.DataFrame()

    comparison = _comparison_rows()
    top_features = _top_screened_features(PVAL_DIR / "all_pvalues.csv", n=10)
    selected_counts = _selected_counts(selected)

    best_model = None
    if not comparison.empty:
        best_model = comparison.iloc[0]["model"]

    lines: list[str] = []
    lines.append("Experimental HF modelling pipeline")
    lines.append("=" * 70)
    lines.append(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"P-value cutoff: {alpha:.3f}")
    lines.append(f"Feature mode: {'all imputed features' if use_all_features else 'screened features'}")
    lines.append("")
    lines.append("Screening summary")
    lines.append("-" * 70)
    if screening:
        lines.append(f"Sample size: {screening.get('n', 'n/a')}")
        lines.append(f"Events: {screening.get('events', 'n/a')} ({screening.get('prevalence', 'n/a')})")
        lines.append(f"Total tested features: {screening.get('tested', 'n/a')}")
        lines.append(f"Selected features: {screening.get('selected', 'n/a')}")
    else:
        lines.append("Screening report not available.")
    lines.append("")
    if not selected_counts.empty:
        lines.append("Selected features by sheet")
        lines.append(selected_counts.to_string(index=False))
        lines.append("")
    lines.append("Top screened features")
    lines.append("-" * 70)
    if not top_features.empty:
        lines.append(top_features.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        lines.append("No screening table available.")
    lines.append("")
    lines.append("Model comparison")
    lines.append("-" * 70)
    if not comparison.empty:
        lines.append(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lines.append("")
        lines.append(f"Best mean AUC: {best_model}")
    else:
        lines.append("No model comparison data available.")
    report = "\n".join(lines) + "\n"
    (PIPELINE_DIR / "report.txt").write_text(report, encoding="utf-8")
    if not comparison.empty:
        comparison.to_csv(PIPELINE_DIR / "model_comparison.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full experimental HF pipeline.")
    parser.add_argument("--alpha", type=float, default=0.05, help="P-value cutoff for screening.")
    parser.add_argument(
        "--use-all-features",
        action="store_true",
        help="Fit downstream models on the full imputed feature matrix instead of the screened subset.",
    )
    args = parser.parse_args()

    run_step("experimental_pvalues.py", "--alpha", str(args.alpha))

    selected_file = PVAL_DIR / "selected_features.csv"
    downstream_scripts = [
        "experimental_pen_logreg2.py",
        "experimental_logistic_regression.py",
        "experimental_random_forest.py",
        "experimental_gradient_boosting.py",
    ]
    for script in downstream_scripts:
        cmd_args = ["--selected-features-file", str(selected_file)]
        if args.use_all_features:
            cmd_args.insert(0, "--use-all-features")
        run_step(script, *cmd_args)

    report = build_aggregate_report(alpha=args.alpha, use_all_features=args.use_all_features)
    print("\n>>> Aggregate report written to:", PIPELINE_DIR / "report.txt")
    print(report)


if __name__ == "__main__":
    main()

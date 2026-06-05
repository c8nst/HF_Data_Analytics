"""Calculate experimental screening p-values once and save them to disk.

This script is the only place that runs the t-tests / chi-squared tests for the
experimental workflow. The model scripts consume the saved outputs.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

from hf_experimental_features import (
    ROOT,
    build_experimental_frames,
    screen_features,
)

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = ROOT / "model_outputs" / "experimental_pvalues"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_report(
    alpha: float,
    raw_frame: pd.DataFrame,
    y: pd.Series,
    all_pvalues: pd.DataFrame,
    selected: pd.DataFrame,
) -> str:
    buf = io.StringIO()
    print("Experimental p-value screening - Main_Data + LABS + ECHO", file=buf)
    print("=" * 70, file=buf)
    print(f"n = {len(raw_frame):,}   events = {int(y.sum()):,} ({y.mean():.2%})", file=buf)
    print(f"Screening alpha = {alpha:.3f}", file=buf)
    print(f"Total tested features = {len(all_pvalues):,}", file=buf)
    print(f"Selected features = {len(selected):,}", file=buf)
    print("\nSelected features by sheet", file=buf)
    print("-" * 70, file=buf)
    counts = selected.groupby("sheet").size().rename("count").reset_index()
    if counts.empty:
        print("No features passed screening.", file=buf)
    else:
        print(counts.to_string(index=False), file=buf)
    print("\nTop ranked features", file=buf)
    print("-" * 70, file=buf)
    top = all_pvalues.sort_values("p_value", na_position="last").head(20)
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"), file=buf)
    return buf.getvalue()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calculate and save experimental p-values.")
    parser.add_argument("--alpha", type=float, default=0.05, help="P-value cutoff for feature screening.")
    args = parser.parse_args()

    raw_frame, imputed_frame, y, meta = build_experimental_frames()
    all_pvalues, main_pvalues, labs_pvalues, echo_numeric_pvalues, echo_chi2_pvalues, selected = screen_features(
        raw_frame, y, meta, alpha=args.alpha
    )

    all_pvalues.to_csv(OUT_DIR / "all_pvalues.csv", index=False)
    main_pvalues.to_csv(OUT_DIR / "main_pvalues.csv", index=False)
    labs_pvalues.to_csv(OUT_DIR / "labs_pvalues.csv", index=False)
    echo_numeric_pvalues.to_csv(OUT_DIR / "echo_numeric_pvalues.csv", index=False)
    echo_chi2_pvalues.to_csv(OUT_DIR / "echo_chisquared_pvalues.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_features.csv", index=False)
    (OUT_DIR / "selected_features.txt").write_text(
        "\n".join(selected["feature_name"].astype(str).tolist()) + ("\n" if not selected.empty else ""),
        encoding="utf-8",
    )

    report = build_report(args.alpha, raw_frame, y, all_pvalues, selected)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

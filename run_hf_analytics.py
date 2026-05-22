"""Generate HF analytics charts from Processed_HF_Project.xlsx."""

from __future__ import annotations

import argparse
from pathlib import Path

from hf_analytics_charts import OUTPUT_DIR, build_all_charts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HF charts from processed Excel.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Path to Processed_HF_Project.xlsx (default: next to this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PNG outputs (default: output_charts/)",
    )
    args = parser.parse_args()
    paths = build_all_charts(excel_path=args.excel, out=args.out)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()

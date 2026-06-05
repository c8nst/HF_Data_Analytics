"""Experimental feature pipeline for LABS and ECHO models.

This module builds a joined feature set from ``Main_Data``, ``Labs`` and
``Echo`` in ``Processed_HF_Project.xlsx``. It also computes univariate
screening tables so the model scripts can fit either the screened subset or
the full candidate set.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROCESSED_EXCEL = ROOT / "Processed_HF_Project.xlsx"


def _sanitize_name(name: str, fallback: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def _to_float(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return np.nan
    sign = 1.0
    if text.startswith(">"):
        text = text[1:].strip()
        sign = 1.0
    elif text.startswith("<"):
        text = text[1:].strip()
        sign = -1.0
    try:
        num = float(re.findall(r"-?\d+(?:\.\d+)?", text)[0])
        if sign > 0 and str(value).strip().startswith(">"):
            return num + 1e-6
        if sign < 0 and str(value).strip().startswith("<"):
            return num - 1e-6
        return num
    except Exception:
        return np.nan


def _contains_any(series: pd.Series, patterns: tuple[str, ...]) -> pd.Series:
    pattern = "|".join(re.escape(p) for p in patterns)
    return series.str.contains(pattern, regex=True, na=False)


def _load_excel(excel_path: Path | str = PROCESSED_EXCEL) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sheets = pd.read_excel(Path(excel_path), sheet_name=None)
    return sheets["Main_Data"], sheets["Labs"], sheets["Echo"], sheets["Echo Ranges Lookup"]


def _main_features(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = main.copy()
    main.columns = main.columns.astype(str).str.strip()
    main["REHOSPITAL"] = main["REHOSPITAL"].astype(bool).astype(int)

    # Target-free feature frame.
    feat = pd.DataFrame({"historyID": main["historyID"]})

    # Numeric admission features.
    numeric_cols = ["AGE", "BMI", "HEIGHT", "ADMISSION_WEIGHT", "HEART_RATE", "SBP", "DBP", "SPO2"]
    for col in numeric_cols:
        feat[f"MAIN__{col}"] = pd.to_numeric(main[col], errors="coerce")

    # Sex as binary.
    sex = main["SEX"].astype(str).str.lower().str.strip()
    sex = sex.replace({"nan": np.nan, "none": np.nan})
    feat["MAIN__SEX_MALE"] = sex.map({"male": 1.0, "female": 0.0})

    # Diagnosis-derived flags.
    diag_text = (
        main["MAIN"].fillna("").astype(str)
        + " | "
        + main["COMPLICATION"].fillna("").astype(str)
        + " | "
        + main["FOLLOWING"].fillna("").astype(str)
    )
    feat["MAIN__COPD"] = _contains_any(diag_text, ("J44",)).astype(float)
    feat["MAIN__CVA_TIA"] = _contains_any(diag_text, ("I63", "I64", "I69", "G45")).astype(float)
    feat["MAIN__PCI_HISTORY"] = _contains_any(diag_text, ("Z95.5",)).astype(float)
    feat["MAIN__ICD_PACEMAKER"] = _contains_any(diag_text, ("Z95.0",)).astype(float)

    # Prior HF admission: same personID had an I50 code on another admission row.
    has_i50 = _contains_any(diag_text, ("I50",)).astype(int)
    person_total_i50 = pd.Series(has_i50, index=main.index).groupby(main["personID"]).transform("sum")
    feat["MAIN__PRIOR_HF_ADMISSION"] = ((person_total_i50 - has_i50) > 0).astype(float)

    target = main.loc[:, ["historyID", "REHOSPITAL"]].drop_duplicates("historyID")
    return feat, target


def _labs_features(labs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labs = labs.copy()
    labs["LABANS_NUM"] = labs["LABANS"].apply(_to_float)
    wide = (
        labs.groupby(["historyID", "OUR_VARIABLE_NAME"], dropna=False)["LABANS_NUM"]
        .median()
        .unstack("OUR_VARIABLE_NAME")
    )
    wide = wide.loc[:, wide.columns.notna()]

    original = list(wide.columns)
    new_cols = []
    seen: dict[str, int] = {}
    for i, col in enumerate(original, start=1):
        base = _sanitize_name(col, fallback=f"LAB_{i:03d}")
        name = f"LAB__{base}"
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    wide.columns = new_cols
    wide = wide.reset_index()

    meta = pd.DataFrame(
        {
            "sheet": "LABS",
            "base_name": [str(x) for x in original],
            "feature_name": new_cols,
            "representation": "numeric",
            "test_type": "ttest",
            "fill_strategy": "mean",
            "original_label": [str(x) for x in original],
        }
    )
    return wide, meta


def _echo_ranges(lookup: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup = lookup.copy()
    labels = lookup.iloc[:, 0].astype(str).str.strip()
    headers = [str(c).strip() for c in lookup.iloc[0, 1:].tolist()]

    row_map = {label: lookup.iloc[i, 1:].tolist() for i, label in enumerate(labels) if i > 0}

    def pick_row(prefix: str) -> list[object] | None:
        for label, values in row_map.items():
            if label.lower().startswith(prefix.lower()):
                return values
        return None

    male_min = pick_row("Male Norm Min")
    male_max = pick_row("Male Norm Max")
    female_min = pick_row("Female Norm Min")
    female_max = pick_row("Female Norm Max")
    if any(x is None for x in (male_min, male_max, female_min, female_max)):
        return {}

    out: dict[str, dict[str, float]] = {}
    for i, header in enumerate(headers):
        out[header] = {
            "male_min": _to_float(male_min[i]),
            "male_max": _to_float(male_max[i]),
            "female_min": _to_float(female_min[i]),
            "female_max": _to_float(female_max[i]),
        }
    return out


def _echo_features(echo: pd.DataFrame, sex_by_history: pd.Series, lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    echo = echo.copy()
    echo.columns = echo.columns.astype(str).str.strip()
    ranges = _echo_ranges(lookup)

    feat = pd.DataFrame({"historyID": echo["historyID"]})
    meta_rows: list[dict] = []

    for idx, col in enumerate([c for c in echo.columns if c not in {"historyID", "EQOREF"}], start=1):
        feature_name = f"ECHO__C{idx:02d}__{_sanitize_name(col, fallback='ECHO')}"
        raw = echo[col].apply(_to_float)
        feat[feature_name] = raw
        meta_rows.append(
            {
                "sheet": "ECHO",
                "base_name": col,
                "feature_name": feature_name,
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": col,
            }
        )

        if col not in ranges:
            continue

        bounds = ranges[col]
        sex = sex_by_history.reindex(echo["historyID"]).fillna(sex_by_history.mode().iloc[0] if not sex_by_history.mode().empty else "female")
        male_mask = sex.astype(str).str.lower().eq("male")
        female_mask = ~male_mask
        flag = pd.Series(np.nan, index=feat.index, dtype=float)

        for mask, low, high in (
            (male_mask, bounds["male_min"], bounds["male_max"]),
            (female_mask, bounds["female_min"], bounds["female_max"]),
        ):
            if np.isnan(low) or np.isnan(high):
                continue
            subset = raw.loc[mask]
            flag.loc[mask] = ((subset < low) | (subset > high)).astype(float)

        abn_name = f"ECHO_ABN__C{idx:02d}__{_sanitize_name(col, fallback='ECHO_ABN')}"
        feat[abn_name] = flag
        meta_rows.append(
            {
                "sheet": "ECHO",
                "base_name": col,
                "feature_name": abn_name,
                "representation": "abnormal",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": col,
            }
        )

    meta = pd.DataFrame(meta_rows)
    return feat, meta


def build_experimental_frames(
    excel_path: Path | str = PROCESSED_EXCEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return raw and imputed feature frames plus target and metadata."""
    main, labs, echo, lookup = _load_excel(excel_path)
    main_feat, target = _main_features(main)

    # Join LABS.
    labs_feat, labs_meta = _labs_features(labs)
    raw = target.merge(main_feat, on="historyID", how="left")
    raw = raw.merge(labs_feat, on="historyID", how="left")

    # Join ECHO and use main sex for abnormal flag screening.
    sex_map = main.copy()
    sex_map.columns = sex_map.columns.astype(str).str.strip()
    sex_series = sex_map.set_index("historyID")["SEX"].astype(str).str.lower().str.strip()
    sex_series = sex_series.replace({"nan": np.nan, "none": np.nan})

    echo_feat, echo_meta = _echo_features(echo, sex_series, lookup)
    raw = raw.merge(echo_feat, on="historyID", how="left")

    # Metadata for main features.
    main_meta = pd.DataFrame(
        [
            {
                "sheet": "MAIN",
                "base_name": "AGE",
                "feature_name": "MAIN__AGE",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "AGE",
            },
            {
                "sheet": "MAIN",
                "base_name": "BMI",
                "feature_name": "MAIN__BMI",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "BMI",
            },
            {
                "sheet": "MAIN",
                "base_name": "HEIGHT",
                "feature_name": "MAIN__HEIGHT",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "HEIGHT",
            },
            {
                "sheet": "MAIN",
                "base_name": "ADMISSION_WEIGHT",
                "feature_name": "MAIN__ADMISSION_WEIGHT",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "ADMISSION_WEIGHT",
            },
            {
                "sheet": "MAIN",
                "base_name": "HEART_RATE",
                "feature_name": "MAIN__HEART_RATE",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "HEART_RATE",
            },
            {
                "sheet": "MAIN",
                "base_name": "SBP",
                "feature_name": "MAIN__SBP",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "SBP",
            },
            {
                "sheet": "MAIN",
                "base_name": "DBP",
                "feature_name": "MAIN__DBP",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "DBP",
            },
            {
                "sheet": "MAIN",
                "base_name": "SPO2",
                "feature_name": "MAIN__SPO2",
                "representation": "numeric",
                "test_type": "ttest",
                "fill_strategy": "mean",
                "original_label": "SPO2",
            },
            {
                "sheet": "MAIN",
                "base_name": "SEX_MALE",
                "feature_name": "MAIN__SEX_MALE",
                "representation": "binary",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": "SEX",
            },
            {
                "sheet": "MAIN",
                "base_name": "COPD",
                "feature_name": "MAIN__COPD",
                "representation": "binary",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": "Derived from MAIN/COMPLICATION/FOLLOWING",
            },
            {
                "sheet": "MAIN",
                "base_name": "CVA_TIA",
                "feature_name": "MAIN__CVA_TIA",
                "representation": "binary",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": "Derived from MAIN/COMPLICATION/FOLLOWING",
            },
            {
                "sheet": "MAIN",
                "base_name": "PCI_HISTORY",
                "feature_name": "MAIN__PCI_HISTORY",
                "representation": "binary",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": "Derived from MAIN/COMPLICATION/FOLLOWING",
            },
            {
                "sheet": "MAIN",
                "base_name": "ICD_PACEMAKER",
                "feature_name": "MAIN__ICD_PACEMAKER",
                "representation": "binary",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": "Derived from MAIN/COMPLICATION/FOLLOWING",
            },
            {
                "sheet": "MAIN",
                "base_name": "PRIOR_HF_ADMISSION",
                "feature_name": "MAIN__PRIOR_HF_ADMISSION",
                "representation": "binary",
                "test_type": "chi2",
                "fill_strategy": "mode",
                "original_label": "Derived from personID history",
            },
        ]
    )

    meta = pd.concat([main_meta, labs_meta, echo_meta], ignore_index=True)
    meta = meta.loc[:, ["sheet", "base_name", "feature_name", "representation", "test_type", "fill_strategy", "original_label"]]

    # Fill missing values realistically: mean for numeric, mode for binary.
    imputed = raw.copy()
    for _, row in meta.iterrows():
        col = row["feature_name"]
        if col not in imputed.columns:
            continue
        if row["fill_strategy"] == "mean":
            imputed[col] = pd.to_numeric(imputed[col], errors="coerce")
            imputed[col] = imputed[col].fillna(imputed[col].mean())
        else:
            series = pd.to_numeric(imputed[col], errors="coerce")
            mode = series.mode(dropna=True)
            fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
            imputed[col] = series.fillna(fill_value)

    # Any remaining numeric gaps should not survive into model fitting.
    for col in imputed.columns:
        if col == "historyID":
            continue
        if imputed[col].isna().any():
            imputed[col] = pd.to_numeric(imputed[col], errors="coerce").fillna(0.0)

    y = raw["REHOSPITAL"].astype(int)
    raw = raw.drop(columns=["REHOSPITAL"])
    imputed = imputed.drop(columns=["REHOSPITAL"])
    return raw, imputed, y, meta


def _ttest_pvalue(x0: pd.Series, x1: pd.Series) -> tuple[float, int, float, float, float]:
    a = pd.to_numeric(x0, errors="coerce").dropna()
    b = pd.to_numeric(x1, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, len(a), len(b), float(a.mean()) if len(a) else np.nan, float(b.mean()) if len(b) else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            _stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        except Exception:
            p = np.nan
    return float(p), len(a), len(b), float(a.mean()), float(b.mean())


def _chi2_pvalue(flag: pd.Series, y: pd.Series) -> tuple[float, pd.DataFrame]:
    tmp = pd.DataFrame({"flag": pd.to_numeric(flag, errors="coerce"), "y": y.astype(int)})
    tmp = tmp.dropna(subset=["flag", "y"])
    tmp["flag"] = tmp["flag"].astype(int)
    table = pd.crosstab(tmp["flag"], tmp["y"])
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    if table.values.sum() == 0:
        return np.nan, table
    try:
        _chi2, p, _dof, _expected = stats.chi2_contingency(table)
    except Exception:
        p = np.nan
    return float(p), table


def screen_features(
    raw_frame: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute p-values and a selected-feature table."""
    rows = []
    for _, row in meta.iterrows():
        feature = row["feature_name"]
        if feature not in raw_frame.columns:
            continue
        series = raw_frame[feature]
        if row["test_type"] == "ttest":
            p, n0, n1, m0, m1 = _ttest_pvalue(series[y == 0], series[y == 1])
            rows.append(
                {
                    **row.to_dict(),
                    "p_value": p,
                    "n_group_0": n0,
                    "n_group_1": n1,
                    "mean_group_0": m0,
                    "mean_group_1": m1,
                }
            )
        else:
            p, table = _chi2_pvalue(series, y)
            rows.append(
                {
                    **row.to_dict(),
                    "p_value": p,
                    "n_group_0": int(table.loc[0, 0] + table.loc[1, 0]) if 0 in table.columns else 0,
                    "n_group_1": int(table.loc[0, 1] + table.loc[1, 1]) if 1 in table.columns else 0,
                    "mean_group_0": np.nan,
                    "mean_group_1": np.nan,
                }
            )

    base_columns = [
        "sheet",
        "base_name",
        "feature_name",
        "representation",
        "test_type",
        "fill_strategy",
        "original_label",
        "p_value",
        "n_group_0",
        "n_group_1",
        "mean_group_0",
        "mean_group_1",
    ]
    all_pvalues = pd.DataFrame(rows, columns=base_columns)
    all_pvalues = all_pvalues.sort_values(["sheet", "representation", "p_value"], na_position="last").reset_index(drop=True)

    main_pvalues = all_pvalues.loc[all_pvalues["sheet"] == "MAIN"].copy()
    labs_pvalues = all_pvalues.loc[all_pvalues["sheet"] == "LABS"].copy()
    echo_numeric_pvalues = all_pvalues.loc[
        (all_pvalues["sheet"] == "ECHO") & (all_pvalues["representation"] == "numeric")
    ].copy()
    echo_chi2_pvalues = all_pvalues.loc[
        (all_pvalues["sheet"] == "ECHO") & (all_pvalues["representation"] == "abnormal")
    ].copy()

    selected_rows = []
    for (sheet, base_name), grp in all_pvalues.groupby(["sheet", "base_name"], dropna=False, sort=False):
        if sheet == "ECHO":
            abn = grp.loc[grp["representation"] == "abnormal"].sort_values("p_value", na_position="last")
            num = grp.loc[grp["representation"] == "numeric"].sort_values("p_value", na_position="last")
            pick = None
            if not abn.empty and pd.notna(abn.iloc[0]["p_value"]) and abn.iloc[0]["p_value"] <= alpha:
                pick = abn.iloc[0]
            elif not num.empty and pd.notna(num.iloc[0]["p_value"]) and num.iloc[0]["p_value"] <= alpha:
                pick = num.iloc[0]
            if pick is not None:
                selected_rows.append(
                    {
                        "sheet": pick["sheet"],
                        "base_name": pick["base_name"],
                        "feature_name": pick["feature_name"],
                        "representation": pick["representation"],
                        "test_type": pick["test_type"],
                        "p_value": pick["p_value"],
                    }
                )
        else:
            pick = grp.sort_values("p_value", na_position="last").head(1)
            if not pick.empty and pd.notna(pick.iloc[0]["p_value"]) and pick.iloc[0]["p_value"] <= alpha:
                row = pick.iloc[0]
                selected_rows.append(
                    {
                        "sheet": row["sheet"],
                        "base_name": row["base_name"],
                        "feature_name": row["feature_name"],
                        "representation": row["representation"],
                        "test_type": row["test_type"],
                        "p_value": row["p_value"],
                    }
                )

    selected = pd.DataFrame(selected_rows, columns=["sheet", "base_name", "feature_name", "representation", "test_type", "p_value"])
    if not selected.empty:
        selected = selected.sort_values(["sheet", "p_value"]).reset_index(drop=True)

    return all_pvalues, main_pvalues, labs_pvalues, echo_numeric_pvalues, echo_chi2_pvalues, selected


def get_feature_columns(raw_frame: pd.DataFrame, selected: pd.DataFrame | None = None, use_all: bool = False) -> list[str]:
    if use_all or selected is None or selected.empty:
        return [c for c in raw_frame.columns if c != "historyID"]
    return [c for c in selected["feature_name"].tolist() if c in raw_frame.columns and c != "historyID"]


def select_feature_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    cols = ["historyID"] + [c for c in feature_names if c in frame.columns and c != "historyID"]
    return frame.loc[:, cols].copy()


def load_selected_feature_names(
    selected_features_path: Path | str,
    frame: pd.DataFrame,
    use_all: bool = False,
    alpha: float = 0.05,
) -> list[str]:
    """Load the screened feature names saved by the separate p-value script."""
    if use_all:
        return [c for c in frame.columns if c != "historyID"]
    path = Path(selected_features_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run experimental_pvalues.py first to generate the screened feature list, "
            "or rerun the model with --use-all-features."
        )
    selected = pd.read_csv(path)
    if "feature_name" not in selected.columns:
        raise ValueError(f"{path} must contain a 'feature_name' column.")
    if "p_value" in selected.columns:
        selected = selected.loc[pd.to_numeric(selected["p_value"], errors="coerce") <= float(alpha)].copy()
    feature_names = [c for c in selected["feature_name"].astype(str).tolist() if c in frame.columns and c != "historyID"]
    if not feature_names:
        raise ValueError(
            f"No usable feature names were found in {path}. "
            "Check that the p-value script was run against the same workbook."
        )
    return feature_names

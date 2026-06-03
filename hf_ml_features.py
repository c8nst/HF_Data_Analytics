"""Shared feature engineering for HF rehospitalisation models.

This module centralizes feature construction so different model scripts
(logistic regression, random forest, etc.) use the exact same predictors.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "working.xlsx"

# Generic-name vocabularies (lowercased, substring-matched against Generic).
ACE_INHIBITOR_TERMS = (
    "ramipril",
    "perindopril",
    "enalapril",
    "lisinopril",
    "captopril",
    "fosinopril",
    "trandolapril",
    "quinapril",
)
ARB_TERMS = (
    "losartan",
    "candesartan",
    "valsartan",
    "telmisartan",
    "azilsartan",
    "olmesartan",
    "irbesartan",
    "eprosartan",
)
STATIN_TERMS = (
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "pravastatin",
    "fluvastatin",
    "lovastatin",
    "pitavastatin",
)
DIGOXIN_TERMS = ("digoxin", "digoxinum")
DIURETIC_TERMS = (
    "furosemide",
    "torasemide",
    "torsemide",
    "bumetanide",
    "spironolactone",
    "eplerenone",
    "hydrochlorothiazide",
    "indapamide",
    "chlorthalidone",
)
NITRATE_TERMS = (
    "isosorbide mononitrate",
    "isosorbide dinitrate",
    "isosorbid  mononitrate",  # spelling seen in the data
    "nitroglycerin",
    "pentaerithrityl tetranitrate",
    "molsidomine",
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


def build_features(excel_path: Path | str = EXCEL) -> pd.DataFrame:
    """Baseline feature set used for the stepwise logistic regression.

    Note: performs complete-case analysis on SBP/SCr/Hb (as in thesis table).
    """
    sheets = pd.read_excel(Path(excel_path), sheet_name=None)
    main = sheets["MAIN"].copy()
    labs = sheets["LABS"].copy()
    meds = sheets["MEDS"].copy()

    # Normalise column names (SBP has trailing space in MAIN).
    main.columns = main.columns.astype(str).str.strip()

    # Target.
    main["REHOSPITAL"] = main["REHOSPITAL"].notna().astype(int)

    # ---- Numeric admission predictors ----
    main["SBP"] = pd.to_numeric(main["SBP"], errors="coerce")

    # SCr & Hb come from LABS — take median per historyID (multiple draws/admission).
    labs["LABANS_NUM"] = labs["LABANS"].apply(to_float)
    scr = (
        labs.loc[labs["OUR_VARIABLE_NAME"] == "SERUM_CREATININE"]
        .groupby("historyID")["LABANS_NUM"]
        .median()
        .rename("SCR")
    )
    hb = (
        labs.loc[labs["OUR_VARIABLE_NAME"] == "HEMOGLOBIN"]
        .groupby("historyID")["LABANS_NUM"]
        .median()
        .rename("HB")
    )
    main = main.merge(scr, left_on="historyID", right_index=True, how="left")
    main = main.merge(hb, left_on="historyID", right_index=True, how="left")

    # ---- Medication flags (per historyID, from MEDS) ----
    meds["Generic"] = meds["Generic"].astype(str).str.lower().str.strip()
    g = meds["Generic"]
    flags = pd.DataFrame({"historyID": meds["historyID"]})
    flags["ACE_INHIBITOR"] = _has_any_term(g, ACE_INHIBITOR_TERMS).astype(int)
    flags["ARB"] = _has_any_term(g, ARB_TERMS).astype(int)
    flags["STATIN"] = _has_any_term(g, STATIN_TERMS).astype(int)
    flags["DIGOXIN"] = _has_any_term(g, DIGOXIN_TERMS).astype(int)
    flags["DIURETIC"] = _has_any_term(g, DIURETIC_TERMS).astype(int)
    flags["NITRATES"] = _has_any_term(g, NITRATE_TERMS).astype(int)
    med_flags = flags.groupby("historyID").max()
    main = main.merge(med_flags, left_on="historyID", right_index=True, how="left")
    for c in ["ACE_INHIBITOR", "ARB", "STATIN", "DIGOXIN", "DIURETIC", "NITRATES"]:
        main[c] = main[c].fillna(0).astype(int)

    # ---- ICD-code-based diagnostic flags (any of MAIN/COMPLICATION/FOLLOWING) ----
    diag_text = (
        main["MAIN"].fillna("").astype(str)
        + " | "
        + main["COMPLICATION"].fillna("").astype(str)
        + " | "
        + main["FOLLOWING"].fillna("").astype(str)
    )
    main["COPD"] = diag_text.str.contains(r"J44", regex=True).astype(int)
    main["CVA_TIA"] = diag_text.str.contains(r"I63|I64|I69|G45", regex=True).astype(int)
    main["PCI_HISTORY"] = diag_text.str.contains(r"Z95\.5", regex=True).astype(int)
    main["ICD_PACEMAKER"] = diag_text.str.contains(r"Z95\.0", regex=True).astype(int)

    # Prior HF admission: same personID has an I50 code in ANOTHER admission row.
    main["_has_I50"] = diag_text.str.contains(r"I50", regex=True).astype(int)
    person_total_I50 = main.groupby("personID")["_has_I50"].transform("sum")
    main["PRIOR_HF_ADMISSION"] = ((person_total_I50 - main["_has_I50"]) > 0).astype(int)
    main = main.drop(columns="_has_I50")

    # Rescale SBP so a unit increase = 10 mmHg DECREASE.
    main["SBP_PER10_DECREASE"] = -main["SBP"] / 10.0

    keep = [
        "REHOSPITAL",
        "SBP_PER10_DECREASE",
        "SCR",
        "HB",
        "ACE_INHIBITOR",
        "ARB",
        "STATIN",
        "DIGOXIN",
        "DIURETIC",
        "NITRATES",
        "COPD",
        "CVA_TIA",
        "PRIOR_HF_ADMISSION",
        "PCI_HISTORY",
        "ICD_PACEMAKER",
    ]
    feat = main[keep].copy()
    feat = feat.dropna(subset=["SBP_PER10_DECREASE", "SCR", "HB"]).reset_index(drop=True)
    return feat


def _sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "UNKNOWN"


def build_features_with_labs_wide(
    excel_path: Path | str = EXCEL,
    min_lab_non_null: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrix with baseline predictors + wide lab variables.

    Wide lab features: median per historyID per OUR_VARIABLE_NAME.
    Unlike build_features(), this keeps rows with missing labs and relies on
    downstream model imputation.
    """
    sheets = pd.read_excel(Path(excel_path), sheet_name=None)
    main = sheets["MAIN"].copy()
    labs = sheets["LABS"].copy()
    meds = sheets["MEDS"].copy()

    main.columns = main.columns.astype(str).str.strip()
    main["REHOSPITAL"] = main["REHOSPITAL"].notna().astype(int)
    main["SBP"] = pd.to_numeric(main["SBP"], errors="coerce")

    labs["LABANS_NUM"] = labs["LABANS"].apply(to_float)
    labs_wide = (
        labs.groupby(["historyID", "OUR_VARIABLE_NAME"], dropna=False)["LABANS_NUM"]
        .median()
        .unstack("OUR_VARIABLE_NAME")
    )

    non_null = labs_wide.notna().sum(axis=0).sort_values(ascending=False)
    keep_lab_vars = non_null[non_null >= int(min_lab_non_null)].index.tolist()
    labs_wide = labs_wide[keep_lab_vars].copy()

    original_names = list(labs_wide.columns)
    sanitized = [_sanitize_name(c) for c in original_names]

    seen: dict[str, int] = {}
    final_cols: list[str] = []
    for s in sanitized:
        if s not in seen:
            seen[s] = 0
            final_cols.append(s)
        else:
            seen[s] += 1
            final_cols.append(f"{s}__{seen[s]}")

    labs_wide.columns = [f"LAB__{c}" for c in final_cols]

    lab_name_map = (
        pd.DataFrame(
            {
                "OUR_VARIABLE_NAME": original_names,
                "feature_name": list(labs_wide.columns),
                "non_null_historyID_count": [int(non_null.loc[n]) for n in original_names],
            }
        )
        .sort_values("non_null_historyID_count", ascending=False)
        .reset_index(drop=True)
    )

    main = main.merge(labs_wide, left_on="historyID", right_index=True, how="left")

    meds["Generic"] = meds["Generic"].astype(str).str.lower().str.strip()
    g = meds["Generic"]
    flags = pd.DataFrame({"historyID": meds["historyID"]})
    flags["ACE_INHIBITOR"] = _has_any_term(g, ACE_INHIBITOR_TERMS).astype(int)
    flags["ARB"] = _has_any_term(g, ARB_TERMS).astype(int)
    flags["STATIN"] = _has_any_term(g, STATIN_TERMS).astype(int)
    flags["DIGOXIN"] = _has_any_term(g, DIGOXIN_TERMS).astype(int)
    flags["DIURETIC"] = _has_any_term(g, DIURETIC_TERMS).astype(int)
    flags["NITRATES"] = _has_any_term(g, NITRATE_TERMS).astype(int)
    med_flags = flags.groupby("historyID").max()
    main = main.merge(med_flags, left_on="historyID", right_index=True, how="left")
    for c in ["ACE_INHIBITOR", "ARB", "STATIN", "DIGOXIN", "DIURETIC", "NITRATES"]:
        main[c] = main[c].fillna(0).astype(int)

    diag_text = (
        main["MAIN"].fillna("").astype(str)
        + " | "
        + main["COMPLICATION"].fillna("").astype(str)
        + " | "
        + main["FOLLOWING"].fillna("").astype(str)
    )
    main["COPD"] = diag_text.str.contains(r"J44", regex=True).astype(int)
    main["CVA_TIA"] = diag_text.str.contains(r"I63|I64|I69|G45", regex=True).astype(int)
    main["PCI_HISTORY"] = diag_text.str.contains(r"Z95\.5", regex=True).astype(int)
    main["ICD_PACEMAKER"] = diag_text.str.contains(r"Z95\.0", regex=True).astype(int)

    main["_has_I50"] = diag_text.str.contains(r"I50", regex=True).astype(int)
    person_total_I50 = main.groupby("personID")["_has_I50"].transform("sum")
    main["PRIOR_HF_ADMISSION"] = ((person_total_I50 - main["_has_I50"]) > 0).astype(int)
    main = main.drop(columns="_has_I50")

    main["SBP_PER10_DECREASE"] = -main["SBP"] / 10.0

    base_cols = [
        "REHOSPITAL",
        "SBP_PER10_DECREASE",
        "ACE_INHIBITOR",
        "ARB",
        "STATIN",
        "DIGOXIN",
        "DIURETIC",
        "NITRATES",
        "COPD",
        "CVA_TIA",
        "PRIOR_HF_ADMISSION",
        "PCI_HISTORY",
        "ICD_PACEMAKER",
    ]
    lab_cols = [c for c in main.columns if c.startswith("LAB__")]
    feat = main[base_cols + lab_cols].copy()
    feat = feat.dropna(subset=["SBP_PER10_DECREASE"]).reset_index(drop=True)
    return feat, lab_name_map


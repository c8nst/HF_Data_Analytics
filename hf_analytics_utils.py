"""Load and normalize data from Processed_HF_Project.xlsx for analytics."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

DEFAULT_EXCEL = Path(__file__).resolve().parent / "Processed_HF_Project.xlsx"
MAIN_SHEET = "Main_Data"
LOOKUP_SHEET = "Diagnosis_Lookup"


def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def parse_code_list(value) -> list[str]:
    """Parse MAIN/COMPLICATION/FOLLOWING cells saved as stringified Python lists."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def load_lookup(excel_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(excel_path or DEFAULT_EXCEL)
    lu = pd.read_excel(path, sheet_name=LOOKUP_SHEET)
    return _strip_column_names(lu)


def load_main_dataframe(excel_path: Path | str | None = None) -> pd.DataFrame:
    return load_sheet_dataframe(excel_path, MAIN_SHEET)


def load_sheet_dataframe(
    excel_path: Path | str | None = None,
    sheet_name: str = MAIN_SHEET,
) -> pd.DataFrame:
    path = Path(excel_path or DEFAULT_EXCEL)
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = _strip_column_names(df)
    for col in ("MAIN", "COMPLICATION", "FOLLOWING"):
        if col in df.columns:
            df[col] = df[col].apply(parse_code_list)
    return df


def label_for_code(code: str, lookup: pd.DataFrame) -> str:
    if lookup is None or lookup.empty or "Code" not in lookup.columns:
        return code
    row = lookup.loc[lookup["Code"] == code]
    if row.empty:
        return code
    desc = row.iloc[0].get("Description")
    if pd.isna(desc) or str(desc).strip() == "":
        return code
    short = str(desc).strip()
    if len(short) > 60:
        short = short[:57] + "..."
    return f"{code} — {short}"


def normalize_rehospital(series: pd.Series) -> pd.Series:
    """Boolean REHOSPITAL series (NaN -> False) for grouping and ML target EDA."""
    col = series
    if col.dtype == object:
        mapped = col.map({"True": True, "False": False, True: True, False: False})
        col = mapped.fillna(col.map(lambda x: str(x).lower() == "true"))
    return col.fillna(False).astype(bool)


def code_list_length(value) -> int:
    return len(parse_code_list(value))


def first_code(value) -> str | None:
    codes = parse_code_list(value)
    return codes[0] if codes else None


NUMERIC_FEATURE_COLS = [
    "AGE",
    "BMI",
    "HEIGHT",
    "ADMISSION_WEIGHT",
    "HEART_RATE",
    "SBP",
    "DBP",
    "SPO2",
]


def explode_codes(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per (original index, code)."""
    tmp = df[[column]].copy()
    tmp["_idx"] = tmp.index
    rows = []
    for _, r in tmp.iterrows():
        codes = r[column] if isinstance(r[column], list) else parse_code_list(r[column])
        for c in codes:
            rows.append({"_idx": r["_idx"], "code": c})
    return pd.DataFrame(rows)


def count_diagnosis_codes(
    value,
    columns: tuple[str, ...] = ("MAIN", "COMPLICATION", "FOLLOWING"),
) -> int:
    """Count ICD codes across MAIN / COMPLICATION / FOLLOWING on one row."""
    if isinstance(value, pd.Series):
        total = 0
        for col in columns:
            if col in value.index:
                codes = value[col]
                if not isinstance(codes, list):
                    codes = parse_code_list(codes)
                total += len(codes)
        return total
    return 0


def add_diagnosis_burden(df: pd.DataFrame) -> pd.Series:
    """Total diagnosis codes per admission (useful ML feature / EDA)."""
    cols = [c for c in ("MAIN", "COMPLICATION", "FOLLOWING") if c in df.columns]
    if not cols:
        return pd.Series(0, index=df.index, dtype=int)
    return df.apply(lambda row: count_diagnosis_codes(row, tuple(cols)), axis=1)

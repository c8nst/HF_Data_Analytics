"""Build charts from Processed_HF_Project.xlsx (Main_Data sheet)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hf_analytics_utils import (
    DEFAULT_EXCEL,
    explode_codes,
    label_for_code,
    load_lookup,
    load_main_dataframe,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_charts"


def _ensure_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    _ensure_output_dir(out)
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def normalize_rehospital(series: pd.Series) -> pd.Series:
    """Return boolean Series for REHOSPITAL (NaN treated as False for grouping charts)."""
    col = series
    if col.dtype == object:
        mapped = col.map({"True": True, "False": False, True: True, False: False})
        col = mapped.fillna(col.map(lambda x: str(x).lower() == "true"))
    return col.fillna(False).astype(bool)


def chart_age_histogram_kde(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    ages = pd.to_numeric(df["AGE"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(ages, kde=True, ax=ax, bins="auto", color="steelblue", edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of admissions")
    ax.set_title("Age distribution (histogram with KDE)")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out, "01_age_histogram_kde.png")


def chart_person_visit_counts(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "personID" not in df.columns:
        raise KeyError("personID column required for visit-frequency chart")
    visits = df.groupby("personID").size().value_counts().sort_index()
    x_labels = visits.index.astype(str)
    x_pos = range(len(visits))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x_pos, visits.values, color="steelblue")
    ax.set_xticks(list(x_pos), labels=x_labels)
    ax.set_xlabel("Admissions per person (same personID)")
    ax.set_ylabel("Number of people")
    ax.set_title("How often the same person appears in the dataset")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in visits.values], padding=3, fontsize=8)
    ax.margins(y=0.12)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out, "02_person_visit_frequency.png")


def chart_sex_distribution(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "SEX" not in df.columns:
        raise KeyError("SEX column required")
    counts = df["SEX"].astype(str).replace("nan", "Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("Sex distribution")
    ax.axis("equal")
    return _save(fig, out, "03_sex_distribution.png")


def _top_code_chart(
    df: pd.DataFrame,
    column: str,
    lookup: pd.DataFrame,
    title: str,
    filename: str,
    top_n: int = 15,
    horizontal: bool = True,
    out: Path = OUTPUT_DIR,
) -> Path:
    exploded = explode_codes(df, column)
    if exploded.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No data in {column}", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, filename)
    freq = exploded["code"].value_counts().head(top_n)
    labels = [label_for_code(c, lookup) for c in freq.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        ax.barh(labels[::-1], freq.values[::-1], color="slategray")
        ax.set_xlabel("Occurrences (admissions)")
        plt.setp(ax.get_yticklabels(), fontsize=8)
    else:
        xpos = range(len(freq))
        ax.bar(xpos, freq.values, color="steelblue")
        ax.set_xticks(list(xpos), labels=labels, rotation=55, ha="right")
        ax.set_ylabel("Occurrences (admissions)")
        plt.setp(ax.get_xticklabels(), fontsize=7)
    ax.set_title(title)
    ax.grid(True, axis="x" if horizontal else "y", alpha=0.3)
    return _save(fig, out, filename)


def chart_main_diagnosis(df: pd.DataFrame, lookup: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    return _top_code_chart(
        df,
        "MAIN",
        lookup,
        "Main diagnosis — top occurrences (horizontal bar)",
        "04_main_diagnosis.png",
        horizontal=True,
        out=out,
    )


def chart_following(df: pd.DataFrame, lookup: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    return _top_code_chart(
        df,
        "FOLLOWING",
        lookup,
        "Following diagnoses — top occurrences (horizontal bar)",
        "05_following_diagnoses.png",
        horizontal=True,
        out=out,
    )


def chart_complications(df: pd.DataFrame, lookup: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    return _top_code_chart(
        df,
        "COMPLICATION",
        lookup,
        "Complications — top occurrences (horizontal bar)",
        "06_complications.png",
        horizontal=True,
        out=out,
    )


def chart_outcome_pie(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "OUTCOME" not in df.columns:
        raise KeyError("OUTCOME column required")
    counts = df["OUTCOME"].astype(str).replace("nan", "Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.72,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_color("0.15")
    ax.legend(
        wedges,
        list(counts.index),
        title="Outcome",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
        borderaxespad=0.0,
    )
    ax.set_title("Outcome distribution")
    ax.axis("equal")
    fig.subplots_adjust(right=0.62)
    return _save(fig, out, "07_outcome_pie.png")


def chart_rehospital_pie(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns:
        raise KeyError("REHOSPITAL column required")
    col = normalize_rehospital(df["REHOSPITAL"])
    counts = col.value_counts()
    labels = ["Rehospitalised" if bool(idx) else "Not rehospitalised" for idx in counts.index]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["seagreen", "coral"],
        textprops={"fontsize": 10},
    )
    ax.set_title("Rehospitalisation (TRUE / FALSE)")
    ax.axis("equal")
    return _save(fig, out, "08_rehospitalisation_pie.png")


def chart_rehospitalisation_by_outcome(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    """Supplementary: rehospitalisation counts by discharge outcome."""
    if "REHOSPITAL" not in df.columns or "OUTCOME" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "REHOSPITAL and OUTCOME columns required", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "09_rehospitalisation_by_outcome.png")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["OUTCOME"] = tmp["OUTCOME"].astype(str).replace("nan", "Unknown")
    ct = pd.crosstab(tmp["OUTCOME"], tmp["_reh"])
    ct = ct.reindex(sorted(ct.index), axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ct.plot(kind="bar", stacked=False, ax=ax, color=["seagreen", "coral"])
    ax.set_ylabel("Admissions")
    ax.set_title("Rehospitalisation by outcome")
    ax.legend(["Not rehospitalised", "Rehospitalised"])
    ax.tick_params(axis="x", rotation=45)
    for container in ax.containers:
        heights = [patch.get_height() for patch in container]
        ax.bar_label(
            container,
            labels=[f"{int(h):,}" for h in heights],
            padding=2,
            fontsize=7,
        )
    ax.margins(y=0.14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "09_rehospitalisation_by_outcome.png")


def chart_age_vs_rehospital_boxplot(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns:
        raise KeyError("REHOSPITAL column required")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["AGE"] = pd.to_numeric(tmp["AGE"], errors="coerce")
    plot_df = tmp.dropna(subset=["AGE"])[["AGE", "_reh"]].copy()
    plot_df["_reh_label"] = plot_df["_reh"].map({False: "Not rehospitalised", True: "Rehospitalised"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df,
        x="_reh_label",
        y="AGE",
        hue="_reh_label",
        ax=ax,
        palette=["seagreen", "coral"],
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Age (years)")
    ax.set_title("Age vs rehospitalisation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "10_age_vs_rehospital_boxplot.png")


def chart_age_vs_rehospital_violin(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    """Violin plot: age distribution by rehospitalisation (complements boxplot)."""
    if "REHOSPITAL" not in df.columns:
        raise KeyError("REHOSPITAL column required")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["AGE"] = pd.to_numeric(tmp["AGE"], errors="coerce")
    plot_df = tmp.dropna(subset=["AGE"])[["AGE", "_reh"]].copy()
    plot_df["_reh_label"] = plot_df["_reh"].map({False: "Not rehospitalised", True: "Rehospitalised"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=plot_df,
        x="_reh_label",
        y="AGE",
        hue="_reh_label",
        ax=ax,
        palette=["seagreen", "coral"],
        legend=False,
        inner="box",
        cut=0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Age (years)")
    ax.set_title("Age vs rehospitalisation (violin)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "13_age_vs_rehospital_violin.png")


def chart_bmi_vs_rehospital_boxplot(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns or "BMI" not in df.columns:
        raise KeyError("REHOSPITAL and BMI columns required")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["BMI"] = pd.to_numeric(tmp["BMI"], errors="coerce")
    plot_df = tmp.dropna(subset=["BMI"])[["BMI", "_reh"]].copy()
    plot_df["_reh_label"] = plot_df["_reh"].map({False: "Not rehospitalised", True: "Rehospitalised"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df,
        x="_reh_label",
        y="BMI",
        hue="_reh_label",
        ax=ax,
        palette=["seagreen", "coral"],
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("BMI")
    ax.set_title("BMI vs rehospitalisation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "11_bmi_vs_rehospital_boxplot.png")


def chart_sex_vs_rehospital_stacked(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns or "SEX" not in df.columns:
        raise KeyError("REHOSPITAL and SEX columns required")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["SEX"] = tmp["SEX"].astype(str).replace("nan", "Unknown")
    ct = pd.crosstab(tmp["SEX"], tmp["_reh"])
    ct = ct.rename(columns={False: "Not rehospitalised", True: "Rehospitalised"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=["seagreen", "coral"])
    ax.set_ylabel("Number of admissions")
    ax.set_xlabel("Sex")
    ax.set_title("Sex vs rehospitalisation (stacked bar)")
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "12_sex_vs_rehospital_stacked.png")


def build_all_charts(
    excel_path: Path | str | None = None,
    out: Path | None = None,
) -> list[Path]:
    out = out or OUTPUT_DIR
    path = Path(excel_path or DEFAULT_EXCEL)
    df = load_main_dataframe(path)
    lookup = load_lookup(path)
    paths: list[Path] = []
    paths.append(chart_age_histogram_kde(df, out))
    paths.append(chart_person_visit_counts(df, out))
    paths.append(chart_sex_distribution(df, out))
    paths.append(chart_main_diagnosis(df, lookup, out))
    paths.append(chart_following(df, lookup, out))
    paths.append(chart_complications(df, lookup, out))
    paths.append(chart_outcome_pie(df, out))
    paths.append(chart_rehospital_pie(df, out))
    paths.append(chart_rehospitalisation_by_outcome(df, out))
    paths.append(chart_age_vs_rehospital_boxplot(df, out))
    paths.append(chart_age_vs_rehospital_violin(df, out))
    paths.append(chart_bmi_vs_rehospital_boxplot(df, out))
    paths.append(chart_sex_vs_rehospital_stacked(df, out))
    return paths

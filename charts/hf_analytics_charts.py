"""Build charts from Processed_HF_Project.xlsx (Main_Data sheet)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hf_analytics_utils import (
    DEFAULT_EXCEL,
    add_diagnosis_burden,
    explode_codes,
    label_for_code,
    load_lookup,
    load_sheet_dataframe,
    normalize_rehospital,
)

# Columns most relevant for upcoming ML (rehospitalisation target)
_ML_FEATURE_COLS = (
    "AGE",
    "BMI",
    "SEX",
    "OUTCOME",
    "MAIN",
    "COMPLICATION",
    "FOLLOWING",
    "REHOSPITAL",
    "personID",
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_charts"
MAIN_SHEET = "Main_Data"
ADDITIONAL_SHEETS = ("Rehospital_True", "Rehospital_False")


def _ensure_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    _ensure_output_dir(out)
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _sheet_output_dir(out: Path, sheet_name: str) -> Path:
    if sheet_name == MAIN_SHEET:
        return out
    return out / sheet_name


def _rehospital_subset(df: pd.DataFrame, rehospitalised: bool) -> pd.DataFrame:
    if "REHOSPITAL" not in df.columns:
        return df.iloc[0:0].copy()
    mask = normalize_rehospital(df["REHOSPITAL"]) == rehospitalised
    return df.loc[mask].copy()


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


def chart_outcome_bar_of_pie(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "OUTCOME" not in df.columns:
        raise KeyError("OUTCOME column required")
    counts = df["OUTCOME"].astype(str).replace("nan", "Unknown").value_counts()
    if counts.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No OUTCOME values", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "07_outcome_bar_of_pie.png")

    labels = list(counts.index)
    colors = list(sns.color_palette("tab10", n_colors=len(counts)))
    fig, (ax_pie, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.05, 1.2]},
    )
    wedges, _, autotexts = ax_pie.pie(
        counts.values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.72,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_color("0.15")
    ax_pie.set_title("Outcome share")
    ax_pie.axis("equal")

    y_pos = range(len(counts))
    bars = ax_bar.barh(y_pos, counts.values, color=colors, edgecolor="white")
    ax_bar.set_yticks(list(y_pos), labels=labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Admissions")
    ax_bar.set_title("Outcome counts")
    ax_bar.bar_label(bars, labels=[f"{int(v):,}" for v in counts.values], padding=3, fontsize=8)
    ax_bar.grid(True, axis="x", alpha=0.3)

    fig.legend(
        wedges,
        labels,
        title="Outcome",
        loc="lower center",
        ncol=min(4, len(labels)),
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Outcome distribution", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return _save(fig, out, "07_outcome_bar_of_pie.png")


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
    ct = ct.reindex(columns=[False, True], fill_value=0)
    ct = ct.reindex(sorted(ct.index), axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ct.plot(kind="bar", stacked=False, ax=ax, color=["seagreen", "coral"])
    ax.set_ylabel("Admissions")
    ax.set_title("Rehospitalisation by outcome")
    ax.legend(
        ["Not rehospitalised", "Rehospitalised"],
        title="Rehospitalisation",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
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
    fig.subplots_adjust(bottom=0.3, top=0.82)
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


def _numeric_columns(df: pd.DataFrame, min_non_null: int = 30) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= min_non_null and s.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def chart_missingness(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    """Missing-data profile for ML feature columns."""
    cols = [c for c in _ML_FEATURE_COLS if c in df.columns] or list(df.columns)
    miss_pct = (df[cols].isna().mean() * 100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(miss_pct))))
    bars = ax.barh(miss_pct.index.astype(str), miss_pct.values, color="slategray")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing values by column (ML feature check)")
    ax.set_xlim(0, 100)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in miss_pct.values], padding=3, fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "14_missingness_ml_features.png")


def chart_numeric_correlation(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    """Correlation heatmap for numeric predictors (multicollinearity check)."""
    num_cols = _numeric_columns(df)
    if len(num_cols) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Need at least two numeric columns", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "15_numeric_correlation_heatmap.png")
    numeric = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    size = max(5, 0.6 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation among numeric features")
    fig.tight_layout()
    return _save(fig, out, "15_numeric_correlation_heatmap.png")


def chart_bmi_histogram_kde(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "BMI" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "BMI column not found", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "16_bmi_histogram_kde.png")
    bmi = pd.to_numeric(df["BMI"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    if bmi.empty:
        ax.text(0.5, 0.5, "No valid BMI values", ha="center", va="center")
        ax.axis("off")
    else:
        sns.histplot(bmi, kde=True, ax=ax, bins="auto", color="teal", edgecolor="white")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Number of admissions")
        ax.set_title("BMI distribution (histogram with KDE)")
        ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out, "16_bmi_histogram_kde.png")


def chart_diagnosis_burden_vs_rehospital(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    """Total ICD codes per admission vs rehospitalisation (comorbidity burden)."""
    if "REHOSPITAL" not in df.columns:
        raise KeyError("REHOSPITAL column required")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["_code_count"] = add_diagnosis_burden(tmp)
    plot_df = tmp[["_code_count", "_reh"]].copy()
    plot_df["_reh_label"] = plot_df["_reh"].map({False: "Not rehospitalised", True: "Rehospitalised"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df,
        x="_reh_label",
        y="_code_count",
        hue="_reh_label",
        ax=ax,
        palette=["seagreen", "coral"],
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Diagnosis codes per admission")
    ax.set_title("Diagnosis burden vs rehospitalisation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "17_diagnosis_burden_vs_rehospital.png")


def chart_rehospital_rate_by_age_group(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns or "AGE" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "AGE and REHOSPITAL required", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "18_rehospital_rate_by_age_group.png")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["AGE"] = pd.to_numeric(tmp["AGE"], errors="coerce")
    tmp = tmp.dropna(subset=["AGE"])
    tmp["_age_bin"] = pd.cut(
        tmp["AGE"],
        bins=[0, 50, 60, 70, 80, 120],
        labels=["≤50", "51–60", "61–70", "71–80", "81+"],
        right=True,
    )
    rates = tmp.groupby("_age_bin", observed=True)["_reh"].mean() * 100
    counts = tmp.groupby("_age_bin", observed=True).size()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(rates.index.astype(str), rates.values, color="steelblue")
    ax.set_ylabel("Rehospitalisation rate (%)")
    ax.set_xlabel("Age group (years)")
    ax.set_title("Rehospitalisation rate by age group")
    ax.set_ylim(0, min(100, max(rates.values, default=0) * 1.25 + 5))
    ax.bar_label(
        bars,
        labels=[f"{r:.1f}%\n(n={int(counts.loc[i])})" for i, r in zip(rates.index, rates.values)],
        padding=3,
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "18_rehospital_rate_by_age_group.png")


def chart_rehospital_rate_by_sex(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns or "SEX" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "SEX and REHOSPITAL required", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "19_rehospital_rate_by_sex.png")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["SEX"] = tmp["SEX"].astype(str).replace("nan", "Unknown")
    rates = tmp.groupby("SEX")["_reh"].mean() * 100
    counts = tmp.groupby("SEX").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rates.index.astype(str), rates.values, color="steelblue")
    ax.set_ylabel("Rehospitalisation rate (%)")
    ax.set_xlabel("Sex")
    ax.set_title("Rehospitalisation rate by sex")
    ax.bar_label(
        bars,
        labels=[f"{r:.1f}%\n(n={int(counts.loc[i])})" for i, r in zip(rates.index, rates.values)],
        padding=3,
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "19_rehospital_rate_by_sex.png")


def chart_outcome_rehospital_heatmap(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    """Row-normalised crosstab: share of each outcome that rehospitalises."""
    if "REHOSPITAL" not in df.columns or "OUTCOME" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "OUTCOME and REHOSPITAL required", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "20_outcome_rehospital_heatmap.png")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["OUTCOME"] = tmp["OUTCOME"].astype(str).replace("nan", "Unknown")
    ct = pd.crosstab(tmp["OUTCOME"], tmp["_reh"], normalize="index") * 100
    if True not in ct.columns:
        ct[True] = 0.0
    plot_ct = ct.reindex(columns=[True]).rename(columns={True: "Rehospitalised (%)"})
    fig, ax = plt.subplots(figsize=(6, max(4, 0.45 * len(plot_ct))))
    sns.heatmap(
        plot_ct,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% within outcome"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Outcome")
    ax.set_title("Rehospitalisation rate within each outcome")
    fig.tight_layout()
    return _save(fig, out, "20_outcome_rehospital_heatmap.png")


def chart_age_bmi_scatter_rehospital(df: pd.DataFrame, out: Path = OUTPUT_DIR) -> Path:
    if "REHOSPITAL" not in df.columns:
        raise KeyError("REHOSPITAL column required")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    tmp["AGE"] = pd.to_numeric(tmp.get("AGE"), errors="coerce")
    tmp["BMI"] = pd.to_numeric(tmp.get("BMI"), errors="coerce")
    plot_df = tmp.dropna(subset=["AGE", "BMI"])[["AGE", "BMI", "_reh"]].copy()
    plot_df["_reh_label"] = plot_df["_reh"].map({False: "Not rehospitalised", True: "Rehospitalised"})
    fig, ax = plt.subplots(figsize=(8, 6))
    if plot_df.empty:
        ax.text(0.5, 0.5, "Need AGE and BMI with valid values", ha="center", va="center")
        ax.axis("off")
    else:
        sns.scatterplot(
            data=plot_df,
            x="AGE",
            y="BMI",
            hue="_reh_label",
            alpha=0.55,
            ax=ax,
            palette=["seagreen", "coral"],
        )
        ax.set_title("Age vs BMI coloured by rehospitalisation")
        ax.grid(True, alpha=0.3)
        ax.legend(title="", loc="best")
    fig.tight_layout()
    return _save(fig, out, "21_age_bmi_scatter_rehospital.png")


def chart_top_main_codes_rehospital_rate(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    out: Path = OUTPUT_DIR,
    top_n: int = 12,
    min_admissions: int = 25,
) -> Path:
    """Rehospitalisation rate for frequent main diagnoses (label leakage awareness for ML)."""
    if "REHOSPITAL" not in df.columns or "MAIN" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "MAIN and REHOSPITAL required", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "22_top_main_codes_rehospital_rate.png")
    tmp = df.copy()
    tmp["_reh"] = normalize_rehospital(tmp["REHOSPITAL"])
    exploded = explode_codes(tmp, "MAIN")
    if exploded.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No MAIN diagnosis codes", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out, "22_top_main_codes_rehospital_rate.png")
    merged = exploded.merge(tmp[["_reh"]], left_on="_idx", right_index=True)
    freq = merged["code"].value_counts()
    top_codes = freq[freq >= min_admissions].head(top_n).index
    subset = merged[merged["code"].isin(top_codes)]
    rates = subset.groupby("code")["_reh"].mean().sort_values(ascending=True) * 100
    labels = [label_for_code(c, lookup) for c in rates.index]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(rates))))
    ax.barh(labels, rates.values, color="indianred", alpha=0.85)
    ax.set_xlabel("Rehospitalisation rate (%)")
    ax.set_title(f"Rehospitalisation rate by main diagnosis (≥{min_admissions} admissions)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, "22_top_main_codes_rehospital_rate.png")


def build_all_charts(
    excel_path: Path | str | None = None,
    out: Path | None = None,
) -> list[Path]:
    out = out or OUTPUT_DIR
    path = Path(excel_path or DEFAULT_EXCEL)
    lookup = load_lookup(path)
    main_df = load_sheet_dataframe(path, MAIN_SHEET)
    paths: list[Path] = []
    sheet_frames: dict[str, pd.DataFrame] = {MAIN_SHEET: main_df}
    for sheet_name in ADDITIONAL_SHEETS:
        try:
            sheet_frames[sheet_name] = load_sheet_dataframe(path, sheet_name)
        except ValueError:
            sheet_frames[sheet_name] = _rehospital_subset(
                main_df,
                rehospitalised=(sheet_name == "Rehospital_True"),
            )

    for sheet_name in (MAIN_SHEET, *ADDITIONAL_SHEETS):
        df = sheet_frames[sheet_name]
        sheet_out = _sheet_output_dir(out, sheet_name)
        paths.append(chart_age_histogram_kde(df, sheet_out))
        paths.append(chart_person_visit_counts(df, sheet_out))
        paths.append(chart_sex_distribution(df, sheet_out))
        paths.append(chart_main_diagnosis(df, lookup, sheet_out))
        paths.append(chart_following(df, lookup, sheet_out))
        paths.append(chart_complications(df, lookup, sheet_out))
        paths.append(chart_outcome_bar_of_pie(df, sheet_out))
        paths.append(chart_rehospital_pie(df, sheet_out))
        paths.append(chart_rehospitalisation_by_outcome(df, sheet_out))
        paths.append(chart_age_vs_rehospital_boxplot(df, sheet_out))
        paths.append(chart_age_vs_rehospital_violin(df, sheet_out))
        paths.append(chart_bmi_vs_rehospital_boxplot(df, sheet_out))
        paths.append(chart_sex_vs_rehospital_stacked(df, sheet_out))
        paths.append(chart_missingness(df, sheet_out))
        paths.append(chart_numeric_correlation(df, sheet_out))
        paths.append(chart_bmi_histogram_kde(df, sheet_out))
        paths.append(chart_diagnosis_burden_vs_rehospital(df, sheet_out))
        paths.append(chart_rehospital_rate_by_age_group(df, sheet_out))
        paths.append(chart_rehospital_rate_by_sex(df, sheet_out))
        paths.append(chart_outcome_rehospital_heatmap(df, sheet_out))
        paths.append(chart_age_bmi_scatter_rehospital(df, sheet_out))
        paths.append(chart_top_main_codes_rehospital_rate(df, lookup, sheet_out))
    return paths

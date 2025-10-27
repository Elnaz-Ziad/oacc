# ======================================================================
# SECTION: Exploration — Baseline Characteristics (Whole Cohort, snake_case)
# PURPOSE: Summarize cohort without age stratification using snake_case cols.
# ======================================================================

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Sequence, Iterable
import matplotlib.pyplot as plt
import plotly.express as px



def _mean_sd(series: pd.Series) -> str:
    """Return 'mean (sd)' using numeric coercion; 'NA' if no numeric values."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return f"{s.mean():.2f} ({s.std(ddof=1):.1f})"
    return "NA"

def _counts_with_pct(series: pd.Series) -> List[Tuple[object, int, float]]:
    """(value, count, percent) including NaN, percent over full length."""
    counts = series.value_counts(dropna=False)
    total = len(series)
    out: List[Tuple[object, int, float]] = []
    for val, cnt in counts.items():
        pct = (cnt / total * 100.0) if total else 0.0
        out.append((val, int(cnt), pct))
    return out

def build_baseline_characteristics(
    df: pd.DataFrame,
    *,
    # snake_case columns only
    age_col: str = "age",
    categorical_cols: Sequence[Tuple[str, str]] = (
        # core
        ("gender", "Gender"),
        ("treatment_intent", "Treatment intent"),
        ("treatment_stage", "Treatment stage"),        
        ("disease_stage", "Disease stage"),
        ("mapped_disease_site", "Disease site"),
        ("mapped_ves13_score", "VES-13 bucket"),
        ("carg_toxicity_risk","CARG score"),
        # mapped domains
        ("mapped_treatment_impact", "Treatment impact"),
        ("mapped_comorbidities", "Comorbidities"),
        ("mapped_functional_status_iadls", "Functional status IADLs"),
        ("mapped_functional_status_phys", "Functional status physical"),
        ("mapped_functional_status", "Functional status"),
        ("mapped_falls_risk", "Falls risk"),
        ("mapped_medication_optimization", "Medication optimization"),
        ("mapped_social_supports", "Social supports"),
        ("mapped_nutrition", "Nutrition"),
        ("mapped_mood", "Mood"),
        ("mapped_cognition", "Cognition"),
        # enhancements
        ("enhance_trmt_delivery", "Enhancement: treatment delivery"),
        ("enhance_comorb_mngmt", "Enhancement: comorbidity management"),
        ("enhance_edu_support", "Enhancement: education support"),
        ("enhance_disease_symptoms", "Enhancement: symptoms management"),
        # ("enhance_future_trmt", "Enhancement: future treatment"),
        ("enhance_peri_op_mgmt", "Enhancement: peri-operative management"),
    ),
    
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build baseline characteristics for the whole cohort (snake_case columns).

    Returns a DataFrame with: ['Characteristics', 'Values', 'Total patients'].
    """
    rows = []

    # Age mean (SD) row (if present)
    if age_col in df.columns:
        rows.append({
            "Characteristics": "Age, mean (SD), years",
            "Values": "",
            "Total patients": _mean_sd(df[age_col]),
        })

    # Categorical summaries
    for col, label in categorical_cols:
        if col not in df.columns:
            continue
        for val, cnt, pct in _counts_with_pct(df[col]):
            rows.append({
                "Characteristics": label,
                "Values": "NaN" if pd.isna(val) else str(val),
                "Total patients": f"{cnt} ({pct:.1f})",
            })

    out = pd.DataFrame(rows, columns=["Characteristics", "Values", "Total patients"])

    if save_csv:
        out.to_csv(save_csv, index=False)

    return out


# ======================================================================
# SECTION: Exploration — Plots
# PURPOSE: Visual summaries of cohort metrics.
# ======================================================================


def plot_referrals_by_year(
    df: pd.DataFrame,
    *,
    date_col: str = "date_referred",
    mrn_col: str = "mrn",
    distinct_mrn: bool = True,      # True → count unique MRNs, False → count rows
    annotate: bool = True,          # write counts above bars
    min_year: int | None = None,    # optional year range filters
    max_year: int | None = None,
    save_path: str | None = None,   # e.g. "referrals_by_year.png"
    return_counts: bool = False,    # return counts DataFrame
):
    """
    Plot Number of Referrals by Year based on `date_referred` and counting `mrn`.
    """
    # lazy import so the module doesn't require matplotlib just to import
    import matplotlib.pyplot as plt

    if date_col not in df.columns or mrn_col not in df.columns:
        missing = [c for c in (date_col, mrn_col) if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    # derive year and (optionally) clamp to a range
    d["year"] = d[date_col].dt.year
    if min_year is not None:
        d = d[d["year"] >= int(min_year)]
    if max_year is not None:
        d = d[d["year"] <= int(max_year)]

    if distinct_mrn:
        counts = d.groupby("year")[mrn_col].nunique().rename("n_referrals").reset_index()
    else:
        counts = d.groupby("year")[mrn_col].size().rename("n_referrals").reset_index()

    # sort by year and ensure integer type
    counts = counts.sort_values("year", kind="stable")
    counts["year"] = counts["year"].astype(int)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts["year"].astype(str), counts["n_referrals"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of referrals")
    ax.set_title("Number of Referrals by Year")
    ax.yaxis.get_major_locator().set_params(integer=True)

    if annotate:
        for x, y in zip(counts["year"].astype(str), counts["n_referrals"]):
            ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if return_counts:
        return counts







def plot_stage_distribution(filtered_df):
    # Row-wise percentages
    ct_pct = pd.crosstab(
        filtered_df["disease_site"],
        filtered_df["disease_stage"],
        normalize="index"
    ) * 100

    # Sort disease sites by frequency
    order = filtered_df["disease_site"].value_counts().index
    ct_pct = ct_pct.loc[order].fillna(0)

    # Plot (fixed style)
    fig, ax = plt.subplots(figsize=(14, 10))
    palette = plt.get_cmap("tab20")
    ct_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[palette(i) for i in range(len(ct_pct.columns))],
        edgecolor="black",
        width=0.7
    )

    # Vertical % labels
    for i, site in enumerate(ct_pct.index):
        cum = 0.0
        for stage in ct_pct.columns:
            v = float(ct_pct.at[site, stage])
            if v > 0:
                ax.text(i, cum + v/2, f"{v:.0f}%", ha="center", va="center",
                        rotation=90, fontsize=12, color="black")
                cum += v
            else:
                cum += 0

    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Disease Site")
    ax.set_title("Distribution of Disease Stage by Disease Site (Percentage)")
    ax.legend(title="Disease Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()







def plot_disease_site_organ_sunburst(filtered_df, save_path=None):
    counts = (
        filtered_df.groupby(["disease_site", "disease_organ"])
        .size()
        .reset_index(name="count")
    )

    fig = px.sunburst(
        counts,
        path=["disease_site", "disease_organ"],
        values="count",
        color="disease_site"
    )

    fig.update_layout(
        title="Disease Site → Disease Organ",
        width=800,
        height=650,
        margin=dict(t=60, l=30, r=30, b=30),
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )

    fig.show()  # keep this

    if save_path is not None:
        fig.write_html(save_path)

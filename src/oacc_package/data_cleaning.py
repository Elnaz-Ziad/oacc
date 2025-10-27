# ======================================================================
# SECTION: Imports & Module Configuration
# PURPOSE: Third-party and stdlib imports used across this module.
# ======================================================================


import os
from typing import Optional, Sequence, Tuple
import pandas as pd
import pyodbc
import numpy as np

# ======================================================================
# SECTION: I/O — Access Reader
# PURPOSE: Read a table from an Access (.accdb/.mdb) database into a DataFrame.
# ======================================================================

def read_access_table(
    db_path: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    driver: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a table from a Microsoft Access database (.accdb/.mdb) into a pandas DataFrame
    using a pyodbc cursor (avoids pandas' SQLAlchemy warning).
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    # choose an Access driver
    installed = set(pyodbc.drivers())
    if driver:
        if driver not in installed:
            raise RuntimeError(
                f"Requested ODBC driver not found: {driver}. Installed drivers: {sorted(installed)}"
            )
        drv = driver
    else:
        drv = next(
            (c for c in ("Microsoft Access Driver (*.mdb, *.accdb)", "Microsoft Access Driver (*.mdb)")
             if c in installed),
            None,
        )
        if drv is None:
            raise RuntimeError(
                "No Microsoft Access ODBC driver found. Install the Access Database Engine "
                "that matches your Python bitness (32/64-bit)."
            )

    conn_str = f"DRIVER={{{drv}}};DBQ={db_path};"

    # build query (quote identifiers with [] to handle spaces/reserved words)
    select_list = "*"
    if columns:
        select_list = ", ".join(f"[{c}]" for c in columns)
    table_quoted = f"[{table}]"
    query = f"SELECT {select_list} FROM {table_quoted}"
    if where:
        query += f" WHERE {where}"

    try:
        with pyodbc.connect(conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            # column names from cursor.description
            col_names = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            # convert pyodbc.Row objects to tuples for DataFrame
            data = [tuple(r) for r in rows]
            df = pd.DataFrame.from_records(data, columns=col_names)
    except pyodbc.Error as e:
        msg = str(e)
        if "architecture mismatch" in msg.lower():
            msg += (
                "\nHint: Ensure Python, the Access ODBC driver, and Office (if installed) "
                "are all the same bitness (all 64-bit or all 32-bit)."
            )
        raise RuntimeError(f"Failed to read table '{table}': {msg}") from e

    return df



# ======================================================================
# SECTION: Notebook UI — Column Tree Widget
# PURPOSE: Interactive accordion view for exploring column_tree.
# ======================================================================

from typing import Dict, List, Union, Any, Tuple
from .constants import column_tree as DEFAULT_COLUMN_TREE


def _count_columns(node) -> int:
    if isinstance(node, list):
        return len(node)
    if isinstance(node, dict):
        return sum(_count_columns(v) for v in node.values())
    return 0


def _count_immediate_subcats(node) -> int:
    return len(node) if isinstance(node, dict) else 0


def _list_to_html(cols: List[str]):
    # lazy import so the module doesn't require ipywidgets just to import
    from ipywidgets import HTML
    import html as _html
    items = "\n".join(f"<li><code>{_html.escape(c)}</code></li>" for c in cols)
    return HTML(
        f"""
        <div style="font-family: ui-sans-serif, system-ui; font-size:14px">
          <ul style="margin:6px 0 6px 18px">{items}</ul>
        </div>
        """
    )


def _build_node_widget(key: str, node) -> Tuple["object", str]:
    # lazy import
    from ipywidgets import Accordion, VBox

    if isinstance(node, list):
        title = f"{key} [{len(node)}]"
        return VBox([_list_to_html(node)]), title

    # dict → build nested accordions
    children_widgets = []
    titles = []
    for subk, subv in node.items():
        w, t = _build_node_widget(subk, subv)
        children_widgets.append(w)
        titles.append(t)

    acc = Accordion(children=children_widgets)
    for i, t in enumerate(titles):
        acc.set_title(i, t)
    acc.selected_index = None  # collapsed by default

    title = f"{key} ({_count_immediate_subcats(node)}) [{_count_columns(node)}]"
    # return container so we can stack header + accordion later if needed
    from ipywidgets import VBox
    return VBox([acc]), title


def make_column_tree_widget(
    tree: Dict[str, Union[List[str], Dict[str, Any]]] | None = None
):
    """
    Return an ipywidgets widget (Accordion) that lets you expand/collapse
    your column categories and subcategories. Leaf nodes show the column list.
    """
    try:
        from ipywidgets import Accordion, VBox, HTML  # noqa: F401
    except Exception as e:
        raise ImportError(
            "ipywidgets is required for make_column_tree_widget(). "
            "Install it with: pip install ipywidgets"
        ) from e

    if tree is None:
        tree = DEFAULT_COLUMN_TREE

    # Top-level accordion
    tops = []
    titles = []
    for k, v in tree.items():
        w, t = _build_node_widget(k, v)
        tops.append(w)
        titles.append(t)

    from ipywidgets import Accordion, VBox, HTML
    acc = Accordion(children=tops)
    for i, t in enumerate(titles):
        acc.set_title(i, t)
    acc.selected_index = None

    header = HTML(
        """
        <div style="font-family: ui-sans-serif, system-ui; font-size:14px; margin:6px 0 10px;">
          <strong>Legend:</strong> <em>(n)</em>= immediate subcategories,
          <em>[m]</em>= total leaf columns under the category
        </div>
        """
    )
    return VBox([header, acc])


def display_column_tree(tree: Dict[str, Union[List[str], Dict[str, Any]]] | None = None):
    """
    Convenience wrapper: build and display the widget in one call.
    """
    try:
        from IPython.display import display
    except Exception as e:
        raise RuntimeError("This function should be used inside a Jupyter notebook.") from e
    display(make_column_tree_widget(tree))



# preprocessing.py

def display_column_tree_(tree: dict, indent: int = 0):
    """
    Recursively print a structured view of a nested column tree.

    Parameters
    ----------
    tree : dict
        Nested dictionary/lists of columns.
    indent : int, default=0
        Indentation level (used internally for recursion).
    """
    spacer = "    " * indent
    for key, value in tree.items():
        print(f"{spacer}- {key}:")
        if isinstance(value, dict):  # nested dict → recurse
            display_column_tree_(value, indent + 1)
        elif isinstance(value, list):  # list of columns
            for col in value:
                print(f"{spacer}    • {col}")
        else:
            print(f"{spacer}    (unexpected type: {type(value)})")


# ======================================================================
# SECTION: Mapping — Value Normalization
# PURPOSE: Create mapped_<col> columns for domains, recs, other cols, VES-13.
# ======================================================================

from .constants import (
    domains_map,     # your dict of per-domain mappings
    recs_map,        # single dict applied to all rec_* columns
    other_cols_map,  # dict: {column_name: {orig: mapped, ...}}
    column_tree,     # to derive the rec columns
)

def map_oacc_values(
    df: pd.DataFrame,
    *,
    prefix: str = "mapped_",
    ves13_sources: tuple[str, ...] = ("ves13_score", "VES13Score"),
    ves13_threshold: float = 3,
) -> pd.DataFrame:
    """
    Create mapped_<col> columns for:
      - VES13 (numeric bucket: <threshold vs >=threshold; non-numeric left as-is)
      - Domain columns (from `domains_map`)
      - Recommendation columns (all rec cols from `column_tree`, using `recs_map`)
      - Other columns (from `other_cols_map`)

    Leaves the original columns unchanged. Returns the same DataFrame for chaining.
    """

    # --- 1) VES-13 bucketing ---
    low_label = f"<{ves13_threshold}"
    high_label = f">={ves13_threshold}"
    for col in ves13_sources:
        if col in df.columns:
            s_num = pd.to_numeric(df[col], errors="coerce")
            mapped = pd.Series(
                np.where(s_num >= ves13_threshold, high_label, low_label),
                index=df.index,
            )
            # keep original cell where non-numeric/NaN
            mapped = mapped.where(~s_num.isna(), df[col])
            df[f"{prefix}{col}"] = mapped

    # --- 2) Domain columns ---
    for col, mapping in domains_map.items():
        if col in df.columns:
            df[f"{prefix}{col}"] = df[col].replace(mapping)

    # --- 3) Recommendation columns (derived from column_tree) ---
    rec_cols = column_tree.get("geriatric_assessment", {}).get("recommendations", [])
    for col in rec_cols:
        if col in df.columns:
            df[f"{prefix}{col}"] = df[col].replace(recs_map)

    # --- 4) Other columns (e.g., treatment_impact) ---
    for col, mapping in other_cols_map.items():
        if col in df.columns:
            df[f"{prefix}{col}"] = df[col].replace(mapping)

    return df


# ======================================================================
# SECTION: Cleaning & Derived Variables
# PURPOSE: One-off rules (hematologic stage) and derived flags (functional status).
# ======================================================================

def update_hematologic_stage(
    df: pd.DataFrame,
    *,
    hematologic_sites: list[str] | None = None,
    site_cols: tuple[str, ...] = ("disease_site", "DiseaseSite"),
    stage_cols: tuple[str, ...] = ("disease_stage", "DiseaseStage"),
    stage_value: str = "Hematologic",
) -> pd.DataFrame:
    """
    Set DiseaseStage/disease_stage to 'Hematologic' where DiseaseSite/disease_site
    is in the hematologic_sites list. Returns the same DataFrame for chaining.
    """
    if hematologic_sites is None:
        hematologic_sites = ["Leukemia", "Lymphoma", "Myeloma"]

    # pick the first existing site/stage column name
    site_col = next((c for c in site_cols if c in df.columns), None)
    stage_col = next((c for c in stage_cols if c in df.columns), None)
    if site_col is None or stage_col is None:
        return df  # quietly skip if columns aren't present

    mask = df[site_col].isin(hematologic_sites)
    df.loc[mask, stage_col] = stage_value
    return df


def make_functional_status(
    df: pd.DataFrame,
    *,
    iadls_col: str = "mapped_functional_status_iadls",
    phys_col: str = "mapped_functional_status_phys",
    out_col: str = "functional_status",
    mapped_col: str = "mapped_functional_status",
) -> pd.DataFrame:
    """
    Set functional_status and mapped_functional_status using:
      'abnormal' if functional_status_iadls == 'abnormal' OR
                   functional_status_phys  == 'abnormal'
      else 'normal'.

    Both output columns receive identical values.
    """
    missing = [c for c in (iadls_col, phys_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    iadls_abn = df[iadls_col].astype(str).str.lower().eq("abnormal")
    phys_abn  = df[phys_col].astype(str).str.lower().eq("abnormal")
    result = np.where(iadls_abn | phys_abn, "abnormal", "normal")

    df[out_col] = result
    df[mapped_col] = result
    return df



# ======================================================================
# SECTION: Cohort Filtering — OACC Inclusion/Exclusion Pipeline
# PURPOSE: Apply sequential filters and print remaining counts after each.
# ======================================================================


def filter_oacc_cohort(
    df: pd.DataFrame,
    *,
    start_date: str = "2015-07-01",
    end_date: str = "2024-07-01",
    min_age: int = 65,
    appt_start: str = "2015-07-01",
    appt_end: str = "2024-07-01",
    incomplete_ids: tuple[int, ...] = (734, 357),
    required_patient_status: str = "New patient",
    allowed_treatment_stages: tuple[str, ...] = ("Pre-treatment", "Pre-treatment new modality"),
    allowed_treatment_impact: tuple[str, ...] = ("treatment unchanged", "treatment changed"),
    
    # ---- NEW simple switches ----
    apply_date_referred: bool = True,
    apply_patient_status: bool = True,
    apply_age_filter: bool = True,
    apply_first_appt_filter: bool = True,
    apply_incomplete_filter: bool = True,
    apply_treatment_stage_filter: bool = True,
    apply_treatment_impact_filter: bool = True,
    apply_carg_risk_filter: bool = True,
    
    verbose: bool = True,
):
    required_cols = [
        "date_referred",
        "patient_referral_status",
        "age",
        "1st_appointment_date",
        "id",
        "treatment_stage",
        "treatment_impact",
        "mapped_treatment_impact",
        "carg_toxicity_risk",
    ]

    d = df.copy()
    d["date_referred"] = pd.to_datetime(d["date_referred"], errors="coerce")
    d["1st_appointment_date"] = pd.to_datetime(d["1st_appointment_date"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    ap_start = pd.to_datetime(appt_start)
    ap_end = pd.to_datetime(appt_end)

    counts = {}
    counts["total"] = len(d)

    # 1) date_referred
    if apply_date_referred:
        d = d[(d["date_referred"] >= start) & (d["date_referred"] < end)]
    counts["after_date_referred"] = len(d)

    # 2) patient_referral_status
    if apply_patient_status:
        d = d[d["patient_referral_status"] == required_patient_status]
    counts["after_patient_referral_status"] = len(d)

    # 3) age
    if apply_age_filter:
        d = d[d["age"] >= min_age]
    counts["after_age"] = len(d)

    # 4) 1st_appointment_date
    if apply_first_appt_filter:
        d = d[(d["1st_appointment_date"] >= ap_start) & (d["1st_appointment_date"] < ap_end)]
    counts["after_1st_appointment_date"] = len(d)

    # 5) incomplete assessments
    if apply_incomplete_filter:
        d = d[~d["id"].isin(incomplete_ids)]
    counts["after_assessment"] = len(d)

    # 6) treatment_stage
    if apply_treatment_stage_filter:
        d = d[d["treatment_stage"].isin(allowed_treatment_stages)]
    counts["after_treatment_stage"] = len(d)

    # 7) treatment_impact
    impact_col = "mapped_treatment_impact" if "mapped_treatment_impact" in d.columns else "treatment_impact"
    if apply_treatment_impact_filter:
        d = d[d[impact_col].isin(allowed_treatment_impact)]
    counts["after_treatment_impact"] = len(d)

    # 8) carg risk
    if apply_carg_risk_filter:
        d = d[d["carg_toxicity_risk"].isin(["Low", "Moderate", "High"])]
    counts["after_carg_toxicity_risk_na"] = len(d)

    if verbose:
        print("Total records in the OACC database:\n", counts["total"])

        if apply_date_referred:
            print("Records after filtering date_referred:\n", counts["after_date_referred"])

        if apply_patient_status:
            print("Records after excluding patients with a status other than 'New patient':\n",
                  counts["after_patient_referral_status"])

        if apply_age_filter:
            print("Records after excluding occasional patients under 65:\n", counts["after_age"])

        if apply_first_appt_filter:
            print("Records after filtering 1st_appointment_date:\n", counts["after_1st_appointment_date"])

        if apply_incomplete_filter:
            print("Records after excluding patients with incomplete assessments:\n", counts["after_assessment"])

        if apply_treatment_stage_filter:
            print("Records after excluding patients who were not referred for pre-treatment advice:\n",
                  counts["after_treatment_stage"])

        if apply_treatment_impact_filter:
            print("Records after excluding patients for whom the treatment impact was not applicable or who experienced unique circumstances:\n",
                  counts["after_treatment_impact"])

        if apply_carg_risk_filter:
            print("Records after excluding patients for whom CARG score is not NA:\n",
                  counts["after_carg_toxicity_risk_na"])

    return d, counts


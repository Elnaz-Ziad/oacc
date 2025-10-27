from typing import Dict, List
import difflib

# ======================================================================
# SECTION: Info Topics (Markdown)
# PURPOSE: Short, reusable explanations shown in notebooks.
# ======================================================================

INFO_TOPICS: Dict[str, str] = {
    "disease_stages": """\
**Disease Stages — quick definitions**

- **Metastatic** — Cancer has spread to distant organs or distant lymph nodes.
- **Locally advanced** — Grown into nearby tissues and/or regional nodes; no distant spread.
- **Localized** — Confined to the organ/site of origin; no regional nodes or distant spread.
- **Hematologic** — Blood/lymph malignancies (e.g., leukemia, lymphoma, myeloma) where solid-tumor staging doesn’t apply.
- **Biochemical relapse** — Rising disease marker (e.g., PSA) suggesting recurrence without visible/radiographic disease.
- **In remission** — No clinical evidence of active disease after treatment.
- **No disease** — No current evidence of disease (NED); disease-free at present.

""",
    "referral_criteria": """\
**Referral Criteria**

- Age ≥ 65 **AND**
- “Active” cancer diagnosis **AND at least one of the following:**
  - Perceived increased vulnerability to adverse effects of cancer treatment  
  - Impaired functional status  
  - Impaired mobility (e.g., falls)  
  - Cognitive impairment  
  - Polypharmacy  
  - Multiple comorbidities  

""",
}

# ======================================================================
# SECTION: API
# PURPOSE: Retrieve and display topics (with fuzzy suggestions).
# ======================================================================

def list_info_topics() -> List[str]:
    """Return available topic keys."""
    return sorted(INFO_TOPICS.keys())

def get_info(topic: str) -> str:
    """Return the Markdown string for a topic key (case-insensitive)."""
    key = topic.strip().lower()
    if key in INFO_TOPICS:
        return INFO_TOPICS[key]
    # fuzzy suggestions
    choices = list(INFO_TOPICS.keys())
    suggestions = difflib.get_close_matches(key, choices, n=3, cutoff=0.4)
    msg = f"Topic '{topic}' not found."
    if suggestions:
        msg += f" Did you mean: {', '.join(suggestions)}?"
    raise KeyError(msg)

def display_info(topic: str) -> None:
    """Render a topic's Markdown in Jupyter; prints suggestions if not found."""
    try:
        from IPython.display import display, Markdown  # lazy import
    except Exception:
        print(get_info(topic))
        return
    try:
        display(Markdown(get_info(topic)))
    except KeyError as e:
        display(Markdown(f"**{e}**\n\nAvailable topics:\n\n- " + "\n- ".join(list_info_topics())))

__all__ = ["INFO_TOPICS", "list_info_topics", "get_info", "display_info"]

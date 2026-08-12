"""Reproducible staged selection beyond the six quick-example lakes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .quick_examples import DEFAULT_CANDIDATE_PATH, QUICK_EXAMPLE_SETS
from .quick_sampling import QuickSamplingError, load_candidate_inventory

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    REPOSITORY_ROOT / "data" / "regional" / "expanded_lake_selection.csv"
)
DEFAULT_NOTES_PATH = (
    REPOSITORY_ROOT / "data" / "regional" / "expanded_lake_selection_notes.md"
)
HYDROLAKES_ASSET = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
INVENTORY_GENERATION_DATE = "2026-08-10"
INITIAL_MIN_AREA_KM2 = 25.0

CURRENT_IDS = QUICK_EXAMPLE_SETS["saline6"]
TIER1_ADDITIONS = (780, 769, 789, 774, 787, 788)
TIER2_ADDITIONS = (792, 777, 730, 760, 719, 8478, 809, 831, 9422, 9244)
TIER1_IDS = (*CURRENT_IDS, *TIER1_ADDITIONS)
TIER2_IDS = (*TIER1_IDS, *TIER2_ADDITIONS)

TYPE_LABELS = {
    1: "lake/default",
    2: "reservoir",
    3: "regulated natural lake",
}

SELECTION_REASONS: Mapping[int, str] = {
    **{
        lake_id: "Existing six-lake saline/terminal quick-transfer benchmark."
        for lake_id in CURRENT_IDS
    },
    780: "USGS IWAA priority saline lake; transboundary Great Basin contrast.",
    769: "USGS IWAA priority terminal-lake ecosystem; shallow wetland-rich setting.",
    789: "USGS IWAA priority saline lake; episodic California terminal basin.",
    774: "USGS IWAA priority saline lake; managed shallow wetland/lake complex.",
    787: "USGS IWAA priority saline lake; deeper perennial saline contrast.",
    788: (
        "Perennial regulated freshwater contrast in a watershed draining toward "
        "Great Salt Lake."
    ),
    792: "Deep, clear regulated natural lake contrast spanning CA/NV.",
    777: "Large shallow eutrophic natural/regulated lake contrast in Oregon.",
    730: "Large northern Rocky Mountain lake/reservoir-coded contrast in Montana.",
    760: "High-elevation natural lake contrast in Wyoming.",
    719: "Large northern Idaho lake/reservoir-coded contrast.",
    8478: "Urban low-elevation natural lake contrast in Washington.",
    809: "Large arid Colorado River reservoir contrast in Nevada/Arizona.",
    831: "Arid Rio Grande reservoir contrast in New Mexico.",
    9422: "Arid central-Arizona reservoir contrast.",
    9244: "High-elevation Colorado reservoir contrast; completes state coverage.",
}

DEFERRED: Mapping[int, tuple[str, str]] = {
    783: (
        "Bonneville salt-flat/playa",
        "Salt-flat feature rather than a persistent open-water experimental unit.",
    ),
    771: (
        "Harney Basin linked system",
        (
            "Hydrologically linked to Malheur in wet periods and often dry; avoid "
            "pseudoreplication until treated as one system."
        ),
    ),
    793: (
        "Sevier playa",
        (
            "Priority terminal system, but its static HydroLAKES footprint is "
            "largely dry playa in recent windows; defer pending two-class "
            "availability check."
        ),
    ),
    782: (
        "Bonneville/GSL fragment",
        (
            "Unnamed same-elevation western-Utah feature; identity and "
            "independence from Great Salt/Bonneville need review."
        ),
    ),
    785: (
        "Bonneville/GSL fragment",
        (
            "Unnamed same-elevation western-Utah feature; identity and "
            "independence from Great Salt/Bonneville need review."
        ),
    ),
    9173: (
        "Bonneville/GSL fragment",
        (
            "Unnamed same-elevation western-Utah feature; identity and "
            "independence from Great Salt/Bonneville need review."
        ),
    ),
    9179: (
        "Bonneville/GSL fragment",
        (
            "Unnamed same-elevation western-Utah feature; identity and "
            "independence from Great Salt/Bonneville need review."
        ),
    ),
    802: (
        "Lake Powell split features",
        (
            "HydroLAKES represents the reservoir with another same-elevation "
            "feature; defer until the full system is handled as one unit."
        ),
    ),
    799: (
        "Lake Powell split features",
        (
            "Unnamed same-elevation polygon appears to be part of Lake Powell; "
            "do not weight it as an independent lake."
        ),
    ),
    9118: (
        "Warner linked-lake chain",
        "Adjacent connected Warner-system polygons should not be independent units.",
    ),
    9103: (
        "Warner linked-lake chain",
        "Unnamed adjacent Warner-system polygon; identity/independence unresolved.",
    ),
    9121: (
        "Warner linked-lake chain",
        "Unnamed adjacent Warner-system polygon; identity/independence unresolved.",
    ),
}


class SelectionError(ValueError):
    """Raised when the candidate inventory cannot support the staged manifest."""


def selected_ids(tier: str) -> tuple[int, ...]:
    """Return the cumulative stable lake order for a requested tier."""

    normalized = str(tier).strip().lower()
    if normalized == "tier1":
        return TIER1_IDS
    if normalized == "tier2":
        return TIER2_IDS
    raise SelectionError("tier must be 'tier1' or 'tier2'.")


def build_selection_manifest(inventory: pd.DataFrame) -> pd.DataFrame:
    """Annotate every candidate with its staged selection or deferral status."""

    required = {"Hylak_id", "Lake_name", "state", "Lake_area", "Lake_type"}
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise SelectionError("Candidate inventory is missing: " + ", ".join(missing))
    if inventory["Hylak_id"].duplicated().any():
        raise SelectionError("Candidate inventory contains duplicate Hylak_id values.")
    available = set(inventory["Hylak_id"].astype(int))
    requested = set(TIER2_IDS) | set(DEFERRED)
    absent = sorted(requested.difference(available))
    if absent:
        raise SelectionError(f"Manifest IDs absent from inventory: {absent}.")

    rank = {lake_id: order for order, lake_id in enumerate(TIER2_IDS, start=1)}
    frame = inventory.copy()
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        lake_id = int(row.Hylak_id)
        if lake_id in CURRENT_IDS:
            tier = "current6"
            include = True
        elif lake_id in TIER1_ADDITIONS:
            tier = "tier1"
            include = True
        elif lake_id in TIER2_ADDITIONS:
            tier = "tier2"
            include = True
        else:
            tier = "deferred"
            include = False
        deferred = DEFERRED.get(lake_id)
        reason = SELECTION_REASONS.get(lake_id, "")
        exclude_reason = (
            deferred[1]
            if deferred is not None
            else (
                "Not prioritized in the staged 22-lake design; context and "
                "independence require later review."
                if not include
                else ""
            )
        )
        records.append(
            {
                "Hylak_id": lake_id,
                "Lake_name": row.Lake_name,
                "state": row.state,
                "Lake_area": float(row.Lake_area),
                "Lake_type": int(row.Lake_type),
                "lake_type_label": TYPE_LABELS.get(
                    int(row.Lake_type), f"unknown ({int(row.Lake_type)})"
                ),
                "tier": tier,
                "include": include,
                "selection_order": rank.get(lake_id, pd.NA),
                "analysis_role": (
                    "existing benchmark"
                    if lake_id in CURRENT_IDS
                    else "coherent expansion"
                    if lake_id in TIER1_ADDITIONS
                    else "diversity expansion"
                    if lake_id in TIER2_ADDITIONS
                    else "not selected"
                ),
                "system_group": deferred[0] if deferred is not None else "",
                "selection_reason": reason,
                "exclude_reason": exclude_reason,
                "notes": (
                    "State uses the deterministic HydroLAKES centroid rule."
                    if lake_id in {780, 792, 809}
                    else ""
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["_sort"] = result["selection_order"].fillna(1_000_000)
    return (
        result.sort_values(
            ["include", "_sort", "Lake_area", "Hylak_id"],
            ascending=[False, True, False, True],
            kind="mergesort",
        )
        .drop(columns="_sort")
        .reset_index(drop=True)
    )


def validate_selection_manifest(manifest: pd.DataFrame) -> None:
    """Validate exact tiers, unique IDs, and nonempty scientific rationales."""

    required = {
        "Hylak_id",
        "Lake_name",
        "state",
        "Lake_area",
        "Lake_type",
        "tier",
        "include",
        "selection_order",
        "selection_reason",
        "exclude_reason",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise SelectionError("Manifest is missing: " + ", ".join(missing))
    if manifest["Hylak_id"].isna().any() or manifest["Hylak_id"].duplicated().any():
        raise SelectionError("Manifest Hylak_id values must be unique and complete.")
    selected = manifest[manifest["include"].astype(bool)]
    if set(selected["Hylak_id"].astype(int)) != set(TIER2_IDS):
        raise SelectionError("Manifest selected IDs do not match the staged design.")
    for tier, ids in (
        ("current6", CURRENT_IDS),
        ("tier1", TIER1_ADDITIONS),
        ("tier2", TIER2_ADDITIONS),
    ):
        actual = tuple(
            selected[selected["tier"] == tier]
            .sort_values("selection_order")["Hylak_id"]
            .astype(int)
        )
        if actual != ids:
            raise SelectionError(f"{tier} order/IDs do not match the design.")
    if selected["selection_reason"].fillna("").str.strip().eq("").any():
        raise SelectionError("Every selected lake requires a selection reason.")
    excluded = manifest[~manifest["include"].astype(bool)]
    if excluded["exclude_reason"].fillna("").str.strip().eq("").any():
        raise SelectionError("Every excluded candidate requires an exclusion reason.")


def selection_notes(manifest: pd.DataFrame) -> str:
    """Document tier intent, HydroLAKES caveats, and authoritative sources."""

    tier1 = manifest[manifest["Hylak_id"].isin(TIER1_IDS)]
    tier2 = manifest[manifest["Hylak_id"].isin(TIER2_IDS)]
    return f"""# Expanded western-lake selection notes

This is a staged exploratory design, not a claim of regional representativeness.
Tier 1 contains {len(tier1)} lakes: the original six plus six additional systems
listed by the U.S. Geological Survey Saline Lake Ecosystems IWAA and Utah Lake as
a perennial regulated freshwater terminal-basin contrast. Tier 2 contains
{len(tier2)} lakes and broadens the design to all 11 configured western states,
deeper natural lakes, urban lakes, high-elevation systems, and arid reservoirs.
Area is used for screening context, not as the sole selection rule.

## Provenance

- HydroLAKES Earth Engine asset: `{HYDROLAKES_ASSET}`
- Candidate inventory generated: {INVENTORY_GENERATION_DATE}
- Initial experimental candidate screen: `Lake_area >= {INITIAL_MIN_AREA_KM2:g} km2`
- Regional membership: HydroLAKES polygon centroid inside one of the 11
  configured contiguous western states

The 25 km2 value is an experimental screening threshold for this Landsat
transfer study, not a scientific minimum lake size. Tier 1 deliberately retains
the original six saline/terminal examples, adds five USGS-priority saline or
terminal systems, and adds Utah Lake as a regulated freshwater terminal-basin
contrast. It is neither random nor statistically representative.

In the manifest, cumulative Tier 1 is defined by `selection_order <= 12`: rows
labeled `current6` are the retained benchmark and rows labeled `tier1` are its
six additions. `include = True` also records the authored Tier 2 plan, but Tier 2
has not been sampled or analyzed.

HydroLAKES `Lake_type` is retained as metadata. In the v1.0 technical
documentation, 1 denotes lake/default, 2 reservoir, and 3 regulated natural
lake; type 1 is also the default where regulation information is unavailable,
so it is not treated as definitive field-verified lake history.

Ambiguous playas, unnamed same-system fragments, and connected lake chains are
explicitly deferred. In particular, Bonneville Salt Flat is not treated as an
open-water lake; Lake Powell's split polygons are not independent units; and
Harney/Malheur plus the Warner chain are not independently weighted without a
system-level treatment. Sevier remains scientifically relevant but is deferred
until recent two-class sample availability inside its static polygon is checked.

Sources:

- USGS Saline Lake Ecosystems IWAA priority lakes:
  https://www.usgs.gov/data/saline-lake-ecosystems-iwaa-lakes
- USGS Great Basin terminal-lake science strategy:
  https://pubs.usgs.gov/publication/cir1516
- Utah Division of Water Resources, Great Salt Lake watershed councils:
  https://water.utah.gov/watershed-councils/great-salt-lake/
- HydroLAKES v1.0 technical documentation:
  https://data.hydrosheds.org/file/technical-documentation/HydroLAKES_TechDoc_v10.pdf

The `state` field remains the deterministic state containing each HydroLAKES
polygon centroid. Goose Lake, Lake Tahoe, and Lake Mead cross state boundaries,
so the single abbreviation is membership metadata rather than full extent.
"""


def write_manifest(
    inventory_path: str | Path = DEFAULT_CANDIDATE_PATH,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    notes_path: str | Path = DEFAULT_NOTES_PATH,
) -> pd.DataFrame:
    """Build, validate, and write the ignored local selection manifest."""

    inventory = load_candidate_inventory(inventory_path)
    manifest = build_selection_manifest(inventory)
    validate_selection_manifest(manifest)
    output = Path(output_path)
    notes = Path(notes_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    notes.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    notes.write_text(selection_notes(manifest), encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the reproducible Tier 1/Tier 2 western-lake manifest."
    )
    parser.add_argument(
        "--candidate-csv", type=Path, default=REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = write_manifest(args.candidate_csv, args.output, args.notes)
        selected = manifest[manifest["include"].astype(bool)]
        print(
            selected[
                [
                    "selection_order",
                    "tier",
                    "Hylak_id",
                    "Lake_name",
                    "state",
                    "Lake_area",
                    "Lake_type",
                    "selection_reason",
                ]
            ].to_string(index=False)
        )
        print(f"Wrote {args.output} and {args.notes}")
        return 0
    except (OSError, QuickSamplingError, SelectionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

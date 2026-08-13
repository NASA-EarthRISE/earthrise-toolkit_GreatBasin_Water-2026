"""Normalized elevation--area--volume tables and validation diagnostics."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from .units import canonical_unit, convert_area, convert_length, convert_volume


class CapacityTableError(ValueError):
    """Raised when an official capacity table fails a required validation gate."""


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapacityTableError(f"{label} must be documented.")
    return value.strip()


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise CapacityTableError(f"{label} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CapacityTableError(f"{label} must be finite.") from exc
    if not math.isfinite(result):
        raise CapacityTableError(f"{label} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class CapacityTableMetadata:
    """Identity and units for one authoritative capacity-table release."""

    lake_name: str
    source_doi: str
    source_title: str
    source_file: str
    source_version: str
    source_hash: str
    vertical_datum: str
    elevation_unit: str
    area_unit: str
    storage_unit: str
    parser_version: str = "hydrometry-capacity-table-v1"

    def __post_init__(self) -> None:
        for name in (
            "lake_name",
            "source_doi",
            "source_title",
            "source_file",
            "source_version",
            "vertical_datum",
            "parser_version",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        source_hash = _required_text(self.source_hash, "source_hash").casefold()
        if not _SHA256_PATTERN.fullmatch(source_hash):
            raise CapacityTableError("source_hash must be a 64-character SHA-256.")
        object.__setattr__(self, "source_hash", source_hash)
        for name in ("elevation_unit", "area_unit", "storage_unit"):
            try:
                object.__setattr__(self, name, canonical_unit(getattr(self, name)))
            except ValueError as exc:
                raise CapacityTableError(str(exc)) from exc
        # These calls enforce dimensional compatibility, not just known aliases.
        try:
            convert_length(0.0, self.elevation_unit, "m")
            convert_area(0.0, self.area_unit, "m2")
            convert_volume(0.0, self.storage_unit, "m3")
        except ValueError as exc:
            raise CapacityTableError(str(exc)) from exc

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CapacityTableInputRow:
    """One source row before unit normalization."""

    source_row: int
    elevation: float
    area: float
    storage: float


@dataclass(frozen=True, slots=True)
class NormalizedCapacityRow:
    """One source row with original and normalized values retained."""

    source_row: int
    elevation_original: float
    elevation_original_unit: str
    elevation_m_navd88: float
    area_original: float
    area_original_unit: str
    area_m2: float
    area_km2: float
    area_acres: float
    storage_original: float
    storage_original_unit: str
    storage_m3: float
    storage_acre_ft: float
    source_doi: str
    source_file: str
    source_version: str
    vertical_datum: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CapacityTableValidation:
    """Compact validation record suitable for the Phase 1 CSV output."""

    row_count: int
    elevation_min: float
    elevation_max: float
    area_min: float
    area_max: float
    storage_min: float
    storage_max: float
    elevation_step_min: float
    elevation_step_median: float
    elevation_step_max: float
    monotonic_elevation: bool
    monotonic_area: bool
    monotonic_storage: bool
    duplicate_elevation_count: int
    missing_value_count: int
    source_hash: str
    validation_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AreaVolumeSegmentDiagnostic:
    """Area--volume consistency diagnostic for one adjacent row pair."""

    lower_source_row: int
    upper_source_row: int
    elevation_increment_m: float
    observed_volume_increment_m3: float
    trapezoidal_volume_increment_m3: float
    absolute_discrepancy_m3: float
    relative_discrepancy: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AreaVolumeConsistencySummary:
    """Summary of adjacent-row trapezoidal consistency diagnostics."""

    segment_count: int
    median_absolute_discrepancy_m3: float
    maximum_absolute_discrepancy_m3: float
    median_relative_discrepancy: float
    maximum_relative_discrepancy: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizedCapacityTable:
    """Validated, immutable capacity table used by the interpolation engine."""

    metadata: CapacityTableMetadata
    rows: tuple[NormalizedCapacityRow, ...]
    validation: CapacityTableValidation

    @property
    def elevation_min_m(self) -> float:
        return self.rows[0].elevation_m_navd88

    @property
    def elevation_max_m(self) -> float:
        return self.rows[-1].elevation_m_navd88

    def normalized_records(self) -> list[dict[str, Any]]:
        """Return detached dictionaries in the required normalized schema."""

        return [row.as_dict() for row in self.rows]

    def to_dataframe(self) -> Any:
        """Return the normalized rows as a pandas DataFrame.

        Pandas is imported lazily so the scientific validation model remains
        usable without importing the project's tabular-output stack.
        """

        import pandas as pd

        return pd.DataFrame.from_records(self.normalized_records())

    def area_volume_diagnostics(
        self,
    ) -> tuple[tuple[AreaVolumeSegmentDiagnostic, ...], AreaVolumeConsistencySummary]:
        diagnostics: list[AreaVolumeSegmentDiagnostic] = []
        for lower, upper in pairwise(self.rows):
            height = upper.elevation_m_navd88 - lower.elevation_m_navd88
            observed = upper.storage_m3 - lower.storage_m3
            trapezoidal = 0.5 * (lower.area_m2 + upper.area_m2) * height
            absolute = abs(observed - trapezoidal)
            if observed == 0.0:
                relative = 0.0 if absolute == 0.0 else math.inf
            else:
                relative = absolute / abs(observed)
            diagnostics.append(
                AreaVolumeSegmentDiagnostic(
                    lower_source_row=lower.source_row,
                    upper_source_row=upper.source_row,
                    elevation_increment_m=height,
                    observed_volume_increment_m3=observed,
                    trapezoidal_volume_increment_m3=trapezoidal,
                    absolute_discrepancy_m3=absolute,
                    relative_discrepancy=relative,
                )
            )
        absolute_values = [item.absolute_discrepancy_m3 for item in diagnostics]
        relative_values = [item.relative_discrepancy for item in diagnostics]
        summary = AreaVolumeConsistencySummary(
            segment_count=len(diagnostics),
            median_absolute_discrepancy_m3=statistics.median(absolute_values),
            maximum_absolute_discrepancy_m3=max(absolute_values),
            median_relative_discrepancy=statistics.median(relative_values),
            maximum_relative_discrepancy=max(relative_values),
        )
        return tuple(diagnostics), summary


def normalize_capacity_table(
    source_rows: Iterable[CapacityTableInputRow],
    metadata: CapacityTableMetadata,
    *,
    expected_lake_name: str | None = None,
) -> NormalizedCapacityTable:
    """Validate and normalize an official E-A-V table without reordering rows."""

    if expected_lake_name is not None and (
        metadata.lake_name.casefold()
        != _required_text(expected_lake_name, "expected_lake_name").casefold()
    ):
        raise CapacityTableError(
            f"Capacity table is for {metadata.lake_name!r}, not {expected_lake_name!r}."
        )

    raw = tuple(source_rows)
    if len(raw) < 2:
        raise CapacityTableError("A capacity table requires at least two rows.")
    source_row_ids: set[int] = set()
    normalized: list[NormalizedCapacityRow] = []
    for index, row in enumerate(raw):
        if not isinstance(row, CapacityTableInputRow):
            raise CapacityTableError(
                f"Row {index} must be a CapacityTableInputRow instance."
            )
        if isinstance(row.source_row, bool) or not isinstance(row.source_row, int):
            raise CapacityTableError("source_row must be an integer.")
        if row.source_row in source_row_ids:
            raise CapacityTableError(f"Duplicate source_row {row.source_row}.")
        source_row_ids.add(row.source_row)
        elevation = _finite(row.elevation, f"row {row.source_row} elevation")
        area = _finite(row.area, f"row {row.source_row} area")
        storage = _finite(row.storage, f"row {row.source_row} storage")
        if area < 0 or storage < 0:
            raise CapacityTableError(
                f"Row {row.source_row} has unexplained negative area or storage."
            )
        normalized.append(
            NormalizedCapacityRow(
                source_row=row.source_row,
                elevation_original=elevation,
                elevation_original_unit=metadata.elevation_unit,
                elevation_m_navd88=convert_length(
                    elevation, metadata.elevation_unit, "m"
                ),
                area_original=area,
                area_original_unit=metadata.area_unit,
                area_m2=convert_area(area, metadata.area_unit, "m2"),
                area_km2=convert_area(area, metadata.area_unit, "km2"),
                area_acres=convert_area(area, metadata.area_unit, "acre"),
                storage_original=storage,
                storage_original_unit=metadata.storage_unit,
                storage_m3=convert_volume(storage, metadata.storage_unit, "m3"),
                storage_acre_ft=convert_volume(
                    storage, metadata.storage_unit, "acre_ft"
                ),
                source_doi=metadata.source_doi,
                source_file=metadata.source_file,
                source_version=metadata.source_version,
                vertical_datum=metadata.vertical_datum,
            )
        )

    elevations = [row.elevation_m_navd88 for row in normalized]
    areas = [row.area_m2 for row in normalized]
    storages = [row.storage_m3 for row in normalized]
    duplicate_count = len(elevations) - len(set(elevations))
    monotonic_elevation = all(right > left for left, right in pairwise(elevations))
    monotonic_area = all(right >= left for left, right in pairwise(areas))
    monotonic_storage = all(right >= left for left, right in pairwise(storages))
    if duplicate_count:
        raise CapacityTableError("Capacity table contains duplicate elevations.")
    if not monotonic_elevation:
        raise CapacityTableError("Capacity-table elevation must strictly increase.")
    if not monotonic_area:
        raise CapacityTableError("Capacity-table area must be nondecreasing.")
    if not monotonic_storage:
        raise CapacityTableError("Capacity-table storage must be nondecreasing.")

    steps = [right - left for left, right in pairwise(elevations)]
    validation = CapacityTableValidation(
        row_count=len(normalized),
        elevation_min=min(elevations),
        elevation_max=max(elevations),
        area_min=min(areas),
        area_max=max(areas),
        storage_min=min(storages),
        storage_max=max(storages),
        elevation_step_min=min(steps),
        elevation_step_median=statistics.median(steps),
        elevation_step_max=max(steps),
        monotonic_elevation=monotonic_elevation,
        monotonic_area=monotonic_area,
        monotonic_storage=monotonic_storage,
        duplicate_elevation_count=duplicate_count,
        missing_value_count=0,
        source_hash=metadata.source_hash,
        validation_status="passed",
    )
    return NormalizedCapacityTable(
        metadata=metadata,
        rows=tuple(normalized),
        validation=validation,
    )


__all__ = [
    "AreaVolumeConsistencySummary",
    "AreaVolumeSegmentDiagnostic",
    "CapacityTableError",
    "CapacityTableInputRow",
    "CapacityTableMetadata",
    "CapacityTableValidation",
    "NormalizedCapacityRow",
    "NormalizedCapacityTable",
    "normalize_capacity_table",
]

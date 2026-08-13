"""Credential-free change, seasonal-context, freshness, and quality metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any


class ManagerMetricError(ValueError):
    """Raised when observations cannot be compared without ambiguity."""


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManagerMetricError(f"{label} must be a non-empty string.")
    return value.strip()


def _as_utc(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ManagerMetricError("Observation datetimes must be timezone-aware.")
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    raise ManagerMetricError("Observation timestamp must be a date or datetime.")


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return _as_utc(value).date()
    if isinstance(value, date):
        return value
    raise ManagerMetricError("analysis_end must be a date or datetime.")


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """Minimal daily value used by manager-summary calculations."""

    timestamp: date | datetime
    value: float | None
    metric: str
    unit: str
    method: str
    provisional: bool | None = None
    approval_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _as_utc(self.timestamp))
        for name in ("metric", "unit", "method"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.value is not None:
            if isinstance(self.value, bool):
                raise ManagerMetricError("Observation values must be finite or null.")
            try:
                numeric = float(self.value)
            except (TypeError, ValueError) as exc:
                raise ManagerMetricError(
                    "Observation values must be finite or null."
                ) from exc
            if not math.isfinite(numeric):
                raise ManagerMetricError("Observation values must be finite or null.")
            object.__setattr__(self, "value", numeric)
        if not isinstance(self.provisional, (bool, type(None))):
            raise ManagerMetricError("provisional must be true, false, or null.")
        if self.approval_status is not None:
            object.__setattr__(
                self,
                "approval_status",
                _required_text(self.approval_status, "approval_status"),
            )

    @property
    def calendar_date(self) -> date:
        return self.timestamp.date()


def _identity(
    observations: Sequence[MetricObservation],
) -> tuple[str, str, str] | None:
    if not observations:
        return None
    identities = {(row.metric, row.unit, row.method) for row in observations}
    if len(identities) != 1:
        raise ManagerMetricError(
            "Manager metrics cannot compare different metrics, units, or methods."
        )
    return next(iter(identities))


def _prepare(
    observations: Iterable[MetricObservation], analysis_end: date | datetime
) -> tuple[tuple[MetricObservation, ...], date]:
    rows = tuple(observations)
    if any(not isinstance(row, MetricObservation) for row in rows):
        raise ManagerMetricError("All rows must be MetricObservation instances.")
    _identity(rows)
    end = _as_date(analysis_end)
    valid = tuple(
        sorted(
            (row for row in rows if row.value is not None and row.calendar_date <= end),
            key=lambda row: row.timestamp,
        )
    )
    dates = [row.calendar_date for row in valid]
    if len(dates) != len(set(dates)):
        raise ManagerMetricError(
            "Daily manager metrics require at most one valid row per calendar date."
        )
    return valid, end


def _iso_dict(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    return value


@dataclass(frozen=True, slots=True)
class ChangeMetricResult:
    latest_date: date | None
    latest_value: float | None
    baseline_target_date: date | None
    baseline_actual_date: date | None
    baseline_value: float | None
    requested_window_days: int
    actual_elapsed_days: int | None
    absolute_change: float | None
    percent_change: float | None
    comparison_status: str

    def as_dict(self) -> dict[str, Any]:
        return {key: _iso_dict(value) for key, value in asdict(self).items()}


def calculate_change_metric(
    observations: Iterable[MetricObservation],
    *,
    requested_window_days: int,
    analysis_end: date | datetime,
    baseline_match_tolerance_days: int = 3,
) -> ChangeMetricResult:
    """Calculate a non-interpolated latest-to-nearest-baseline change."""

    if (
        isinstance(requested_window_days, bool)
        or not isinstance(requested_window_days, int)
        or requested_window_days <= 0
    ):
        raise ManagerMetricError("requested_window_days must be a positive integer.")
    if (
        isinstance(baseline_match_tolerance_days, bool)
        or not isinstance(baseline_match_tolerance_days, int)
        or baseline_match_tolerance_days < 0
    ):
        raise ManagerMetricError(
            "baseline_match_tolerance_days must be a non-negative integer."
        )
    rows, _end = _prepare(observations, analysis_end)
    if not rows:
        return ChangeMetricResult(
            latest_date=None,
            latest_value=None,
            baseline_target_date=None,
            baseline_actual_date=None,
            baseline_value=None,
            requested_window_days=requested_window_days,
            actual_elapsed_days=None,
            absolute_change=None,
            percent_change=None,
            comparison_status="no_latest_observation",
        )
    latest = rows[-1]
    target = latest.calendar_date - timedelta(days=requested_window_days)
    candidates = [
        row
        for row in rows[:-1]
        if abs((row.calendar_date - target).days) <= baseline_match_tolerance_days
    ]
    if not candidates:
        return ChangeMetricResult(
            latest_date=latest.calendar_date,
            latest_value=latest.value,
            baseline_target_date=target,
            baseline_actual_date=None,
            baseline_value=None,
            requested_window_days=requested_window_days,
            actual_elapsed_days=None,
            absolute_change=None,
            percent_change=None,
            comparison_status="no_baseline_within_tolerance",
        )
    baseline = min(
        candidates,
        key=lambda row: (
            abs((row.calendar_date - target).days),
            row.calendar_date,
        ),
    )
    absolute = float(latest.value) - float(baseline.value)
    percent = (
        None if baseline.value == 0.0 else 100.0 * absolute / float(baseline.value)
    )
    return ChangeMetricResult(
        latest_date=latest.calendar_date,
        latest_value=latest.value,
        baseline_target_date=target,
        baseline_actual_date=baseline.calendar_date,
        baseline_value=baseline.value,
        requested_window_days=requested_window_days,
        actual_elapsed_days=(latest.calendar_date - baseline.calendar_date).days,
        absolute_change=absolute,
        percent_change=percent,
        comparison_status="available",
    )


def calculate_change_metrics(
    observations: Iterable[MetricObservation],
    *,
    windows_days: Iterable[int],
    analysis_end: date | datetime,
    baseline_match_tolerance_days: int = 3,
) -> tuple[ChangeMetricResult, ...]:
    """Calculate a deterministic set of manager change windows."""

    rows = tuple(observations)
    windows = tuple(windows_days)
    if any(
        isinstance(window, bool) or not isinstance(window, int) for window in windows
    ):
        raise ManagerMetricError("windows_days must contain only integers.")
    if len(windows) != len(set(windows)):
        raise ManagerMetricError("windows_days contains duplicate windows.")
    return tuple(
        calculate_change_metric(
            rows,
            requested_window_days=window,
            analysis_end=analysis_end,
            baseline_match_tolerance_days=baseline_match_tolerance_days,
        )
        for window in windows
    )


def _seasonal_day_distance(left: date, right: date) -> int:
    # Mapping all month/day pairs to a leap reference year gives Feb. 29 an
    # explicit position and makes the Dec./Jan window circular.
    left_day = date(2000, left.month, left.day).timetuple().tm_yday
    right_day = date(2000, right.month, right.day).timetuple().tm_yday
    direct = abs(left_day - right_day)
    return min(direct, 366 - direct)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class SeasonalContextResult:
    latest_date: date | None
    latest_value: float | None
    seasonal_window_days: int
    empirical_percentile: float | None
    median: float | None
    q25: float | None
    q75: float | None
    minimum: float | None
    maximum: float | None
    observation_count: int
    distinct_year_count: int
    status: str
    percentile_method: str = "midrank_empirical_percentile"

    def as_dict(self) -> dict[str, Any]:
        return {key: _iso_dict(value) for key, value in asdict(self).items()}


def calculate_seasonal_context(
    observations: Iterable[MetricObservation],
    *,
    analysis_end: date | datetime,
    seasonal_window_days: int = 15,
    minimum_distinct_years: int = 5,
    minimum_observations: int = 20,
) -> SeasonalContextResult:
    """Compare the latest value with prior-year observations near its season."""

    for value, label, minimum in (
        (seasonal_window_days, "seasonal_window_days", 0),
        (minimum_distinct_years, "minimum_distinct_years", 1),
        (minimum_observations, "minimum_observations", 1),
    ):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ManagerMetricError(f"{label} must be an integer >= {minimum}.")
    rows, _end = _prepare(observations, analysis_end)
    if not rows:
        return SeasonalContextResult(
            latest_date=None,
            latest_value=None,
            seasonal_window_days=seasonal_window_days,
            empirical_percentile=None,
            median=None,
            q25=None,
            q75=None,
            minimum=None,
            maximum=None,
            observation_count=0,
            distinct_year_count=0,
            status="no_latest_observation",
        )
    latest = rows[-1]
    historical = tuple(
        row
        for row in rows
        if row.calendar_date.year != latest.calendar_date.year
        and _seasonal_day_distance(row.calendar_date, latest.calendar_date)
        <= seasonal_window_days
    )
    values = [float(row.value) for row in historical]
    years = {row.calendar_date.year for row in historical}
    common = {
        "latest_date": latest.calendar_date,
        "latest_value": latest.value,
        "seasonal_window_days": seasonal_window_days,
        "empirical_percentile": None,
        "median": None,
        "q25": None,
        "q75": None,
        "minimum": None,
        "maximum": None,
        "observation_count": len(values),
        "distinct_year_count": len(years),
    }
    if len(years) < minimum_distinct_years:
        return SeasonalContextResult(status="insufficient_years", **common)
    if len(values) < minimum_observations:
        return SeasonalContextResult(status="insufficient_observations", **common)

    less = sum(value < float(latest.value) for value in values)
    equal = sum(value == float(latest.value) for value in values)
    percentile = 100.0 * (less + 0.5 * equal) / len(values)
    return SeasonalContextResult(
        latest_date=latest.calendar_date,
        latest_value=latest.value,
        seasonal_window_days=seasonal_window_days,
        empirical_percentile=percentile,
        median=_quantile(values, 0.5),
        q25=_quantile(values, 0.25),
        q75=_quantile(values, 0.75),
        minimum=min(values),
        maximum=max(values),
        observation_count=len(values),
        distinct_year_count=len(years),
        status="available",
    )


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    days_since_latest_observation: int | None
    freshness_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_freshness(
    latest_observation_date: date | datetime | None,
    *,
    analysis_end: date | datetime,
    freshness_warning_days: int = 7,
) -> FreshnessResult:
    """Classify an available latest observation as current or stale."""

    if isinstance(freshness_warning_days, bool) or freshness_warning_days < 0:
        raise ManagerMetricError(
            "freshness_warning_days must be a non-negative integer."
        )
    if latest_observation_date is None:
        return FreshnessResult(None, "unavailable")
    latest = _as_date(latest_observation_date)
    end = _as_date(analysis_end)
    elapsed = (end - latest).days
    if elapsed < 0:
        raise ManagerMetricError("latest observation cannot be after analysis_end.")
    return FreshnessResult(
        days_since_latest_observation=elapsed,
        freshness_status=("current" if elapsed <= freshness_warning_days else "stale"),
    )


def summarize_quality(observations: Iterable[MetricObservation]) -> str:
    """Return approved/provisional/mixed/unknown without discarding detail."""

    statuses: set[str] = set()
    for row in observations:
        if not isinstance(row, MetricObservation):
            raise ManagerMetricError("All rows must be MetricObservation instances.")
        approval = (row.approval_status or "").casefold()
        if row.provisional is True or "provisional" in approval:
            statuses.add("provisional")
        elif row.provisional is False or "approved" in approval:
            statuses.add("approved")
        else:
            statuses.add("unknown")
    if not statuses or statuses == {"unknown"}:
        return "unknown"
    known = statuses - {"unknown"}
    if len(known) == 1 and "unknown" not in statuses:
        return next(iter(known))
    return "mixed"


__all__ = [
    "ChangeMetricResult",
    "FreshnessResult",
    "ManagerMetricError",
    "MetricObservation",
    "SeasonalContextResult",
    "calculate_change_metric",
    "calculate_change_metrics",
    "calculate_freshness",
    "calculate_seasonal_context",
    "summarize_quality",
]

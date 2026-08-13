from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

from regional_analysis.hydrometry.manager_metrics import (
    ManagerMetricError,
    MetricObservation,
    calculate_change_metric,
    calculate_change_metrics,
    calculate_freshness,
    calculate_seasonal_context,
    summarize_quality,
)


def observation(
    day: date,
    value: float | None,
    *,
    metric: str = "total_storage",
    unit: str = "acre_ft",
    method: str = "agency_reported",
    provisional: bool | None = None,
    approval_status: str | None = None,
) -> MetricObservation:
    return MetricObservation(
        timestamp=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
        value=value,
        metric=metric,
        unit=unit,
        method=method,
        provisional=provisional,
        approval_status=approval_status,
    )


class ChangeMetricTests(unittest.TestCase):
    def test_exact_comparison_date_and_percent_change(self) -> None:
        rows = (
            observation(date(2026, 8, 6), 100.0),
            observation(date(2026, 8, 13), 110.0),
        )
        result = calculate_change_metric(
            rows,
            requested_window_days=7,
            analysis_end=date(2026, 8, 13),
        )
        self.assertEqual(result.baseline_target_date, date(2026, 8, 6))
        self.assertEqual(result.baseline_actual_date, date(2026, 8, 6))
        self.assertEqual(result.actual_elapsed_days, 7)
        self.assertEqual(result.absolute_change, 10.0)
        self.assertEqual(result.percent_change, 10.0)
        self.assertEqual(result.comparison_status, "available")

    def test_nearest_within_tolerance_is_used_without_interpolation(self) -> None:
        rows = (
            observation(date(2026, 8, 4), 80.0),
            observation(date(2026, 8, 8), 90.0),
            observation(date(2026, 8, 13), 100.0),
        )
        result = calculate_change_metric(
            rows,
            requested_window_days=7,
            analysis_end=date(2026, 8, 13),
            baseline_match_tolerance_days=3,
        )
        # The dates are tied at two days from Aug. 6; deterministic tie-breaking
        # chooses the earlier observed value, never an interpolated value.
        self.assertEqual(result.baseline_actual_date, date(2026, 8, 4))
        self.assertEqual(result.baseline_value, 80.0)
        self.assertEqual(result.actual_elapsed_days, 9)

    def test_outside_tolerance_is_unavailable(self) -> None:
        result = calculate_change_metric(
            (
                observation(date(2026, 8, 1), 80.0),
                observation(date(2026, 8, 13), 100.0),
            ),
            requested_window_days=7,
            analysis_end=date(2026, 8, 13),
            baseline_match_tolerance_days=2,
        )
        self.assertEqual(result.comparison_status, "no_baseline_within_tolerance")
        self.assertIsNone(result.absolute_change)

    def test_zero_baseline_keeps_absolute_but_not_percent_change(self) -> None:
        result = calculate_change_metric(
            (
                observation(date(2026, 8, 6), 0.0),
                observation(date(2026, 8, 13), 5.0),
            ),
            requested_window_days=7,
            analysis_end=date(2026, 8, 13),
        )
        self.assertEqual(result.absolute_change, 5.0)
        self.assertIsNone(result.percent_change)

    def test_identity_mismatch_is_rejected(self) -> None:
        mismatches = (
            {"unit": "m3"},
            {"metric": "surface_area"},
            {"method": "official_eav_piecewise_linear"},
        )
        for override in mismatches:
            rows = (
                observation(date(2026, 8, 6), 10.0),
                observation(date(2026, 8, 13), 11.0, **override),
            )
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ManagerMetricError, "different metrics"),
            ):
                calculate_change_metric(
                    rows,
                    requested_window_days=7,
                    analysis_end=date(2026, 8, 13),
                )

    def test_fractional_window_is_not_silently_coerced(self) -> None:
        with self.assertRaisesRegex(ManagerMetricError, "only integers"):
            calculate_change_metrics(
                (),
                windows_days=(7, 30.5),
                analysis_end=date(2026, 8, 13),
            )

    def test_standard_window_set_is_deterministic(self) -> None:
        end = date(2026, 8, 13)
        rows = tuple(
            observation(end - timedelta(days=day), 1_000.0 - day) for day in range(370)
        )
        results = calculate_change_metrics(
            rows,
            windows_days=(7, 30, 90, 365),
            analysis_end=end,
        )
        self.assertEqual(
            tuple(result.requested_window_days for result in results),
            (7, 30, 90, 365),
        )
        self.assertTrue(
            all(result.comparison_status == "available" for result in results)
        )


class SeasonalContextTests(unittest.TestCase):
    def test_prior_year_window_percentile_and_quantiles(self) -> None:
        rows: list[MetricObservation] = []
        values = iter(range(25))
        for year in range(2020, 2025):
            for offset in range(-2, 3):
                rows.append(
                    observation(
                        date(year, 8, 15) + timedelta(days=offset),
                        float(next(values)),
                    )
                )
        rows.extend(
            (
                observation(date(2020, 7, 1), -999.0),
                # Same-year values other than the latest must be excluded.
                observation(date(2026, 8, 14), 999.0),
                observation(date(2026, 8, 15), 12.0),
            )
        )
        result = calculate_seasonal_context(
            rows,
            analysis_end=date(2026, 8, 15),
            seasonal_window_days=15,
            minimum_distinct_years=5,
            minimum_observations=20,
        )
        self.assertEqual(result.status, "available")
        self.assertEqual(result.observation_count, 25)
        self.assertEqual(result.distinct_year_count, 5)
        self.assertEqual(result.minimum, 0.0)
        self.assertEqual(result.maximum, 24.0)
        self.assertEqual(result.median, 12.0)
        self.assertEqual(result.empirical_percentile, 50.0)

    def test_minimum_year_and_observation_gates_are_distinct(self) -> None:
        short = tuple(
            observation(date(year, 8, 15), float(year))
            for year in (2023, 2024, 2025, 2026)
        )
        years = calculate_seasonal_context(
            short,
            analysis_end=date(2026, 8, 15),
            minimum_distinct_years=5,
            minimum_observations=1,
        )
        self.assertEqual(years.status, "insufficient_years")

        sparse = tuple(
            observation(date(year, 8, 15), float(year)) for year in range(2020, 2027)
        )
        observations = calculate_seasonal_context(
            sparse,
            analysis_end=date(2026, 8, 15),
            minimum_distinct_years=5,
            minimum_observations=20,
        )
        self.assertEqual(observations.status, "insufficient_observations")

    def test_leap_day_has_explicit_neighboring_window(self) -> None:
        rows = [
            observation(date(year, 2, 28), float(year))
            for year in (2018, 2019, 2020, 2021, 2022)
        ]
        rows.extend(
            observation(date(year, 3, 1), float(year) + 0.5)
            for year in (2018, 2019, 2020, 2021, 2022)
        )
        rows.append(observation(date(2024, 2, 29), 2020.0))
        result = calculate_seasonal_context(
            rows,
            analysis_end=date(2024, 2, 29),
            seasonal_window_days=1,
            minimum_distinct_years=5,
            minimum_observations=10,
        )
        self.assertEqual(result.status, "available")
        self.assertEqual(result.observation_count, 10)

    def test_no_latest_observation(self) -> None:
        result = calculate_seasonal_context(
            (observation(date(2026, 8, 13), None),),
            analysis_end=date(2026, 8, 13),
        )
        self.assertEqual(result.status, "no_latest_observation")
        self.assertIsNone(result.empirical_percentile)


class FreshnessAndQualityTests(unittest.TestCase):
    def test_current_stale_and_unavailable(self) -> None:
        end = date(2026, 8, 13)
        self.assertEqual(
            calculate_freshness(date(2026, 8, 6), analysis_end=end).freshness_status,
            "current",
        )
        self.assertEqual(
            calculate_freshness(date(2026, 8, 5), analysis_end=end).freshness_status,
            "stale",
        )
        unavailable = calculate_freshness(None, analysis_end=end)
        self.assertEqual(unavailable.freshness_status, "unavailable")
        self.assertIsNone(unavailable.days_since_latest_observation)

    def test_quality_distinctions_are_retained(self) -> None:
        approved = observation(
            date(2026, 8, 12),
            1.0,
            provisional=False,
            approval_status="Approved",
        )
        provisional = observation(
            date(2026, 8, 13),
            2.0,
            provisional=True,
            approval_status="Provisional",
        )
        self.assertEqual(summarize_quality((approved,)), "approved")
        self.assertEqual(summarize_quality((provisional,)), "provisional")
        self.assertEqual(summarize_quality((approved, provisional)), "mixed")
        self.assertEqual(
            summarize_quality((observation(date(2026, 8, 13), 1.0),)),
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()

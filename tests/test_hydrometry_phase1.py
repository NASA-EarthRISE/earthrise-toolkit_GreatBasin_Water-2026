from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

from regional_analysis.hydrometry.adapters.usbr_rise import (
    RiseCatalogItem,
    RiseCatalogRecord,
    RiseItemInventory,
    RiseLocation,
    RiseObservation,
)
from regional_analysis.hydrometry.adapters.usgs_waterdata import (
    MonitoringLocation,
    UsgsDataKind,
    UsgsObservation,
    UsgsParameterMetadata,
    UsgsSeriesMetadata,
)
from regional_analysis.hydrometry.cache import CacheMode
from regional_analysis.hydrometry.capacity_tables import (
    CapacityTableInputRow,
    CapacityTableMetadata,
    normalize_capacity_table,
)
from regional_analysis.hydrometry.models import HeightType, HydrometryVariable
from regional_analysis.hydrometry.phase1 import (
    DAILY_COLUMNS,
    MANAGER_SCHEMA_VERSION,
    Phase1Adapters,
    Phase1Error,
    Phase1Settings,
    build_manager_summary,
    build_manager_tables,
    load_phase1_settings,
    normalize_daily_frame,
    run_phase1,
    write_phase1_outputs,
)
from regional_analysis.hydrometry.registry import load_registry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "hydrometry_pilots.yaml"
PINNED_NOW = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)


def _history_dates() -> tuple[date, ...]:
    seasonal = tuple(
        date(year, 8, 13) + timedelta(days=offset)
        for year in range(2019, 2024)
        for offset in (-2, -1, 0, 1)
    )
    comparisons = (
        date(2025, 8, 13),
        date(2026, 5, 15),
        date(2026, 7, 14),
        date(2026, 8, 6),
        date(2026, 8, 13),
    )
    return tuple(sorted(set(seasonal + comparisons)))


def _series_value(location_id: str, index: int) -> float:
    if location_id == "USGS-10055000":
        return 5_920.0 + 0.01 * index
    if location_id == "USGS-10387150":
        return 4_250.0 + 0.02 * index
    if location_id == "USGS-10010000":
        return 4_191.0 + 0.01 * index
    return 4_190.0 + 0.015 * index


class SyntheticUsgsAdapter:
    """A deterministic modern-WDFN-shaped adapter with no HTTP behavior."""

    def fetch_monitoring_location(self, location_id: str) -> MonitoringLocation:
        datum = (
            "NGVD29"
            if location_id
            in {
                "USGS-10010000",
                "USGS-10010100",
            }
            else "NAVD88"
        )
        return MonitoringLocation(
            location_id=location_id,
            monitoring_location_number=location_id.removeprefix("USGS-"),
            name=f"Synthetic {location_id}",
            site_type="Lake, Reservoir, Impoundment",
            state_name=None,
            longitude=-111.0,
            latitude=42.0,
            altitude=None,
            altitude_unit=None,
            vertical_datum=datum,
            vertical_datum_name=datum,
            source_url_public=f"https://example.test/usgs/{location_id}",
        )

    def discover_series(self, location_id: str) -> tuple[UsgsSeriesMetadata, ...]:
        datum = (
            "NGVD29"
            if location_id
            in {
                "USGS-10010000",
                "USGS-10010100",
            }
            else "NAVD88"
        )
        parameter = "62614" if datum == "NGVD29" else "62615"
        return (
            UsgsSeriesMetadata(
                time_series_id=f"daily-{location_id}-{parameter}",
                monitoring_location_id=location_id,
                parameter_code=parameter,
                parameter_name="Lake or reservoir water surface elevation",
                parameter_description=(
                    f"Lake or reservoir water surface elevation above {datum}, feet"
                ),
                unit="ft",
                statistic_code="00003",
                computation_identifier="mean",
                computation_period_identifier="daily",
                begin_utc=datetime(2016, 1, 1, tzinfo=UTC),
                end_utc=PINNED_NOW,
                primary=True,
                web_description="Synthetic daily mean fixture",
                variable=HydrometryVariable.WATER_LEVEL,
                height_type=HeightType.ORTHOMETRIC_HEIGHT,
                vertical_datum=datum,
                data_kind=UsgsDataKind.DAILY,
                source_url_public=f"https://example.test/usgs/{location_id}/series",
            ),
        )

    def fetch_parameter_metadata(self, parameter_code: str) -> UsgsParameterMetadata:
        return UsgsParameterMetadata(
            parameter_code=parameter_code,
            parameter_name="Lake or reservoir water surface elevation",
            parameter_description="Absolute daily water elevation",
            unit="ft",
            medium="Water",
            source_url_public=f"https://example.test/usgs/parameter/{parameter_code}",
        )

    def fetch_observations(
        self,
        series: UsgsSeriesMetadata,
        *,
        start: date,
        end: date,
        **_kwargs,
    ) -> tuple[UsgsObservation, ...]:
        return tuple(
            UsgsObservation(
                time_series_id=series.time_series_id,
                monitoring_location_id=series.monitoring_location_id,
                timestamp_utc=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
                value=_series_value(series.monitoring_location_id, index),
                unit="ft",
                parameter_code=series.parameter_code,
                statistic_code="00003",
                approval_status=("Provisional" if day == end else "Approved"),
                provisional=day == end,
                qualifier="Estimated" if day == end else "Approved",
                last_modified_utc=PINNED_NOW,
            )
            for index, day in enumerate(_history_dates())
            if start <= day <= end
        )


def _rise_item(
    record_id: int,
    item_id: int,
    variable: HydrometryVariable,
    unit: str,
    title: str,
) -> RiseCatalogItem:
    return RiseCatalogItem(
        item_id=item_id,
        record_id=record_id,
        title=title,
        description=f"Synthetic {title}",
        variable=variable,
        provider_variable=variable.value,
        parameter_name=title,
        parameter_group="Lake/Reservoir",
        parameter_id=item_id + 10_000,
        unit=unit,
        timestep="day",
        transformation="mean",
        temporal_start=datetime(2016, 1, 1, tzinfo=UTC),
        temporal_end=PINNED_NOW,
        update_frequency="daily",
        updated_at=PINNED_NOW,
        vertical_datum=None,
        provisional=True,
        disclaimer="Provisional data subject to revision.",
        data_structure="time series",
        binary_file_url=None,
        validation_issues=(),
        source_url_public=f"https://example.test/rise/item/{item_id}",
    )


RISE_ITEMS = {
    2362: (
        _rise_item(
            2362,
            508,
            HydrometryVariable.RESERVOIR_ELEVATION,
            "ft",
            "Lake Powell Reservoir Elevation",
        ),
        _rise_item(
            2362,
            509,
            HydrometryVariable.STORAGE,
            "af",
            "Lake Powell Total Reservoir Storage",
        ),
        _rise_item(
            2362,
            4784,
            HydrometryVariable.SURFACE_AREA,
            "acres",
            "Lake Powell Reservoir Area",
        ),
    ),
    2300: (
        _rise_item(
            2300,
            341,
            HydrometryVariable.RESERVOIR_ELEVATION,
            "ft",
            "Flaming Gorge Reservoir Elevation",
        ),
        _rise_item(
            2300,
            337,
            HydrometryVariable.STORAGE,
            "af",
            "Flaming Gorge Total Reservoir Storage",
        ),
        _rise_item(
            2300,
            4782,
            HydrometryVariable.SURFACE_AREA,
            "acres",
            "Flaming Gorge Reservoir Area",
        ),
        _rise_item(
            2300,
            4275,
            HydrometryVariable.BANK_STORAGE,
            "af",
            "Flaming Gorge Bank Storage",
        ),
    ),
}


class SyntheticRiseAdapter:
    """A deterministic RISE-shaped adapter that records requested item IDs."""

    def __init__(self) -> None:
        self.fetched_item_ids: list[int] = []

    def fetch_record(
        self, record_id: int, *, expected_title_contains: str
    ) -> RiseCatalogRecord:
        location_id = 393 if record_id == 2362 else 1535
        return RiseCatalogRecord(
            record_id=record_id,
            title=f"{expected_title_contains} synthetic operations",
            description="Synthetic record",
            generation_effort=None,
            location_id=location_id,
            item_ids=tuple(item.item_id for item in RISE_ITEMS[record_id]),
            updated_at=PINNED_NOW,
            source_url_public=f"https://example.test/rise/record/{record_id}",
        )

    def fetch_location(
        self,
        location_id: int,
        *,
        expected_name_contains: str,
        expected_record_id: int,
    ) -> RiseLocation:
        return RiseLocation(
            location_id=location_id,
            name=f"{expected_name_contains} synthetic location",
            location_type="Reservoir",
            longitude=-110.0,
            latitude=40.0,
            horizontal_datum="WGS84",
            vertical_datum=None,
            states=("UT",),
            record_ids=(expected_record_id,),
            source_url_public=f"https://example.test/rise/location/{location_id}",
        )

    def inventory_record_items(
        self,
        record: RiseCatalogRecord,
        *,
        expected_location_name: str,
        target_variables,
    ) -> RiseItemInventory:
        del expected_location_name, target_variables
        items = RISE_ITEMS[record.record_id]
        candidates = {
            variable: tuple(item.item_id for item in items if item.variable is variable)
            for variable in HydrometryVariable
        }
        return RiseItemInventory(
            record=record,
            items=items,
            candidates_by_variable=candidates,
            requires_review=False,
        )

    def fetch_timeseries(
        self,
        item: RiseCatalogItem,
        *,
        start: date,
        end: date,
        **_kwargs,
    ) -> tuple[RiseObservation, ...]:
        self.fetched_item_ids.append(item.item_id)
        base = {
            HydrometryVariable.RESERVOIR_ELEVATION: 3_600.0,
            HydrometryVariable.SURFACE_AREA: 100_000.0,
            HydrometryVariable.STORAGE: 10_000_000.0,
        }[item.variable]
        results: list[RiseObservation] = []
        for index, day in enumerate(_history_dates()):
            if not start <= day <= end:
                continue
            # Preserve one provider null as a missing value; it must never
            # become a zero or the manager's latest value.
            value = None if index == 1 else base + 10.0 * index + item.item_id
            results.append(
                RiseObservation(
                    result_id=item.item_id * 10_000 + index,
                    item_id=item.item_id,
                    location_id=393 if item.record_id == 2362 else 1535,
                    parameter_id=item.parameter_id,
                    timestamp_utc=datetime(day.year, day.month, day.day, 7, tzinfo=UTC),
                    value=value,
                    status="Provisional",
                    source_code="fixture",
                    created_at=PINNED_NOW,
                    updated_at=PINNED_NOW,
                )
            )
        return tuple(results)


class UnorderedRiseAdapter(SyntheticRiseAdapter):
    def fetch_timeseries(self, item: RiseCatalogItem, **kwargs):
        ordered = super().fetch_timeseries(item, **kwargs)
        if item.item_id == 508:
            return tuple(reversed(ordered))
        return ordered


def _reference_source() -> SimpleNamespace:
    table = normalize_capacity_table(
        (
            CapacityTableInputRow(2, 5_900.0, 60_000.0, 500_000.0),
            CapacityTableInputRow(3, 5_930.0, 70_000.0, 800_000.0),
            CapacityTableInputRow(4, 5_960.0, 75_000.0, 1_200_000.0),
        ),
        CapacityTableMetadata(
            lake_name="Bear Lake",
            source_doi="10.5066/P14KZS2D",
            source_title="Synthetic official Bear Lake E-A-V fixture",
            source_file="bear_lake_eav_fixture.csv",
            source_version="fixture-v1",
            source_hash="c" * 64,
            vertical_datum="NAVD88",
            elevation_unit="ft",
            area_unit="acres",
            storage_unit="acre-feet",
        ),
        expected_lake_name="Bear Lake",
    )
    _segments, summary = table.area_volume_diagnostics()
    return SimpleNamespace(
        table=table,
        manifest={
            "doi": "10.5066/P14KZS2D",
            "sha256": "c" * 64,
            "source_version": "fixture-v1",
        },
        validation_frame=pd.DataFrame([table.validation.as_dict()]),
        area_volume_diagnostic_frame=pd.DataFrame([summary.as_dict()]),
    )


class Phase1SettingsTests(unittest.TestCase):
    def test_repository_settings_are_pinned_and_no_extrapolation(self) -> None:
        settings = load_phase1_settings(CONFIG_PATH)
        self.assertEqual(settings.history_start, date(2016, 1, 1))
        self.assertEqual(settings.analysis_end, date(2026, 8, 13))
        self.assertEqual(settings.change_windows_days, (7, 30, 90, 365))
        self.assertEqual(settings.baseline_match_tolerance_days, 3)
        self.assertEqual(settings.interpolation_method, "piecewise_linear")
        self.assertFalse(settings.allow_extrapolation)

    def test_settings_reject_extrapolation_and_unpinned_order(self) -> None:
        with self.assertRaisesRegex(Phase1Error, "forbids"):
            Phase1Settings(
                history_start=date(2016, 1, 1),
                analysis_end=date(2026, 8, 13),
                change_windows_days=(7,),
                baseline_match_tolerance_days=3,
                seasonal_window_days=15,
                seasonal_min_distinct_years=5,
                seasonal_min_observations=20,
                freshness_warning_days=7,
                interpolation_method="piecewise_linear",
                allow_extrapolation=True,
            )
        with self.assertRaisesRegex(Phase1Error, "on or before"):
            Phase1Settings(
                history_start=date(2026, 8, 14),
                analysis_end=date(2026, 8, 13),
                change_windows_days=(7,),
                baseline_match_tolerance_days=3,
                seasonal_window_days=15,
                seasonal_min_distinct_years=5,
                seasonal_min_observations=20,
                freshness_warning_days=7,
                interpolation_method="piecewise_linear",
                allow_extrapolation=False,
            )

    def test_registry_freezes_direct_and_bank_storage_item_identities(self) -> None:
        registry = load_registry(CONFIG_PATH)
        powell = registry.get_profile("lake_powell", strict=True)
        flaming = registry.get_profile("flaming_gorge", strict=True)
        powell_ids = {
            source.variable: source.source_item_id for source in powell.sources
        }
        flaming_ids = {
            source.variable: source.source_item_id for source in flaming.sources
        }
        self.assertEqual(powell_ids[HydrometryVariable.STORAGE], "509")
        self.assertEqual(powell_ids[HydrometryVariable.SURFACE_AREA], "4784")
        self.assertEqual(flaming_ids[HydrometryVariable.STORAGE], "337")
        self.assertEqual(flaming_ids[HydrometryVariable.BANK_STORAGE], "4275")
        self.assertNotEqual(
            flaming_ids[HydrometryVariable.STORAGE],
            flaming_ids[HydrometryVariable.BANK_STORAGE],
        )


class Phase1OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(CONFIG_PATH)
        cls.settings = load_phase1_settings(CONFIG_PATH)
        cls.rise = SyntheticRiseAdapter()
        cls.adapters = Phase1Adapters(
            usgs=SyntheticUsgsAdapter(),
            rise=cls.rise,
            client=SimpleNamespace(request_count=0),
        )
        cls.bundle = run_phase1(
            cls.registry,
            cls.settings,
            mode=CacheMode.OFFLINE,
            adapters=cls.adapters,
            reference_source=_reference_source(),
            now=PINNED_NOW,
        )

    def test_daily_schema_and_direct_versus_derived_contract(self) -> None:
        daily = self.bundle.daily
        self.assertEqual(tuple(daily.columns), DAILY_COLUMNS)
        bear_storage = daily.loc[
            daily["profile_id"].eq("bear_lake") & daily["metric"].eq("total_storage")
        ]
        self.assertFalse(bear_storage.empty)
        self.assertEqual(set(bear_storage["method"]), {"official_eav_piecewise_linear"})
        self.assertEqual(set(bear_storage["source_quality_class"]), {"B"})
        self.assertEqual(set(bear_storage["capability_level"]), {4})
        self.assertEqual(set(bear_storage["capacity_table_hash"]), {"c" * 64})
        bear_level = daily.loc[
            daily["profile_id"].eq("bear_lake") & daily["metric"].eq("water_elevation")
        ].set_index("calendar_date")
        bear_storage_by_date = bear_storage.set_index("calendar_date")
        for field in ("quality_status", "approval_status", "provisional"):
            self.assertEqual(
                bear_storage_by_date[field].to_dict(), bear_level[field].to_dict()
            )
        self.assertEqual(set(bear_storage["uncertainty_status"]), {"not_quantified"})
        self.assertTrue(bear_storage["uncertainty_lower"].isna().all())
        self.assertTrue(bear_storage["uncertainty_upper"].isna().all())

        for profile_id, item_id in (("lake_powell", "509"), ("flaming_gorge", "337")):
            storage = daily.loc[
                daily["profile_id"].eq(profile_id) & daily["metric"].eq("total_storage")
            ]
            self.assertEqual(set(storage["method"]), {"agency_reported"})
            self.assertEqual(set(storage["source_quality_class"]), {"A"})
            self.assertEqual(set(storage["source_item_id"]), {item_id})
            self.assertTrue(storage["capacity_table_hash"].isna().all())

    def test_great_salt_arms_are_separate_and_level_only(self) -> None:
        current = self.bundle.current_conditions
        great_salt = current.loc[current["profile_id"].eq("great_salt")]
        self.assertEqual(
            set(great_salt["subprofile_id"]),
            {"great_salt_south_arm", "great_salt_north_arm"},
        )
        self.assertNotIn("great_salt", set(great_salt["subprofile_id"]))
        levels = great_salt.loc[great_salt["metric"].eq("water_elevation")]
        self.assertTrue(levels["latest_value"].notna().all())
        unsupported = great_salt.loc[
            great_salt["metric"].isin(("surface_area", "total_storage"))
        ]
        self.assertTrue(unsupported["latest_value"].isna().all())
        self.assertEqual(
            set(unsupported["availability_status"]), {"unavailable_for_profile"}
        )

    def test_exactly_18_current_rows_and_abert_has_no_storage(self) -> None:
        current = self.bundle.current_conditions
        self.assertEqual(len(current), 18)
        self.assertFalse(
            current.duplicated(["subprofile_id", "metric"], keep=False).any()
        )
        abert_storage = current.loc[
            current["profile_id"].eq("lake_abert")
            & current["metric"].eq("total_storage")
        ].iloc[0]
        self.assertTrue(pd.isna(abert_storage["latest_value"]))
        self.assertEqual(
            abert_storage["availability_status"], "unavailable_for_profile"
        )

    def test_current_quality_comes_from_latest_observation(self) -> None:
        current = self.bundle.current_conditions.loc[
            self.bundle.current_conditions["availability_status"].eq("available")
        ]
        daily = self.bundle.daily.loc[
            self.bundle.daily["availability_status"].eq("available")
            & self.bundle.daily["value"].notna()
        ]
        for row in current.itertuples(index=False):
            latest = (
                daily.loc[
                    daily["subprofile_id"].eq(row.subprofile_id)
                    & daily["metric"].eq(row.metric)
                ]
                .sort_values("calendar_date", kind="stable")
                .iloc[-1]
            )
            expected_approval = (
                "unknown"
                if pd.isna(latest["approval_status"])
                else str(latest["approval_status"])
            )
            expected_provisional = (
                None if pd.isna(latest["provisional"]) else bool(latest["provisional"])
            )
            self.assertEqual(row.quality_status, latest["quality_status"])
            self.assertEqual(row.approval_status, expected_approval)
            self.assertEqual(row.provisional, expected_provisional)

    def test_bank_storage_is_validated_but_never_fetched_or_emitted(self) -> None:
        self.assertIn(337, self.rise.fetched_item_ids)
        self.assertNotIn(4275, self.rise.fetched_item_ids)
        self.assertNotIn("4275", set(self.bundle.daily["source_item_id"].dropna()))
        self.assertNotIn("bank_storage", set(self.bundle.daily["metric"]))
        flaming_total = self.bundle.daily.loc[
            self.bundle.daily["profile_id"].eq("flaming_gorge")
            & self.bundle.daily["metric"].eq("total_storage")
        ]
        self.assertEqual(set(flaming_total["source_item_id"]), {"337"})
        self.assertTrue(self.bundle.provenance["flaming_gorge_bank_storage_excluded"])

    def test_provider_null_is_retained_as_missing_not_zero(self) -> None:
        powell = self.bundle.daily.loc[
            self.bundle.daily["profile_id"].eq("lake_powell")
        ]
        missing = powell.loc[powell["availability_status"].eq("source_value_missing")]
        self.assertFalse(missing.empty)
        self.assertTrue(missing["value"].isna().all())
        self.assertFalse(powell["value"].eq(0.0).any())

    def test_manager_tables_and_json_are_deterministic(self) -> None:
        current, changes, seasonal = build_manager_tables(
            self.bundle.daily.copy(), self.settings, self.registry.profiles
        )
        assert_frame_equal(current, self.bundle.current_conditions)
        assert_frame_equal(changes, self.bundle.change_metrics)
        assert_frame_equal(seasonal, self.bundle.seasonal_context)
        self.assertEqual(len(changes), 48)
        self.assertEqual(len(seasonal), 12)

        rebuilt = build_manager_summary(
            self.registry.profiles,
            current,
            changes,
            seasonal,
            settings=self.settings,
            sources=tuple(
                source
                for profile in self.bundle.manager_summary["profiles"]
                for source in profile["sources"]
            ),
            generated_at=PINNED_NOW,
            provenance=self.bundle.manager_summary["provenance"],
        )
        self.assertEqual(
            json.dumps(rebuilt, sort_keys=True),
            json.dumps(self.bundle.manager_summary, sort_keys=True),
        )
        json.dumps(rebuilt, sort_keys=True, allow_nan=False)
        self.assertEqual(rebuilt["schema_version"], MANAGER_SCHEMA_VERSION)
        self.assertEqual(len(rebuilt["profiles"]), 5)
        rise_conditions = [
            row
            for profile in rebuilt["profiles"]
            if profile["identity"]["profile_id"] in {"lake_powell", "flaming_gorge"}
            for row in profile["current_conditions"]
            if row["availability_status"] == "available"
        ]
        self.assertEqual(
            {row["approval_status"] for row in rise_conditions}, {"unknown"}
        )

    def test_normalization_rejects_duplicate_daily_keys(self) -> None:
        row = self.bundle.daily.iloc[0].to_dict()
        with self.assertRaisesRegex(Phase1Error, "duplicate keys"):
            normalize_daily_frame((row, copy.deepcopy(row)))

    def test_offline_fixture_orchestration_writes_no_network_state(self) -> None:
        self.assertEqual(self.bundle.request_count, 0)
        self.assertEqual(self.bundle.provenance["mode"], "offline")
        self.assertFalse(self.bundle.provenance["earth_engine_used"])
        self.assertFalse(self.bundle.provenance["great_salt_storage_calculated"])
        self.assertFalse(self.bundle.provenance["lake_abert_storage_calculated"])
        self.assertFalse(self.bundle.provenance["lake_powell_storage_reconstructed"])
        self.assertFalse(self.bundle.provenance["flaming_gorge_storage_reconstructed"])
        self.assertTrue(self.bundle.provenance["bear_lake_storage_derived"])

    def test_unordered_direct_storage_source_is_sorted_deterministically(self) -> None:
        bundle = run_phase1(
            self.registry,
            self.settings,
            mode=CacheMode.OFFLINE,
            adapters=Phase1Adapters(
                usgs=SyntheticUsgsAdapter(),
                rise=UnorderedRiseAdapter(),
                client=SimpleNamespace(request_count=0),
            ),
            reference_source=_reference_source(),
            profile_ids=("lake_powell",),
            now=PINNED_NOW,
        )
        storage = bundle.daily.loc[bundle.daily["metric"].eq("total_storage")]
        self.assertEqual(
            storage["calendar_date"].tolist(),
            sorted(storage["calendar_date"].tolist()),
        )

    def test_atomic_output_contract_is_complete_and_byte_deterministic(self) -> None:
        expected = {
            "pilot_daily_hydrometry.csv",
            "pilot_current_conditions.csv",
            "pilot_change_metrics.csv",
            "pilot_seasonal_context.csv",
            "bear_lake_eav_normalized.csv",
            "bear_lake_eav_validation.csv",
            "bear_lake_eav_area_volume_diagnostic.csv",
            "pilot_manager_summary.json",
            "pilot_phase1_provenance.json",
        }
        with (
            tempfile.TemporaryDirectory() as first_name,
            tempfile.TemporaryDirectory() as second_name,
        ):
            first = Path(first_name)
            second = Path(second_name)
            first_paths = write_phase1_outputs(self.bundle, first)
            second_paths = write_phase1_outputs(self.bundle, second)
            self.assertEqual({path.name for path in first_paths}, expected)
            self.assertEqual({path.name for path in second_paths}, expected)
            for name in expected:
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )

            manager = json.loads(
                (first / "pilot_manager_summary.json").read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(
                    f"Manager output contains non-standard JSON value {value}"
                ),
            )
            self.assertEqual(manager["schema_version"], MANAGER_SCHEMA_VERSION)
            self.assertIsNone(
                next(
                    row
                    for profile in manager["profiles"]
                    if profile["identity"]["profile_id"] == "lake_abert"
                    for row in profile["current_conditions"]
                    if row["metric"] == "total_storage"
                )["latest_value"]
            )


if __name__ == "__main__":
    unittest.main()

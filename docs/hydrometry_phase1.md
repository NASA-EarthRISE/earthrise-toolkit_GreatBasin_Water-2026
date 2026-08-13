# Hydrometry Phase 1

Phase 1 turns the five-pilot [Phase 0](hydrometry_phase0.md) source inventory
into reproducible daily hydrometry and manager-facing summaries. It has two
operational pathways: direct agency-reported reservoir conditions and one
controlled, official-table stage-to-storage derivation for Bear Lake. It does
not change the QGIS plug-in, use Earth Engine, download bathymetric rasters, or
construct three-dimensional lake products.

The primary validation interval is pinned in
`configs/hydrometry_pilots.yaml`:

```text
2016-01-01 through 2026-08-13
```

Dates may be overridden explicitly at the command line. Tests and baseline
validation never substitute the execution date implicitly.

## Pilot products and evidence

| Pilot product | Source | Method label | Observation type | Evidence / capability |
|---|---|---|---|---|
| Great Salt south-arm elevation | USGS `USGS-10010000`, daily mean `62614` | `agency_observed` | observed | Level 2; NGVD29 |
| Great Salt north-arm elevation | USGS `USGS-10010100`, daily mean `62614` | `agency_observed` | observed | Level 2; NGVD29 |
| Lake Abert elevation | USGS `USGS-10387150`, daily mean `62615` | `agency_observed` | observed | Level 2; NAVD88 |
| Bear Lake elevation | USGS `USGS-10055000`, daily mean `62615` | `agency_observed` | observed | Level 2 input; NAVD88 |
| Bear Lake area and storage | Official USGS E-A-V relation, DOI `10.5066/P14KZS2D` | `official_eav_piecewise_linear` | derived | Class B, Level 4 after all gates pass |
| Lake Powell elevation, area, and total storage | USBR RISE record `2362`, location `393`, items `508`, `4784`, and `509` | `agency_reported` | reported | Class A, Level 4 |
| Flaming Gorge elevation, area, and total storage | USBR RISE record `2300`, location `1535`, items `341`, `4782`, and `337` | `agency_reported` | reported | Class A, Level 4 |

The distinction between these products is scientific, not cosmetic:

- **Observed water elevation** is the provider's water-surface-elevation
  observation.
- **Agency-reported total storage** is published directly by USBR RISE. It is
  not reconstructed from elevation.
- **Official-table-derived storage** is calculated only for Bear Lake from an
  observed elevation and the official USGS relation.
- **Unavailable** means that a profile has no authoritative Phase 1 pathway for
  that metric. It is serialized as null, never zero.

Lake Powell item `509` is validated total reservoir storage and therefore
supports current Level 4/Class A capability. The unresolved relationship
between RISE elevation and the NAVD88 capacity product blocks independent
capacity-table recomputation and quantitative depth modeling, not the direct
storage series. Flaming Gorge item `337` is total reservoir storage. Item
`4275` remains separately identified as **bank storage** and never populates a
total-storage product.

Great Salt Lake's arms remain independent subprofiles. Phase 1 never computes a
combined level, area, or storage for the lake. Lake Abert remains level-only.

## Official Bear Lake E-A-V source

The reference workflow resolves DOI `10.5066/P14KZS2D` through structured USGS
ScienceBase metadata, validates the release title and lake identity, inventories
the advertised files, and selects exactly one documented small
elevation-area-volume CSV. It does not select the first CSV by order and does
not download the 0.5 m topobathymetric raster.

The selected source must match the documented Bear Lake filename pattern and
CSV schema. The parser preserves source values and validates:

- finite elevation, area, and storage;
- strictly increasing, non-duplicated elevations;
- nondecreasing area and storage;
- nonnegative area and storage;
- documented NAVD88 vertical datum and units;
- Bear Lake identity, release version, and SHA-256;
- at least two valid rows.

The normalized table is written to
`data/hydrometry/phase1/bear_lake_eav_normalized.csv`. It retains source-row,
original-value, original-unit, DOI, filename, version, datum, and normalized SI
and operational-unit fields. Validation statistics are written to
`bear_lake_eav_validation.csv`. A separate adjacent-row diagnostic compares the
observed storage increment with trapezoidal mean area times elevation increment;
it is descriptive because published precision and interpolation can prevent
machine-zero agreement.

One official FGDC entity description calls the CSV a relation for the "north
arm of Bear Lake." That wording conflicts with the release title and abstract,
which identify Bear Lake as a whole, and with the table's 287.08 km² maximum
area (consistent with the whole-lake envelope). Phase 1 treats this as a likely
source-metadata inconsistency, retains it explicitly in run provenance, and does
not silently reinterpret the table as an arm-only product.

Reference files and metadata are retained below
`data/hydrometry/reference/bear_lake/`. The manifest records the DOI, landing and
metadata URLs, download URL, release title and version, filename, content type
and length, retrieval time, provider checksum, SHA-256, license, source units,
datum, parser version, and local content-addressed paths. Writes use temporary
files and atomic replacement, so a failed refresh cannot replace a prior valid
reference.

## Units and vertical-reference gate

Physical conversions use centralized, tested constants for feet and metres,
acres and square metres, square kilometres and square metres, and acre-feet and
cubic metres. Original values and units remain available alongside normalized
values. A physical unit conversion does not constitute a vertical-datum
transformation.

Bear Lake derivation is allowed only when all of the following are explicit:

1. the USGS series is absolute water-surface elevation, not local gage height;
2. its variable and parameter metadata identify water elevation;
3. its source unit is supported;
4. its vertical datum is NAVD88;
5. the official E-A-V table is NAVD88; and
6. the relation passes all identity, unit, hash, and monotonicity checks.

The compatible path performs only a feet-to-metres conversion and records
`compatible_no_vertical_transformation`. An unknown datum, mismatched datum,
local gage height, wrong variable, or unsupported unit blocks derivation. The
workflow does not infer compatibility from similar numerical ranges.

## Bounded interpolation

Bear Lake area and storage use adjacent official table nodes and ordinary
piecewise-linear interpolation. Exact-node elevations reproduce the source node
exactly. An interior elevation cannot overshoot either bracket.

The implementation has no polynomial or PCHIP path and never extrapolates:

- below-table input returns `below_supported_range` with null area and storage;
- above-table input returns `above_supported_range` with null area and storage;
- null input remains null;
- a nonfinite input fails clearly.

Every derived result carries its input and normalized elevation, bracket
elevations, interpolation fraction, range status, method, Class B/Level 4
labels, table DOI, hash, version, and warning flags. Local segment sensitivities
`d_area_dh`, `d_storage_dh`, and storage change per 0.01 m are also recorded.
These slopes describe local sensitivity and are not uncertainty intervals.

## Daily and manager schemas

The long-form daily file uses schema version
`hydrometry-phase1-daily-v1` and one row per
`subprofile_id × calendar_date × metric`. Supported metrics are
`water_elevation`, `surface_area`, and `total_storage`. The stable record groups
are:

- identity: profile, subprofile, HydroLAKES ID, display name, and state;
- time: UTC source timestamp and calendar date;
- value: normalized and original values and units;
- vertical semantics: datum and height type;
- scientific semantics: metric, method, observation type, evidence class, and
  capability level;
- source identity: provider, record, location, item, and parameter identifiers;
- status: provider quality, approval, provisional, range, availability,
  freshness, and warning flags;
- uncertainty: status and nullable bounds;
- capacity provenance: source DOI, hash, and version; and
- Bear-specific interpolation brackets and sensitivity diagnostics.

Provider series are aligned by calendar date with an outer-join interpretation.
Values are not forward-filled and missing days are not interpolated. A provider
null remains unavailable and is never changed to zero. Unsupported area and
storage products are explicit null rows in current conditions, but are not
fabricated as daily observations.

The manager contract uses schema version
`hydrometry-manager-summary-v1` and has this stable top-level shape:

```json
{
  "schema_version": "hydrometry-manager-summary-v1",
  "analysis_start": "2016-01-01",
  "analysis_end": "2026-08-13",
  "generated_at": "...",
  "profiles": [],
  "provenance": {}
}
```

Each profile includes `identity`, `capabilities`, `current_conditions`,
`change_metrics`, `seasonal_context`, `sources`, `warnings`, and `provenance`.
Conditions retain subprofile, metric, method, source, quality, freshness, and
availability so a future client cannot silently substitute one product for
another.

Phase 1 writes the following ignored local outputs under
`data/hydrometry/phase1/`:

```text
pilot_daily_hydrometry.csv
pilot_current_conditions.csv
pilot_change_metrics.csv
pilot_seasonal_context.csv
pilot_manager_summary.json
pilot_phase1_provenance.json
bear_lake_eav_normalized.csv
bear_lake_eav_validation.csv
bear_lake_eav_area_volume_diagnostic.csv
```

Output SHA-256 values, source/reference provenance, execution mode, request
count, profile selection, pinned dates, vertical-gate decision, Bear range
counts, and direct-versus-derived method declarations are recorded in the
Phase 1 provenance file.

## Manager metrics

### Change windows

For each metric/unit/method series, the workflow chooses the latest valid value
at or before the configured analysis end. For each 7-, 30-, 90-, or 365-day
window it targets `latest date - window days` and selects the nearest actual
observation within three days. It does not interpolate. The result retains the
target and actual baseline dates, requested and actual elapsed days, baseline
and latest values, absolute change, and comparison status. Percent change is
reported only for a finite, nonzero baseline.

Values with different metrics, units, or methods cannot be compared. A missing
baseline produces an explicit unavailable status.

### Seasonal context

The seasonal reference excludes the latest observation's calendar year and
uses prior-year observations within ±15 calendar days of its month/day. Leap
day is handled with circular calendar-day distance rather than interpolation.
At least five distinct years and 20 observations are required. When supported,
the output reports empirical percentile, median, Q25, Q75, minimum, maximum,
observation count, and distinct-year count. Otherwise it reports an explicit
status such as `insufficient_years`, `insufficient_observations`, or
`no_latest_observation`.

### Freshness and quality

Freshness is based on days between the latest valid observation and the pinned
analysis end. Seven days or fewer is `current`; a larger gap is `stale`; no
latest observation is `unavailable`. Provider approval and provisional values
remain visible as approved, provisional, mixed, or unknown. Derived Bear Lake
area and storage inherit the source elevation's status.

## Uncertainty

Phase 1 does not invent numeric uncertainty. When no compatible quantitative
stage or relation uncertainty is available, records use
`uncertainty_status = not_quantified` with null lower and upper bounds. The
Bear Lake local interpolation sensitivities remain diagnostics and must not be
presented as confidence intervals. Direct RISE provisional status is retained
rather than translated into a numeric error.

## Figures and manager report

Unless `--skip-plots` is used, matplotlib writes four figure families below
`outputs/hydrometry_phase1/`:

1. pilot elevation and storage time series;
2. the Bear Lake official elevation-area and elevation-storage relations with
   the observed stage range;
3. absolute and percent 7/30/90/365-day storage changes; and
4. seasonal percentiles with insufficient-history cases retained.

Each figure is saved as a 300 dpi PNG, an SVG, and matching PNG/SVG copies in
`no_text/` with textual layers removed while graphical geometry is retained.
`figure_notes.md` documents observed, reported, and derived methods.
`manager_summary.md` uses the required distinctions **Observed water
elevation**, **Agency-reported total storage**, **Official-table-derived
storage**, and **Unavailable**.

## Live, offline, and cache operation

Network behavior is always explicit. From an activated project environment:

```powershell
python -m regional_analysis.hydrometry.phase1 `
    --config configs\hydrometry_pilots.yaml `
    --live `
    --start 2016-01-01 `
    --end 2026-08-13
```

An exact cached replay uses:

```powershell
python -m regional_analysis.hydrometry.phase1 `
    --config configs\hydrometry_pilots.yaml `
    --offline `
    --start 2016-01-01 `
    --end 2026-08-13
```

On the configured Windows machine, the environment can also be invoked
directly:

```powershell
& "C:\Code\Envs\lake-scaling\python.exe" `
    -m regional_analysis.hydrometry.phase1 `
    --config configs\hydrometry_pilots.yaml `
    --offline `
    --start 2016-01-01 `
    --end 2026-08-13
```

`--refresh` refreshes provider responses and `--refresh-reference-data`
refreshes the official Bear Lake table. Neither is legal with `--offline`.
Other options select profiles or override output, provider-cache, reference,
and figure directories. `--skip-plots` writes data and the manager report
without figures. The CLI requires `--live` or `--offline`; it never makes an
implicit network request.

Provider responses use the shared bounded, atomic Phase 0 cache under
`data/hydrometry/cache/`. Reference data use
`data/hydrometry/reference/`. Both are ignored by Git and must be retained
locally for offline replay. Public URLs and manifests contain no API keys,
authorization headers, cookies, or other credentials.

## Known limitations

- Only Bear Lake has an official-table derived storage pathway.
- Great Salt Lake has no datum conversion or combined-arm product.
- Lake Abert has no authoritative area/storage relation.
- Lake Powell and Flaming Gorge storage are not recomputed from capacity or
  ACAP tables.
- Phase 1 does not quantify uncertainty, propagate provider errors, integrate
  bathymetry, produce 3D geometry, or perform independent storage validation.
- No SWOT, ICESat-2, Earth Engine, Landsat, OPERA, JRC, or classification work
  is performed.
- Missing provider observations are not interpolated.

## QGIS boundary

The QGIS plug-in is unchanged. The manager JSON is a future service contract,
not a plug-in-side registry or GUI implementation. A later integration requires:

- asynchronous retrieval using `QgsTask`;
- visible source and method labels;
- an explicit direct-versus-derived distinction;
- provisional/approved status and freshness;
- clear unavailable-product handling;
- no custom-AOI storage fallback;
- no storage display without a validated hydrometry profile; and
- manual QGIS regression testing that confirms classification behavior is
  unchanged.

## Phase 2 decision gates

Phase 2 must not begin merely because Phase 1 produces manager outputs.

### Bear Lake capacity pathway

Proceed only if the official source is unambiguous, the table validates, the
stage/table datum match is explicit, interpolation tests pass, no extrapolation
occurs, and live-to-offline reproduction is exact.

### Lake Powell capacity validation

Do not recalculate storage until the RISE elevation datum is fully documented,
its relationship to the NAVD88 elevation-area-capacity product is established,
and survey version and sedimentation implications are documented.

### Flaming Gorge ACAP validation

Do not recalculate storage until the ACAP table/file is acquired, its project
datum is documented, the elevation item and ACAP relation share a verified
datum, and RISE total-storage item `337` can serve as an independent validation
source.

### Great Salt Lake

Do not derive volume until the NGVD29-to-NAVD88 relationship is documented,
north and south arms are modeled separately, and bathymetric survey age and
uncertainty are incorporated.

### QGIS integration

Do not integrate hydrometry until the Phase 1 service contract is stable,
live/offline/cache behavior is validated, provider failures are represented
cleanly, a `QgsTask` design is reviewed, source and method provenance can be
shown visibly, and existing classification behavior remains unchanged.

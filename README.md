# Lake Classifier QGIS Plug-In - develop-Water_Virtual_2026_1

[![Python: 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EarthRISE: Development](https://img.shields.io/badge/EarthRISE-Development-b50000?labelColor=191f4c)](https://appliedsciences.nasa.gov/what-we-do/capacity-building/develop)

> NASA EarthRISE Developers Academy project level repo

## Overview

This repository is part of the **NASA EarthRISE Developers Academy** — a program
focused on applying Earth observation data and geospatial tools to real-world
challenges across environmental, agricultural, urban, and planetary domains.

This project analyzes water extent, wetland dynamics, and evapotranspiration for
**Lake Abert, OR** and the **Great Salt Lake, UT** using Landsat imagery, in
partnership with the Oregon Department of Water Resources, Surface Water Section
and the Utah Department of Natural Resources, Division of Water Resources.

The study period runs **January 1985 through May 2026**. Classification uses
summer composites (June–September); the historical water-extent time series uses
April–October (see below).

> **On the "wetland" class:** it identifies *moist, vegetated ground at the time
> of the scene*, not wetlands in the conventional or regulatory sense. The name
> was kept at the partners' request. Read
> [What the "wetland" class actually measures](#what-the-wetland-class-actually-measures)
> before using or citing that output.

It has three deliverables:

1. **A QGIS plug-in** (`Lake_Classifer_FD/`) — the primary deliverable. Applies
   the calibrated thresholds interactively and ships with per-lake defaults.
2. **Threshold calibration** (`final_threshold_optimization.py`) — spatial-block
   cross-validated NDWI / NDVI / NDMI cutoffs, validated against JRC Global
   Surface Water, OPERA DSWx-HLS, Dynamic World, NWI, and NLCD. This is what
   produced the plug-in's defaults.
3. **A monthly water-extent time series** (`water_extent_timeseries.py`) for
   each lake, overlaid with USGS water-surface-elevation gauges, plus a
   stage-area relationship.

The plug-in is self-contained: it does not import either script, and runs
without them.

## Project

| Field        | Value                             |
|--------------|-----------------------------------|
| Program      | NASA EarthRISE Developers Academy |
| Year         | 2026                             |
| Repository   | develop-Water_Virtual_2026_1                             |
| Status       | In development                    |
| Team         | Katie Bossert, Sofia Garcia, Dean Gelling, Vivian Maneval, Kayla Sundling |
| Study sites  | Great Salt Lake, UT · Lake Abert, OR |

## Repository structure

```
.
├── README.md
├── requirements.txt                    # standard EarthRISE geospatial stack
├── LICENSE_PACKAGES.txt
│
├── final_threshold_optimization.py     # NDWI/NDVI calibration + validation figures
├── threshold_optimization_ndmi.py      # NDMI calibration vs NWI + NLCD
├── water_extent_timeseries.py          # monthly water extent + stage-area
│
├── Lake_Classifer_FD/                  # the QGIS plug-in
│   ├── __init__.py
│   ├── metadata.txt                    # QGIS plug-in manifest
│   ├── lake_classifier.py              # plug-in entry point, menu/toolbar wiring
│   ├── dock.py                         # the whole UI panel + local classification
│   └── ee_backend.py                   # all Earth Engine calls, GeoTIFF download
│
├── threshold_grid_search_demo.ipynb    # threshold sweep walkthrough
├── csda_Planet_fetcher_final.ipynb     # CSDA / Planet imagery fetcher
├── notebook.ipynb                      # org starter template (unused)
│
├── GEE_Animation                       # GEE JS — MODIS NDVI animation (T. Mayer)
├── example_SLC.js                      # GEE JS — Lake Abert ROI example (T. Mayer)
├── main.js                             # org starter template (unused)
│
└── past_project/                       # 2018 CO Utah & Colorado Water materials
```

> Note: `Lake_Classifer_FD` is missing an "i" in "Classifier". It is left as-is
> because the directory name is the QGIS plug-in folder name; renaming it means
> reinstalling the plug-in.

## Prerequisites

```bash
pip install -r requirements.txt
```

`requirements.txt` is the shared EarthRISE stack and is much broader than this
project needs. The scripts here actually require: `earthengine-api`, `earthaccess`,
`numpy`, `pandas`, `scikit-learn`, `rasterio`, `pyproj`, `shapely`, `matplotlib`,
`requests`, and GDAL (for the plug-in's classification step).

**Google Earth Engine.** You need a registered Cloud project. Both scripts call
`ee.Authenticate()` on first run and read the project ID from the `EE_PROJECT`
environment variable, so no identifier is stored in the repository:

```bash
export EE_PROJECT=your-ee-project-id
```

Without it they fall back to the literal `your-ee-project-id`, which will fail
at `ee.Initialize()` — that failure is intentional, and is the cue to set the
variable. The QGIS plug-in asks for the project ID in its own field instead and
remembers it via `QgsSettings`.

**Local data.** No machine-specific paths are baked into the source. Two inputs
live outside the repo and are located by environment variable:

| Variable | Default | Needed for |
|----------|---------|------------|
| `THRESHOLD_CSV` | `data/threshold_cv_summary.csv` | `water_extent_timeseries.py` |
| `NWI_DIR` | `data/nwi/` | NDMI/wetland calibration only |

See [`data/README.md`](data/README.md) for what belongs there and why it is not
committed.

**NASA Earthdata.** `final_threshold_optimization.py` streams OPERA DSWx-HLS
granules through `earthaccess`, which authenticates from `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

## Where the plug-in's default thresholds come from

The two scripts at the repository root are **calibration tools, not part of the
plug-in**. They produced the `DEFAULT_THRESHOLDS` baked into
`Lake_Classifer_FD/dock.py`, and exist here as the evidence trail for those
numbers. The plug-in does not import them and runs without them; you only need
them to re-derive or update a threshold.

Each shipped default traces to a specific calibration row. IoU below is the
test-set value — the one-time evaluation against the untouched test blocks,
which the technical report describes as "the defensible, final performance of
each index/reference threshold combination." These are the numbers published in
Table 2 of the technical report.

| Site | Index | Default | Reference | IoU | F1 |
|------|-------|---------|-----------|-----|-----|
| Great Salt Lake | NDWI | −0.055 | OPERA DSWx-HLS | 0.793 | 0.885 |
| Great Salt Lake | NDVI | +0.165 | Dynamic World | 0.780 | 0.877 |
| Great Salt Lake | NDMI | +0.053 | NWI | 0.092 | 0.169 |
| Lake Abert | NDWI | −0.063 | OPERA DSWx-HLS | 0.655 | 0.792 |
| Lake Abert | NDVI | +0.130 | Dynamic World | 0.705 | 0.827 |
| Lake Abert | NDMI | +0.123 | NWI | 0.216 | 0.355 |

**NDWI uses OPERA at both sites.** JRC scores higher on the test set (0.870 at
Great Salt Lake, 0.844 at Lake Abert), but as the technical report states,
"OPERA was selected as the primary reference dataset for threshold optimization
due to its frequent updates and greater availability of current data compared to
JRC… While JRC showed higher agreement at both sites, OPERA's classification
performance remained strong, supporting its use as the primary threshold dataset
given its advantages in temporal resolution and update frequency."

The JRC-fit thresholds (+0.0151 at Great Salt Lake, +0.084 at Lake Abert) were
produced for the historical time-series analysis and are **not** used by the
plug-in. Only Great Salt Lake actually uses its JRC threshold downstream — see
[`water_extent_timeseries.py`](#water_extent_timeseriespy) for why Lake Abert's
historical series uses OPERA instead.

NDVI comes from Dynamic World, "used exclusively to calibrate NDVI thresholds by
distinguishing vegetated land cover classes from non-vegetated classes."

NDMI comes from [`threshold_optimization_ndmi.py`](threshold_optimization_ndmi.py),
run against the NWI raster with NLCD 90/95 as a second reference. As the
technical report describes it: "Because the NWI reference dataset offers limited
temporal coverage, a single Landsat composite spanning 2010-2020 was used for
NDMI threshold optimization, chosen to encompass the years in which wetlands
were mapped within the study areas. Open water, classified from the NDWI
threshold, was masked from the composite prior to sampling, since NDMI otherwise
misclassifies open water as wetland."

| Setting | Value |
|---------|-------|
| Window | Summers (Jun–Sep), `START_YEAR` 2010 to `END_YEAR` 2020 |
| Missions | Landsat 8 and 9 only — **not** L5/L7 |
| Composite | Per-pixel **median** (50th percentile) of NDMI |
| Sampling | **4,500 points per site from one static composite** — this is a single-composite sample, not 200 points per scene |
| Open-water mask | Median-NDWI composite above the site's JRC-optimized NDWI\* (+0.0151 Great Salt Lake, +0.0839 Lake Abert) |

Note that **the effective window is narrower than 2010–2020**. Landsat 8 returns
no data before 2013 and Landsat 9 none before 2022, so with `END_YEAR = 2020` the
composite is built from **Landsat 8 only, summers 2013–2020**; Landsat 9
contributes nothing and the first three years are empty.

The report's NDMI sampling description — "Two-hundred-pixel locations were then
randomly sampled from the masked composite" — does not match the code. The
200-point figure is the per-scene sampling used for NDWI and NDVI. NDMI samples
one static composite 4,500 times, which is what the committed table reflects:
4,232 usable points at Lake Abert and 3,648 at Great Salt Lake.

**Read the NDMI row with caution.** Its test IoU is 0.092 at Great Salt Lake and
0.216 at Lake Abert, against 0.66–0.79 for NDWI and 0.71–0.78 for NDVI. As the
technical report puts it, NDMI "produced substantially lower accuracy, which
highlights the challenges of correctly mapping wetlands using a single spectral
index as well as showing the limitations of available wetland reference
datasets." The wetland class should not be presented with the same confidence as
open water — and see
[What the "wetland" class actually measures](#what-the-wetland-class-actually-measures)
before interpreting or reporting it at all.

Two things to know before recalibrating:

* A 99th-percentile composite was also evaluated for NDMI and scored higher at
  Lake Abert (threshold +0.407, IoU 0.423 versus the shipped +0.123 and 0.216)
  but lower at Great Salt Lake (+0.356, IoU 0.085 versus 0.053 and 0.092). The
  median composite was shipped, as it "is more widely accepted as the standard
  for general statistical analyses."
* `final_threshold_optimization.py` writes
  `outputs/tables/final_threshold_summary.csv`, a later calibration whose values
  differ from the shipped defaults — notably a different Abert NDMI (NWI ≈
  +0.006) and no Great Salt Lake NDMI row at all. Re-running it will not
  reproduce the published table.

## Running the calibration scripts

Neither is needed for day-to-day plug-in use. Run them to recalibrate.

### `final_threshold_optimization.py`

Calibrates one threshold per index × reference pair, using spatial-block
cross-validation: points are tagged into 1 km UTM blocks, whole blocks go to
train or test (80/20), and the threshold is picked by GroupKFold CV inside train
and scored on held-out blocks.

```bash
python final_threshold_optimization.py
```

Behaviour is controlled by flags at the top of the file:

| Flag | Default | Effect |
|------|---------|--------|
| `RUN_OPTIMIZATION` | `False` | `True` re-samples and re-fits. `False` is a figures-only pass that reads thresholds from `THRESHOLD_SOURCE_CSV`. |
| `RUN_RESIDUAL_MAPS` | `True` | Residual maps + confusion matrices vs OPERA / Dynamic World. |
| `RUN_TIMESERIES` | `False` | ~500 monthly composites — slow. |
| `RUN_EPOCH_CUTS` | `True` | Re-fits per 5-year epoch and within the OLI era, to test threshold drift. |

Outputs land under `outputs/`:

```
outputs/
├── tables/     final_threshold_summary.csv, threshold_epochs.csv, roc_points.csv
└── figures/    roc/  residual_maps/  confusion_matrices/  threshold_drift/  timeseries/
```

Every figure is written as both PNG and SVG, and each folder carries a
`no_text/` twin with all labels stripped for annotation in Illustrator or Canva.

> A figures-only pass (`RUN_OPTIMIZATION = False`) cannot redraw the ROC curves
> unless `outputs/tables/roc_points.csv` exists — the per-point test scores live
> only in memory during a fitting run.

### `water_extent_timeseries.py`

Builds one median composite per calendar month and measures water extent as
`NDWI >= x AND NDVI < y`. Overlays USGS gauge elevation and produces a
stage-area scatter with Spearman rank correlation.

**This analysis uses a different NDWI reference than the plug-in, per site:**

| Site | NDWI reference | Threshold | Why |
|------|----------------|-----------|-----|
| Great Salt Lake | JRC | +0.0151 | Gauge record starts 1999; 144 of its 175 gauge-paired months fall inside JRC's 1984–2021 window |
| Lake Abert | OPERA | −0.0626 | Gauge came online in 2023, so all 20 gauge-paired months fall **after** JRC stopped being compiled in 2021 |

JRC is the historical reference — it is the only one spanning the long record —
but it cannot calibrate an analysis of a period it does not cover. Lake Abert's
stage-area analysis is entirely 2023–2026, so OPERA (2016–present) is the
appropriate reference there. This is a coverage argument, not a performance one.

These are the settings that produced the published stage-area figures: Lake
Abert ρ = 0.44 (n = 20), Great Salt Lake south arm ρ = 0.87 (n = 175), north arm
ρ = 0.83 (n = 175). The pairing is pinned in `INDEX_REFERENCE` /
`INDEX_REFERENCE_BY_SITE` rather than inferred from scores, so it cannot drift.

```bash
python water_extent_timeseries.py            # full pass — hits Earth Engine
python water_extent_timeseries.py --replot   # redraw figures from existing CSVs
```

`--replot` makes **no Earth Engine calls**. It reuses the water extents already
in `water_extent_monthly_<site>.csv`, refreshes only the USGS gauge columns, and
redraws — seconds rather than tens of minutes.

Outputs, written next to the script:

```
water_extent_monthly_<site>.csv      one row per month, incl. gauge elevation
water_extent_monthly_all.csv         both sites concatenated
water_extent_timeseries_<site>.png   extent over time + gauge overlay
stage_area_<site>.png                extent vs elevation + Spearman rho
```

Key settings:

| Constant | Value | Why |
|----------|-------|-----|
| `ANALYSIS_MONTHS` | `(4, 10)` | As published: "Due to snow and ice artifacts, surface water extent was calculated only in the period from April through October, and some outliers remain, likely also caused by snow events inside that period." At Lake Abert the excluded months reach 2215 km² against a ~150 km² lake, on fully observed scenes. Out-of-season months are never queried. |
| `MIN_VALID_FRACTION` | `0.5` | Months observing less of the AOI are excluded from the figures — an extent extrapolated from under half the AOI is a guess, not a measurement. |
| `MAX_WORKERS` | `2` | See Earth Engine Restricted Mode below. |
| `START_DATE` | `1999-01-01` | Landsat 7 launch. |

The seasonal window is applied at query time, so a full pass requests 193 months
per site instead of ~331 — a 42% reduction in Earth Engine calls.

## Earth Engine Restricted Mode

Unregistered or unverified Cloud projects run in **Restricted Mode**, which
permits only a couple of concurrent `getInfo()` calls. Exceeding it returns:

```
Too Many Requests: Exceeded Earth Engine concurrency limit.
Your project is in Restricted Mode.
```

`water_extent_timeseries.py` handles this with `MAX_WORKERS = 2`, six retries
per month with exponential backoff and jitter, and an explicit end-of-run report
listing any month that never came back. A throttled month is reported as lost —
it is never silently confused with a month that genuinely has no imagery.

Registering the project for noncommercial use removes the restriction, after
which `MAX_WORKERS` can go back to 10.

## QGIS plug-in

Copy `Lake_Classifer_FD/` into your QGIS profile plug-in directory:

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |
| Windows | `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` |
| Linux | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |

Then enable **Saline Lake Classifier** in *Plugins → Manage and Install Plugins*.
The plug-in needs `earthengine-api` available inside QGIS's Python — the
*Google Earth Engine* plug-in is the usual way to get it.

The panel runs top to bottom:

| Stage | Does |
|-------|------|
| 1 · Area of Interest | Preset lake, a vector file, or a box drawn on the canvas |
| 2 · Outputs | True Color / LST / NDWI / NDVI / NDMI for a summer (Jun–Sep) composite |
| 3 · Classification | Three threshold sliders → open water / wetland / land, with class areas |
| 4 · Time Series | Monthly NDWI/NDVI/NDMI CSV for the selected AOI and year |
| 5 · OpenET | Evapotranspiration — **CONUS only** |

Classification is applied locally with GDAL/NumPy. The three classes are
mutually exclusive and assigned in this order:

```
open water : NDWI >= ndwi_threshold
wetland    : NOT water AND NDVI >= ndvi_threshold AND NDMI >= ndmi_threshold
land       : NOT water AND NOT wetland
```

Land is the residual, so no threshold combination can leave a valid pixel
unclassified. Thresholds default to per-lake calibrated values and can be
adjusted live — releasing a slider re-runs the classification.

### What the "wetland" class actually measures

**The wetland class does not identify wetlands in the conventional sense, and
the label should not be read that way.**

The class was trained against wetland reference data (NWI, NLCD), but what the
NDVI + NDMI combination actually detects is **land that is moist and vegetated
at the moment the scene was acquired**. That is not the same thing as a wetland.

The gap is hydroperiod. A wetland is designated a wetland year-round, even
though many are saturated for only part of the year. The reference data encodes
that permanent designation; a spectral index can only report the instantaneous
surface state. Because these classifications are built from single static
composites, they describe a point in time, not a hydrologic regime.

Two consequences follow, and they run in opposite directions:

* A genuine wetland observed outside its saturated months reads as **land** —
  it is dry and possibly senesced at the time of the scene.
* Anything moist and green that is not a wetland — irrigated fields, wet
  meadows, riparian margins, post-precipitation ground — reads as **wetland**.

So disagreement with NWI is expected in both directions and is not by itself
evidence that a threshold is wrong. It also means the class cannot be summed
across dates and interpreted as wetland extent change; consecutive scenes may
differ because the ground dried out, not because wetland was lost.

The name "wetland" was retained **at the partners' request**. A description
closer to what is measured would be *moist vegetated surface*, and that is how
the class should be characterised in any writeup, caption, or briefing where a
reader might otherwise take it as a wetland delineation.

This is also why the NDMI cutoff's weak cross-validated skill (see
[the threshold provenance section](#where-the-plug-ins-default-thresholds-come-from))
matters less than it first appears: the index is not failing to find wetlands so
much as being asked a question it cannot answer from one scene.

## Known gaps

Tracked here rather than hidden. None of these prevent the code from importing —
both scripts and the plug-in are self-contained — but each needs attention:

1. **`data/` inputs are not distributed with the repo** by design — see
   [`data/README.md`](data/README.md). The threshold table and the NWI rasters
   have to be supplied separately. Both paths are configurable (`THRESHOLD_CSV`,
   `NWI_DIR`) and both scripts fail with an explicit message naming the missing
   file rather than a bare traceback.
2. **`threshold_cv_summary.csv` has no producer in this repo.** It came from the
   superseded `threshold_optimization_cv.py`. The current calibration script
   writes `outputs/tables/final_threshold_summary.csv`, which is
   column-compatible — consider standardising on it, bearing in mind it is a
   later calibration and will change the resulting extents.
3. **`Lake_Classifer_FD/metadata.txt` has unfilled placeholders** — `email`,
   `repository`, and `tracker` still read `<insert>` / `<team email or primary
   contact>`. QGIS surfaces these in the plug-in manager.

Smaller items, documented inline in the source:

- `dock.py` — `_fetch_openet` returns its success message instead of writing it
  to the status label, so the label stays on "Fetching…" after a successful
  fetch. It also assigns the OpenET layer to `self.lst_layer`, overwriting the
  LST handle.
- `dock.py` — `_classify_raster` silently drops the NDMI condition if the index
  raster has fewer than three bands, widening the wetland class. Not reachable
  with the current `ee_backend`, which always emits three bands.

## License

NASA / EarthRISE Developers Academy — 2026

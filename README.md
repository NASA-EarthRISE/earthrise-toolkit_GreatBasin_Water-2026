# Lake Classifier QGIS Plug-In - develop-Water_Virtual_2026_1

[![Python: 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EarthRISE: Development](https://img.shields.io/badge/EarthRISE-Development-b50000?labelColor=191f4c)](https://appliedsciences.nasa.gov/what-we-do/capacity-building/develop)

> NASA EarthRISE Developers Academy project level repo

## Overview

Saline terminal lakes in the Great Basin provide critical socio-economic and
ecological benefits. These lakes have undergone substantial long-term decline due
to increased water demand, groundwater extraction, and climate change. This
project partnered with the Oregon Department of Water Resources, Surface Water
Section and Utah Department of Natural Resources, Division of Water Resources to
analyze water extent, wetland dynamics, and evapotranspiration for Lake Abert and
the Great Salt Lake using Landsat imagery from 1985 to 2026.

To support basin-wide water and ecosystem management, we created a QGIS plug-in
with a user-friendly graphical interface (GUI) to generate seasonal composites of
Landsat-derived environmental indices. The workflow uses Landsat 5, 7, 8, and 9
imagery to generate seasonal composites, calculate NDWI, NDVI, NDMI, and Land
Surface Temperature (LST), and classify land cover into open water, wetlands, and
terrestrial land.

The study period extends from January 1985 through May 2026. Classification uses
summer composites (June through September). The historical water-extent time
series uses April through October, for reasons described below.

> **On the "wetland" class.** It identifies moist, vegetated ground at the time
> the scene was acquired. It does not identify wetlands in the conventional or
> regulatory sense, and the name was kept at the partners' request. Please read
> [What the "wetland" class actually measures](#what-the-wetland-class-actually-measures)
> before using or citing that output.

The repository holds three deliverables:

1. **A QGIS plug-in** (`Lake_Classifer_FD/`), the primary product. It applies the
   calibrated thresholds interactively and ships with per-lake defaults.
2. **Threshold calibration** (`final_threshold_optimization.py` and
   `threshold_optimization_ndmi.py`), using optimized thresholds calibrated
   against OPERA Dynamic Surface Water Extent, Google Dynamic World, and the
   National Wetlands Inventory. `threshold_optimization_ndmi.py` produced the
   shipped NDMI defaults. The shipped NDWI and NDVI defaults came from an earlier
   run whose script is not in this repository, which is covered under
   [Known gaps](#known-gaps).
3. **A monthly water-extent time series** (`water_extent_timeseries.py`) for each
   lake, overlaid with USGS water-surface-elevation gauges, plus a stage-area
   relationship.

The plug-in is self-contained. It does not import the calibration scripts and
runs without them.

## Project

| Field        | Value                             |
|--------------|-----------------------------------|
| Program      | NASA EarthRISE Developers Academy |
| Year         | 2026                             |
| Repository   | develop-Water_Virtual_2026_1                             |
| Status       | In development                    |
| Team         | Katie Bossert, Sofia Garcia, Dean Gelling, Vivian Maneval, Kayla Sundling |
| Partners     | ODWR Surface Water Section · UDWR Division of Water Resources |
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
├── GEE_Animation                       # GEE JS, MODIS NDVI animation (T. Mayer)
├── example_SLC.js                      # GEE JS, Lake Abert ROI example (T. Mayer)
├── main.js                             # org starter template (unused)
│
└── past_project/                       # 2018 CO Utah & Colorado Water materials
```

## Prerequisites

```bash
pip install -r requirements.txt
```

`requirements.txt` is the shared EarthRISE stack and covers far more than this
project uses. The scripts here require `earthengine-api`, `earthaccess`, `numpy`,
`pandas`, `scikit-learn`, `rasterio`, `pyproj`, `shapely`, `matplotlib`,
`requests`, and GDAL for the plug-in's classification step.

**Google Earth Engine.** To use GEE, an active Google account and a registered
Google Cloud Project are required. To create a new Cloud Project, log into the
Google Earth Engine Code Editor at https://code.earthengine.google.com/, click
the user icon in the upper-right corner, and select "Register a new Cloud
Project". Cloud Project names cannot be changed later.

The scripts call `ee.Authenticate()` on first run and read the project ID from
the `EE_PROJECT` environment variable, so no identifier is stored in the
repository:

```bash
export EE_PROJECT=your-ee-project-id
```

If the variable is unset, the scripts fall back to the literal
`your-ee-project-id` and fail at `ee.Initialize()`.

**Local data.** No machine-specific paths are baked into the source. Two inputs
live outside the repository and are located by environment variable:

| Variable | Default | Needed for |
|----------|---------|------------|
| `THRESHOLD_CSV` | `data/threshold_cv_summary.csv` | `water_extent_timeseries.py` |
| `NWI_DIR` | `data/nwi/` | NDMI/wetland calibration only |

See [`data/README.md`](data/README.md) for what belongs there and why it stays
out of version control.

**NASA Earthdata.** `final_threshold_optimization.py` streams OPERA DSWx-HLS
granules through `earthaccess`, which authenticates from `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

**QGIS.** The tutorial and plug-in were developed using QGIS 3.44 LTR, available
at https://www.qgis.org/download/. Older versions are in the "Archive" section of
that page.

## Where the plug-in's default thresholds come from

The calibration scripts at the repository root are tools for deriving thresholds.
They are not part of the plug-in, and you only need them to re-derive or update a
threshold.

Each of the six `DEFAULT_THRESHOLDS` in `Lake_Classifer_FD/dock.py` traces to one
of two committed tables:

| Shipped default | Table | Producer |
|-----------------|-------|----------|
| NDWI, NDVI (all four values) | `data/final_threshold_summary.csv` | `final_threshold_optimization.py` |
| NDMI (both values) | `data/ndmi_threshold_cv_all.csv` | `threshold_optimization_ndmi.py` |

### Which script owns which index

The two calibration scripts divide the work, and a complete threshold set is the
union of their outputs:

| Index | Owner | Why |
|-------|-------|-----|
| NDWI, NDVI | `final_threshold_optimization.py` | Per-scene sampling time-joined to JRC, OPERA, and Dynamic World |
| NDMI | `threshold_optimization_ndmi.py` | Its narrower 2010–2020 Landsat 8/9 window succeeds at both sites |

Each shipped default traces to a specific calibration row. The IoU column below
holds test-set values, from the one-time evaluation conducted against the
untouched test data after the mean threshold was finalized for each index. These
test-set metrics represent the defensible, final performance of each
index/reference threshold combination, and they are the numbers published in
Table 2 of the tech paper.

| Site | Index | Default | Reference | IoU | F1 |
|------|-------|---------|-----------|-----|-----|
| Great Salt Lake | NDWI | −0.055 | OPERA DSWx-HLS | 0.793 | 0.885 |
| Great Salt Lake | NDVI | +0.165 | Dynamic World | 0.780 | 0.877 |
| Great Salt Lake | NDMI | +0.053 | NWI | 0.092 | 0.169 |
| Lake Abert | NDWI | −0.063 | OPERA DSWx-HLS | 0.655 | 0.792 |
| Lake Abert | NDVI | +0.130 | Dynamic World | 0.705 | 0.827 |
| Lake Abert | NDMI | +0.123 | NWI | 0.216 | 0.355 |

IoU is the primary metric because it penalizes false positives and false
negatives symmetrically and is a common standard for surface-water and land-cover
mapping accuracy. F1, precision, and recall are reported alongside it to give a
fuller picture of each threshold's error behavior.

### NDWI uses OPERA at both sites

OPERA was selected as the primary reference dataset for threshold optimization
due to its frequent updates and greater availability of current data compared to
JRC. JRC showed higher agreement at both sites, scoring 0.870 at Great Salt Lake
and 0.844 at Lake Abert, but OPERA's classification performance remained strong,
supporting its use as the primary threshold dataset given its advantages in
temporal resolution and update frequency.

The JRC-fit thresholds, +0.0151 at Great Salt Lake and +0.084 at Lake Abert, were
produced for the historical time-series analysis. The plug-in never uses them.
Only Great Salt Lake uses its JRC threshold downstream when constructing a historical timeseries; see
[`water_extent_timeseries.py`](#water_extent_timeseriespy) for why Lake Abert's
historical series uses OPERA.

### NDVI and NDMI

Dynamic World was used exclusively to calibrate NDVI thresholds, by
distinguishing vegetated land cover classes from non-vegetated classes.

NDMI comes from [`threshold_optimization_ndmi.py`](threshold_optimization_ndmi.py),
run against the NWI raster with NLCD classes 90 and 95 as a second reference.
Because the NWI reference dataset offers limited temporal coverage, a single
Landsat composite spanning 2010-2020 was used for NDMI threshold optimization,
chosen to encompass the years in which wetlands were mapped within the study
areas. Open water, classified from the NDWI threshold, was masked from the
composite prior to sampling, since NDMI otherwise misclassifies open water as
wetland.

| Setting | Value |
|---------|-------|
| Window | Summers (Jun–Sep), `START_YEAR` 2010 to `END_YEAR` 2020 |
| Missions | Landsat 8 and 9 only |
| Composite | Per-pixel median (50th percentile) of NDMI |
| Sampling | 4,500 points per site, drawn once from the single static composite |
| Open-water mask | Median-NDWI composite above the site's JRC-optimized NDWI\* (+0.0151 Great Salt Lake, +0.0839 Lake Abert) |

It is worth noting that the effective window is narrower than 2010–2020: Landsat 8 returns no data before 2013 and Landsat 9 none before 2022, so with `END_YEAR = 2020` the composite draws on Landsat 8 alone across summers 2013–2020, leaving the first three years empty.

### Reading the NDMI row

NDMI's test IoU is 0.092 at Great Salt Lake and 0.216 at Lake Abert, against
0.66–0.79 for NDWI and 0.71–0.78 for NDVI. NDMI produced substantially lower
accuracy, which highlights the challenges of correctly mapping wetlands using a
single spectral index as well as showing the limitations of available wetland
reference datasets. Give the wetland class less confidence than open water in any
writeup, and see
[What the "wetland" class actually measures](#what-the-wetland-class-actually-measures)
before interpreting it.

Two things to know before recalibrating:

* A 99th-percentile composite was also evaluated for NDMI. It scored higher at
  Lake Abert (threshold +0.407, IoU 0.423 against the shipped +0.123 and 0.216)
  and lower at Great Salt Lake (+0.356, IoU 0.085 against 0.053 and 0.092). The
  median composite was shipped, being more widely accepted as the standard for
  general statistical analyses.

## Running the calibration scripts

Neither script is needed for day-to-day plug-in use. Only run them to reoptimize, based on different parameters or AOIs.

### `final_threshold_optimization.py`

Calibrates one threshold per index × reference pair. All sample points are
projected into a local UTM coordinate system and aggregated into 1 km x 1 km
spatial blocks, and those blocks are randomly assigned to training (80%) or test
(20%) subsets. This ensures spatial separation between the data used to fit
thresholds and the data used to evaluate them. Within the training set, each
index's optimal threshold is determined using five-fold cross-validation, with
spatial blocks as the grouping unit. For each fold a grid search runs over the
candidate threshold range, and the final threshold is the mean of the five
fold-level thresholds.

```bash
python final_threshold_optimization.py
```

Behaviour is controlled by flags at the top of the file:

| Flag | Default | Effect |
|------|---------|--------|
| `RUN_OPTIMIZATION` | `False` | `True` re-samples and re-fits. `False` runs a figures-only pass that reads thresholds from `THRESHOLD_SOURCE_CSV`. |
| `RUN_RESIDUAL_MAPS` | `True` | Residual maps and confusion matrices against OPERA and Dynamic World. |
| `RUN_TIMESERIES` | `False` | Roughly 500 monthly composites, so it is slow. |
| `RUN_EPOCH_CUTS` | `True` | Re-fits per 5-year epoch and within the OLI era, to test threshold drift. |

Outputs land under `outputs/`:

```
outputs/
├── tables/     final_threshold_summary.csv, threshold_epochs.csv, roc_points.csv
└── figures/    roc/  residual_maps/  confusion_matrices/  threshold_drift/  timeseries/
```

Every figure is written as both PNG and SVG, and each folder carries a `no_text/`
twin with all labels stripped, for annotation in Illustrator or Canva.

> A figures-only pass (`RUN_OPTIMIZATION = False`) cannot redraw the ROC curves
> unless `outputs/tables/roc_points.csv` exists. The per-point test scores live
> only in memory during a fitting run.

### `water_extent_timeseries.py`

Builds one median composite per calendar month and measures water extent as
`NDWI >= x AND NDVI < y`. It overlays USGS gauge elevation and produces a
stage-area scatter. Since surface elevation and surface area are monotonically
but not linearly related, Spearman rank correlation is used for that
relationship.

This analysis uses a different NDWI reference than the plug-in, and the reference
differs by site:

| Site | NDWI reference | Threshold | Why |
|------|----------------|-----------|-----|
| Great Salt Lake | JRC | +0.0151 | Gauge record starts 1999, and 144 of its 175 gauge-paired months fall inside JRC's 1984–2021 window |
| Lake Abert | OPERA | −0.0626 | Gauge came online in 2023, so all 20 gauge-paired months fall after JRC stopped being compiled in 2021 |

JRC is the historical reference, being the only one that spans the long record,
but it cannot calibrate an analysis of a period it does not cover. Lake Abert's
stage-area analysis runs entirely from 2023 to 2026, so OPERA (2016 to present)
is the appropriate reference there. Coverage of the analysis period is what
decides this. JRC simply has no observations left to calibrate against after
2021.

These settings produced the published stage-area figures: Lake Abert ρ = 0.44
(n = 20), Great Salt Lake south arm ρ = 0.87 (n = 175), north arm ρ = 0.83
(n = 175). Since the USGS monitoring station at Lake Abert was deployed much more
recently, its sample size is much lower and the relationship more difficult to
constrain than at the Great Salt Lake stations. The pairing is pinned in
`INDEX_REFERENCE` and `INDEX_REFERENCE_BY_SITE` so it cannot drift with future
scores.

```bash
python water_extent_timeseries.py            # full pass, hits Earth Engine
python water_extent_timeseries.py --replot   # redraw figures from existing CSVs
```

`--replot` makes no Earth Engine calls. It reuses the water extents already in
`water_extent_monthly_<site>.csv`, refreshes only the USGS gauge columns, and
redraws, taking seconds instead of tens of minutes.

Outputs are written next to the script:

```
water_extent_monthly_<site>.csv      one row per month, incl. gauge elevation
water_extent_monthly_all.csv         both sites concatenated
water_extent_timeseries_<site>.png   extent over time + gauge overlay
stage_area_<site>.png                extent vs elevation + Spearman rho
```

Key settings:

| Constant | Value | Why |
|----------|-------|-----|
| `ANALYSIS_MONTHS` | `(4, 10)` | Due to snow and ice artifacts, surface water extent is calculated only in the period from April through October, and some outliers remain, likely also caused by snow events inside that period. At Lake Abert the excluded months reach 2215 km² against a lake of roughly 150 km², on fully observed scenes. Out-of-season months are never queried. |
| `MIN_VALID_FRACTION` | `0.5` | Months observing less of the AOI are excluded from the figures, since an extent extrapolated from under half the AOI is closer to a guess than a measurement. |
| `MAX_WORKERS` | `2` | See Earth Engine Restricted Mode below. |
| `START_DATE` | `1999-01-01` | Landsat 7 launch. |

The seasonal window is applied at query time, so a full pass requests 193 months
per site instead of roughly 331, cutting Earth Engine calls by 42%.

## Earth Engine Restricted Mode

Unregistered or unverified Cloud projects run in Restricted Mode, which permits
only a couple of concurrent `getInfo()` calls. Exceeding it returns:

```
Too Many Requests: Exceeded Earth Engine concurrency limit.
Your project is in Restricted Mode.
```

`water_extent_timeseries.py` handles this with `MAX_WORKERS = 2`.

Registering the project for noncommercial use removes the restriction, after
which `MAX_WORKERS` can go back to 10.

## QGIS plug-in

Copy `Lake_Classifer_FD/` into your QGIS user plugins directory:

| OS | Path |
|----|------|
| Windows | `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins` |
| macOS | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |
| Linux | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |

Open QGIS and create a New Empty Project. From the menu bar, select "Plugins",
then "Manage and Install Plugins." In the "All" tab, search for "Google Earth
Engine" and click "Install Plugin" if it is not already installed, then verify
that your GEE Cloud Project is set under "Set Project". In the same tab, search
for Saline Lake Classifier and install it. Restart QGIS if prompted, then check
that both plugins appear under the Plugins menu.

The panel runs top to bottom:

| Stage | Does |
|-------|------|
| 1 · Area of Interest | Choose a predefined lake, upload a shapefile, or draw a custom AOI on the map |
| 2 · Outputs | True Color, LST, NDWI, NDVI, and NDMI for the summer composite (June–September) |
| 3 · Classification | Three threshold sliders producing open water, wetland, and land, with class areas |
| 4 · Time Series | Monthly NDWI/NDVI/NDMI CSV for the selected AOI and year |
| 5 · OpenET | Evapotranspiration, CONUS coverage only |

NDWI, NDVI, and NDMI must first be fetched in Section 2 before a threshold-based
land classification image can be generated. Default threshold values are provided
for the predefined Great Salt Lake and Lake Abert AOIs, and can be adjusted using
the sliders or by entering custom values. Releasing a slider re-runs the
classification.

Classification is applied locally with GDAL and NumPy. The three classes are
mutually exclusive and assigned in this order:

```
open water : NDWI >= ndwi_threshold
wetland    : NOT water AND NDVI >= ndvi_threshold AND NDMI >= ndmi_threshold
land       : NOT water AND NOT wetland
```

Open water is identified by pixels equaling or exceeding the optimized NDWI
threshold. Wetlands are identified by pixels with NDVI and NDMI values greater
than or equal to their respective optimized thresholds, and all remaining pixels
are classified as land. Land is the residual, so no threshold combination can
leave a valid pixel unclassified.

The plug-in then computes summary statistics for each land cover class by
counting the number of classified pixels and converting these counts to area
using the spatial resolution of the Landsat imagery (30 m x 30 m). Class areas
are reported in square kilometers together with the proportion of the total
classified area represented by each class.

### What the "wetland" class actually measures

The wetland class does not identify wetlands in the conventional sense, and the
label should not be read that way.

The class was trained against wetland reference data (NWI and NLCD), but what the
NDVI and NDMI combination detects is land that is moist and vegetated at the
moment the scene was acquired. Those are different things.

A wetland carries that designation year-round, even though many are saturated for only part of the year. The reference data encodes the permanent designation, while a spectral index reports the instantaneous surface state. These classifications are built from single static composites, so
they describe a point in time. A hydrologic regime cannot be read out of one scene.

Two consequences follow, running in opposite directions:

* A genuine wetland observed outside its saturated months reads as land, being
  dry and possibly senesced when the scene was acquired.
* Anything moist and green that happens not to be a wetland, such as irrigated
  fields, wet meadows, riparian margins, or ground after recent precipitation,
  reads as wetland.

Disagreement with NWI is therefore expected in both directions, and on its own it
is weak evidence that a threshold needs changing. It also means the class cannot
be summed across dates and read as wetland extent change. Two consecutive scenes
can differ simply because the ground dried out between them, with no wetland lost
at all.

The name "wetland" was retained at the partners' request. A description closer to
what is measured would be *moist vegetation*, and that is how the class
should be characterised in any writeup, caption, or briefing where a reader might
otherwise take it for a wetland delineation.

This also puts NDMI's weak skill in context (see
[the threshold provenance section](#where-the-plug-ins-default-thresholds-come-from)).
The index is being asked a question that one scene cannot answer. The reference
data carries a matching limitation: the NWI is spatially static, and updates
arrive project by project across the United States on no fixed schedule. Given
those temporal limitations, the NWI may not accurately represent wetland
conditions at the time the Landsat imagery was acquired, which contributes to the
relatively lower NDMI classification performance.

## Known gaps

Recorded here so the next reader does not have to discover them. None of them
stop the code from importing, since the scripts and the plug-in are all
self-contained, but each needs attention:

1. **`data/` inputs are not distributed with the repository**, by design. See
   [`data/README.md`](data/README.md). The threshold table and the NWI rasters
   have to be supplied separately. Both paths are configurable through
   `THRESHOLD_CSV` and `NWI_DIR`, and both scripts fail with an explicit message
   naming the missing file instead of a bare traceback.

2. **`Lake_Classifer_FD/metadata.txt` has unfilled placeholders.** The `email`,
   `repository`, and `tracker` fields still read `<insert>` and `<team email or
   primary contact>`. QGIS surfaces these in the plug-in manager.

## Known uncertainties

The sources of error and uncertainty that bear most directly on how these outputs
should be read:

- **Spectral mixing.** A single 30 m Landsat pixel may contain multiple land
  cover types, which can result in the misclassification of land cover within the
  validation sets. This is especially likely in transitional areas between open
  water, wetlands, and the surrounding land.
- **Satellite-derived reference data.** JRC, OPERA, and Dynamic World are all
  satellite-derived model outputs. None of them are field-level ground truth
  observations.
- **Threshold sensitivity.** Small changes in threshold can significantly shift
  the mapped land classifications.
- **Landsat 7 SLC failure.** Imagery acquired after the 2003 Scan Line Corrector
  failure contains data gaps that may reduce the accuracy of classification
  despite preprocessing. This is a sensor-level complication.
- **Snow and ice.** Winter imagery can alter spectral response and interfere with
  the effectiveness of NDWI for classifying open water, which may lead to
  increased omission or commission errors. This is why the time series is
  restricted to April through October.

## License

NASA / EarthRISE Developers Academy — 2026

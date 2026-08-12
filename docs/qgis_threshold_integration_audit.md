# QGIS threshold integration audit

## Current threshold flow

The QGIS dock's current UI semantics are:

* `PRESETS` contains the hard-coded Great Salt Lake and Lake Abert rectangles.
* The three threshold controls initialize from the Great Salt Lake profile.
* `lake_combo.currentIndexChanged` calls `_update_default_thresholds()`.
  Consequently, changing the dropdown changes both the displayed values and
  reset targets before **Use selected lake** is clicked; it does not yet change
  the AOI.
* `_use_preset()` applies the selected rectangle and calls
  `_update_default_thresholds()` again.
* An unknown preset key makes `_update_default_thresholds()` return without
  changing the sliders or reset targets.
* File-loaded and drawn AOIs set `aoi_extent`, but neither path establishes
  threshold provenance nor resets the controls.
* A custom AOI therefore retains whichever threshold values and reset targets
  were active previously. This is a stale-threshold/provenance risk, not a
  validated custom-AOI threshold policy.

This scaffold does not change those semantics. The preset labels remain lookup
keys into `Lake_Classifer_FD.threshold_profiles.DEFAULT_THRESHOLDS`. The
registry is standard-library-only, so its provenance can be tested without
importing QGIS.

The registry contains exactly the two profiles already shipped by the plugin:
Great Salt Lake (`Hylak_id=67`) and Lake Abert (`Hylak_id=776`). Both have
`scope=lake_specific` and `status=shipped`. It contains no pooled, regional,
Tier 1, or quick-transfer profile.

## Classification contract

The registry changes where default numbers are stored, not how classification
works. `_classify_raster()` continues to apply these mutually exclusive rules
in order:

```text
open water = valid AND NDWI >= ndwi_threshold
wetland    = valid AND NOT water AND NDVI >= ndvi_threshold
             AND NDMI >= ndmi_threshold
land       = valid AND NOT water AND NOT wetland
```

Current three-band rasters use all three conditions. The existing two-band
compatibility path omits the NDMI condition. Class codes remain 1 water,
2 wetland, 3 land, and 0 NoData. Neither `ee_backend.py` nor
`lake_classifier.py` changes in this integration.

## Plugin independence

* The plugin imports neither the calibration scripts nor `regional_analysis`.
* Earth Engine access is isolated in `ee_backend.py`.
* Threshold classification is local GDAL/NumPy processing in `dock.py`.
* `lake_classifier.py` controls plugin lifecycle and the time-series workflow.
* The threshold registry contains only standard-library data objects and does
  not create a dependency from the plugin to the research workflow.

## Shipped values versus committed calibration rows

The registry intentionally preserves the plugin's historical three-decimal
defaults. It does not silently replace them with fuller-precision values from
the source tables.

| Lake | Index | Shipped | Source row | Difference | Reference | Source |
|---|---:|---:|---:|---:|---|---|
| Great Salt Lake | NDWI | -0.055 | -0.0555 | +0.0005 | OPERA | `data/threshold_cv_summary.csv` |
| Great Salt Lake | NDVI | +0.165 | +0.1653 | -0.0003 | Dynamic World | `data/threshold_cv_summary.csv` |
| Great Salt Lake | NDMI | +0.053 | +0.0530 | 0.0000 | NWI | `data/ndmi_threshold_cv_all.csv` |
| Lake Abert | NDWI | -0.063 | -0.0626 | -0.0004 | OPERA | `data/threshold_cv_summary.csv` |
| Lake Abert | NDVI | +0.130 | +0.1304 | -0.0004 | Dynamic World | `data/threshold_cv_summary.csv` |
| Lake Abert | NDMI | +0.123 | +0.1230 | 0.0000 | NWI | `data/ndmi_threshold_cv_all.csv` |

The NDWI/NDVI table was produced by the superseded
`threshold_optimization_cv.py` workflow; that producer script is not present in
this repository. The NDMI table was produced by
`threshold_optimization_ndmi.py`, which is present. This is provenance, not a
request to change the current defaults.

## Scientific domain mismatch

The regional experiment and operational plugin do not currently implement the
same measurement procedure:

| Component | Regional Tier 1 analysis | QGIS plugin |
|---|---|---|
| Scene cloud filter | `<20%` | `<60%` |
| Pixel cloud mask | High-confidence cloud confidence bits | `QA_PIXEL` cloud bit 3 |
| Reflectance clipping | Clamp to `[0,1]` | Floor at zero with `.max(0)` |
| Index timing | Per-scene index sampling | Index from a seasonal median composite |
| Temporal domain | Multi-year, reference-specific windows | One selected June-September year |
| AOI | HydroLAKES polygon + 5 km | Hard-coded rectangle or custom bounding rectangle |
| Reference fitting | OPERA/JRC per-scene labels | Thresholds only; no reference product in classification |
| Threshold status | Exploratory regional/Tier 1 | Shipped lake-specific defaults |

The shipped profiles additionally record legacy rectangular AOIs; `Hylak_id`
is identity metadata only and does not make the plugin use a HydroLAKES polygon.
No pooled, LOLO, per-lake Tier 1, or custom-AOI fallback threshold is registered
or applied by the plugin.

> Tier 1 regional thresholds cannot be inserted into the QGIS plugin as validated defaults without a separate validation using the plugin’s exact preprocessing, compositing, temporal window, and AOI semantics.

## Decision gates

### Tier 2 expansion gate

Tier 2 should not run merely because the tooling supports it. It is justified
only if at least one of these conditions is met:

1. Tier 1 explanatory metrics identify a plausible mechanism that Tier 2 can
   deliberately test.
2. Tier 1 has a clear covariate gap that prevents evaluation of a hypothesis.
3. One or more results are controlled by a single influential lake and
   additional targeted systems can test robustness.

Known design gaps are the presence of only one Tier 1 type-2 lake and no type-3
lake, plus limited deep-lake, reservoir, and high-elevation contrasts and
possible underrepresentation of stable freshwater systems. If explanatory
relationships are unstable under leave-one-lake-out influence analysis, the
appropriate response is targeted Tier 2 sampling, not a claimed association.

### QGIS regional-threshold integration gate

Do not expose a regional threshold in the QGIS GUI until all of the following
exist:

1. Validation using the plugin's exact cloud filtering, pixel mask,
   reflectance scaling, seasonal compositing, index calculation, and AOI
   semantics.
2. Complete threshold provenance.
3. An explicit status model such as `shipped`, `validated`, and `exploratory`.
4. An explicit custom-AOI threshold-selection policy.
5. A visible threshold-source/provenance indication in the GUI.
6. Regression tests for profile selection and stale-profile prevention.
7. A manual smoke test in the supported QGIS version.
8. A decision whether regional thresholds are general defaults, selectable
   exploratory profiles, or inappropriate for operational use.

Until those gates are cleared, the registry contains only the two shipped
lake-specific profiles. Tier 1 remains exploratory.

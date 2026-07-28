"""
water_extent_timeseries.py
───────────────────────────
Monthly median-composite surface-water extent for Great Salt Lake and Lake
Abert, from Landsat 7/8/9, using the site-specific NDWI/NDVI thresholds
already fit in threshold_cv_summary.csv (see threshold_optimization_cv.py).

Per site, picks the best-performing (highest cv_iou) NDWI row and NDVI row
from the summary table — these are usually different references (e.g. JRC
for NDWI, Dynamic World for NDVI) — and combines them as:

    water = (NDWI >= ndwi_threshold) AND (NDVI < ndvi_threshold)

The NDVI term suppresses emergent vegetation / algae that can otherwise read
as "water" under NDWI alone; this mirrors why threshold_optimization_cv.py
validates NDVI against Dynamic World in the first place.

Each calendar month is collapsed to one cloud-masked median composite (across
whatever L7/8/9 scenes and adjacent path/rows fell in that month) before the
indices are computed — this fills most cloud and Landsat-7 SLC-off gaps with
other looks from the same month, so monthly valid_fraction is typically far
better than any single scene's.

Thresholds were fit on JRC/OPERA/DW references from 2015 onward; applying
them back across the full L7/8/9 archive (1999–present) is out-of-sample for
the pre-2015 portion, which is expected — that's the whole point of a long
time series.

Each site's monthly extent is also overlaid with USGS water-surface-elevation
gauge(s) (daily values service, resampled to monthly mean) — Lake Abert has
one gauge, Great Salt Lake has two (its rail causeway splits it into a south
arm and a north arm that sit at different elevations, so both are shown).

Surface elevation and surface area are related through the lake's basin
hypsometry, not proportionally, so two things are produced for that
relationship: a dual-axis version of the time series (a quick "do they move
together" glance — but axis scaling is arbitrary, so it can't show the actual
curve or be read as a correlation), and a separate stage-area scatter (extent
vs. elevation, colored by year) with Spearman rank correlation annotated,
which is the honest way to see a monotonic-but-nonlinear relationship.

Only April–October is considered (ANALYSIS_MONTHS). Nov–Mar are never queried,
never written to the CSV, and never plotted: snow, ice and wet playa satisfy
`NDWI >= x AND NDVI < y` and register as water — at Lake Abert this reaches
1877 km² (Feb 2025) and 2097 km² (Nov 2022) against a ~150 km² lake, on fully
observed AOIs, so valid_fraction does not catch it. Being seasonal and fixed in
advance, this criterion is independent of the extent value — unlike a magnitude
cap, which would select on the very quantity the scatter correlates. Within the
window every observation is kept at face value; there is no cap on extent.

Low-coverage months (valid_fraction below MIN_VALID_FRACTION) are excluded from
both figures — an extent extrapolated from under half the AOI is a guess about
the unobserved remainder, not a measurement. They cannot be skipped at query
time the way the seasonal window is, because valid_fraction is only known once
the composite exists, so they remain in the CSV and are filtered at plot time.
The gauge elevation overlay is NOT coverage-filtered: a water-level reading is
valid regardless of what Landsat could see that month.

Outputs per site:
    water_extent_monthly_<site>.csv    — one row per calendar month, incl.
                                          elevation column(s) where available
    water_extent_timeseries_<site>.png — dual-axis time series
    stage_area_<site>.png              — extent-vs-elevation scatter + Spearman ρ

Run: python water_extent_timeseries.py            # full pass, hits Earth Engine
     python water_extent_timeseries.py --replot   # redraw from existing CSVs only
"""

from __future__ import annotations

import os
import datetime
import sys
import time
import random
import concurrent.futures as cf
from pathlib import Path

import ee
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ── Landsat mission configuration ───────────────────────────────────────────
# Previously imported from gee_pipeline.py, which is retired. Inlined verbatim
# so this script is self-contained; the band mappings match the plugin's
# Lake_Classifer_FD/ee_backend.py, which is the authoritative copy for the QGIS
# side. L5 is defined but unused — LANDSAT_SENSORS starts at L7.
#
# Collection 2, Tier 1, Level 2 surface reflectance.
COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",   # 1984-01-01 – 2012-05-05
    "L7": "LANDSAT/LE07/C02/T1_L2",   # 1999-01-01 – 2022-04-06 (SLC-off 2003+)
    "L8": "LANDSAT/LC08/C02/T1_L2",   # 2013-04-11 – present
    "L9": "LANDSAT/LC09/C02/T1_L2",   # 2022-02-10 – present
}

# Mission-specific band numbers -> the shared names the index math uses. TM/ETM+
# and OLI number their bands differently; this is what lets one threshold be
# applied across all missions.
BAND_MAP = {
    "L5": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
           "nir":  "SR_B4", "swir1": "SR_B5"},
    "L7": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
           "nir":  "SR_B4", "swir1": "SR_B5"},
    "L8": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
           "nir":  "SR_B5", "swir1": "SR_B6"},
    "L9": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
           "nir":  "SR_B5", "swir1": "SR_B6"},
}

# Operational window per mission, so a date filter never asks a satellite for
# imagery it could not have taken. date.max means "still operating".
SENSOR_WINDOWS = {
    "L5": (datetime.date(1984,  1,  1), datetime.date(2012,  5,  5)),
    "L7": (datetime.date(1999,  1,  1), datetime.date(2022,  4,  6)),
    "L8": (datetime.date(2013,  4, 11), datetime.date.max),
    "L9": (datetime.date(2022,  2, 10), datetime.date.max),
}

SCRIPT_DIR = Path(__file__).resolve().parent

# Calibrated thresholds. Any CSV with site / index / reference / threshold_mean /
# cv_iou columns works; the best-scoring NDWI and NDVI row per site is selected
# (see load_thresholds). Override with:
#     export THRESHOLD_CSV=/path/to/thresholds.csv
#
# This file is NOT produced by anything in this repository. It comes from the
# superseded threshold_optimization_cv.py. final_threshold_optimization.py writes
# outputs/tables/final_threshold_summary.csv instead, which is column-compatible
# and can be pointed at directly — note it holds a different (later) calibration,
# so switching changes the thresholds the time series is built on.
THRESHOLD_CSV = Path(os.environ.get(
    'THRESHOLD_CSV', SCRIPT_DIR / 'data' / 'threshold_cv_summary.csv'))

# Earth Engine Cloud project. Kept out of source control: set EE_PROJECT in the
# environment rather than hardcoding an ID here, so a project identifier cannot
# be committed by accident.
#     export EE_PROJECT=your-ee-project-id
EE_PROJECT = os.environ.get('EE_PROJECT', 'your-ee-project-id')

AOI_BOUNDS = {
    'Great Salt Lake': (-113.3061047, 40.51087812, -111.8534658, 41.87473984),
    'Lake Abert':      (-120.6098988, 42.40054901, -119.9981325, 42.85172324),
}

LANDSAT_SENSORS   = ['L7', 'L8', 'L9']
START_DATE        = datetime.date(1999, 1, 1)   # Landsat 7 launch
CLOUD_COVER_MAX   = 20
SCALE_M           = 30
MIN_VALID_FRACTION = 0.5   # months observing less of the AOI than this are flagged, not dropped

# The only months this script looks at, inclusive: April–October.
#
# Nov–Mar are excluded because snow, ice and wet playa satisfy
# `NDWI >= x AND NDVI < y` and register as water. At Lake Abert (~4250 ft, high
# desert) the effect is severe and not caught by valid_fraction, because these
# are fully observed pixels that are simply misclassified:
#     Feb 2025  1877 km²  |  Nov 2022  2097 km² raw, valid_fraction 1.00
# against a lake whose real full-pool area is ~150 km². Every Abert month above
# 300 km² in an Apr–Nov window fell in November; Apr–Oct leaves a maximum of
# 282 km² and no implausible months at all.
#
# Being seasonal and fixed in advance, this is independent of the extent value,
# so unlike an "extent > N km²" cap it does not select on the quantity the
# stage-area scatter correlates.
#
# NOTE — this window is uniform across sites, which costs Great Salt Lake 25
# sound November months: GSL's November peaks at 5665 km² against a ~5700 km²
# high stand, i.e. it never had the freeze artifact Abert does (a far larger
# saline lake resists freezing). If GSL's sample size ever becomes the binding
# constraint, a per-site window is the fix — Abert (4, 10), GSL (4, 11) — which
# means threading `site` through _month_starts, _in_season and _season_label.
#
# This governs QUERYING as well as plotting: out-of-season months are never
# requested from Earth Engine, which cuts the monthly composites per site by
# 45% — a material saving under Earth Engine's Restricted Mode concurrency
# limit. Widening the window means re-running a full pass to backfill.
ANALYSIS_MONTHS = (4, 10)
# Concurrent Earth Engine interactive requests. Earth Engine's noncommercial
# "Restricted Mode" allows only a couple of concurrent getInfo() calls, and once
# that ceiling is hit every worker starts failing at once — so on a restricted
# project this must be small. Raise it only if the run reports zero throttling.
MAX_WORKERS       = 2
EE_MAX_RETRIES    = 6      # per month, on throttling only
EE_BACKOFF_BASE_S = 2.0    # 2, 4, 8, 16, 32s (+jitter) between attempts

# USGS gauges from "Data Appendix (USGS Monitoring Sites).csv" that actually carry
# a lake/reservoir water-surface-elevation parameter (62614 or 62615) — the other
# rows in that table (405356112205601, 410401112134801) only report air/water
# temperature and are excluded here.
NWIS_DV_URL = 'https://waterservices.usgs.gov/nwis/dv/'

# NWIS encodes a missing daily value as -999999 rather than omitting the row, and
# that sentinel is valid numeric text, so pd.to_numeric() lets it straight through.
# Averaging even two such days into a monthly mean drags it to ~-80000 ft, which is
# what produced the impossible negative-elevation points on Lake Abert's stage-area
# scatter. Anything below this floor is treated as no-data, not as a reading.
USGS_MIN_PLAUSIBLE_FT = -1000.0

USGS_ELEVATION_STATIONS = {
    'Lake Abert': [
        {'label': 'Lake Abert (USGS 10387150)', 'site': '10387150', 'param': '62615'},  # NAVD88 ft
    ],
    'Great Salt Lake': [
        {'label': 'GSL south arm – Saltair (USGS 10010000)', 'site': '10010000', 'param': '62614'},  # NGVD29 ft
        {'label': 'GSL north arm – Saline (USGS 10010100)',  'site': '10010100', 'param': '62614'},  # NGVD29 ft
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Thresholds — one pinned reference per index, both sites
# ─────────────────────────────────────────────────────────────────────────────

# Which validation reference supplies each index's threshold. Pinned explicitly
# rather than chosen by best cv_iou, because that rule picked the reference by
# accident of scoring rather than by whether the reference covers the period
# being analysed.
#
# NDWI defaults to JRC Global Surface Water. This script is the HISTORICAL
# analysis, and JRC's monthly history (1984–2021) is the reference that spans
# that record. The QGIS plug-in uses OPERA at both sites instead, because it
# classifies a single recent summer composite — see the README.
#
# LAKE ABERT IS THE EXCEPTION, and it is a coverage argument, not a scoring one.
# Its USGS gauge came online in 2023, so every gauge-paired month in the
# stage-area analysis falls in 2023–2026 — entirely after JRC stopped being
# compiled in 2021. Calibrating Abert against JRC would fit a threshold on a
# period the analysis does not cover. OPERA (2016–present) does cover it.
# Great Salt Lake keeps JRC: its gauge record starts in 1999, and 144 of its 175
# gauge-paired months fall inside JRC's window.
#
# This reproduces the published technical report exactly (Abert rho = 0.44,
# n = 20; GSL south arm rho = 0.87, n = 175).
#
# NDVI → Dynamic World, the only vegetation reference in the table.
#
# MNDWI rows are ignored: this script computes no MNDWI band.
INDEX_REFERENCE = {
    'NDWI': 'JRC',
    'NDVI': 'DW',
}

INDEX_REFERENCE_BY_SITE = {
    ('NDWI', 'Lake Abert'): 'OPERA',
}


def _reference_for(index: str, site: str) -> str:
    """The reference dataset supplying `index` at `site`."""
    return INDEX_REFERENCE_BY_SITE.get((index, site), INDEX_REFERENCE[index])


def _pick(rows: pd.DataFrame, index: str, site: str, csv_path) -> pd.Series:
    """The row for `index` from its pinned reference, or a clear error."""
    want = _reference_for(index, site)
    hit = rows[rows['reference'] == want]
    if hit.empty:
        available = ', '.join(sorted(rows['reference'].unique())) or 'none'
        raise ValueError(
            f"'{site}' has no {index} row from reference '{want}' in {csv_path}.\n"
            f"    Available {index} references: {available}.\n"
            f"    Adjust INDEX_REFERENCE if this calibration uses a different one.")
    # Defensive: a table with repeated site/index/reference rows would otherwise
    # pick an arbitrary one. Prefer the best-scoring duplicate.
    return hit.loc[hit['cv_iou'].idxmax()]


def load_thresholds(csv_path: Path) -> dict:
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Threshold table not found: {csv_path}\n"
            f"    Set THRESHOLD_CSV to your calibration table, or point it at "
            f"outputs/tables/final_threshold_summary.csv from "
            f"final_threshold_optimization.py — see the README.")

    df = pd.read_csv(csv_path)
    thresholds = {}
    for site, g in df.groupby('site'):
        ndwi_rows = g[g['index'] == 'NDWI']
        ndvi_rows = g[g['index'] == 'NDVI']
        if ndwi_rows.empty or ndvi_rows.empty:
            raise ValueError(f"'{site}' is missing an NDWI or NDVI row in {csv_path}")
        best_ndwi = _pick(ndwi_rows, 'NDWI', site, csv_path)
        best_ndvi = _pick(ndvi_rows, 'NDVI', site, csv_path)
        thresholds[site] = {
            'ndwi_threshold': float(best_ndwi['threshold_mean']),
            'ndwi_reference': best_ndwi['reference'],
            'ndwi_cv_iou':    float(best_ndwi['cv_iou']),
            'ndvi_threshold': float(best_ndvi['threshold_mean']),
            'ndvi_reference': best_ndvi['reference'],
            'ndvi_cv_iou':    float(best_ndvi['cv_iou']),
        }
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# 1b. USGS water-surface-elevation gauges
# ─────────────────────────────────────────────────────────────────────────────

def fetch_usgs_elevation(site: str, parameter_cd: str,
                          start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Daily water-surface elevation (feet) for one USGS gauge, via the NWIS
    daily-values service. Returns an empty frame (not an error) if the gauge
    has no data in range — most of these stations have short or gappy records."""
    resp = requests.get(NWIS_DV_URL, params={
        'format': 'json', 'sites': site, 'parameterCd': parameter_cd,
        'startDT': start.isoformat(), 'endDT': end.isoformat(), 'siteStatus': 'all',
    }, timeout=60)
    resp.raise_for_status()
    series = resp.json().get('value', {}).get('timeSeries', [])
    if not series:
        return pd.DataFrame(columns=['date', 'elevation_ft'])

    values = series[0]['values'][0]['value']
    df = pd.DataFrame(values)
    df['date'] = pd.to_datetime(df['dateTime']).dt.tz_localize(None)
    df['elevation_ft'] = pd.to_numeric(df['value'], errors='coerce')

    # Drop NWIS no-data sentinels BEFORE any averaging. Doing it here rather than
    # after the monthly resample means a month with a few bad days keeps a correct
    # mean from its good days instead of being discarded outright.
    sentinel = df['elevation_ft'] < USGS_MIN_PLAUSIBLE_FT
    if sentinel.any():
        print(f"    [dropped {int(sentinel.sum())} no-data reading(s) "
              f"below {USGS_MIN_PLAUSIBLE_FT:.0f} ft]")
        df.loc[sentinel, 'elevation_ft'] = float('nan')

    return df[['date', 'elevation_ft']].dropna()


def load_elevation_series(site: str, start: datetime.date,
                           end: datetime.date) -> dict[str, pd.DataFrame]:
    """Monthly-mean elevation per configured gauge for this site, keyed by label."""
    out = {}
    for st in USGS_ELEVATION_STATIONS.get(site, []):
        daily = fetch_usgs_elevation(st['site'], st['param'], start, end)
        if daily.empty:
            print(f"  [no USGS data returned for {st['label']}]")
            continue
        monthly = (
            daily.set_index('date')['elevation_ft']
            .resample('MS').mean()
            .rename(st['label'])
            .reset_index()
        )
        out[st['label']] = monthly
        print(f"  fetched {st['label']}: {len(daily)} daily readings, "
              f"{len(monthly)} monthly means")
    return out


def attach_elevation(df: pd.DataFrame, elevation_series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-joins each gauge's monthly series onto df['date'] as its own column."""
    merged = df.copy()
    for label, monthly in elevation_series.items():
        merged = merged.merge(monthly, on='date', how='left')
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing — cloud mask, scale, harmonize bands per sensor
# ─────────────────────────────────────────────────────────────────────────────

def _cloud_mask(image: ee.Image) -> ee.Image:
    """High-confidence cloud only (QA_PIXEL bits 8-9 == 3), matching the mask
    used when the thresholds were fit in threshold_optimization_cv.py."""
    qa = image.select('QA_PIXEL')
    cloud_high = qa.rightShift(8).bitwiseAnd(3).eq(3)
    return image.updateMask(cloud_high.Not())


def _scale_and_mask(image: ee.Image) -> ee.Image:
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2).clamp(0, 1)
    return _cloud_mask(image.addBands(optical, overwrite=True))


def _harmonize(image: ee.Image, sensor: str) -> ee.Image:
    src = list(BAND_MAP[sensor].values())
    dst = list(BAND_MAP[sensor].keys())
    return image.select(src).rename(dst)


def _sensor_collection(sensor: str, aoi: ee.Geometry,
                        start: datetime.date, end: datetime.date) -> ee.ImageCollection | None:
    win_start, win_end = SENSOR_WINDOWS[sensor]
    s = max(start, win_start)
    e = end if win_end == datetime.date.max else min(end, win_end)
    if s > e:
        return None

    return (
        ee.ImageCollection(COLLECTIONS[sensor])
        .filterBounds(aoi)
        .filterDate(s.isoformat(), (e + datetime.timedelta(days=1)).isoformat())
        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
        .map(_scale_and_mask)
        .map(lambda img: _harmonize(img, sensor).set('sensor', sensor))
    )


def build_landsat_collection(aoi: ee.Geometry, start: datetime.date,
                              end: datetime.date) -> ee.ImageCollection:
    cols = [c for c in (_sensor_collection(s, aoi, start, end) for s in LANDSAT_SENSORS)
            if c is not None]
    merged = cols[0]
    for c in cols[1:]:
        merged = merged.merge(c)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 3. Monthly median composite → water extent
# ─────────────────────────────────────────────────────────────────────────────

def _month_starts(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Month-start dates from `start` to `end`, restricted to ANALYSIS_MONTHS.

    Out-of-season months are filtered out HERE, before any Earth Engine work is
    scheduled, so they are never composited, never downloaded, and never reach
    the CSV. That is the cheapest place to drop them — one skipped month is one
    fewer getInfo() round trip.
    """
    lo, hi = ANALYSIS_MONTHS
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        if lo <= m <= hi:
            months.append(datetime.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _next_month(d: datetime.date) -> datetime.date:
    return datetime.date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _monthly_feature(col_month: ee.ImageCollection, aoi: ee.Geometry, ndwi_threshold: float,
                      ndvi_threshold: float, total_area_m2: ee.Number, month_str: str) -> ee.Feature:
    n = col_month.size()

    def _empty():
        return ee.Feature(None, {
            'month': month_str, 'n_images': 0, 'sensors': '',
            'water_km2': None, 'water_km2_raw': None,
            'valid_km2': None, 'valid_fraction': None,
        })

    def _composited():
        composite = col_month.median()
        ndwi = composite.normalizedDifference(['green', 'nir']).rename('NDWI')
        ndvi = composite.normalizedDifference(['nir', 'red']).rename('NDVI')

        valid_mask = ndwi.mask().And(ndvi.mask())
        water_mask = ndwi.gte(ndwi_threshold).And(ndvi.lt(ndvi_threshold))

        pixel_area = ee.Image.pixelArea()
        areas = (
            ee.Image.cat([
                pixel_area.updateMask(water_mask).rename('water_m2'),
                pixel_area.updateMask(valid_mask).rename('valid_m2'),
            ])
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=SCALE_M,
                          maxPixels=1e9, bestEffort=True)
        )

        water_m2 = ee.Number(areas.get('water_m2'))
        valid_m2 = ee.Number(areas.get('valid_m2'))

        # Extrapolate through any remaining cloud/gap holes by assuming
        # unobserved pixels have the same water proportion as observed ones.
        water_km2 = water_m2.divide(valid_m2.max(1)).multiply(total_area_m2).divide(1e6)

        return ee.Feature(None, {
            'month':          month_str,
            'n_images':       n,
            'sensors':        col_month.aggregate_array('sensor').distinct().join(','),
            'water_km2':      water_km2,
            'water_km2_raw':  water_m2.divide(1e6),
            'valid_km2':      valid_m2.divide(1e6),
            'valid_fraction': valid_m2.divide(total_area_m2),
        })

    return ee.Feature(ee.Algorithms.If(n.eq(0), _empty(), _composited()))


# A month that Earth Engine refused is NOT the same as a month with no imagery:
# the first leaves a hole in the series that must be reported, the second is a
# real, expected absence. Returning None for both is what let rate-limit errors
# silently thin the time series.
_EE_FAILED = object()


def _is_rate_limited(exc: Exception) -> bool:
    """True for transient throttling, which is worth retrying. Deliberately does
    NOT match 'user memory limit exceeded' — that is a per-request cost problem
    and retrying it unchanged just burns quota."""
    msg = str(exc).lower()
    return any(marker in msg for marker in (
        'too many requests', 'concurrency limit', 'rate limit',
        'quota exceeded', 'restricted mode', 'backend error', 'timed out',
    ))


def _evaluate_month(col: ee.ImageCollection, aoi: ee.Geometry, ndwi_threshold: float,
                     ndvi_threshold: float, total_area_m2: ee.Number,
                     month_start: datetime.date):
    month_end = _next_month(month_start)
    col_month = col.filterDate(month_start.isoformat(), month_end.isoformat())
    feat = _monthly_feature(col_month, aoi, ndwi_threshold, ndvi_threshold,
                             total_area_m2, month_start.isoformat())

    for attempt in range(EE_MAX_RETRIES):
        try:
            props = feat.getInfo()['properties']
            return props if props.get('n_images', 0) else None
        except ee.ee_exception.EEException as e:
            if not _is_rate_limited(e):
                print(f"    [{month_start.isoformat()} failed: {e} — skipping]")
                return _EE_FAILED
            if attempt == EE_MAX_RETRIES - 1:
                print(f"    [{month_start.isoformat()} still throttled after "
                      f"{EE_MAX_RETRIES} attempts — LOST]")
                return _EE_FAILED
            # Exponential backoff with jitter; jitter matters because all workers
            # get throttled at once and would otherwise retry in lockstep.
            delay = EE_BACKOFF_BASE_S * (2 ** attempt) + random.uniform(0, 1.0)
            time.sleep(delay)
    return _EE_FAILED


def compute_monthly_extent(col: ee.ImageCollection, aoi: ee.Geometry, start: datetime.date,
                            end: datetime.date, ndwi_threshold: float,
                            ndvi_threshold: float) -> pd.DataFrame:
    """One median composite per calendar month, evaluated with a small pool of
    concurrent getInfo() calls. Each request is still scoped to a single month
    — that's what avoids Earth Engine's 'User memory limit exceeded' error,
    which batching hundreds of AOI-wide reduceRegions into one request reliably
    trips — but running MAX_WORKERS of them at once instead of strictly one at
    a time cuts wall-clock time roughly proportionally.
    """
    total_area_m2 = aoi.area(1)
    months = _month_starts(start, end)
    print(f"  {len(months)} calendar months to evaluate ({months[0]} – {months[-1]}), "
          f"{MAX_WORKERS} concurrent requests")

    rows, done, failed = [], 0, []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_evaluate_month, col, aoi, ndwi_threshold, ndvi_threshold,
                        total_area_m2, month_start): month_start
            for month_start in months
        }
        for future in cf.as_completed(futures):
            props = future.result()
            if props is _EE_FAILED:
                failed.append(futures[future])
            elif props is not None:
                rows.append(props)
            done += 1
            if done % 25 == 0 or done == len(months):
                print(f"    {done}/{len(months)} months checked, {len(rows)} with data"
                      + (f", {len(failed)} LOST to errors" if failed else ''))

    # Loud, because a silently short series looks identical to a genuinely sparse
    # archive once it reaches the CSV and the figures.
    if failed:
        print(f"\n  !! {len(failed)} month(s) could not be retrieved and are MISSING "
              f"from this series, not absent from the archive:")
        for m in sorted(failed)[:12]:
            print(f"       {m.isoformat()}")
        if len(failed) > 12:
            print(f"       ... and {len(failed) - 12} more")
        print("     Lower MAX_WORKERS and re-run to fill these in.\n")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot
# ─────────────────────────────────────────────────────────────────────────────

_ELEVATION_COLORS = ['darkorange', 'firebrick', 'purple']

def _in_season(df: pd.DataFrame) -> pd.Series:
    """Months inside ANALYSIS_MONTHS (inclusive).

    A fresh run contains only in-season months already — _month_starts never
    requests the others. This still matters for --replot, which reads CSVs
    written before the seasonal window existed and therefore may hold Dec–Mar
    rows that must not reach the figures.
    """
    lo, hi = ANALYSIS_MONTHS
    return df['date'].dt.month.between(lo, hi)


def _season_label() -> str:
    """'Apr–Nov' — derived from the constant so figure labels can never drift
    out of sync with the window actually applied."""
    names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    lo, hi = ANALYSIS_MONTHS
    return f'{names[lo - 1]}–{names[hi - 1]}'


def _low_confidence(df: pd.DataFrame) -> pd.Series:
    if 'low_confidence' not in df.columns:
        return pd.Series(False, index=df.index)
    return df['low_confidence'].astype(bool)


def _plot_timeseries(site: str, df: pd.DataFrame, slug: str, elevation_labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    # Out-of-season months are dropped outright rather than drawn in a muted
    # style. A fresh run never collects them, so the only way they can appear
    # here is a --replot over a CSV written before the seasonal window existed;
    # filtering makes the two paths produce the same figure.
    df = df[_in_season(df)]

    # Low-coverage months are dropped from the extent series too. Unlike the
    # seasonal window this cannot be applied at query time — valid_fraction is
    # only known after the composite is built — so these months stay in the CSV
    # and are filtered here. An extent extrapolated from under half the AOI is a
    # guess about the unobserved remainder, not a measurement.
    good = df[~_low_confidence(df)]

    ax.plot(good['date'], good['water_km2'], '-o', color='steelblue', ms=3, lw=1,
             label=f'water extent ({_season_label()})')

    ax.set_title(f'{site} — monthly median-composite water extent, '
                 f'Landsat 7/8/9 ({START_DATE.isoformat()} – present)')
    ax.set_xlabel('month')
    ax.set_ylabel('water extent (km²)')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    handles, labels = ax.get_legend_handles_labels()
    # Deliberately `df`, not `good`: the gauge series is measured in the water,
    # independent of what Landsat could see that month, so cloud cover over the
    # AOI is no reason to drop a valid water-level reading. Filtering it here
    # would put gaps in the elevation curve for reasons unrelated to the gauge.
    present_labels = [lbl for lbl in elevation_labels if lbl in df.columns]
    if present_labels:
        ax2 = ax.twinx()
        for label, color in zip(present_labels, _ELEVATION_COLORS):
            sub = df.dropna(subset=[label])
            ax2.plot(sub['date'], sub[label], '--', color=color, lw=1.3, label=label)
        ax2.set_ylabel('water surface elevation (ft, USGS gauge)')
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, fontsize=7, loc='upper right')

    out_png = SCRIPT_DIR / f'water_extent_timeseries_{slug}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_png.name}")


def _plot_stage_area(site: str, df: pd.DataFrame, elevation_labels: list[str], slug: str) -> None:
    """Extent-vs-elevation scatter, one panel per gauge, colored by year. This is
    the stage-area curve — the direct, non-misleading way to show how a
    nonlinear elevation/area relationship actually behaves, versus a dual-axis
    line chart which can't show curvature and whose apparent agreement depends
    on arbitrary axis scaling."""
    present_labels = [lbl for lbl in elevation_labels if lbl in df.columns]
    if not present_labels:
        return

    n = len(present_labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, label in zip(axes[0], present_labels):
        sub = df.dropna(subset=[label, 'water_km2'])

        # Exclude only months that cannot represent a real stage-area pair:
        # out-of-season (snow/ice reads as water) and partially observed AOIs.
        # No cap on extent — every in-season observation is kept at face value.
        # The season is stated in the figure title; the per-panel n reports what
        # survived, so the exclusion counts are not repeated here.
        sub = sub[_in_season(sub) & ~_low_confidence(sub)]

        if len(sub) < 3:
            ax.set_title(f'{label}\n(not enough overlapping months)')
            ax.axis('off')
            continue

        sc = ax.scatter(sub[label], sub['water_km2'], c=sub['date'].dt.year,
                         cmap='viridis', s=20)
        rho = sub[[label, 'water_km2']].corr(method='spearman').iloc[0, 1]
        ax.set_title(f'{label}\nSpearman ρ = {rho:.2f}  (n={len(sub)} months)',
                     fontsize=10)
        ax.set_xlabel('water surface elevation (ft)')
        ax.set_ylabel('water extent (km²)')
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label='year')

    # Season stated once in the figure title rather than repeated per panel.
    # Derived from _season_label() so it tracks ANALYSIS_MONTHS automatically.
    fig.suptitle(f'{site} — stage-area relationship ({_season_label()} Only)')
    fig.tight_layout()

    out_png = SCRIPT_DIR / f'stage_area_{slug}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_png.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-site driver
# ─────────────────────────────────────────────────────────────────────────────

def run_site(site: str, thresholds: dict) -> pd.DataFrame:
    print(f"\n=== {site} ===")
    xmin, ymin, xmax, ymax = AOI_BOUNDS[site]
    aoi = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

    t = thresholds[site]
    print(f"  NDWI threshold {t['ndwi_threshold']:+.4f} "
          f"(from {t['ndwi_reference']}, cv_iou={t['ndwi_cv_iou']:.3f})")
    print(f"  NDVI threshold {t['ndvi_threshold']:+.4f} "
          f"(from {t['ndvi_reference']}, cv_iou={t['ndvi_cv_iou']:.3f})")

    end = datetime.date.today()
    col = build_landsat_collection(aoi, START_DATE, end)

    print("  querying Earth Engine for monthly composite water extent (this can take a while)...")
    df = compute_monthly_extent(col, aoi, START_DATE, end, t['ndwi_threshold'], t['ndvi_threshold'])
    if df.empty:
        print(f"  [no months with imagery found for {site}]")
        return df

    df['date'] = pd.to_datetime(df['month'])
    df = df.sort_values('date').reset_index(drop=True)
    df['low_confidence'] = df['valid_fraction'] < MIN_VALID_FRACTION

    print("  fetching USGS water-surface-elevation gauge(s)...")
    elevation_series = load_elevation_series(site, START_DATE, end)
    elevation_labels = [st['label'] for st in USGS_ELEVATION_STATIONS.get(site, [])]
    df = attach_elevation(df, elevation_series)

    slug    = site.lower().replace(' ', '_')
    out_csv = SCRIPT_DIR / f'water_extent_monthly_{slug}.csv'
    df.to_csv(out_csv, index=False)
    print(f"  saved {out_csv.name} ({len(df)} months, "
          f"{int(df['low_confidence'].sum())} flagged low-coverage)")

    _plot_timeseries(site, df, slug, elevation_labels)
    _plot_stage_area(site, df, elevation_labels, slug)
    return df


def replot_site(site: str) -> pd.DataFrame:
    """Redraw both figures for one site from its existing monthly-extent CSV.

    The water extents are reused exactly as previously computed — no Earth Engine
    calls, no re-thresholding. Only the gauge columns are refreshed, which is what
    lets the NWIS sentinel fix reach the figures without a multi-hour recompute."""
    print(f"\n=== {site} (replot from existing extents) ===")
    slug     = site.lower().replace(' ', '_')
    csv_path = SCRIPT_DIR / f'water_extent_monthly_{slug}.csv'
    if not csv_path.exists():
        print(f"  [{csv_path.name} not found — run a full pass for {site} first]")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date' if 'date' in df.columns else 'month'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  loaded {len(df)} months of existing water extents from {csv_path.name}")

    labels = [st['label'] for st in USGS_ELEVATION_STATIONS.get(site, [])]

    # Blank contaminated monthly means already baked into the CSV first, so that a
    # failed re-fetch degrades to "those months are missing" rather than
    # "every gauge value is missing".
    for lbl in labels:
        if lbl in df.columns:
            bad = df[lbl] < USGS_MIN_PLAUSIBLE_FT
            if bad.any():
                print(f"  [{lbl}: blanked {int(bad.sum())} contaminated monthly mean(s) "
                      f"— {', '.join(df.loc[bad, 'date'].dt.strftime('%Y-%m'))}]")
                df.loc[bad, lbl] = float('nan')

    # Re-fetch so those months return with a correct mean over their good days.
    print("  re-fetching USGS gauge(s)...")
    try:
        fresh = load_elevation_series(site, START_DATE, datetime.date.today())
    except Exception as e:
        print(f"  [gauge re-fetch failed ({type(e).__name__}) — keeping blanked values]")
        fresh = {}
    if fresh:
        df = df.drop(columns=[l for l in fresh if l in df.columns])
        df = attach_elevation(df, fresh)

    df.to_csv(csv_path, index=False)
    print(f"  rewrote {csv_path.name}")

    _plot_timeseries(site, df, slug, labels)
    _plot_stage_area(site, df, labels, slug)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # --replot redraws the figures from the existing water_extent_monthly_*.csv
    # files. Earth Engine is not touched, so the water extents are exactly the
    # ones already computed.
    replot_only = '--replot' in sys.argv

    if not replot_only:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
        thresholds = load_thresholds(THRESHOLD_CSV)

    all_dfs = []
    for site in AOI_BOUNDS:
        df = replot_site(site) if replot_only else run_site(site, thresholds)
        if not df.empty:
            all_dfs.append(df.assign(site=site))

    if all_dfs:
        combined_path = SCRIPT_DIR / 'water_extent_monthly_all.csv'
        pd.concat(all_dfs, ignore_index=True).to_csv(combined_path, index=False)
        print(f"\nSaved {combined_path.name}")

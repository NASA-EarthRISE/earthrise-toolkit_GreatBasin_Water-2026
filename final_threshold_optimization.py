"""
final_threshold_optimization.py
────────────────────────────────
Final, consolidated threshold optimization for the Spectral Watch plugin.

Supersedes threshold_optimization_cv.py, threshold_optimization.py, and
threshold_optimization_ndmi_abert.py by running all four indices through one
spatial-block cross-validation, over as much of the Landsat archive as each
reference supports.

INDICES / REFERENCES
    NDWI         vs JRC Global Surface Water monthly history (1984-03 → 2021-12)
    NDWI         vs OPERA DSWx-HLS  (2016–17 and 2023 → present)
    NDVI         vs Dynamic World   (2015-06 → present)
    NDMI         vs NWI raster + NLCD 90/95 (static; growing-season MEDIAN composite)

METHOD
    Per-scene, time-joined sampling with a per-scene seed (NDWI/NDVI), or a
    single growing-season median composite sampled once (NDMI).  Points are tagged
    with 1 km UTM blocks; whole blocks go to train or test (80/20), and the
    threshold is picked by GroupKFold CV *inside* train, scored on held-out folds.
    The 20% test blocks are never touched until the final metrics.

PLUGIN CLASSES (mutually exclusive, assigned in this order)
    open water : NDWI >= x
    wetland    : NDWI <  x AND NDVI >= y AND NDMI >= z
    open land  : NDWI <  x AND NOT wetland
    NDVI is used only to classify wetland; open land is everything that is not
    water and does not meet the wetland standard.

KNOWN LIMITATIONS
  * Cross-sensor spectral differences.  TM (L5), ETM+ (L7) and OLI (L8/L9) have
    different band responses; one threshold is applied across all of them.
    Harmonization renames bands, it does not radiometrically cross-calibrate.
  * L7 SLC-off (post 2003-05-31) leaves stripe gaps.  Monthly median compositing
    fills most of them; valid_fraction flags what it cannot.
  * Reference coverage is uneven in time — JRC ends 2021-12, Dynamic World starts
    2015-06, OPERA has a 2018–2022 gap.  No single reference spans the whole
    series, which is why JRC (historical) and OPERA (recent) are both reported.
  * NDMI is applied out of its fitting domain in the time series: it is fit on a
    growing-season median composite but applied to individual monthly composites
    including winter.  The wetland band is the least certain of the three.
  * NDMI skill is low in absolute terms (earlier Abert runs: cv_iou 0.21–0.42).
    Report the number; do not oversell the wetland class.

Run: python final_threshold_optimization.py
"""

from __future__ import annotations

import io
import os
import datetime
import concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

# A TauDEM runtime can add its own older GDAL/PROJ data, driver, and DLL
# directories to the machine environment. Those paths are incompatible with
# newer Rasterio wheels, but valid conda/system GIS paths must be preserved.
# Filter only TauDEM entries, before importing any binary GIS package; failed
# imports cannot be safely repaired afterward.
def _remove_taudem_geo_paths():
    names = ('PROJ_DATA', 'PROJ_LIB', 'GDAL_DATA',
             'GDAL_DRIVER_PATH', 'OGR_DRIVER_PATH', 'PATH')
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        entries = [entry for entry in raw.split(os.pathsep) if entry]
        kept = [
            entry for entry in entries
            if not any(part.casefold().startswith('taudem')
                       for part in Path(entry.strip().strip('"')).parts)
        ]
        if len(kept) == len(entries):
            continue
        if kept:
            os.environ[name] = os.pathsep.join(kept)
        else:
            os.environ.pop(name, None)


_remove_taudem_geo_paths()
del _remove_taudem_geo_paths

import ee
import earthaccess
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling, transform_bounds
from pyproj import Transformer
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyArrowPatch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

# ── Landsat mission configuration ───────────────────────────────────────────
# Previously imported from gee_pipeline.py, which is retired. Inlined verbatim
# so this script is self-contained; the band mappings match the plugin's
# Lake_Classifer_FD/ee_backend.py, which is the authoritative copy for the QGIS
# side. Sentinel-2 is dropped — SENSORS below is Landsat only.
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
# applied across all four missions.
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

# Earth Engine Cloud project. Kept out of source control: set EE_PROJECT in the
# environment rather than hardcoding an ID here, so a project identifier cannot
# be committed by accident.
#     export EE_PROJECT=your-ee-project-id
EE_PROJECT = os.environ.get('EE_PROJECT', 'your-ee-project-id')

def _initialize_earth_engine():
    """Initialize from existing credentials without launching a browser."""
    try:
        ee.Initialize(project=EE_PROJECT)
    except ee.EEException as exc:
        if 'Please authorize access' not in str(exc):
            raise
        raise RuntimeError(
            "Earth Engine credentials are unavailable. Authenticate outside "
            "this script (for example, run 'earthengine authenticate' in a "
            "terminal), then rerun. Automatic browser authentication is "
            "intentionally disabled."
        ) from exc


_initialize_earth_engine()
del _initialize_earth_engine

SCRIPT_DIR = Path(__file__).resolve().parent

# Every figure is written in each of these formats (plus a no_text/ twin of each).
# SVG keeps <text> as real text rather than outlines, so labels stay editable in
# PowerPoint / Illustrator / Canva; the font list favours faces that exist on
# Windows so PowerPoint doesn't substitute and reflow them.
FIGURE_FORMATS = ('png', 'svg')
plt.rcParams['svg.fonttype']  = 'none'
plt.rcParams['font.family']   = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ── output layout ───────────────────────────────────────────────────────────
# outputs/
#   tables/                          final_threshold_summary.csv, landcover_monthly_*.csv
#   cache/opera_b01/                 downloaded OPERA water layers (restart-safe)
#   figures/roc/                     roc_curves.png
#   figures/residual_maps/           residual_opera_<lake>.png
#   figures/confusion_matrices/      confusion_matrix_<index>_opera_<lake>.png
#   figures/assembled/               residual_confusion_assembled.png
#   figures/timeseries/              landcover_stacked_*.png, stage_area_*.png
# Each figure folder keeps its text-free twins in a no_text/ subfolder under the
# SAME basename, so a pair is obvious and the whole text-free set can be grabbed
# at once (e.g. for annotating in Canva).
OUT_DIR    = SCRIPT_DIR / 'outputs'
TABLES_DIR = OUT_DIR / 'tables'
OPERA_CACHE_DIR = OUT_DIR / 'cache' / 'opera_b01'
FIG_DIRS = {
    'roc':                OUT_DIR / 'figures' / 'roc',
    'residual_maps':      OUT_DIR / 'figures' / 'residual_maps',
    'confusion_matrices': OUT_DIR / 'figures' / 'confusion_matrices',
    'assembled':          OUT_DIR / 'figures' / 'assembled',
    'threshold_drift':    OUT_DIR / 'figures' / 'threshold_drift',
    'timeseries':         OUT_DIR / 'figures' / 'timeseries',
}


def fig_path(kind, filename):
    """Path for a figure of `kind`, creating the folder and its no_text/ twin."""
    d = FIG_DIRS[kind]
    (d / 'no_text').mkdir(parents=True, exist_ok=True)
    return d / filename


def table_path(filename):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    return TABLES_DIR / filename


def _rel(p):
    """Path relative to the script dir when possible, for tidy logging."""
    try:
        return Path(p).relative_to(SCRIPT_DIR)
    except ValueError:
        return Path(p)

# ═══════════════════════ configuration ═══════════════════════

RUN_OPTIMIZATION  = False   # figures-only pass; thresholds come from THRESHOLD_SOURCE_CSV
RUN_RESIDUAL_MAPS = True
RUN_ASSEMBLED_FIGURE = True  # paper-ready 2-site residual + confusion plate
RUN_TIMESERIES    = False   # ~511 months; enable once the table looks right

ASSEMBLED_FIGSIZE = (7.5, 4.1)  # full-width paper figure, matching the reference aspect
ASSEMBLED_DPI     = 300
ASSEMBLED_FONT_FAMILY = 'Garamond'
ASSEMBLED_TEXT_SCALES = (1.15, 1.25, 1.35)  # fixed-layout comparison versions

# Figures-only runs intentionally default to the committed published calibration
# used by the plug-in defaults and paper figures. This legacy fit differs from a
# fresh RUN_OPTIMIZATION=True result. To redraw from this script's latest fitted
# table, set this to 'final_threshold_summary.csv'; a bare filename resolves under
# outputs/tables/, while an absolute path is used as given. Any CSV with
# site / index / reference / threshold_mean columns works.
THRESHOLD_SOURCE_CSV = SCRIPT_DIR / 'data' / 'threshold_cv_summary.csv'

# Epoch cuts — how the optimal threshold has drifted, so a recommendation can be
# based on recent data rather than a 40-year average.
#   * NDWI/NDVI epochs are FREE of new Earth Engine calls: they re-filter the
#     already-sampled dataframe and re-run the CV locally.
#   * NDMI epochs are NOT free — each needs its own median composite + sample, so
#     they sit behind their own flag with a cheap scene-count precheck.
RUN_EPOCH_CUTS   = True
RUN_NDMI_EPOCHS  = True    # extra GEE round trip per epoch per site
EPOCH_YEARS      = 5        # width of the calendar bins
OLI_EPOCH_COUNT  = 3        # within-OLI (L8/L9) sub-epochs — the sensor-free test
OLI_ERA_START    = 2013     # L8 launch; TM/ETM+ excluded so radiometry is constant
MIN_EPOCH_POINTS = 200      # below this an epoch is reported n/a, not fitted to noise
MIN_EPOCH_BLOCKS = 20
MIN_EPOCH_SCENES = 15       # NDMI only: skip building a composite from too few scenes

aoi_gsl   = ee.Geometry.Rectangle([-113.3061047, 40.51087812, -111.8534658, 41.87473984])
aoi_abert = ee.Geometry.Rectangle([-120.6098988, 42.40054901, -119.9981325, 42.85172324])

AOI_BOUNDS = {
    'Great Salt Lake': (-113.3061047, 40.51087812, -111.8534658, 41.87473984),
    'Lake Abert':      (-120.6098988, 42.40054901, -119.9981325, 42.85172324),
}
AOIS = {'Great Salt Lake': aoi_gsl, 'Lake Abert': aoi_abert}

# Landsat 4 is skipped: L5 already starts 1984, matching JRC's first month, so
# adding L4 would widen the mission list without widening the usable training
# window. Any sensor listed here must have an entry in COLLECTIONS, BAND_MAP,
# and SENSOR_WINDOWS above.
SENSORS         = ['L5', 'L7', 'L8', 'L9']
ARCHIVE_START   = datetime.date(1984, 1, 1)
ARCHIVE_END     = datetime.date.today()
CLOUD_COVER_MAX = 20

# Each reference is sampled over its own availability, not a common window.
JRC_START,   JRC_END   = '1984-03-01', '2021-12-31'   # GSW 1.4 monthly history
DW_START,    DW_END    = '2015-06-27', ARCHIVE_END.isoformat()
OPERA_START, OPERA_END = '2016-01-01', ARCHIVE_END.isoformat()

# NWI wetland masks (1 = wetland), one local raster per site.
#
# These are NOT committed: GSL_Wetland_Raster.tif alone is ~114 MB, over GitHub's
# 100 MB per-file hard limit, and a binary raster in git history is permanent
# weight. Keep them outside the repo and point NWI_DIR at them:
#     export NWI_DIR=/path/to/your/wetland/rasters
# The default is data/nwi/ beside this script, which is gitignored.
#
# Only the NDMI/wetland calibration reads these; the NDWI and NDVI paths run
# without them (see the guard in optimize_site).
NWI_DIR = Path(os.environ.get('NWI_DIR', SCRIPT_DIR / 'data' / 'nwi'))
NWI_RASTERS = {
    'Great Salt Lake': NWI_DIR / 'GSL_Wetland_Raster.tif',
    'Lake Abert':      NWI_DIR / 'Abert_Wetlands_Raster.tif',
}

NLCD_COLLECTION      = 'USGS/NLCD_RELEASES/2019_REL/NLCD'
NLCD_YEAR            = '2019'
NLCD_WETLAND_CLASSES = [90, 95]        # woody + emergent herbaceous

# NDMI: growing-season MEDIAN composite over the full archive, sampled once.
NDMI_MONTHS       = (6, 9)
NDMI_SAMPLES      = 4500
# Escalating tileScale for the NDMI sample. A 40-year, 4-sensor median over the
# GSL AOI is heavy enough to trip "User memory limit exceeded" at the default
# tileScale=1; larger tiles trade speed for a lower per-tile memory ceiling.
NDMI_TILE_SCALES  = (4, 8, 16)

SAMPLES_PER_IMAGE = 200
SNOW_MAX_FRAC     = 0.05               # drop a DW scene only if >5% of AOI is snow
OPERA_WINDOW_DAYS = 5

RANDOM_SEED      = 42
BLOCK_SIZE_M     = 1000
SPLIT_FRACS      = (0.80, 0.20)
CV_SPLITS        = 5
GRID_STEPS_CV    = 150
SELECTION_METRIC = 'iou'

_DW_VEG    = [1, 2, 3, 4, 5]
_DW_NONVEG = [0, 6, 7]

# Every index × reference pair that gets optimized. (index, direction)
REFERENCES = {
    'JRC':   [('NDWI', 'above')],
    'OPERA': [('NDWI', 'above')],
    'DW':    [('NDVI', 'above')],
    'NWI':   [('NDMI', 'above')],
    'NLCD':  [('NDMI', 'above')],
}

# The (index, reference) pairs this table must carry for every site. These are
# what water_extent_timeseries.py reads and what supplies the plug-in's NDWI and
# NDVI defaults. A run is checked against this list before it finishes.
#
# NDMI is deliberately absent. threshold_optimization_ndmi.py owns the wetland
# moisture cutoff, and its narrower 2010-2020 Landsat 8/9 window succeeds at both
# sites where the full-archive composite below exhausts Earth Engine's memory on
# the larger Great Salt Lake AOI. This script still fits NDMI where it can, but
# those rows are secondary; data/ndmi_threshold_cv_all.csv is authoritative, and
# the complete threshold set is the union of the two outputs.
REQUIRED_ROWS = [
    ('NDWI', 'JRC'),      # historical water extent (Great Salt Lake)
    ('NDWI', 'OPERA'),    # plug-in default at both sites; historical at Abert
    ('NDVI', 'DW'),       # vegetation cutoff
]

# Residual maps / confusion matrices use the OPERA-window scene.
OPERA_RESIDUAL_YEAR = 2023
OPERA_MAX_DAYS      = 6
OPERA_MIN_COVERAGE  = 0.90
OPERA_SCALE_M       = 120
MIN_AOI_COVERAGE    = 0.90

# Time series (GSL only)
TS_START           = datetime.date(1984, 1, 1)
TS_SCALE_M         = 30
MIN_VALID_FRACTION = 0.5
MAX_WORKERS        = 10
NWIS_DV_URL        = 'https://waterservices.usgs.gov/nwis/dv/'
USGS_ELEVATION_STATIONS = {
    'Great Salt Lake': [
        {'label': 'GSL south arm — Saltair (USGS 10010000)', 'site': '10010000', 'param': '62614'},
        {'label': 'GSL north arm — Saline (USGS 10010100)',  'site': '10010100', 'param': '62614'},
    ],
}

_EARTHACCESS_READY = False


def _ensure_earthaccess():
    global _EARTHACCESS_READY
    if not _EARTHACCESS_READY:
        # "all" tries EARTHDATA_* environment credentials, netrc, then an
        # interactive prompt.  Requiring netrc alone makes a first run fail on
        # machines where credentials have not already been persisted.
        guidance = (
            "NASA Earthdata authentication failed. Set EARTHDATA_TOKEN, set "
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD, configure a netrc "
            "file, or enter credentials at the interactive prompt."
        )
        try:
            auth = earthaccess.login(strategy="all")
        except Exception as exc:
            raise RuntimeError(guidance) from exc
        if not auth.authenticated:
            raise RuntimeError(guidance)
        _EARTHACCESS_READY = True


# ═══════════════════════ figure helper ═══════════════════════

def _require_assembled_font():
    """Fail clearly instead of silently substituting another paper font."""
    properties = font_manager.FontProperties(family=[ASSEMBLED_FONT_FAMILY])
    try:
        return font_manager.findfont(properties, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            f"The assembled figure requires {ASSEMBLED_FONT_FAMILY}, but "
            "Matplotlib cannot find it. Install a licensed copy of the font, "
            "restart Python, and rerun; no substitute font will be used."
        ) from exc


def _save_textless_copy(fig, out_path, dpi=150, bbox_inches='tight'):
    """Save a second copy of `fig` with every character removed — titles, axis
    labels, tick labels, and legends stripped; plot geometry/imagery kept.
    Mutates the figure, so call it AFTER the as-is copy has been saved."""
    for t in list(fig.texts):
        t.remove()
    for lg in list(fig.legends):
        lg.remove()
    for ax in fig.axes:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for t in list(ax.texts):
            t.remove()
    out_path = Path(out_path)
    dest_dir = out_path.parent / 'no_text'
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(dest_dir / f'{out_path.stem}.{ext}', dpi=dpi,
                    bbox_inches=bbox_inches)
    print(f"  saved {_rel(dest_dir / out_path.stem)}.{{{','.join(FIGURE_FORMATS)}}}")


def _save_fig(fig, out_path, dpi=150, bbox_inches='tight'):
    """Save the as-is figure in every FIGURE_FORMATS, then its text-free twin."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(out_path.with_suffix(f'.{ext}'), dpi=dpi,
                    bbox_inches=bbox_inches)
    print(f"Saved {_rel(out_path.with_suffix(''))}.{{{','.join(FIGURE_FORMATS)}}}")
    _save_textless_copy(fig, out_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


# ═══════════════ harmonized multi-sensor collection ═══════════════
# Index math moves to the common green/red/nir/swir1 schema so TM and ETM+ can be
# sampled alongside OLI. _cloud_mask and the SR scaling are valid for every
# Collection-2 L2 sensor, so only band selection/renaming changes.

def _cloud_mask(image):
    """High-confidence cloud only (QA_PIXEL bits 8-9 == 3). Shadow is deliberately
    not masked: bright salt flats and exposed lakebed trip CFMask into false
    shadow flags."""
    qa = image.select('QA_PIXEL')
    cloud_high = qa.rightShift(8).bitwiseAnd(3).eq(3)
    return image.updateMask(cloud_high.Not())


def _scale_and_mask(image):
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2).clamp(0, 1)
    return _cloud_mask(image.addBands(optical, overwrite=True))


def _harmonize(image, sensor):
    src = list(BAND_MAP[sensor].values())
    dst = list(BAND_MAP[sensor].keys())
    return image.select(src).rename(dst)


def _sensor_collection(sensor, aoi, start, end):
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


def build_landsat_collection(aoi, start=ARCHIVE_START, end=ARCHIVE_END, sensors=SENSORS):
    """Merged, cloud-masked, band-harmonized L5/L7/L8/L9 over the archive."""
    cols = [c for c in (_sensor_collection(s, aoi, start, end) for s in sensors)
            if c is not None]
    merged = cols[0]
    for c in cols[1:]:
        merged = merged.merge(c)
    return merged


def _ndwi(img):  return img.normalizedDifference(['green', 'nir']).rename('NDWI')
def _ndvi(img):  return img.normalizedDifference(['nir', 'red']).rename('NDVI')
def _ndmi(img):  return img.normalizedDifference(['nir', 'swir1']).rename('NDMI')


# ═══════════════════════ sampling ═══════════════════════

def _time_join(primary, secondary, window_days, match_key):
    filt = ee.Filter.maxDifference(
        difference=int(window_days * 86400 * 1000),
        leftField='system:time_start', rightField='system:time_start',
    )
    return ee.Join.saveBest(matchKey=match_key, measureKey='_dt').apply(
        primary, secondary, filt)


def _sampler(aoi):
    """Per-scene sampler: the seed varies with the scene so points don't collapse
    onto the same fixed locations on every date.

    img.sample() propagates band values only, so the acquisition time and sensor
    are stamped back onto each feature — the epoch cuts need both, and `sensor`
    is what lets a real change be told apart from a TM/ETM+/OLI artifact.
    """
    def _sample(img):
        img  = ee.Image(img)
        seed = ee.Number(img.get('system:time_start')).mod(1000000)
        fc = img.sample(region=aoi, scale=30, numPixels=SAMPLES_PER_IMAGE,
                        seed=seed, geometries=True, dropNulls=True)
        return fc.map(lambda f: f.set('t_ms',   img.get('system:time_start'),
                                      'sensor', img.get('sensor')))
    return _sample


def _fc_to_df(fc, page_size=4500):
    """Paginate past GEE's 5000-element getInfo limit, keeping lon/lat."""
    rows, offset = [], 0
    while True:
        batch = fc.toList(page_size, offset).getInfo()
        if not batch:
            break
        for f in batch:
            props = dict(f['properties'])
            lon, lat = f['geometry']['coordinates'][:2]
            props['lon'], props['lat'] = lon, lat
            rows.append(props)
        if len(batch) < page_size:
            break
        offset += page_size
    return pd.DataFrame(rows)


def collect_water_indices_jrc(ls_col, aoi):
    """NDWI labelled by JRC open water (class 2), 1984-03 → 2021-12."""
    jrc_col = (ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
               .filterBounds(aoi).filterDate(JRC_START, JRC_END))

    def prep(feature):
        img   = ee.Image(feature)
        jrc   = ee.Image(img.get('jrc')).select('water')
        valid = jrc.neq(0)                                   # 0 = JRC no-data
        label = jrc.eq(2).rename('label').updateMask(valid)  # water = 1
        return (_ndwi(img).addBands(label)
                .set('system:time_start', img.get('system:time_start'),
                     'sensor', img.get('sensor')))

    joined = _time_join(ls_col.filterDate(JRC_START, JRC_END), jrc_col,
                        window_days=31, match_key='jrc').map(prep)
    return _fc_to_df(joined.map(_sampler(aoi)).flatten())


def collect_ndvi_dw(ls_col, aoi):
    """NDVI labelled by Dynamic World vegetation, 2015-06 → present."""
    dw_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterBounds(aoi).filterDate(DW_START, DW_END))

    def flag_snow(img):
        # Areal snow fraction at 300 m (cheap over the large GSL AOI); mean()
        # ignores masked pixels, so this is snow / valid-observed.
        frac = (img.select('label').eq(8)
                .reduceRegion(ee.Reducer.mean(), aoi, 300, maxPixels=1e8)
                .values().get(0))
        return img.set('snow_frac', frac)

    dw_no_snow = dw_col.map(flag_snow).filter(ee.Filter.lte('snow_frac', SNOW_MAX_FRAC))

    def prep(feature):
        img   = ee.Image(feature)
        dw    = ee.Image(img.get('dw')).select('label')
        label = dw.remap(_DW_VEG + _DW_NONVEG,
                         [1] * len(_DW_VEG) + [0] * len(_DW_NONVEG),
                         defaultValue=-1).rename('label')
        label = label.updateMask(label.gte(0))               # drop residual snow
        return (_ndvi(img).addBands(label)
                .set('system:time_start', img.get('system:time_start'),
                     'sensor', img.get('sensor')))

    joined = _time_join(ls_col.filterDate(DW_START, DW_END), dw_no_snow,
                        window_days=3, match_key='dw').map(prep)
    return _fc_to_df(joined.map(_sampler(aoi)).flatten())


# ---- OPERA DSWx-HLS (NASA Earthdata, not GEE) ----
# B01 = WTR: 0 not water, 1 open water, 2 partial surface water,
# 252 snow/ice, 253 cloud/cloud shadow, 254 ocean masked, 255 no data.
# Only classes 0 and 1 are evaluated below; all other classes are excluded.

def _opera_search(bounds, start=OPERA_START, end=OPERA_END):
    _ensure_earthaccess()
    try:
        return earthaccess.search_data(short_name="OPERA_L3_DSWx-HLS_V1",
                                       bounding_box=bounds, temporal=(start, end))
    except Exception as e:
        print(f"    [OPERA search failed: {type(e).__name__}]")
        return []


def _opera_date_index(results):
    idx = defaultdict(list)
    for g in results:
        dt = g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        idx[pd.Timestamp(dt, tz='UTC').normalize()].append(g)
    return idx


def _closest_opera(ls_ts_ms, opera_idx, window_days=OPERA_WINDOW_DAYS):
    ls_day    = pd.Timestamp(ls_ts_ms, unit='ms', tz='UTC').normalize()
    best_diff, best = window_days + 1, []
    for day, granules in opera_idx.items():
        diff = abs((day - ls_day).days)
        if diff < best_diff:
            best_diff, best = diff, granules
    return best if best_diff <= window_days else []


def _read_opera_b01(granule, aoi_bounds):
    """Download/cache B01_WTR and clip it to the AOI.

    Passing earthaccess's file-like object to rasterio makes rasterio read the
    entire remote TIFF into memory. An explicit local cache avoids that slow
    path and lets an interrupted figure run resume without downloading again.
    """
    b01_name = 'unknown B01_WTR file'
    try:
        b01_url = next((link for link in granule.data_links()
                        if urlsplit(link).path.lower().endswith('_b01_wtr.tif')),
                       None)
        if b01_url is None:
            return None
        b01_name = Path(urlsplit(b01_url).path).name
        provider = granule.get('meta', {}).get('provider-id')
        OPERA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        b01_path = OPERA_CACHE_DIR / b01_name
        if not b01_path.exists():
            # CMR search is public; authenticate only when a protected asset
            # really needs downloading. Cached OPERA reads skip this login.
            _ensure_earthaccess()
            print(f"      downloading {b01_name}...")
            downloaded = earthaccess.download(
                b01_url, local_path=OPERA_CACHE_DIR, provider=provider,
                threads=1, show_progress=False)
            if not downloaded:
                return None
            b01_path = Path(downloaded[0])
        with rasterio.open(b01_path) as src:
            if src.crs is None:
                raise rasterio.errors.CRSError(f'{b01_name} has no CRS metadata')
            # Transform the full rectangle boundary. Transforming only its SW
            # and NE corners can clip the opposite edges after rotation into a
            # projected OPERA tile CRS.
            minx, miny, maxx, maxy = transform_bounds(
                'EPSG:4326', src.crs, *aoi_bounds, densify_pts=21)
            data, transform = rio_mask(src, [mapping(box(minx, miny, maxx, maxy))],
                                       crop=True)
            data = data[0]
            return data, transform, src.crs
    except rasterio.errors.CRSError as exc:
        # A CRS error is normally a process-wide GDAL/PROJ configuration issue;
        # retrying every tile only wastes downloads and hides the real cause.
        raise RuntimeError(f'CRS failure reading {b01_name}: {exc}') from exc
    except Exception as e:
        print(f"      [OPERA {b01_name} skipped: {type(e).__name__}]")
        return None


def collect_water_indices_opera(ls_col, aoi, aoi_bounds):
    """NDWI labelled by OPERA open water (WTR class 1). Client-side scene
    loop, since OPERA lives on NASA Earthdata rather than in Earth Engine."""
    results = _opera_search(aoi_bounds)
    cols = ['NDWI', 'label', 'lon', 'lat', 't_ms', 'sensor']
    if not results:
        return pd.DataFrame(columns=cols)

    opera_idx = _opera_date_index(results)
    sub       = ls_col.filterDate(OPERA_START, OPERA_END)
    ls_ts     = sub.aggregate_array('system:time_start').getInfo()
    ls_sensor = sub.aggregate_array('sensor').getInfo()      # one call, for epoch cuts
    ls_list   = sub.toList(sub.size())
    rows      = []

    for i, ts in enumerate(ls_ts):
        matched = _closest_opera(ts, opera_idx)
        if not matched:
            continue
        img_h = ee.Image(ls_list.get(i))
        img   = _ndwi(img_h)
        feats = img.sample(region=aoi, scale=30, numPixels=SAMPLES_PER_IMAGE,
                           seed=int(ts % 1000000), geometries=True, dropNulls=True
                           ).getInfo()['features']
        if not feats:
            continue

        opera_data = None
        for g in matched:
            opera_data = _read_opera_b01(g, aoi_bounds)
            if opera_data is not None:
                break
        if opera_data is None:
            continue
        data, transform, crs = opera_data
        t_proj = Transformer.from_crs('EPSG:4326', crs, always_xy=True)

        for f in feats:
            lon, lat = f['geometry']['coordinates'][:2]
            x, y = t_proj.transform(lon, lat)
            r, c = rowcol(transform, x, y)
            if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                cls = int(data[r, c])
                if cls in (0, 1):                    # exclude partial/cloud/fill
                    rows.append({'NDWI':  f['properties'].get('NDWI'),
                                 'label': int(cls == 1), 'lon': lon, 'lat': lat,
                                 't_ms': ts,
                                 'sensor': ls_sensor[i] if i < len(ls_sensor) else None})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _safe_opera(site, ls_col, aoi, bounds):
    """OPERA streams from Earthdata and is prone to transient failures; never let
    it block the GEE-native references."""
    cols = ['NDWI', 'label', 'lon', 'lat', 't_ms', 'sensor']
    try:
        return collect_water_indices_opera(ls_col, aoi, bounds)
    except Exception as e:
        print(f"    [{site} OPERA collection failed: {type(e).__name__} — skipping]")
        return pd.DataFrame(columns=cols)


# ---- NDMI vs NWI raster + NLCD (growing-season MEDIAN composite) ----

def nlcd_wetland():
    """NLCD wetland mask: 1 for classes 90 (woody) / 95 (emergent herbaceous)."""
    lc = (ee.ImageCollection(NLCD_COLLECTION)
          .filter(ee.Filter.eq('system:index', NLCD_YEAR))
          .first().select('landcover'))
    mask = lc.eq(NLCD_WETLAND_CLASSES[0])
    for cls in NLCD_WETLAND_CLASSES[1:]:
        mask = mask.Or(lc.eq(cls))
    return mask.rename('NLCD')


def _growing_season(ls_col):
    return ls_col.filter(ee.Filter.calendarRange(NDMI_MONTHS[0], NDMI_MONTHS[1], 'month'))


def growing_season_median(ls_col):
    """ONE median reflectance composite over every growing-season scene.

    Both NDMI and NDWI are derived from this single composite rather than from two
    separate median-of-index reductions. Two reductions over ~40 years x 4 sensors
    is what tripped 'User memory limit exceeded' on the GSL AOI, and index-of-median
    also matches how the time series computes its indices (_monthly_feature medians
    the bands first), so the threshold is fit on the same basis it is applied on.
    Only the three bands the two indices need are carried.
    """
    return _growing_season(ls_col).select(['green', 'nir', 'swir1']).median()


def _sample_with_retry(image, aoi, num_pixels, seed=RANDOM_SEED):
    """image.sample(), escalating tileScale before giving up.

    tileScale splits the computation into smaller tiles, trading speed for a lower
    per-tile memory ceiling — the standard remedy for 'User memory limit exceeded'.
    """
    last = None
    for ts in NDMI_TILE_SCALES:
        try:
            return image.sample(region=aoi, scale=30, numPixels=num_pixels,
                                seed=seed, geometries=True,
                                tileScale=ts).getInfo()['features']
        except ee.ee_exception.EEException as e:
            if 'memory' not in str(e).lower():
                raise
            print(f"    [tileScale={ts} hit the memory limit — escalating]")
            last = e
    raise last


def collect_ndmi(ls_col, aoi, nwi_raster_path, ndwi_water_threshold):
    """Sample the NDMI median composite once; label from NWI raster AND NLCD.

    Persistent open water is dropped from the evaluation entirely (neither
    intersection nor union) using the site's JRC-fit NDWI threshold — NDMI is a
    vegetation-moisture index and cannot be expected to flag open lake water.
    """
    if not Path(nwi_raster_path).exists():
        raise FileNotFoundError(
            f"NWI raster not found: {nwi_raster_path}\n"
            f"    These rasters are too large to commit (GSL is ~114 MB). "
            f"Set NWI_DIR to wherever you keep them, or place them in "
            f"{NWI_DIR}/ — see the README.")

    with rasterio.open(nwi_raster_path) as src:
        nwi, nwi_transform, nwi_crs = src.read(1), src.transform, src.crs
    to_nwi = Transformer.from_crs('EPSG:4326', nwi_crs, always_xy=True)

    med = growing_season_median(ls_col)          # one reduction, not two
    composite = _ndmi(med).addBands(_ndwi(med)).addBands(nlcd_wetland())
    feats = _sample_with_retry(composite, aoi, NDMI_SAMPLES)
    cols = ['NDMI', 'NWI', 'NLCD', 'ndwi', 'lon', 'lat']
    if not feats:
        return pd.DataFrame(columns=cols)

    rows = []
    for f in feats:
        v = f['properties'].get('NDMI')
        if v is None:
            continue
        lon, lat = f['geometry']['coordinates'][:2]
        x, y = to_nwi.transform(lon, lat)
        r, c = rowcol(nwi_transform, x, y)
        if 0 <= r < nwi.shape[0] and 0 <= c < nwi.shape[1]:
            rows.append({'NDMI': v,
                         'NWI':  int(nwi[r, c] == 1),
                         'NLCD': f['properties'].get('NLCD'),
                         'ndwi': f['properties'].get('NDWI'),
                         'lon': lon, 'lat': lat})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    if len(df) and ndwi_water_threshold is not None:
        open_water = df['ndwi'] > ndwi_water_threshold
        print(f"    excluding {int(open_water.sum())} persistent open-water pts "
              f"(NDWI > {ndwi_water_threshold:+.4f}) of {len(df)}")
        df = df[~open_water].reset_index(drop=True)
    return df


# ═══════════════ spatial blocking + block CV ═══════════════

def _utm_epsg(aoi):
    lon, lat = aoi.centroid(1).coordinates().getInfo()
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone if lat >= 0 else 32700 + zone), lon, lat


def assign_blocks_and_split(df, epsg, block_size_m=BLOCK_SIZE_M,
                            fracs=SPLIT_FRACS, seed=RANDOM_SEED):
    """Tag each point with its 1 km UTM block, then assign whole blocks to
    train/test so spatially-correlated neighbours never straddle the split."""
    if not len(df):
        return df
    to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = to_utm.transform(df['lon'].values, df['lat'].values)
    df = df.copy()
    df['block_id'] = [f'{a}_{b}' for a, b in
                      zip(np.floor(x / block_size_m).astype(int),
                          np.floor(y / block_size_m).astype(int))]
    blocks = np.array(sorted(df['block_id'].unique()))
    np.random.default_rng(seed).shuffle(blocks)
    n_tr = int(len(blocks) * fracs[0])
    split_of = {b: 'train' for b in blocks[:n_tr]}
    split_of.update({b: 'test' for b in blocks[n_tr:]})
    df['split'] = df['block_id'].map(split_of)
    return df


def _metrics(labels, pred):
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return dict(precision=prec, recall=sens, specificity=spec, f1=f1, iou=iou,
                youden_j=sens + spec - 1)


def grid_search_threshold(values, labels, direction, n_steps=GRID_STEPS_CV):
    """Sweep thresholds across the index's central 99% range and score each."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    lo, hi = np.percentile(values, [0.5, 99.5])
    rows = []
    for t in np.linspace(lo, hi, n_steps):
        pred = (values >= t) if direction == 'above' else (values <= t)
        rows.append(dict(threshold=t, **_metrics(labels, pred)))
    return pd.DataFrame(rows)


def eval_threshold(values, labels, threshold, direction):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (values >= threshold) if direction == 'above' else (values <= threshold)
    return dict(n=len(labels), **_metrics(labels, pred))


def cv_optimal_threshold(values, labels, groups, direction,
                         metric=SELECTION_METRIC, n_splits=CV_SPLITS):
    """Block-grouped CV. Each fold fits on its training part and is SCORED on its
    held-out part, so cv_iou is a clean generalization estimate that never touches
    the test blocks."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    k = int(min(n_splits, len(np.unique(groups))))
    if k < 2:
        gs = grid_search_threshold(values, labels, direction)
        best = float(gs.loc[gs[metric].idxmax(), 'threshold'])
        return best, 0.0, float('nan')

    gkf = GroupKFold(n_splits=k)
    fold_t, fold_s = [], []
    for tr_idx, ho_idx in gkf.split(values, labels, groups):
        gs = grid_search_threshold(values[tr_idx], labels[tr_idx], direction)
        t = float(gs.loc[gs[metric].idxmax(), 'threshold'])
        fold_t.append(t)
        fold_s.append(eval_threshold(values[ho_idx], labels[ho_idx], t, direction)[metric])
    return float(np.mean(fold_t)), float(np.std(fold_t)), float(np.mean(fold_s))


def optimize_index(df, value_col, label_col, direction, site, index, reference,
                   sensors, date_range):
    """Block-CV threshold for one index × reference, evaluated on held-out test."""
    d = df.dropna(subset=[value_col]).copy()
    d['_lab'] = d[label_col].fillna(0).astype(int)
    tr, te = d[d['split'] == 'train'], d[d['split'] == 'test']
    if len(tr) < 10 or len(te) < 10 or d['_lab'].nunique() < 2:
        print(f"    [skip {index} vs {reference}: too few usable samples]")
        return None, None

    mean_t, std_t, cv_iou = cv_optimal_threshold(
        tr[value_col].values, tr['_lab'].values, tr['block_id'].values, direction)
    test_m = eval_threshold(te[value_col].values, te['_lab'].values, mean_t, direction)

    y_true, y_score = te['_lab'].values, te[value_col].values
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:                     # single class in test
        auc = float('nan')

    summary = dict(
        site=site, index=index, reference=reference,
        sensors=sensors, date_range=date_range,
        threshold_mean=round(mean_t, 4), threshold_std=round(std_t, 4),
        n_train=len(tr), n_test=len(te), n_blocks=int(d['block_id'].nunique()),
        cv_iou=round(cv_iou, 3) if cv_iou == cv_iou else float('nan'),
        test_iou=round(test_m['iou'], 3), test_f1=round(test_m['f1'], 3),
        test_precision=round(test_m['precision'], 3),
        test_recall=round(test_m['recall'], 3),
        test_auc=round(auc, 3) if auc == auc else float('nan'),
    )
    roc = dict(site=site, index=index, reference=reference, threshold=mean_t,
               y_true=y_true, y_score=y_score, auc=auc)
    return summary, roc


# ═══════════════════════ optimization driver ═══════════════════════

# ═══════════════════════ epoch cuts ═══════════════════════
# Two cuts, answering two different questions:
#   'cal5'  fixed calendar bins across the whole archive — the headline drift
#           curve, but the sensor mix changes between bins, so a shift can be a
#           TM/ETM+/OLI radiometry artifact rather than real change.
#   'oli'   within L8/L9 only.  Radiometry is held constant, so a shift here is
#           real change on the ground.  This is the cut that says whether a
#           recommendation should use recent data.

def _calendar_epochs(width=EPOCH_YEARS, start=None, end=None):
    start = start or ARCHIVE_START.year
    end   = end or ARCHIVE_END.year
    out = []
    y = start
    while y <= end:
        hi = min(y + width - 1, end)
        out.append((f'{y}-{hi}', y, hi))
        y += width
    return out


def _oli_epochs(count=OLI_EPOCH_COUNT, start=OLI_ERA_START, end=None):
    """Equal-width sub-epochs inside the OLI era (L8/L9 only)."""
    end = end or ARCHIVE_END.year
    span = max(end - start + 1, count)
    width = int(np.ceil(span / count))
    out, y = [], start
    while y <= end:
        hi = min(y + width - 1, end)
        out.append((f'{y}-{hi}', y, hi))
        y += width
    return out


def _sensor_mix(sub):
    """'L8:1204,L9:388' — makes the sensor confound visible per epoch."""
    if 'sensor' not in sub.columns or sub['sensor'].isna().all():
        return ''
    vc = sub['sensor'].dropna().value_counts()
    return ','.join(f'{k}:{int(v)}' for k, v in vc.items())


def _epoch_subset(df, y0, y1, oli_only=False):
    if 't_ms' not in df.columns:
        return df.iloc[0:0]
    d = df.dropna(subset=['t_ms']).copy()
    yr = pd.to_datetime(d['t_ms'], unit='ms').dt.year
    d = d[(yr >= y0) & (yr <= y1)]
    if oli_only and 'sensor' in d.columns:
        d = d[d['sensor'].isin(['L8', 'L9'])]
    return d


def optimize_epochs(df, value_col, label_col, direction, site, index, reference,
                    epsg, cut, epochs, pooled, oli_only=False):
    """Re-fit `value_col` inside each epoch. Blocks and the 80/20 split are
    re-derived WITHIN each epoch, so the held-out test stays spatially honest at
    every bucket rather than inheriting the pooled split."""
    rows = []
    for label, y0, y1 in epochs:
        sub = _epoch_subset(df, y0, y1, oli_only)
        if len(sub) < MIN_EPOCH_POINTS:
            rows.append(dict(cut=cut, site=site, index=index, reference=reference,
                             epoch=label, n_points=len(sub), note='too few points'))
            continue
        sub = assign_blocks_and_split(sub, epsg)
        if sub['block_id'].nunique() < MIN_EPOCH_BLOCKS:
            rows.append(dict(cut=cut, site=site, index=index, reference=reference,
                             epoch=label, n_points=len(sub), note='too few blocks'))
            continue

        s, _ = optimize_index(sub, value_col, label_col, direction, site, index,
                              reference, _sensor_mix(sub), label)
        if s is None:
            rows.append(dict(cut=cut, site=site, index=index, reference=reference,
                             epoch=label, n_points=len(sub), note='CV failed'))
            continue

        drift = s['threshold_mean'] - pooled if pooled is not None else float('nan')
        rows.append(dict(
            cut=cut, site=site, index=index, reference=reference, epoch=label,
            year_start=y0, year_end=y1, sensors=s.pop('sensors'),
            n_points=len(sub), n_train=s['n_train'], n_test=s['n_test'],
            n_blocks=s['n_blocks'],
            threshold_mean=s['threshold_mean'], threshold_std=s['threshold_std'],
            cv_iou=s['cv_iou'], test_iou=s['test_iou'], test_auc=s['test_auc'],
            pooled_threshold=pooled,
            drift=round(drift, 4) if drift == drift else float('nan'),
            # the decision rule: drift only matters if it exceeds the fold noise
            drift_exceeds_fold_std=bool(abs(drift) > max(s['threshold_std'], 1e-9))
            if drift == drift else False,
            note=''))
    return rows


def _ndmi_epoch_rows(site, aoi, ls_col, epsg, nwi_path, jrc_ndwi, pooled, epochs):
    """NDMI per epoch. Unlike the other indices this is NOT free — each epoch needs
    its own growing-season median composite and its own sample."""
    rows = []
    for label, y0, y1 in epochs:
        window = ls_col.filter(ee.Filter.calendarRange(y0, y1, 'year'))
        n_scenes = _growing_season(window).size().getInfo()   # cheap precheck
        if n_scenes < MIN_EPOCH_SCENES:
            print(f"    [{label}: {n_scenes} growing-season scene(s) — skipped]")
            rows.append(dict(cut='cal5', site=site, index='NDMI', reference='NWI',
                             epoch=label, n_points=0, note='too few scenes'))
            continue
        print(f"    {label}: compositing {n_scenes} growing-season scene(s)...")
        sub = collect_ndmi(window, aoi, nwi_path, jrc_ndwi)
        if len(sub) < MIN_EPOCH_POINTS:
            rows.append(dict(cut='cal5', site=site, index='NDMI', reference='NWI',
                             epoch=label, n_points=len(sub), note='too few points'))
            continue
        sub = assign_blocks_and_split(sub, epsg)
        s, _ = optimize_index(sub, 'NDMI', 'NWI', 'above', site, 'NDMI', 'NWI',
                              f'{"+".join(SENSORS)}', label)
        if s is None:
            continue
        drift = s['threshold_mean'] - pooled if pooled is not None else float('nan')
        rows.append(dict(
            cut='cal5', site=site, index='NDMI', reference='NWI', epoch=label,
            year_start=y0, year_end=y1, sensors=s.pop('sensors'),
            n_points=len(sub), n_train=s['n_train'], n_test=s['n_test'],
            n_blocks=s['n_blocks'],
            threshold_mean=s['threshold_mean'], threshold_std=s['threshold_std'],
            cv_iou=s['cv_iou'], test_iou=s['test_iou'], test_auc=s['test_auc'],
            pooled_threshold=pooled,
            drift=round(drift, 4) if drift == drift else float('nan'),
            drift_exceeds_fold_std=bool(abs(drift) > max(s['threshold_std'], 1e-9))
            if drift == drift else False,
            note=''))
    return rows


def plot_threshold_drift(epoch_df, cut, out_path):
    """Threshold vs epoch, with a ±threshold_std band and the pooled value as a
    dashed reference. If a point's band overlaps the dashed line, the drift is
    inside the fitting noise and the pooled threshold is still fine."""
    d = epoch_df[(epoch_df['cut'] == cut) & epoch_df['threshold_mean'].notna()]
    if not len(d):
        return
    indices = [i for i in ('NDWI', 'NDVI', 'NDMI') if i in set(d['index'])]
    fig, axes = plt.subplots(1, len(indices), figsize=(5.2 * len(indices), 4.6),
                             squeeze=False)
    for ax, index in zip(axes[0], indices):
        di = d[d['index'] == index]
        for (site, ref), g in di.groupby(['site', 'reference']):
            g = g.sort_values('year_start')
            x = g['year_start'] + (g['year_end'] - g['year_start']) / 2
            color = _SITE_COLOR.get(site, '#666666')
            ls = _REF_STYLE.get(ref, '-')
            ax.plot(x, g['threshold_mean'], ls, marker='o', ms=5, lw=1.8,
                    color=color, label=f'{site} vs {ref}')
            ax.fill_between(x, g['threshold_mean'] - g['threshold_std'],
                            g['threshold_mean'] + g['threshold_std'],
                            color=color, alpha=0.15, lw=0)
            pooled = g['pooled_threshold'].dropna()
            if len(pooled):
                ax.axhline(pooled.iloc[0], color=color, ls=':', lw=1.2, alpha=0.8)
        ax.set_title(f'{index} — threshold by epoch ({cut})', fontsize=10)
        ax.set_xlabel('epoch midpoint (year)')
        ax.set_ylabel(f'{index} threshold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path)


def optimize_site(site):
    """Every index × reference for one lake. Returns (summaries, roc_records)."""
    aoi    = AOIS[site]
    bounds = AOI_BOUNDS[site]
    print(f"\n=== {site} ===")
    epsg, lon, lat = _utm_epsg(aoi)
    print(f"  centroid ({lon:.3f}, {lat:.3f}) -> UTM EPSG:{epsg}")

    ls_col = build_landsat_collection(aoi)
    print(f"  harmonized L5/L7/L8/L9 collection built ({ARCHIVE_START} → {ARCHIVE_END})")

    frames = {}
    print("  sampling NDWI vs JRC (1984-2021) ...")
    frames['JRC'] = assign_blocks_and_split(collect_water_indices_jrc(ls_col, aoi), epsg)
    print("  sampling NDVI vs Dynamic World (2015-) ...")
    frames['DW'] = assign_blocks_and_split(collect_ndvi_dw(ls_col, aoi), epsg)
    print("  sampling NDWI vs OPERA ...")
    opera_df = _safe_opera(site, ls_col, aoi, bounds)
    if len(opera_df):
        frames['OPERA'] = assign_blocks_and_split(opera_df, epsg)
    else:
        print("  [OPERA: no usable samples — skipping this reference]")

    spans = {'JRC':   f'{JRC_START[:7]}..{JRC_END[:7]}',
             'DW':    f'{DW_START[:7]}..{DW_END[:7]}',
             'OPERA': f'{OPERA_START[:7]}..{OPERA_END[:7]}',
             'NWI':   f'{ARCHIVE_START.year}..{ARCHIVE_END.year} (Jun-Sep median)',
             'NLCD':  f'{ARCHIVE_START.year}..{ARCHIVE_END.year} (Jun-Sep median)'}
    sensors_str = '+'.join(SENSORS)

    summaries, rocs = [], []

    # Water/vegetation indices first — NDMI needs the JRC-fit NDWI threshold.
    for ref in ('JRC', 'OPERA', 'DW'):
        frame = frames.get(ref)
        if frame is None or not len(frame):
            continue
        for index, direction in REFERENCES[ref]:
            if index not in frame.columns:
                continue
            s, r = optimize_index(frame, index, 'label', direction, site, index,
                                  ref, sensors_str, spans[ref])
            if s:
                summaries.append(s)
                rocs.append(r)

    # NDMI last, using this site's JRC-fit NDWI threshold for open-water exclusion.
    jrc_ndwi = next((s['threshold_mean'] for s in summaries
                     if s['index'] == 'NDWI' and s['reference'] == 'JRC'), None)
    print(f"  sampling NDMI (Jun-Sep median composite, full archive) ...")
    try:
        ndmi_df = assign_blocks_and_split(
            collect_ndmi(ls_col, aoi, NWI_RASTERS[site], jrc_ndwi), epsg)
    except Exception as e:
        # NDMI is the heaviest collection (full-archive median). Never let it lose
        # the JRC/OPERA/DW results already gathered for this site.
        print(f"  [{site} NDMI collection failed: {type(e).__name__}: {e}]")
        ndmi_df = pd.DataFrame()
    if len(ndmi_df):
        for ref in ('NWI', 'NLCD'):
            s, r = optimize_index(ndmi_df, 'NDMI', ref, 'above', site, 'NDMI',
                                  ref, sensors_str, spans[ref])
            if s:
                summaries.append(s)
                rocs.append(r)
    else:
        print("  [NDMI: no usable samples]")

    # ── epoch cuts ────────────────────────────────────────────────────────
    epoch_rows = []
    if RUN_EPOCH_CUTS:
        cal5, oli = _calendar_epochs(), _oli_epochs()
        pooled_of = {(s['index'], s['reference']): s['threshold_mean']
                     for s in summaries}
        print(f"  epoch cuts: {len(cal5)} calendar bins + {len(oli)} within-OLI bins "
              f"(no new GEE calls)")
        for ref in ('JRC', 'OPERA', 'DW'):
            frame = frames.get(ref)
            if frame is None or not len(frame):
                continue
            for index, direction in REFERENCES[ref]:
                if index not in frame.columns:
                    continue
                pooled = pooled_of.get((index, ref))
                epoch_rows += optimize_epochs(frame, index, 'label', direction, site,
                                              index, ref, epsg, 'cal5', cal5, pooled)
                epoch_rows += optimize_epochs(frame, index, 'label', direction, site,
                                              index, ref, epsg, 'oli', oli, pooled,
                                              oli_only=True)
        if RUN_NDMI_EPOCHS and len(ndmi_df):
            print("  NDMI epochs (one composite + sample per epoch)...")
            epoch_rows += _ndmi_epoch_rows(
                site, aoi, ls_col, epsg, NWI_RASTERS[site], jrc_ndwi,
                pooled_of.get(('NDMI', 'NWI')), cal5)

    return summaries, rocs, epoch_rows


# ═══════════════════════ ROC figure ═══════════════════════

_SITE_COLOR = {'Great Salt Lake': '#4575b4', 'Lake Abert': '#f16913'}
_REF_STYLE  = {'JRC': '-', 'OPERA': '--', 'DW': '-', 'NWI': '-', 'NLCD': '--'}


def plot_roc(rocs, out_path):
    """ROC on the held-out test blocks, one panel per index, both lakes overlaid.
    The dot marks the CV-selected operating point — the threshold actually used."""
    indices = [i for i in ('NDWI', 'NDVI', 'NDMI')
               if any(r['index'] == i for r in rocs)]
    if not indices:
        return
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4.6),
                             squeeze=False)
    for ax, index in zip(axes[0], indices):
        for r in [r for r in rocs if r['index'] == index]:
            if len(np.unique(r['y_true'])) < 2:
                continue
            fpr, tpr, thr = roc_curve(r['y_true'], r['y_score'])
            ax.plot(fpr, tpr, lw=2,
                    color=_SITE_COLOR.get(r['site'], '#666666'),
                    ls=_REF_STYLE.get(r['reference'], '-'),
                    label=f"{r['site']} vs {r['reference']} (AUC {r['auc']:.3f})")
            # operating point of the CV-selected threshold
            k = int(np.argmin(np.abs(thr - r['threshold'])))
            ax.plot(fpr[k], tpr[k], 'o', ms=6, mfc='white', mew=1.8,
                    color=_SITE_COLOR.get(r['site'], '#666666'))
        ax.plot([0, 1], [0, 1], color='#999999', lw=1, ls=':', label='chance')
        ax.set_title(f'{index} — ROC (held-out test blocks)', fontsize=10)
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ═══════════════ residual maps + confusion matrices ═══════════════
# 0 TN, 1 TP, 2 FP, 3 FN. CVD-validated distinct hues: grey / green / orange / red.

_RESID_CMAP = ListedColormap(['#e8e8e8', '#1a9850', '#f16913', '#d73027'])
_RESID_NORM = BoundaryNorm([0, 1, 2, 3, 4], _RESID_CMAP.N)
_RESID_LEGEND = [
    Patch(facecolor='#e8e8e8', label='TN — true negative'),
    Patch(facecolor='#1a9850', label='TP — true positive'),
    Patch(facecolor='#f16913', label='FP — over-detected'),
    Patch(facecolor='#d73027', label='FN — missed'),
]
_CM_CODES = np.array([[3, 1],     # reference positive: FN | TP
                      [0, 2]])    # reference negative: TN | FP
_CM_LIGHT = {1, 3}


def _coverage(mosaic, band, aoi):
    """Fraction of AOI pixels where the mosaic has valid data."""
    result = (ee.Image.constant(0)
              .where(mosaic.select(band).mask(), 1)
              .reduceRegion(ee.Reducer.mean(), aoi, 1000, maxPixels=1e6)
              .values().get(0).getInfo())
    return result if result is not None else 0.0


def _month_bounds(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    return start, start.advance(1, 'month')


def _landsat_median_for_month(ls_col, aoi, start, end):
    col = ls_col.filterDate(start, end)
    if col.size().getInfo() == 0:
        return None
    composite = col.median()
    if _coverage(composite, 'red', aoi) < MIN_AOI_COVERAGE:
        return None
    return composite


def _dw_water_fraction(dw_col, aoi, start, end):
    """AOI water fraction per Dynamic World — used only to rank water stage.
    JRC stops in 2021 so it cannot rank a 2023 month, and ranking by NDWI would
    let the index pick the scene it is about to be scored on."""
    col = dw_col.filterDate(start, end)
    if col.size().getInfo() == 0:
        return None
    mosaic = col.select('label').mode()
    if _coverage(mosaic, 'label', aoi) < MIN_AOI_COVERAGE:
        return None
    return (mosaic.eq(0)
            .reduceRegion(ee.Reducer.mean(), aoi, 300, maxPixels=1e7)
            .values().get(0).getInfo())


def select_median_stage_month(ls_col, dw_col, aoi, year):
    """(start, end, date_str, frac) for the median-water-stage month of `year`."""
    candidates = []
    for month in range(5, 11):
        start, end = _month_bounds(year, month)
        if _landsat_median_for_month(ls_col, aoi, start, end) is None:
            continue
        frac = _dw_water_fraction(dw_col, aoi, start, end)
        if frac is None:
            continue
        candidates.append((frac, month, start, end))
    if not candidates:
        raise RuntimeError(f"No {year} month with usable Landsat + DW coverage.")
    candidates.sort()
    frac, month, start, end = candidates[len(candidates) // 2]
    date_str = f'{year}-{month:02d}'
    print(f"  {len(candidates)} eligible {year} month(s); median stage is {date_str} "
          f"({frac:.1%} water by Dynamic World)")
    return start, end, date_str, frac


def _download_grid(image, aoi, bands, scale=OPERA_SCALE_M):
    """Fetch an ee.Image as GeoTIFF, keeping the dataset open so the OPERA
    reference can be warped onto exactly this grid."""
    url = image.select(bands).getDownloadURL({
        'region': aoi, 'scale': scale, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'})
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    src = MemoryFile(resp.content).open()
    return src.read(), src


def _download_residual(image, aoi, scale=OPERA_SCALE_M):
    url = image.getDownloadURL({'region': aoi, 'scale': scale,
                                'format': 'GEO_TIFF', 'crs': 'EPSG:4326'})
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with rasterio.open(io.BytesIO(resp.content)) as src:
        arr = src.read(1).astype(float)
        arr[arr < 0] = np.nan
        return arr, src.bounds


def _download_truecolor(ls_img, aoi, scale=OPERA_SCALE_M):
    """Harmonized RGB, percentile-stretched to [0,1]."""
    url = ls_img.select(['red', 'green', 'blue']).getDownloadURL({
        'region': aoi, 'scale': scale, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'})
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with rasterio.open(io.BytesIO(resp.content)) as src:
        arr, bnds = src.read([1, 2, 3]).astype(float), src.bounds
    arr = np.moveaxis(arr, 0, -1)
    for i in range(3):
        p2, p98 = np.nanpercentile(arr[..., i], [2, 98])
        arr[..., i] = np.clip((arr[..., i] - p2) / max(p98 - p2, 1e-6), 0, 1)
    return arr, bnds


def _ndvi_residual_image(ls_img, dw_img, ndvi_thresh):
    """0 TN, 1 TP, 2 FP, 3 FN vs Dynamic World vegetation."""
    ndvi = _ndvi(ls_img)
    pred_veg = ndvi.gte(ndvi_thresh)
    dw_label = dw_img.select('label')
    actual = dw_label.remap(_DW_VEG + _DW_NONVEG,
                            [1] * len(_DW_VEG) + [0] * len(_DW_NONVEG),
                            defaultValue=-1)
    # The constant image passed to where() is unmasked, so explicitly exclude
    # Landsat cloud/gap/no-data pixels; otherwise a masked prediction falls
    # through as code 0 and is incorrectly counted as a true negative.
    valid = actual.gte(0).And(dw_label.neq(8)).And(ndvi.mask())
    return (ee.Image(0)
            .where(pred_veg.Not().And(actual.eq(0)), 0)
            .where(pred_veg.And(actual.eq(1)), 1)
            .where(pred_veg.And(actual.eq(0)), 2)
            .where(pred_veg.Not().And(actual.eq(1)), 3)
            .updateMask(valid).unmask(-1).rename('residual'))


def _opera_reference_on_grid(opera_idx, grid_src, aoi_bounds, target_day):
    """Majority-vote OPERA B01_WTR onto the downloaded grid -> (water, valid).
    Granules are taken a whole overpass day at a time, nearest first, until the
    AOI is covered — one day rarely spans a multi-tile AOI."""
    shape = (grid_src.height, grid_src.width)
    water_votes = np.zeros(shape, dtype=np.int16)
    valid_votes = np.zeros(shape, dtype=np.int16)

    for day in sorted(opera_idx, key=lambda d: abs((d - target_day).days))[:OPERA_MAX_DAYS]:
        used = 0
        for granule in opera_idx[day]:
            out = _read_opera_b01(granule, aoi_bounds)
            if out is None:
                continue
            data, transform, crs = out
            dst = np.full(shape, 255, dtype=np.uint8)
            reproject(source=data, destination=dst,
                      src_transform=transform, src_crs=crs,
                      dst_transform=grid_src.transform, dst_crs=grid_src.crs,
                      # nearest: WTR is categorical and must not be averaged
                      resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
            usable = (dst == 0) | (dst == 1)
            water_votes += (dst == 1) & usable
            valid_votes += usable
            used += 1
        cov = (valid_votes > 0).mean()
        print(f"      {day.date()}: {used} granule(s) used, AOI coverage now {cov:.1%}")
        if cov >= OPERA_MIN_COVERAGE:
            break

    valid = valid_votes > 0
    return (water_votes > (valid_votes / 2)) & valid, valid


def _residual_codes_np(evaluated, pred, ref):
    codes = np.full(pred.shape, np.nan)
    codes[evaluated & ~pred & ~ref] = 0
    codes[evaluated &  pred &  ref] = 1
    codes[evaluated &  pred & ~ref] = 2
    codes[evaluated & ~pred &  ref] = 3
    return codes


def _confusion_counts(resid_arr):
    codes = resid_arr[~np.isnan(resid_arr)].astype(int)
    return {name: int((codes == code).sum())
            for code, name in [(0, 'TN'), (1, 'TP'), (2, 'FP'), (3, 'FN')]}


def _plot_confusion(ax, resid_arr, ref_name, pos_label):
    """2x2 confusion matrix in the residual palette. Counts are panel pixels at the
    download scale, not the samples the threshold was fit on — read the shares."""
    c = _confusion_counts(resid_arr)
    n_evaluated = sum(c.values())
    total = max(n_evaluated, 1)
    iou = c['TP'] / max(c['TP'] + c['FP'] + c['FN'], 1)
    ax.imshow(_CM_CODES, cmap=_RESID_CMAP, norm=_RESID_NORM, interpolation='nearest')
    for (row, col), code in np.ndenumerate(_CM_CODES):
        name = {0: 'TN', 1: 'TP', 2: 'FP', 3: 'FN'}[code]
        ax.text(col, row, f'{name}\n{c[name]:,}\n{c[name] / total:.1%}',
                ha='center', va='center', fontsize=9,
                color='white' if code in _CM_LIGHT else '#222222')
    ax.set_xticks([0, 1]); ax.set_xticklabels([f'not {pos_label}', pos_label], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([pos_label, f'not {pos_label}'], fontsize=8, rotation=90, va='center')
    ax.set_xlabel('predicted (index)', fontsize=8)
    ax.set_ylabel(f'reference ({ref_name})', fontsize=8)
    ax.set_title(f'IoU = {iou:.3f}  ({n_evaluated:,} evaluated px)', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _plot_confusion_compact(ax, resid_arr, pos_label, text_scale=1.0):
    """Compact matrix cell for the assembled paper figure."""
    c = _confusion_counts(resid_arr)
    n_evaluated = sum(c.values())
    denominator = max(n_evaluated, 1)
    ax.imshow(_CM_CODES, cmap=_RESID_CMAP, norm=_RESID_NORM,
              interpolation='nearest')
    for (row, col), code in np.ndenumerate(_CM_CODES):
        name = {0: 'TN', 1: 'TP', 2: 'FP', 3: 'FN'}[code]
        ax.text(col, row, f'{name}\n{c[name]:,}\n{c[name] / denominator:.1%}',
                ha='center', va='center', fontsize=5.8 * text_scale,
                linespacing=1.05,
                color='white' if code in _CM_LIGHT else '#222222')

    positive = 'water' if pos_label == 'water' else 'veg.'
    negative = 'not water' if pos_label == 'water' else 'not veg.'
    ax.set_xticks([0, 1])
    ax.set_xticklabels([negative, positive], fontsize=5.7 * text_scale)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([positive, negative], fontsize=5.7 * text_scale,
                       rotation=90, va='center')
    ax.tick_params(length=0, pad=1)
    ax.set_box_aspect(1)
    ax.set_anchor('C')
    for spine in ax.spines.values():
        spine.set_visible(False)


@plt.rc_context({'font.family': ASSEMBLED_FONT_FAMILY})
def plot_assembled_residual_figure(results, out_path, text_scale=1.0):
    """Paper-ready two-site plate matching the reference 2 x 5 assembly."""
    _require_assembled_font()
    if len(results) != len(AOIS):
        raise ValueError('The assembled figure requires one result for every AOI.')

    fig = plt.figure(figsize=ASSEMBLED_FIGSIZE)
    gs = fig.add_gridspec(
        2, 6, left=0.072, right=0.985, bottom=0.13, top=0.89,
        width_ratios=[1, 1, 1, 0.05, 0.72, 0.72],
        height_ratios=[1, 1], wspace=0.18, hspace=0.055)

    rows = []
    for row, result in enumerate(results):
        tc_ax = fig.add_subplot(gs[row, 0])
        ndwi_ax = fig.add_subplot(gs[row, 1])
        ndvi_ax = fig.add_subplot(gs[row, 2])
        ndwi_cm_ax = fig.add_subplot(gs[row, 4])
        ndvi_cm_ax = fig.add_subplot(gs[row, 5])
        rows.append((tc_ax, ndwi_ax, ndvi_ax, ndwi_cm_ax, ndvi_cm_ax))

        tc_ax.imshow(result['true_color'], extent=result['true_color_extent'],
                     origin='upper', aspect='auto')
        ndwi_ax.imshow(np.ma.masked_invalid(result['ndwi_residual']),
                       cmap=_RESID_CMAP, norm=_RESID_NORM,
                       extent=result['ndwi_extent'], origin='upper',
                       aspect='auto', interpolation='nearest')
        ndvi_ax.imshow(np.ma.masked_invalid(result['ndvi_residual']),
                       cmap=_RESID_CMAP, norm=_RESID_NORM,
                       extent=result['ndvi_extent'], origin='upper',
                       aspect='auto', interpolation='nearest')

        for ax in (tc_ax, ndwi_ax, ndvi_ax):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(0.55)

        _plot_confusion_compact(ndwi_cm_ax, result['ndwi_residual'], 'water', text_scale)
        _plot_confusion_compact(ndvi_cm_ax, result['ndvi_residual'],
                                'vegetation', text_scale)

    # Draw once so figure-coordinate labels can be centered on the final axes.
    fig.canvas.draw()
    column_titles = [
        'True color (context)', 'NDWI residual vs OPERA', 'NDVI residual vs DW',
        'NDWI confusion\nmatrix', 'NDVI confusion\nmatrix']
    for ax, title in zip(rows[0], column_titles):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.94, title,
                 ha='center', va='center', fontsize=7.4 * text_scale,
                 fontweight='bold')

    map_left = rows[0][0].get_position().x0
    for result, row_axes in zip(results, rows):
        pos = row_axes[0].get_position()
        fig.text(map_left - 0.038, (pos.y0 + pos.y1) / 2, result['site'],
                 ha='center', va='center', rotation=90,
                 fontsize=8.2 * text_scale, fontweight='bold')

    matrix_positions = [ax.get_position() for row_axes in rows for ax in row_axes[3:]]
    matrix_left = min(pos.x0 for pos in matrix_positions)
    matrix_right = max(pos.x1 for pos in matrix_positions)
    matrix_bottom = min(pos.y0 for pos in matrix_positions)
    matrix_top = max(pos.y1 for pos in matrix_positions)
    fig.text(matrix_left - 0.025, (matrix_bottom + matrix_top) / 2,
             'Reference truth', ha='center', va='center', rotation=90,
             fontsize=6.8 * text_scale)
    fig.text((matrix_left + matrix_right) / 2, 0.068,
             'Predicted (index threshold)', ha='center', va='center',
             fontsize=6.8 * text_scale)

    # Reference-style north arrow and spatial/matrix panel letters.
    bottom_map = rows[-1][0].get_position()
    arrow_x = map_left - 0.040
    arrow_y0 = bottom_map.y0 + 0.010
    arrow = FancyArrowPatch(
        (arrow_x, arrow_y0), (arrow_x, arrow_y0 + 0.060),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9,
        linewidth=0.8, color='black')
    fig.add_artist(arrow)
    fig.text(arrow_x, arrow_y0 - 0.012, 'N', ha='center', va='top',
             fontsize=7.0 * text_scale, fontweight='bold')
    fig.text(map_left - 0.060, 0.025, 'a', fontsize=9 * text_scale,
             fontweight='bold')
    fig.text(matrix_left - 0.020, 0.025, 'b', fontsize=9 * text_scale,
             fontweight='bold')

    _save_fig(fig, out_path, dpi=ASSEMBLED_DPI, bbox_inches=None)


def residual_maps_for_site(site, thresholds):
    """OPERA-window residual figure + NDWI/NDVI confusion matrices for one lake."""
    aoi, bounds = AOIS[site], AOI_BOUNDS[site]
    slug = site.lower().replace(' ', '_')
    ls_col = build_landsat_collection(aoi)
    dw_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterBounds(aoi).filterDate(DW_START, DW_END))

    t_ndwi = thresholds.get(('NDWI', 'OPERA'))
    t_ndvi = thresholds.get(('NDVI', 'DW'))
    if t_ndwi is None or t_ndvi is None:
        print(f"  [{site}: missing OPERA NDWI or DW NDVI threshold — skipping maps]")
        return None

    print(f"\n{site} (OPERA): selecting the median-stage scene of {OPERA_RESIDUAL_YEAR}...")
    start, end, date_str, frac = select_median_stage_month(
        ls_col, dw_col, aoi, OPERA_RESIDUAL_YEAR)
    ls_img = _landsat_median_for_month(ls_col, aoi, start, end)

    print("    Downloading true color...")
    tc_arr, tc_bnds = _download_truecolor(ls_img, aoi)

    print("    Downloading NDWI grid...")
    grid_img = _ndwi(ls_img).addBands(_ndwi(ls_img).mask().rename('valid'))
    arr, grid_src = _download_grid(grid_img, aoi, ['NDWI', 'valid'])
    ndwi_arr, ls_valid = arr[0], arr[1] > 0.5

    try:
        print(f"    Searching OPERA granules for {date_str}...")
        month_start = pd.Timestamp(f'{date_str}-01')
        month_end = month_start + pd.offsets.MonthEnd(1)
        try:
            results = earthaccess.search_data(
                short_name="OPERA_L3_DSWx-HLS_V1", bounding_box=bounds,
                temporal=(month_start.strftime('%Y-%m-%d'),
                          month_end.strftime('%Y-%m-%d')),
                count=2000)
        except IndexError:                  # earthaccess raises this on empty results
            results = []
        if not results:
            print(f"  [{site}: no OPERA granules for {date_str} — skipping]")
            return None

        opera_idx = _opera_date_index(results)
        print(f"      {len(results)} granule(s) across {len(opera_idx)} overpass day(s)")
        opera_water, opera_valid = _opera_reference_on_grid(
            opera_idx, grid_src, bounds, pd.Timestamp(f'{date_str}-15', tz='UTC'))
        if not opera_valid.any():
            print(f"  [{site}: no usable OPERA pixels for {date_str} — skipping]")
            return None
        opera_coverage = float(opera_valid.mean())
        if opera_coverage < OPERA_MIN_COVERAGE:
            print(f"    [warning: final OPERA coverage is {opera_coverage:.1%}, below the "
                  f"{OPERA_MIN_COVERAGE:.0%} target; uncovered pixels are excluded]")

        resid_ndwi = _residual_codes_np(ls_valid & opera_valid,
                                        ndwi_arr >= t_ndwi, opera_water)

        print("    Downloading NDVI residual (vs Dynamic World)...")
        dw_img = dw_col.filterDate(start, end).select('label').mode()
        ndvi_arr, ndvi_bnds = _download_residual(
            _ndvi_residual_image(ls_img, dw_img, t_ndvi), aoi)

        b = grid_src.bounds
        ndwi_extent = [b.left, b.right, b.bottom, b.top]
        true_color_extent = [tc_bnds.left, tc_bnds.right,
                             tc_bnds.bottom, tc_bnds.top]
        ndvi_extent = [ndvi_bnds.left, ndvi_bnds.right,
                       ndvi_bnds.bottom, ndvi_bnds.top]
    finally:
        grid_src.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'{site} — Spatial Residual Maps, {date_str} ({frac:.1%} water)\n'
                 f'NDWI >= {t_ndwi:+.3f} (OPERA validation) | '
                 f'NDVI >= {t_ndvi:+.3f} (Dynamic World validation)', fontsize=12)

    axes[0].imshow(tc_arr, extent=true_color_extent,
                   origin='upper', aspect='auto')
    axes[0].set_title('True color (context)', fontsize=10)

    # nearest-neighbor: categorical classes must not be blended when downsampled
    axes[1].imshow(np.ma.masked_invalid(resid_ndwi), cmap=_RESID_CMAP, norm=_RESID_NORM,
                   extent=ndwi_extent, origin='upper', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'NDWI residual vs OPERA\n(threshold {t_ndwi:+.3f})', fontsize=10)

    axes[2].imshow(np.ma.masked_invalid(ndvi_arr), cmap=_RESID_CMAP, norm=_RESID_NORM,
                   extent=ndvi_extent,
                   origin='upper', aspect='auto', interpolation='nearest')
    axes[2].set_title(f'NDVI residual vs Dynamic World\n(threshold {t_ndvi:+.3f})',
                      fontsize=10)

    for ax in axes:
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.legend(handles=_RESID_LEGEND, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save_fig(fig, fig_path('residual_maps', f'residual_opera_{slug}.png'))

    for arr_cm, ref, pos, tag, thr in [
        (resid_ndwi, 'OPERA', 'water', 'ndwi', t_ndwi),
        (ndvi_arr, 'Dynamic World', 'vegetation', 'ndvi', t_ndvi),
    ]:
        cm_fig, cm_ax = plt.subplots(figsize=(4, 4.4))
        _plot_confusion(cm_ax, arr_cm, ref, pos)
        cm_fig.suptitle(f'{site} — {tag.upper()} vs {ref}, {date_str}\n'
                        f'threshold {thr:+.3f}', fontsize=11)
        cm_fig.tight_layout()
        _save_fig(cm_fig, fig_path('confusion_matrices',
                                   f'confusion_matrix_{tag}_opera_{slug}.png'))
    return {
        'site': site,
        'date_str': date_str,
        'water_fraction': frac,
        't_ndwi': t_ndwi,
        't_ndvi': t_ndvi,
        'true_color': tc_arr,
        'true_color_extent': true_color_extent,
        'ndwi_residual': resid_ndwi,
        'ndwi_extent': ndwi_extent,
        'ndvi_residual': ndvi_arr,
        'ndvi_extent': ndvi_extent,
    }


# ═══════════════ GSL time series: three-class stacked area ═══════════════

_CLASS_COLORS = {'open_water': '#4575b4', 'open_land': '#d2b48c', 'wetland': '#1a9850'}


def _month_starts(start, end):
    months, (y, m) = [], (start.year, start.month)
    while (y, m) <= (end.year, end.month):
        months.append(datetime.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _next_month(d):
    return datetime.date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _monthly_feature(col_month, aoi, t, total_area_m2, month_str):
    """One median composite -> area of each of the three mutually exclusive classes.

        open water : NDWI >= x
        wetland    : NDWI <  x AND NDVI >= y AND NDMI >= z
        open land  : NDWI <  x AND NOT wetland
    """
    n = col_month.size()

    def _empty():
        return ee.Feature(None, {
            'month': month_str, 'n_images': 0, 'sensors': '',
            'open_water_km2': None, 'open_land_km2': None, 'wetland_km2': None,
            'valid_km2': None, 'valid_fraction': None})

    def _composited():
        composite = col_month.median()
        ndwi, ndvi, ndmi = _ndwi(composite), _ndvi(composite), _ndmi(composite)

        valid_mask = ndwi.mask().And(ndvi.mask()).And(ndmi.mask())
        water   = ndwi.gte(t['ndwi'])
        wetland = ndwi.lt(t['ndwi']).And(ndvi.gte(t['ndvi'])).And(ndmi.gte(t['ndmi']))
        land    = ndwi.lt(t['ndwi']).And(wetland.Not())

        px = ee.Image.pixelArea()
        areas = (ee.Image.cat([
            px.updateMask(water.And(valid_mask)).rename('water_m2'),
            px.updateMask(land.And(valid_mask)).rename('land_m2'),
            px.updateMask(wetland.And(valid_mask)).rename('wet_m2'),
            px.updateMask(valid_mask).rename('valid_m2'),
        ]).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=TS_SCALE_M,
                        maxPixels=1e9, bestEffort=True))

        valid_m2 = ee.Number(areas.get('valid_m2'))

        def _extrap(key):
            # Extrapolate through cloud/SLC-off holes: unobserved pixels are
            # assumed to share the observed class proportions.
            return (ee.Number(areas.get(key)).divide(valid_m2.max(1))
                    .multiply(total_area_m2).divide(1e6))

        return ee.Feature(None, {
            'month': month_str, 'n_images': n,
            'sensors': col_month.aggregate_array('sensor').distinct().join(','),
            'open_water_km2': _extrap('water_m2'),
            'open_land_km2':  _extrap('land_m2'),
            'wetland_km2':    _extrap('wet_m2'),
            'valid_km2':      valid_m2.divide(1e6),
            'valid_fraction': valid_m2.divide(total_area_m2)})

    return ee.Feature(ee.Algorithms.If(n.eq(0), _empty(), _composited()))


def _evaluate_month(col, aoi, t, total_area_m2, month_start):
    col_month = col.filterDate(month_start.isoformat(), _next_month(month_start).isoformat())
    feat = _monthly_feature(col_month, aoi, t, total_area_m2, month_start.isoformat())
    try:
        props = feat.getInfo()['properties']
    except ee.ee_exception.EEException as e:
        print(f"    [{month_start.isoformat()} failed: {e} — skipping]")
        return None
    return props if props.get('n_images', 0) else None


def compute_monthly_classes(col, aoi, t, start, end):
    """One median composite per calendar month, evaluated with a small pool of
    concurrent getInfo() calls. Each request stays scoped to a single month —
    that is what avoids Earth Engine's user-memory limit."""
    total_area_m2 = aoi.area(1)
    months = _month_starts(start, end)
    print(f"  {len(months)} calendar months to evaluate ({months[0]} → {months[-1]}), "
          f"{MAX_WORKERS} concurrent requests")
    rows, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_evaluate_month, col, aoi, t, total_area_m2, ms)
                   for ms in months]
        for future in cf.as_completed(futures):
            props = future.result()
            if props is not None:
                rows.append(props)
            done += 1
            if done % 25 == 0 or done == len(months):
                print(f"    {done}/{len(months)} months checked, {len(rows)} with data")
    return pd.DataFrame(rows)


def fetch_usgs_elevation(site_id, parameter_cd, start, end):
    """Daily water-surface elevation (ft) for one USGS gauge; empty frame if the
    gauge has no data in range."""
    resp = requests.get(NWIS_DV_URL, params={
        'format': 'json', 'sites': site_id, 'parameterCd': parameter_cd,
        'startDT': start.isoformat(), 'endDT': end.isoformat(), 'siteStatus': 'all',
    }, timeout=60)
    resp.raise_for_status()
    series = resp.json().get('value', {}).get('timeSeries', [])
    if not series:
        return pd.DataFrame(columns=['date', 'elevation_ft'])
    df = pd.DataFrame(series[0]['values'][0]['value'])
    df['date'] = pd.to_datetime(df['dateTime']).dt.tz_localize(None)
    df['elevation_ft'] = pd.to_numeric(df['value'], errors='coerce')
    return df[['date', 'elevation_ft']].dropna()


def load_elevation_series(site, start, end):
    out = {}
    for st in USGS_ELEVATION_STATIONS.get(site, []):
        daily = fetch_usgs_elevation(st['site'], st['param'], start, end)
        if daily.empty:
            print(f"  [no USGS data returned for {st['label']}]")
            continue
        out[st['label']] = (daily.set_index('date')['elevation_ft']
                            .resample('MS').mean().rename(st['label']).reset_index())
        print(f"  fetched {st['label']}: {len(daily)} daily readings")
    return out


def plot_stacked_classes(site, df, labels, out_path):
    """Stacked area of the three mutually exclusive classes, with gauge overlay."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    d = df.sort_values('date')
    ax.stackplot(d['date'],
                 d['open_water_km2'], d['open_land_km2'], d['wetland_km2'],
                 labels=['open water', 'open land', 'wetland'],
                 colors=[_CLASS_COLORS['open_water'], _CLASS_COLORS['open_land'],
                         _CLASS_COLORS['wetland']])
    ax.set_title(f'{site} — monthly land cover from median composites, '
                 f'Landsat {"/".join(s[1:] for s in SENSORS)} '
                 f'({TS_START.year}–present)')
    ax.set_xlabel('month')
    ax.set_ylabel('area (km²)')
    ax.grid(True, alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()

    present = [l for l in labels if l in df.columns]
    if present:
        ax2 = ax.twinx()
        for label, color in zip(present, ['#d73027', '#762a83']):
            sub = d.dropna(subset=[label])
            ax2.plot(sub['date'], sub[label], '--', color=color, lw=1.3, label=label)
        ax2.set_ylabel('water surface elevation (ft, USGS gauge)')
        h2, l2 = ax2.get_legend_handles_labels()
        handles, lbls = handles + h2, lbls + l2

    ax.legend(handles, lbls, fontsize=7, loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    _save_fig(fig, out_path)


def plot_stage_area(site, df, labels, out_path):
    """Extent-vs-elevation scatter — the honest way to show a monotonic but
    nonlinear stage-area relationship, versus a dual-axis line chart."""
    present = [l for l in labels if l in df.columns]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False)
    for ax, label in zip(axes[0], present):
        sub = df.dropna(subset=[label, 'open_water_km2'])
        if len(sub) < 3:
            ax.set_title(f'{label}\n(not enough overlapping months)')
            ax.axis('off')
            continue
        sc = ax.scatter(sub[label], sub['open_water_km2'], c=sub['date'].dt.year,
                        cmap='viridis', s=20)
        rho = sub[[label, 'open_water_km2']].corr(method='spearman').iloc[0, 1]
        ax.set_title(f'{label}\nSpearman ρ = {rho:.2f}  (n={len(sub)} months)', fontsize=10)
        ax.set_xlabel('water surface elevation (ft)')
        ax.set_ylabel('open water extent (km²)')
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label='year')
    fig.suptitle(f'{site} — stage-area relationship')
    fig.tight_layout()
    _save_fig(fig, out_path)


def run_timeseries(site, thresholds):
    """Monthly three-class land cover + USGS gauge overlay + stage-area, GSL only."""
    print(f"\n=== {site}: monthly land-cover time series ===")
    aoi = AOIS[site]
    t = {'ndwi': thresholds.get(('NDWI', 'JRC')),      # historical record → JRC-fit
         'ndvi': thresholds.get(('NDVI', 'DW')),
         'ndmi': thresholds.get(('NDMI', 'NWI'))}
    if any(v is None for v in t.values()):
        print(f"  [{site}: missing a threshold {t} — skipping time series]")
        return
    print(f"  NDWI >= {t['ndwi']:+.4f} (JRC-fit) | NDVI >= {t['ndvi']:+.4f} (DW-fit) | "
          f"NDMI >= {t['ndmi']:+.4f} (NWI-fit)")

    end = datetime.date.today()
    col = build_landsat_collection(aoi, TS_START, end)
    df = compute_monthly_classes(col, aoi, t, TS_START, end)
    if df.empty:
        print(f"  [no months with imagery for {site}]")
        return

    df['date'] = pd.to_datetime(df['month'])
    df = df.sort_values('date').reset_index(drop=True)
    df['low_confidence'] = df['valid_fraction'] < MIN_VALID_FRACTION
    # open_land_total preserves the plugin's nested definition (wetland ⊂ open land)
    df['open_land_total_km2'] = df['open_land_km2'] + df['wetland_km2']

    print("  fetching USGS water-surface-elevation gauge(s)...")
    elev = load_elevation_series(site, TS_START, end)
    labels = [st['label'] for st in USGS_ELEVATION_STATIONS.get(site, [])]
    for label, monthly in elev.items():
        df = df.merge(monthly, on='date', how='left')

    slug = site.lower().replace(' ', '_')
    out_csv = table_path(f'landcover_monthly_{slug}.csv')
    df.to_csv(out_csv, index=False)
    print(f"  saved {_rel(out_csv)} ({len(df)} months, "
          f"{int(df['low_confidence'].sum())} flagged low-coverage)")

    plot_stacked_classes(site, df, labels,
                         fig_path('timeseries', f'landcover_stacked_{slug}.png'))
    plot_stage_area(site, df, labels,
                    fig_path('timeseries', f'stage_area_{slug}.png'))


# ═══════════════════════ main ═══════════════════════

if __name__ == '__main__':
    thresholds_by_site = {}

    if RUN_OPTIMIZATION:
        all_summaries, all_rocs, all_epochs = [], [], []
        for site in AOIS:
            summaries, rocs, epochs = optimize_site(site)
            all_summaries.extend(summaries)
            all_rocs.extend(rocs)
            all_epochs.extend(epochs)

        summary = pd.DataFrame(all_summaries)
        if len(summary):
            summary = summary.sort_values(['site', 'index', 'cv_iou'],
                                          ascending=[True, True, False]).reset_index(drop=True)
            out_csv = table_path('final_threshold_summary.csv')
            summary.to_csv(out_csv, index=False)
            print(f"\n=== Final thresholds (test_* is the defensible number) ===")
            print(summary.to_string(index=False))
            print(f"\nSaved {_rel(out_csv)}")

            # Completeness check against what actually consumes this table. A
            # reference can drop out silently when its collection fails, leaving
            # a short table that still looks valid, so state plainly which rows
            # are missing before anything downstream reads it.
            missing = [(site, index, ref)
                       for site in AOIS
                       for index, ref in REQUIRED_ROWS
                       if summary[(summary['site'] == site)
                                  & (summary['index'] == index)
                                  & (summary['reference'] == ref)].empty]
            if missing:
                print(f"\n!! {len(missing)} of {len(AOIS) * len(REQUIRED_ROWS)} required "
                      f"row(s) are MISSING from {_rel(out_csv)}:")
                for site, index, ref in missing:
                    print(f"     {site:18} {index} vs {ref}")
                print("   Downstream consumers will fail on these. Scroll up for the "
                      "collection error that caused each one.\n")
            else:
                print("All NDWI/NDVI rows required by water_extent_timeseries.py "
                      "and the plug-in defaults are present.")
                print("NDMI comes from threshold_optimization_ndmi.py; the complete "
                      "threshold set is the union of both outputs.")

            # Persist the ROC inputs. They live only in memory otherwise, which
            # makes roc_curves.png the one figure a RUN_OPTIMIZATION=False pass
            # cannot redraw.
            if all_rocs:
                pd.DataFrame([
                    {'site': r['site'], 'index': r['index'], 'reference': r['reference'],
                     'threshold': r['threshold'], 'auc': r['auc'],
                     'y_true': int(yt), 'y_score': float(ys)}
                    for r in all_rocs for yt, ys in zip(r['y_true'], r['y_score'])
                ]).to_csv(table_path('roc_points.csv'), index=False)
                print(f"Saved {_rel(table_path('roc_points.csv'))}")

            plot_roc(all_rocs, fig_path('roc', 'roc_curves.png'))

            if all_epochs:
                ep = pd.DataFrame(all_epochs)
                ep_csv = table_path('threshold_epochs.csv')
                ep.to_csv(ep_csv, index=False)
                print(f"\nSaved {_rel(ep_csv)}")
                fitted = ep[ep.get('threshold_mean').notna()] if 'threshold_mean' in ep else ep.iloc[0:0]
                if len(fitted):
                    print("\n=== Threshold drift (epochs where drift exceeds the CV "
                          "fold spread are the ones worth acting on) ===")
                    show = ['cut', 'site', 'index', 'reference', 'epoch', 'sensors',
                            'n_points', 'threshold_mean', 'threshold_std',
                            'pooled_threshold', 'drift', 'drift_exceeds_fold_std',
                            'cv_iou', 'test_iou']
                    print(fitted[[c for c in show if c in fitted.columns]]
                          .to_string(index=False))
                for cut in ('cal5', 'oli'):
                    plot_threshold_drift(ep, cut,
                                         fig_path('threshold_drift',
                                                  f'threshold_drift_{cut}.png'))

            for site in AOIS:
                thresholds_by_site[site] = {
                    (r['index'], r['reference']): r['threshold_mean']
                    for r in all_summaries if r['site'] == site}
        else:
            print("\n[no thresholds produced — nothing downstream can run]")
    else:
        # Figures-only pass: redraw everything from the persisted tables, no
        # sampling and no CV. Useful for restyling or adding an export format.
        src = Path(THRESHOLD_SOURCE_CSV)
        if not src.is_absolute():          # bare filename → this run's own output
            src = table_path(src)
        if not src.exists():
            raise FileNotFoundError(
                f"Threshold table not found: {src}\n"
                f"    Set THRESHOLD_SOURCE_CSV, or run with RUN_OPTIMIZATION=True "
                f"to fit thresholds first.")

        prev = pd.read_csv(src)
        for site, g in prev.groupby('site'):
            thresholds_by_site[site] = {
                (r['index'], r['reference']): float(r['threshold_mean'])
                for _, r in g.iterrows()}
        print(f"Loaded thresholds from {_rel(src)}")
        for site, t in thresholds_by_site.items():
            print(f"  {site}: " + ', '.join(f'{i} vs {r} = {v:+.4f}'
                                            for (i, r), v in sorted(t.items())))

        roc_csv = table_path('roc_points.csv')
        if roc_csv.exists():
            pts = pd.read_csv(roc_csv)
            rocs = [dict(site=s, index=i, reference=r,
                         threshold=float(g['threshold'].iloc[0]),
                         auc=float(g['auc'].iloc[0]),
                         y_true=g['y_true'].values, y_score=g['y_score'].values)
                    for (s, i, r), g in pts.groupby(['site', 'index', 'reference'])]
            plot_roc(rocs, fig_path('roc', 'roc_curves.png'))
        else:
            print(f"[optional {_rel(roc_csv)} not found — skipping ROC redraw; "
                  f"create it with RUN_OPTIMIZATION=True]")

        ep_csv = table_path('threshold_epochs.csv')
        if ep_csv.exists():
            ep = pd.read_csv(ep_csv)
            for cut in ('cal5', 'oli'):
                plot_threshold_drift(ep, cut,
                                     fig_path('threshold_drift',
                                              f'threshold_drift_{cut}.png'))
        else:
            print(f"[optional {_rel(ep_csv)} not found — skipping threshold-drift redraw]")

    if RUN_RESIDUAL_MAPS:
        if RUN_ASSEMBLED_FIGURE:
            # Check before the two-site network/download pass so another machine
            # never spends minutes fetching data only to discover the font is absent.
            _require_assembled_font()
        residual_failures = []
        residual_results = {}
        for site in AOIS:
            try:
                result = residual_maps_for_site(site, thresholds_by_site.get(site, {}))
                if result is None:
                    residual_failures.append(site)
                else:
                    residual_results[site] = result
            except Exception as e:
                print(f"  [{site}: residual maps failed ({type(e).__name__}: {e})]")
                residual_failures.append(site)
        if residual_failures:
            raise RuntimeError(
                "Residual maps/confusion matrices were not generated for: "
                + ", ".join(residual_failures)
            )
        if RUN_ASSEMBLED_FIGURE:
            print("\nAssembling the two-site paper figure...")
            assembled_stem = 'residual_confusion_assembled'
            assembled_results = [residual_results[site] for site in AOIS]
            plot_assembled_residual_figure(
                assembled_results,
                fig_path('assembled', f'{assembled_stem}.png'))
            for text_scale in ASSEMBLED_TEXT_SCALES:
                scale_pct = round(text_scale * 100)
                print(f"  rendering {scale_pct}% text version...")
                plot_assembled_residual_figure(
                    assembled_results,
                    fig_path('assembled', f'{assembled_stem}_{scale_pct}pct.png'),
                    text_scale=text_scale)

    if RUN_TIMESERIES:
        run_timeseries('Great Salt Lake', thresholds_by_site.get('Great Salt Lake', {}))

    print("\nDone.")

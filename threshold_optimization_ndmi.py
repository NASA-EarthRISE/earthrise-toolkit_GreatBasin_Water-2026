"""
NDMI wetland-threshold optimization for Lake Abert.

Validates an NDMI moisture threshold against two independent wetland references:
  1. Your rasterized NWI wetland mask (static, local GeoTIFF, EPSG:4326, 1=wetland)
  2. NLCD wetland (classes 90/95, static, in GEE)

The threshold is selected by spatial block cross-validation (1 km UTM blocks,
block-grouped 80/20 train/test, GroupKFold held-out CV IoU) — the same approach as
threshold_optimization_cv.py — so neighbouring pixels can't leak across the split.

Lake Abert only. No OPERA / earthaccess needed for this comparison.
"""

import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import urllib.request
import rasterio
from rasterio.transform import rowcol
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

# Earth Engine Cloud project. Kept out of source control: set EE_PROJECT in the
# environment rather than hardcoding an ID here, so a project identifier cannot
# be committed by accident.
#     export EE_PROJECT=your-ee-project-id
EE_PROJECT = os.environ.get('EE_PROJECT', 'your-ee-project-id')

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

# --- settings ---------------------------------------------------------------

aoi_abert = ee.Geometry.Rectangle([-120.6098988, 42.40054901, -119.9981325, 42.85172324])
aoi_gsl   = ee.Geometry.Rectangle([-113.3061047, 40.51087812, -111.8534658, 41.87473984])
START_YEAR         = 2010     # 2020-21 were near-record dry for Abert; widen the
END_YEAR           = 2020      # window so wetter years contribute a wet peak
SUMMER_START_MONTH = 6
SUMMER_END_MONTH   = 9
 

#  NWI wetland masks (1 = wetland, 0 = not; EPSG:4326), one local raster per site.
#  Not committed — GSL_Wetland_Raster.tif alone is ~114 MB, over GitHub's 100 MB
#  hard limit. Point NWI_DIR at wherever you keep them:
#      export NWI_DIR=/path/to/your/wetland/rasters
#  Default is data/nwi/ beside this script, which is gitignored. See data/README.md.
SCRIPT_DIR = Path(__file__).resolve().parent
NWI_DIR    = Path(os.environ.get('NWI_DIR', SCRIPT_DIR / 'data' / 'nwi'))

NWI_RASTER_PATH     = NWI_DIR / 'Abert_Wetlands_Raster.tif'
NWI_RASTER_PATH_GSL = NWI_DIR / 'GSL_Wetland_Raster.tif'

# NLCD — second, independent wetland reference (national, in GEE). Wetland =
# class 90 (woody) or 95 (emergent herbaceous). Static like NWI, so one image
# labels every point.
NLCD_COLLECTION      = 'USGS/NLCD_RELEASES/2019_REL/NLCD'
NLCD_YEAR            = '2019'
NLCD_WETLAND_CLASSES = [90, 95]

# Per-pixel upper-percentile NDMI composite. Sampling each scene separately
# labels a seasonally-dry wetland as "dry" in most scenes and buries it; a high
# percentile keeps each pixel's peak-moisture state, which is what the static
# NWI boundary encodes. Raise toward 90 / max for more sensitivity, lower toward
# 50 (median) for more robustness against residual haze.
NDMI_PERCENTILE   = 50     # median — the documented method
TOTAL_SAMPLES     = 4500   # single composite now, so sample the whole AOI once

# Open-water exclusion. NDMI is a vegetation-moisture index and can't be expected
# to flag open lake water / salt playa, so those pixels are dropped from the IoU
# union entirely (neither intersection nor union) rather than scored as misses.
#
# The cutoff is each lake's JRC-optimized NDWI threshold, from
# data/threshold_cv_summary.csv. It is per-site because the two lakes' NDWI
# distributions differ: Abert's shallow, turbid margins sit higher than GSL's.
NDWI_WATER_THRESHOLD = {
    'Lake Abert':      0.0839,   # JRC-optimized NDWI*, cf. threshold_cv_summary.csv
    'Great Salt Lake': 0.0151,   # JRC-optimized NDWI*
}
NDWI_WATER_PERCENTILE = 50     # median NDWI => exclude *persistent* open water,
                               # not a wetland that briefly flooded

#Landsat prep

def _mask_qa(image):
    """Drop cloud (QA_PIXEL bit 3) and cloud-shadow (bit 4) pixels before use."""
    qa = image.select('QA_PIXEL')
    keep = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(keep)


def scale_optical(image):
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)


def _summer_filter():
    """Months SUMMER_START_MONTH..SUMMER_END_MONTH of every year START_YEAR..END_YEAR."""
    return ee.Filter.And(
        ee.Filter.calendarRange(START_YEAR, END_YEAR, 'year'),
        ee.Filter.calendarRange(SUMMER_START_MONTH, SUMMER_END_MONTH, 'month'),
    )

def _landsat_collection(sensor, aoi):
    return (
        ee.ImageCollection(f'LANDSAT/{sensor}/C02/T1_L2')
        .filterBounds(aoi)
        .filter(_summer_filter())              # summer months across all years
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .map(_mask_qa)                         # per-pixel cloud/shadow mask
        .map(scale_optical)
    )
 
 
abert_landsat = _landsat_collection('LC08', aoi_abert).merge(
                _landsat_collection('LC09', aoi_abert))
gsl_landsat   = _landsat_collection('LC08', aoi_gsl).merge(
                _landsat_collection('LC09', aoi_gsl))


def _add_ndmi(image):
    return image.addBands(
        image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI'))


def ndmi_composite(ls_col, percentile=NDMI_PERCENTILE):
    """Reduce the scene stack to one per-pixel upper-percentile NDMI image.

    This is the compositing step: every masked, scaled summer scene contributes,
    and each output pixel takes its `percentile`-th NDMI over the whole period.
    Seasonally-flooded wetlands that dry out late in summer still register,
    because the composite keeps their wet peak instead of averaging it away.
    """
    return (ls_col.map(_add_ndmi).select('NDMI')
            .reduce(ee.Reducer.percentile([percentile]))
            .rename('NDMI'))


def _add_ndwi(image):
    # McFeeters NDWI: (Green - NIR) / (Green + NIR) => SR_B3, SR_B5 on Landsat 8/9
    return image.addBands(
        image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI'))


def ndwi_composite(ls_col, percentile=NDWI_WATER_PERCENTILE):
    """Per-pixel NDWI composite used to mask persistent open water."""
    return (ls_col.map(_add_ndwi).select('NDWI')
            .reduce(ee.Reducer.percentile([percentile]))
            .rename('NDWI'))


def nlcd_wetland():
    """NLCD wetland mask (1 = class 90/95, 0 = not) for the configured year."""
    lc = (ee.ImageCollection(NLCD_COLLECTION)
          .filter(ee.Filter.eq('system:index', NLCD_YEAR))
          .first().select('landcover'))
    mask = lc.eq(NLCD_WETLAND_CLASSES[0])
    for cls in NLCD_WETLAND_CLASSES[1:]:
        mask = mask.Or(lc.eq(cls))
    return mask.rename('NLCD')


# helpers

def _fc_to_records(fc, page_size=4500):
    """Paginate a FeatureCollection past GEE's 5000-element getInfo limit."""
    rows, offset = [], 0
    while True:
        batch = fc.toList(page_size, offset).getInfo()
        if not batch:
            break
        rows.extend(f['properties'] for f in batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _time_join(primary, secondary, window_days, match_key):
    filt = ee.Filter.maxDifference(
        difference=int(window_days * 86400 * 1000),
        leftField='system:time_start',
        rightField='system:time_start',
    )
    return ee.Join.saveBest(matchKey=match_key, measureKey='_dt').apply(
        primary, secondary, filt)


# --- spatial block cross-validation (ported from threshold_optimization_cv.py) ---

RANDOM_SEED      = 42
BLOCK_SIZE_M     = 1000
SPLIT_FRACS      = (0.80, 0.20)   # block-grouped train / test (no separate val set)
CV_SPLITS        = 5
GRID_STEPS_CV    = 150
GRID_STEPS_PLOT  = 200
SELECTION_METRIC = 'iou'


def _utm_epsg(aoi):
    lon, lat = aoi.centroid(1).coordinates().getInfo()
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone if lat >= 0 else 32700 + zone), lon, lat


def assign_blocks_and_split(df, epsg, block_size_m=BLOCK_SIZE_M,
                            fracs=SPLIT_FRACS, seed=RANDOM_SEED):
    """Tag each point with its 1 km UTM block, then assign whole blocks to
    train/test so spatially-correlated neighbours never straddle the split."""
    to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = to_utm.transform(df['lon'].values, df['lat'].values)
    bx = np.floor(x / block_size_m).astype(int)
    by = np.floor(y / block_size_m).astype(int)
    df = df.copy()
    df['block_id'] = [f'{a}_{b}' for a, b in zip(bx, by)]
    blocks = np.array(sorted(df['block_id'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    n_tr = int(len(blocks) * fracs[0])
    split_of = {b: 'train' for b in blocks[:n_tr]}
    split_of.update({b: 'test' for b in blocks[n_tr:]})
    df['split'] = df['block_id'].map(split_of)
    return df


def grid_search_threshold(values, labels, direction, n_steps=GRID_STEPS_PLOT):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    lo, hi = np.percentile(values, [0.5, 99.5])
    thresholds = np.linspace(lo, hi, n_steps)
    rows = []
    for t in thresholds:
        pred = (values >= t) if direction == 'above' else (values <= t)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        rows.append(dict(threshold=t, precision=prec, recall=sens, specificity=spec,
                         f1=f1, iou=iou, youden_j=sens + spec - 1))
    return pd.DataFrame(rows)


def eval_threshold(values, labels, threshold, direction):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (values >= threshold) if direction == 'above' else (values <= threshold)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return dict(n=len(labels), f1=f1, iou=iou, precision=prec, recall=sens, specificity=spec)


def cv_optimal_threshold(values, labels, groups, direction, metric=SELECTION_METRIC,
                         n_splits=CV_SPLITS, n_steps=GRID_STEPS_CV):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    k = int(min(n_splits, len(np.unique(groups))))
    if k < 2:                                   # too few blocks to cross-validate
        gs = grid_search_threshold(values, labels, direction, n_steps)
        best = float(gs.loc[gs[metric].idxmax(), 'threshold'])
        return best, 0.0, [best], float('nan')
    gkf = GroupKFold(n_splits=k)
    fold_thresholds, fold_scores = [], []
    for tr_idx, ho_idx in gkf.split(values, labels, groups):
        gs = grid_search_threshold(values[tr_idx], labels[tr_idx], direction, n_steps)
        t  = float(gs.loc[gs[metric].idxmax(), 'threshold'])
        fold_thresholds.append(t)
        # held-out fold score: clean generalization estimate, used to pick the
        # threshold without ever touching the test blocks
        fold_scores.append(eval_threshold(values[ho_idx], labels[ho_idx], t, direction)[metric])
    return (float(np.mean(fold_thresholds)), float(np.std(fold_thresholds)),
            fold_thresholds, float(np.mean(fold_scores)))


def optimize_ndmi(df, label_col, ref_name, direction='above'):
    """Block-CV NDMI threshold for one wetland reference.

    NDMI is always the index (wetland = higher NDMI => 'above'); the reference
    only changes the label column. cv_iou is the held-out selection metric,
    test_* the untouched final number.
    """
    d = df.dropna(subset=['NDMI']).copy()
    d['_lab'] = d[label_col].fillna(0).astype(int)
    tr = d[d['split'] == 'train']
    te = d[d['split'] == 'test']

    mean_t, std_t, folds, cv_iou = cv_optimal_threshold(
        tr['NDMI'].values, tr['_lab'].values, tr['block_id'].values, direction)
    test_m = eval_threshold(te['NDMI'].values, te['_lab'].values, mean_t, direction)
    curve  = grid_search_threshold(tr['NDMI'].values, tr['_lab'].values,
                                   direction, GRID_STEPS_PLOT)

    summary = dict(
        reference=ref_name, threshold_mean=round(mean_t, 4), threshold_std=round(std_t, 4),
        n_train=len(tr), n_test=len(te), n_blocks=int(d['block_id'].nunique()),
        n_wetland=int(d['_lab'].sum()),
        cv_iou=round(cv_iou, 3) if cv_iou == cv_iou else float('nan'),
        test_iou=round(test_m['iou'], 3), test_f1=round(test_m['f1'], 3),
        test_precision=round(test_m['precision'], 3), test_recall=round(test_m['recall'], 3),
    )
    return summary, curve, mean_t


def _save_textless_copy(fig, out_path):
    """Save a second copy of `fig` with every character removed — titles, axis
    labels, tick labels, and legends stripped; plot geometry/imagery kept.
    Written next to the original as <name>_notext<ext>. Mutates the figure, so
    call it AFTER the as-is copy has already been saved.
    """
    for t in list(fig.texts):              # figure-level text (incl. suptitle)
        t.remove()
    for lg in list(fig.legends):           # figure-level legends
        lg.remove()
    for ax in fig.axes:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for t in list(ax.texts):           # in-axes annotations
            t.remove()
    base, ext = os.path.splitext(out_path)
    fig.savefig(f'{base}_notext{ext}', dpi=150, bbox_inches='tight')
    print(f'  saved {base}_notext{ext}')


def _plot_curves(curves, optima, site, out_path):
    n = len(curves)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, (name, gs) in zip(axes[0], curves.items()):
        ax.plot(gs['threshold'], gs['iou'], color='steelblue', lw=2, label='IoU')
        ax.plot(gs['threshold'], gs['f1'],  color='seagreen',  lw=2, label='F1')
        if optima.get(name) is not None:
            ax.axvline(optima[name], color='red', ls='--', alpha=0.7,
                       label=f'CV threshold = {optima[name]:+.3f}')
        ax.set_title(f'{site} — NDMI vs {name}')
        ax.set_xlabel('NDMI threshold'); ax.set_ylabel('score (train grid)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  saved {out_path}')
    _save_textless_copy(fig, out_path)
    plt.show()


def _label_points_from_raster(coords, vals, arr, transform, to_crs,
                              positive_value=1, ndwi=None, nlcd=None):
    """Look up a static-raster label for each sampled point.

    coords : list of (lon, lat) in EPSG:4326 (GEE sample geometries)
    vals   : the index value (NDMI) at each point
    arr    : the NWI label raster as a 2-D array
    ndwi   : optional NDWI value per point, carried through for open-water masking
    nlcd   : optional NLCD wetland label (0/1) per point, the second reference
    Returns list of {'NDMI', 'label' (NWI), 'nlcd', 'ndwi', 'lon', 'lat'};
    points outside the NWI raster are dropped.
    """
    ndwi = ndwi if ndwi is not None else [None] * len(coords)
    nlcd = nlcd if nlcd is not None else [None] * len(coords)
    rows = []
    for (lon, lat), v, w, nl in zip(coords, vals, ndwi, nlcd):
        if v is None:
            continue
        x, y = to_crs.transform(lon, lat)
        r, c = rowcol(transform, x, y)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            rows.append({'NDMI': v, 'label': int(arr[r, c] == positive_value),
                         'nlcd': nl, 'lon': lon, 'lat': lat, 'ndwi': w})
    return rows


# NDMI vs NWI (local raster) and NLCD (GEE) — sampled at the same points

def collect_samples(ls_col, nwi_raster_path, aoi, ndwi_water_threshold):
    """Sample the NDMI composite once; label each point from NWI *and* NLCD.

    Both references are static, so one set of points serves the whole period.
    'label' = NWI wetland (0/1), 'nlcd' = NLCD wetland (0/1); NDWI is carried for
    open-water exclusion. Sampling once means the two references share identical
    points, so their IoU curves are directly comparable.
    """
    with rasterio.open(nwi_raster_path) as src:
        nwi           = src.read(1)
        nwi_transform = src.transform
        nwi_crs       = src.crs
    to_nwi = Transformer.from_crs('EPSG:4326', nwi_crs, always_xy=True)

    composite = (ndmi_composite(ls_col)
                 .addBands(ndwi_composite(ls_col))
                 .addBands(nlcd_wetland()))
    feats = composite.sample(
        region=aoi, scale=30, numPixels=TOTAL_SAMPLES,
        seed=42, geometries=True,
    ).getInfo()['features']
    cols = ['NDMI', 'label', 'nlcd', 'lon', 'lat', 'ndwi']
    if not feats:
        return pd.DataFrame(columns=cols)

    coords     = [tuple(f['geometry']['coordinates']) for f in feats]
    ndmi_vals  = [f['properties'].get('NDMI') for f in feats]
    ndwi_vals  = [f['properties'].get('NDWI') for f in feats]
    nlcd_vals  = [f['properties'].get('NLCD') for f in feats]
    rows = _label_points_from_raster(coords, ndmi_vals, nwi, nwi_transform, to_nwi,
                                     ndwi=ndwi_vals, nlcd=nlcd_vals)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    # Drop persistent open water: NDMI can't score those pixels, so they leave the
    # IoU union entirely instead of counting as false negatives.
    if len(df):
        open_water = df['ndwi'] > ndwi_water_threshold
        print(f'  excluding {int(open_water.sum())} open-water pts '
              f'(NDWI > {ndwi_water_threshold:+.4f}) of {len(df)}')
        df = df[~open_water].reset_index(drop=True)
    return df



def run_site(name, aoi, ls_col, nwi_path):
    """Full block-CV NDMI optimization + residual maps for one AOI.

    NDMI is the index; each reference ('NWI' = local raster, 'NLCD' = GEE) only
    changes the label column. Wetland pixels have higher NDMI => 'above'.
    """
    slug = name.lower().replace(' ', '_')
    print(f'\n=== {name}: NDMI vs NWI + NLCD (block-CV) ===')

    # Each lake masks open water at its own JRC-optimized NDWI threshold.
    try:
        ndwi_water = NDWI_WATER_THRESHOLD[name]
    except KeyError:
        raise KeyError(
            f"No open-water NDWI threshold for '{name}'. Add its JRC-optimized "
            f"NDWI* to NDWI_WATER_THRESHOLD (see data/threshold_cv_summary.csv).")

    df = collect_samples(ls_col, nwi_path, aoi, ndwi_water)

    # 1 km spatial blocks -> block-grouped 80/20 train/test (partitioning step)
    epsg = _utm_epsg(aoi)[0]
    df = assign_blocks_and_split(df, epsg)

    summaries, curves, optima = [], {}, {}
    for ref_name, label_col in [('NWI', 'label'), ('NLCD', 'nlcd')]:
        if not len(df) or df[label_col].fillna(0).nunique() < 2:
            print(f'  [skip {ref_name}: no usable labelled samples]')
            optima[ref_name] = None
            continue
        summ, curve, star = optimize_ndmi(df, label_col, ref_name)
        summaries.append(summ)
        curves[ref_name] = curve
        optima[ref_name] = star

    summary = None
    if summaries:
        summary = (pd.DataFrame(summaries)
                   .sort_values('cv_iou', ascending=False).reset_index(drop=True))
        summary.insert(0, 'site', name)
        print(f'\n  {name} — NDMI threshold summary (ranked by held-out CV IoU; '
              f'test_* is the defensible final number):')
        print(summary.to_string(index=False))
        summary.to_csv(f'ndmi_threshold_cv_{slug}.csv', index=False)
        _plot_curves(curves, optima, name, f'ndmi_threshold_{slug}.png')

    # Residual maps hold the NDMI prediction at the NWI CV-selected threshold.
    ndmi_star = optima.get('NWI')
    if ndmi_star is not None and len(df):
        try:
            residual_map(ls_col, ndmi_star, nwi_path, aoi,
                         f'residual_{slug}.png', title=name,
                         ndwi_water_threshold=ndwi_water)
        except Exception as e:
            # Large AOIs (GSL) can exceed GEE's getDownloadURL size cap at
            # DISPLAY_SCALE; don't let a map failure lose the CV results.
            print(f'  [{name}: residual map failed ({type(e).__name__}: {e}); '
                  f'try a coarser DISPLAY_SCALE]')
    elif ndmi_star is None:
        print(f'  [{name}: no NWI CV threshold — skipping residual maps]')
    return summary

# --- spatial residual maps (raster) -----------------------------------------
#
# Same approach as the NDWI/NDVI residual figure: pull actual rasters out of GEE
# and render classification + residual panels, rather than scatter the sample
# points. NWI is static (no water-stage rows), so the grid is a single row:
#   true color | NDMI classification | NDMI residual (vs NWI)

DISPLAY_SCALE = 150   # metres/pixel for the downloaded panels (coarser = faster)

# classification: 0 dry, 1 wetland. Wetland = green to match TP below, so
# "predicted wetland" reads the same across the classification and residual panels.
_CLS_CMAP = mcolors.ListedColormap(['#e8e8e8', '#1a9850'])
_CLS_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], _CLS_CMAP.N)
_CLS_LEGEND = [mpatches.Patch(color='#e8e8e8', label='dry'),
               mpatches.Patch(color='#1a9850', label='wetland')]

# residual: 0 TN, 1 TP, 2 FP, 3 FN, 4 open water excluded from union.
# Maximally distinct hues (CVD-validated): grey / green / orange / red / blue —
# no two colours are close, and blue reads intuitively as water.
_RESID_CMAP = mcolors.ListedColormap(
    ['#e8e8e8', '#1a9850', '#f16913', '#d73027', '#4575b4'])
_RESID_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], _RESID_CMAP.N)
_RESID_LEGEND = [mpatches.Patch(color='#e8e8e8', label='TN — agree dry'),
                 mpatches.Patch(color='#1a9850', label='TP — agree wetland'),
                 mpatches.Patch(color='#f16913', label='FP — over-detected (false wetland)'),
                 mpatches.Patch(color='#d73027', label='FN — missed wetland'),
                 mpatches.Patch(color='#4575b4', label='open water (excluded)')]


def _download_geotiff(image, aoi, scale, bands):
    """Fetch an ee.Image as a GeoTIFF and return (array(bands,H,W), rasterio src)."""
    url = image.select(bands).getDownloadURL({
        'region': aoi, 'scale': scale, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326',
    })
    data = urllib.request.urlopen(url).read()
    src  = MemoryFile(data).open()
    return src.read(), src


def _truecolor_panel(ls_col, aoi, lo=0.0, hi=0.30):
    """Median SR RGB, contrast-stretched to [0,1] for display."""
    rgb = ls_col.select(['SR_B4', 'SR_B3', 'SR_B2']).median()
    arr, src = _download_geotiff(rgb, aoi, DISPLAY_SCALE, ['SR_B4', 'SR_B3', 'SR_B2'])
    img = np.clip((np.moveaxis(arr, 0, -1) - lo) / (hi - lo), 0, 1)
    return img, src.bounds


def _composite_grid(ls_col, aoi, ndwi_water_threshold):
    """Download NDMI, its valid mask, NDWI (open water), and NLCD on one grid."""
    comp  = ndmi_composite(ls_col)
    water = ndwi_composite(ls_col)
    stack = (comp.addBands(comp.mask().rename('valid'))
                 .addBands(water).addBands(nlcd_wetland()))
    arr, src = _download_geotiff(stack, aoi, DISPLAY_SCALE,
                                 ['NDMI', 'valid', 'NDWI', 'NLCD'])
    ndmi       = arr[0]
    valid      = arr[1] > 0.5
    open_water = valid & (arr[2] > ndwi_water_threshold)
    nlcd_ref   = arr[3] > 0.5
    return ndmi, valid, open_water, nlcd_ref, src


def _nwi_on_grid(nwi_raster_path, grid_src):
    """Reproject the static NWI mask onto the downloaded composite's grid."""
    with rasterio.open(nwi_raster_path) as nwi_src:
        dst = np.zeros((grid_src.height, grid_src.width), dtype=np.float32)
        reproject(
            source=nwi_src.read(1), destination=dst,
            src_transform=nwi_src.transform, src_crs=nwi_src.crs,
            dst_transform=grid_src.transform, dst_crs=grid_src.crs,
            resampling=Resampling.nearest,
        )
    return dst >= 0.5


def _residual_codes(ev, pred, ref, open_water):
    """0 TN, 1 TP, 2 FP, 3 FN, 4 open-water (excluded); NaN elsewhere."""
    r = np.full(pred.shape, np.nan)
    r[ev & ~pred & ~ref] = 0
    r[ev &  pred &  ref] = 1
    r[ev &  pred & ~ref] = 2
    r[ev & ~pred &  ref] = 3
    r[open_water]        = 4
    return r


def residual_map(ls_col, threshold, nwi_raster_path, aoi, out_path,
                 title, ndwi_water_threshold):
    """Classification + per-reference residual panels for the NDMI composite.

    The NDMI prediction is held at `threshold`; its residual is mapped against
    both references (NWI, NLCD) so where the two references disagree is visible.
    `ndwi_water_threshold` is the site's JRC-optimized NDWI*, used to mask
    persistent open water out of the evaluated area.
    """
    tc_img, tc_bounds = _truecolor_panel(ls_col, aoi)
    ndmi, valid, open_water, nlcd_ref, grid_src = _composite_grid(
        ls_col, aoi, ndwi_water_threshold)
    nwi_ref = _nwi_on_grid(nwi_raster_path, grid_src)
    pred    = ndmi > threshold
    ev      = valid & ~open_water   # evaluated (union) pixels

    cls        = np.where(ev, pred.astype(float), np.nan)
    resid_nwi  = _residual_codes(ev, pred, nwi_ref,  open_water)
    resid_nlcd = _residual_codes(ev, pred, nlcd_ref, open_water)

    b   = grid_src.bounds
    ext = [b.left, b.right, b.bottom, b.top]
    tcb = tc_bounds
    tc_ext = [tcb.left, tcb.right, tcb.bottom, tcb.top]

    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    fig.suptitle(f'{title} — Spatial Residual Maps  '
                 f'(NDMI > {threshold:+.2f}, {NDMI_PERCENTILE}th-pct composite '
                 f'{START_YEAR}–{END_YEAR})', fontsize=12)

    axes[0].imshow(tc_img, extent=tc_ext, origin='upper', aspect='auto')
    axes[0].set_title('True color (median)', fontsize=10)

    # nearest-neighbor: categorical classes must not be blended when the raster
    # is downsampled to the panel, which produces false grain.
    axes[1].imshow(np.ma.masked_invalid(cls), cmap=_CLS_CMAP, norm=_CLS_NORM,
                   extent=ext, origin='upper', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'NDMI classification (threshold {threshold:+.2f})', fontsize=10)

    for ax, resid, ref_name in [(axes[2], resid_nwi,  'NWI'),
                                (axes[3], resid_nlcd, 'NLCD 90/95')]:
        ax.imshow(np.ma.masked_invalid(resid), cmap=_RESID_CMAP, norm=_RESID_NORM,
                  extent=ext, origin='upper', aspect='auto', interpolation='nearest')
        ax.set_title(f'NDMI residual (vs {ref_name})', fontsize=10)

    for ax in axes:
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.legend(handles=_CLS_LEGEND + _RESID_LEGEND, loc='lower center',
               ncol=7, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')
    _save_textless_copy(fig, out_path)
    plt.show()


# run both sites

SITES = [
    ('Lake Abert',      aoi_abert, abert_landsat, NWI_RASTER_PATH),
    ('Great Salt Lake', aoi_gsl,   gsl_landsat,   NWI_RASTER_PATH_GSL),
]

all_summaries = [run_site(*s) for s in SITES]
combined = [s for s in all_summaries if s is not None]
if combined:
    pd.concat(combined, ignore_index=True).to_csv('ndmi_threshold_cv_all.csv', index=False)
    print('\nSaved ndmi_threshold_cv_all.csv')

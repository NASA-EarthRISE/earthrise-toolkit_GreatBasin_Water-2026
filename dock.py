#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dock.py
───────
The Saline Lake Classifier dock widget — the entire user-facing panel of the
QGIS plugin, plus the local raster classification step.

WHAT THIS FILE OWNS vs WHAT IT DOESN'T
    dock.py        builds the panel, holds all UI state, and runs the pixel
                   classification locally with GDAL/NumPy (_classify_raster).
    ee_backend.py  every Earth Engine call: compositing, index math, GeoTIFF
                   download. This file never touches `ee` directly — it imports
                   ee_backend lazily inside each handler so that a QGIS install
                   without earthengine-api can still load the plugin and show a
                   readable message instead of failing at import time.

PANEL LAYOUT (built once in _build_ui, top to bottom)
    1 · Area of Interest     preset lake, vector file, or a box drawn on canvas
    2 · Outputs              True Color / LST / NDWI / NDVI / NDMI checkboxes
    3 · Classification       three threshold sliders + water/wetland/land
    4 · Time Series          CSV button, wired up externally (see note below)
    5 · OpenET               evapotranspiration, CONUS only

THE CLASSIFICATION RULE (applied in _classify_raster, mutually exclusive and
assigned in this order)
    open water : NDWI >= ndwi_thr
    wetland    : NOT water AND NDVI >= ndvi_thr AND NDMI >= ndmi_thr
    land       : NOT water AND NOT wetland
NDVI and NDMI only ever narrow the wetland class; land is the residual. The
thresholds default to per-lake values calibrated externally against JRC/OPERA
(water) and Dynamic World (vegetation) — see DEFAULT_THRESHOLDS.

CONVENTIONS WORTH KNOWING BEFORE EDITING
  * All AOI geometry is stored in self.aoi_extent as EPSG:4326, whatever the
    project CRS is. Every entry point transforms into WGS84 immediately, so
    downstream code never has to ask which CRS it is holding.
  * The Stage 2 fetch helpers RETURN a status string rather than setting the
    label themselves, so _fetch_selected can join several outputs into one
    message. _fetch_openet does not follow this convention — see its docstring.
  * Long Earth Engine calls are synchronous and block the GUI thread. Each
    handler calls QApplication.processEvents() after setting its "fetching…"
    label so the text actually paints before the freeze.
  * Section 4's button is deliberately left unconnected here; lake_classifier.py
    connects it, which is why self.btn_timeseries is stored as an attribute.
"""

import os
import math
import tempfile

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QFrame, QFileDialog, QApplication, QSlider,
    QDoubleSpinBox, QScrollArea, QCheckBox
)
from qgis.core import (
    QgsProject, QgsRectangle, QgsPointXY, QgsWkbTypes,
    QgsVectorLayer, QgsRasterLayer, QgsSettings,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsColorRampShader, QgsRasterShader,
    QgsSingleBandPseudoColorRenderer, QgsMultiBandColorRenderer,
    QgsPalettedRasterRenderer,
)
from qgis.gui import QgsMapTool, QgsRubberBand


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in AOIs, in EPSG:4326. The keys are shown verbatim in the lake dropdown
# and are also the lookup keys into DEFAULT_THRESHOLDS, so the two dicts must be
# kept in step — a preset with no matching threshold entry silently keeps
# whatever thresholds were last on the sliders (see _update_default_thresholds).
PRESETS = {
    "Great Salt Lake, UT": QgsRectangle(
        -113.31, 40.51, -111.85, 41.87
    ),
    "Lake Abert, OR": QgsRectangle(
        -120.61, 42.40, -120.00, 42.85
    ),
}

# Every AOI is normalized to this CRS on the way in, no matter what the project
# or the loaded vector file uses.
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

# Newest first so the dropdown opens on the most recent year. The lower bound is
# 1984 because that is where the Landsat archive usable here begins (L5 onward).
YEARS = [str(y) for y in range(2026, 1984, -1)]

# Per-lake starting thresholds, applied whenever the preset changes. These are
# not guesses: they come from the external spatial-block cross-validation runs
# (NDWI vs OPERA, NDVI vs Dynamic World), which is why they are lake-specific
# rather than one global default — Abert's water cutoff sits lower than GSL's
# because its shallow, turbid margins depress NDWI.
# NDMI is the least certain of the three; it drives only the wetland class.
DEFAULT_THRESHOLDS = {
    "Great Salt Lake, UT": {
        "ndwi": -0.055,
        "ndvi": 0.165,
        "ndmi": 0.053,
    },
    "Lake Abert, OR": {
        "ndwi": -0.063,
        "ndvi": 0.130,
        "ndmi": 0.123,
    },
}

# Colour ramps for the continuous index layers, as (value, hex, legend label).
# Entries with an empty label are interpolation anchors only — they shape the
# gradient but stay out of the legend, which keeps the legend to the three or so
# stops a reader actually needs. QgsColorRampShader interpolates between them,
# so values outside the listed range clamp to the end colours rather than going
# transparent. The ramps are diverging around a meaningful midpoint (0 for the
# normalized-difference indices), so the neutral colour marks the class boundary
# rather than the middle of the data.
NDWI_STOPS = [
    (-0.6, "#8c510a", "dry land"),
    (-0.2, "#d8b365", ""),
    (0.0, "#f6f6f4", "shoreline (0)"),
    (0.2, "#80b1d3", ""),
    (0.6, "#2166ac", "open water"),
]

NDVI_STOPS = [
    (-0.30, "#4575b4", "water"),
    (0.05, "#d9c9a3", "bare / soil"),
    (0.20, "#fee08b", "sparse veg."),
    (0.40, "#a6d96a", ""),
    (0.60, "#1a9850", "dense veg."),
]

NDMI_STOPS = [
    (-0.40, "#8c510a", "very dry"),
    (-0.10, "#d8b365", ""),
    (0.10, "#f6f6f4", ""),
    (0.30, "#5ab4ac", ""),
    (0.60, "#01665e", "high moisture"),
]

# LST is land-surface temperature in Celsius rather than a -1 to 1 index.
LST_STOPS = [
    (5.0, "#2166ac", "5 \u00b0C"),
    (20.0, "#92c5de", ""),
    (30.0, "#fddbc7", ""),
    (40.0, "#f4a582", ""),
    (55.0, "#b2182b", "55 \u00b0C"),
]

# Combined classification: value -> (color, label). These pixel values are the
# contract between _classify_raster (which writes them) and the paletted
# renderer in _classify (which colours them); 0 is reserved for NoData. Changing
# a number here means changing it in _classify_raster too.
CLASS_DEFS = [
    (1, "#2166ac", "Open water"),
    (2, "#8a9b54", "Wetland"),
    (3, "#d6cab2", "Land"),
]

OPENET_STOPS = [
    (0.0,   "#9e6212", "0 mm"),
    (100.0, "#c8b434", ""),
    (200.0, "#61b95c", ""),
    (300.0, "#4dc2a2", ""),
    (400.0, "#2d4c6e", "400 mm"),
]

# Hover tooltips shown on each output checkbox in Section 2.
OUTPUT_INFO = {
    "true_color": (
        "True Color is a natural-color composite built from the red, "
        "green, and blue surface-reflectance bands."
    ),
    "lst": (
        "Land Surface Temperature (LST) is defined as the radiative "
        "skin temperature of the land, derived from thermal infrared "
        "radiance, which measures the temperature of the surface "
        "itself rather than the air above it."
    ),
    "ndwi": (
        "NDWI (Normalized Difference Water Index) uses the green and "
        "near-infrared bands to highlight open water. Values near +1 "
        "indicate water, while negative "
        "values indicate dry land."
    ),
    "ndvi": (
        "NDVI (Normalized Difference Vegetation Index) uses the "
        "near-infrared and red bands to measure vegetation density "
        "and health. Higher values indicate denser, healthier "
        "vegetation."
    ),
    "ndmi": (
        "NDMI (Normalized Difference Moisture Index) uses the "
        "near-infrared and shortwave-infrared bands and is highly "
        "sensitive to the water content of both soil and vegetation. "
        "NDMI is not the same thing as a classified wetland, it is "
        "a continuous moisture signal, not a land-cover class. Here "
        "it is combined with NDWI and NDVI thresholds in Section 3 "
        "to derive the wetland class."
    ),
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

# These three exist so the panel's visual language lives in one place: section
# titles, muted explanatory text, and separators are each defined once rather
# than restyled at every call site in _build_ui.

def _heading(text):
    """Bold section title, e.g. '1 · Area of Interest'."""
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:600; font-size:13px;")
    return lbl


def _caption(text):
    """Small muted explanatory line under a heading or above a control."""
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#9fb3c1; font-size:11px;")
    return lbl


def _divider():
    """Horizontal rule between stages."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#33485a;")
    return line


def _output_checkbox(label, info_key):
    """Checkbox for one selectable output, with a hover tooltip explaining it.

    info_key must exist in OUTPUT_INFO — a missing key raises KeyError here at
    build time rather than silently producing a checkbox with no explanation.
    """
    box = QCheckBox(label)
    box.setToolTip(OUTPUT_INFO[info_key])
    # The what's-this cursor is the hint that hovering will explain the index;
    # without it the tooltips are easy to miss entirely.
    box.setCursor(Qt.CursorShape.WhatsThisCursor)
    return box


def _make_index_layer(path, band, name, stops):
    """Open one GeoTIFF band as a continuous pseudocolor layer.

    Used for every single-band raster the plugin displays — the three indices,
    LST, and OpenET — with only `stops` changing between them.

    Returns None instead of raising if QGIS cannot open the file, because a
    failed layer is a message to show the user, not a crash: callers report
    which layers failed and carry on loading the rest.
    """
    layer = QgsRasterLayer(path, name)

    if not layer.isValid():
        return None

    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(
                value,
                QColor(color),
                label,
            )
            for value, color, label in stops
        ]
    )

    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)

    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(),
        band,
        shader,
    )

    layer.setRenderer(renderer)
    layer.triggerRepaint()

    return layer


def _make_true_color_layer(path, name):
    """Open a three-band GeoTIFF as a true-color RGB layer.

    Band order is fixed by ee_backend.build_true_color_image, which emits
    red/green/blue in that order; the renderer below maps them positionally, so
    the two must change together.

    Returns None if QGIS cannot open the file — see _make_index_layer.
    """
    layer = QgsRasterLayer(path, name)

    if not layer.isValid():
        return None

    renderer = QgsMultiBandColorRenderer(
        layer.dataProvider(),
        1,  # red
        2,  # green
        3,  # blue
    )

    layer.setRenderer(renderer)
    layer.triggerRepaint()

    return layer


def _classify_raster(
    index_path,
    ndwi_thr,
    ndvi_thr,
    ndmi_thr,
    out_path,
):
    """Classify an index raster into open water, wetland, and land.

    This is the only heavy computation the plugin runs locally — everything else
    is done server-side in Earth Engine. It is cheap enough to re-run on every
    threshold change (see _reclassify_if_active), which is what makes the
    sliders feel live.

    Reads:
        Band 1 = NDWI
        Band 2 = NDVI
        Band 3 = NDMI

    Writes:
        1 = Open water
        2 = Wetland
        3 = Land
        0 = NoData

    The three classes are mutually exclusive and assigned in that order, so a
    pixel that qualifies as water is never reconsidered for wetland. Wetland is
    a valid non-water pixel at or above BOTH the vegetation and moisture
    cutoffs; land is simply everything valid that is left over, which is why no
    threshold can ever leave a valid pixel unclassified.

    Returns:
        (counts, pixel_area_km2) — counts keyed by class value, and the
        approximate ground area of one pixel, for the Stage 3 area readout.
    """
    # Imported here rather than at module scope: GDAL is heavy, and a QGIS
    # install missing it should still be able to load the plugin panel and use
    # the fetch-only features. The cost is paid once, on first classification.
    from osgeo import gdal
    import numpy as np

    ds = gdal.Open(index_path)

    if ds is None:
        raise IOError("Could not open %s" % index_path)

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    b1 = ds.GetRasterBand(1)
    b2 = ds.GetRasterBand(2)

    ndwi = b1.ReadAsArray().astype("float32")
    ndvi = b2.ReadAsArray().astype("float32")

    nd1 = b1.GetNoDataValue()
    nd2 = b2.GetNoDataValue()

    # NDMI is treated as optional so an older two-band raster still classifies.
    # NOTE: when it is absent the wetland test silently drops its moisture
    # condition, making wetland strictly NDVI-based and therefore broader than
    # the documented rule. Anything written by the current ee_backend has three
    # bands, so this is a compatibility path, not the normal one.
    has_ndmi = ds.RasterCount >= 3

    if has_ndmi:
        b3 = ds.GetRasterBand(3)
        ndmi = b3.ReadAsArray().astype("float32")
        nd3 = b3.GetNoDataValue()

    # A pixel counts as valid only where every band in use has real data. Two
    # separate things can mark a gap — an IEEE NaN/inf, or the band's declared
    # NoData sentinel — and cloud-masked Earth Engine exports contain both, so
    # each is tested. Building one shared mask first is what guarantees the
    # three classes stay mutually exclusive and jointly cover the valid area.
    valid = np.isfinite(ndwi) & np.isfinite(ndvi)

    if nd1 is not None:
        valid &= ndwi != nd1

    if nd2 is not None:
        valid &= ndvi != nd2

    if has_ndmi:
        valid &= np.isfinite(ndmi)

        if nd3 is not None:
            valid &= ndmi != nd3

    # Order matters: water wins outright, wetland is carved from what is left,
    # and land takes the remainder. `~water` in the wetland test is what makes
    # a flooded, vegetated pixel read as water rather than wetland.
    water = valid & (ndwi >= ndwi_thr)
    wet = valid & (~water) & (ndvi >= ndvi_thr)

    if has_ndmi:
        wet &= ndmi >= ndmi_thr

    land = valid & (~water) & (~wet)

    # Zero-filled, so every pixel outside `valid` keeps class 0 = NoData without
    # being written explicitly.
    cls = np.zeros(ndwi.shape, dtype="uint8")
    cls[water] = 1
    cls[wet] = 2
    cls[land] = 3

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        out_path,
        xsize,
        ysize,
        1,
        gdal.GDT_Byte,
    )

    out.SetGeoTransform(gt)
    out.SetProjection(proj)

    out_band = out.GetRasterBand(1)
    out_band.WriteArray(cls)
    out_band.SetNoDataValue(0)
    out_band.FlushCache()

    # Dropping both references is what actually closes the datasets and flushes
    # the header to disk — GDAL has no explicit close(), and QGIS will read a
    # truncated file if the writer is still open when the layer is added.
    out = None
    ds = None

    # Estimate pixel area because the downloaded raster is in EPSG:4326, whose
    # pixels are degrees, not meters — their ground area shrinks toward the
    # poles. One degree of latitude is ~111.32 km everywhere; one degree of
    # longitude is that times cos(latitude). Evaluating cos once at the AOI's
    # center latitude is accurate to well under a percent across AOIs this size
    # (~1.4° tall), and avoids reprojecting the raster purely to measure area.
    lat_center = gt[3] + (ysize / 2.0) * gt[5]

    longitude_km = (
        abs(gt[1])
        * 111.32
        * math.cos(math.radians(lat_center))
    )

    latitude_km = abs(gt[5]) * 111.32
    pixel_area_km2 = longitude_km * latitude_km

    counts = {
        1: int(water.sum()),
        2: int(wet.sum()),
        3: int(land.sum()),
    }

    return counts, pixel_area_km2


# ---------------------------------------------------------------------------
# AOI rectangle drawing tool
# ---------------------------------------------------------------------------

class RectangleMapTool(QgsMapTool):
    """Allow the user to draw a rectangular AOI on the map canvas.

    Press-drag-release drives a QgsRubberBand preview; on release the finished
    rectangle is emitted via `rectangleDrawn` and the dock takes over. The tool
    deliberately knows nothing about CRS — it emits in whatever the canvas uses,
    and _finish_drawing_aoi does the transform to WGS84.
    """

    # Emitted once per completed drag, never during it.
    rectangleDrawn = pyqtSignal(QgsRectangle)

    def __init__(self, canvas):
        """Set up drag state and the translucent rubber-band preview."""
        super().__init__(canvas)

        self.canvas = canvas
        # Corners of the drag in canvas CRS; None until the first press.
        self.start_point = None
        self.end_point = None
        # Gates canvasMoveEvent so the preview only follows an active drag.
        self.is_drawing = False

        self.rubber_band = QgsRubberBand(
            canvas,
            QgsWkbTypes.GeometryType.PolygonGeometry,
        )

        self.rubber_band.setColor(
            QColor(72, 201, 214, 100)
        )

        self.rubber_band.setFillColor(
            QColor(72, 201, 214, 40)
        )

        self.rubber_band.setWidth(2)

    def canvasPressEvent(self, event):
        """Anchor the first corner and begin the drag."""
        self.start_point = self.toMapCoordinates(event.pos())
        self.end_point = self.start_point
        self.is_drawing = True

        self._show_rectangle()

    def canvasMoveEvent(self, event):
        """Track the opposite corner. The is_drawing guard is what keeps this
        from redrawing on ordinary mouse movement over the canvas."""
        if not self.is_drawing:
            return

        self.end_point = self.toMapCoordinates(event.pos())
        self._show_rectangle()

    def canvasReleaseEvent(self, event):
        """Finish the drag and publish the rectangle."""
        if not self.is_drawing:
            return

        self.end_point = self.toMapCoordinates(event.pos())
        self.is_drawing = False

        self._show_rectangle()

        rect = QgsRectangle(
            self.start_point,
            self.end_point,
        )

        # normalize() reorders the corners so min < max; without it a drag up
        # or to the left yields an inverted rectangle that reads as empty.
        rect.normalize()

        # A plain click (no drag) collapses to zero area — ignored, so a stray
        # click cannot silently replace a carefully chosen AOI.
        if not rect.isEmpty():
            self.rectangleDrawn.emit(rect)

    def _show_rectangle(self):
        """Redraw the rubber-band preview between the two current corners."""
        if self.start_point is None or self.end_point is None:
            return

        rect = QgsRectangle(
            self.start_point,
            self.end_point,
        )

        rect.normalize()

        p1 = QgsPointXY(
            rect.xMinimum(),
            rect.yMinimum(),
        )

        p2 = QgsPointXY(
            rect.xMinimum(),
            rect.yMaximum(),
        )

        p3 = QgsPointXY(
            rect.xMaximum(),
            rect.yMaximum(),
        )

        p4 = QgsPointXY(
            rect.xMaximum(),
            rect.yMinimum(),
        )

        self.rubber_band.reset(
            QgsWkbTypes.GeometryType.PolygonGeometry
        )

        # Corners added counter-clockwise, closing back on p1. The False flag
        # suppresses the repaint for all but the final point, so the preview
        # updates once per move rather than five times.
        self.rubber_band.addPoint(p1, False)
        self.rubber_band.addPoint(p2, False)
        self.rubber_band.addPoint(p3, False)
        self.rubber_band.addPoint(p4, False)
        self.rubber_band.addPoint(p1, True)

    def clear(self):
        """Reset corners and wipe the preview, so the next activation starts
        clean instead of showing the previous AOI."""
        self.start_point = None
        self.end_point = None
        self.is_drawing = False

        self.rubber_band.reset(
            QgsWkbTypes.GeometryType.PolygonGeometry
        )


# ---------------------------------------------------------------------------
# Reusable threshold control
# ---------------------------------------------------------------------------

class ThresholdControl(QWidget):
    """Slider and spin box for editing one threshold value.

    Qt sliders are integer-only, so the slider stores value/step and converts
    back on the way out; `step` therefore sets the slider's resolution, not just
    the spin box's arrows. The two widgets mirror each other, which is why both
    handlers use blockSignals — without it, setting one would re-trigger the
    other and recurse.

    Two signals, deliberately distinct:
        valueChanged  every intermediate value, including mid-drag.
        committed     only when the user settles (slider released, or editing
                      finished). Reclassification listens to `committed` alone,
                      so dragging the slider does not fire a GDAL pass per pixel
                      of travel.
    """

    valueChanged = pyqtSignal(float)
    committed = pyqtSignal()

    def __init__(
        self,
        title,
        minimum,
        maximum,
        default,
        step=0.01,
        parent=None,
    ):
        """Build the title row (caption + reset link) and the control row
        (slider + spin box), then wire the two-way sync.

        `minimum`, `maximum` and `step` are floats in index units; the slider's
        integer range is derived from them by division, so a step that does not
        divide the range evenly makes the slider unable to reach one endpoint
        exactly. `default` seeds both widgets and becomes the reset target.
        """
        super().__init__(parent)

        self._step = step
        self._default = default

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)

        cap = QLabel(title)
        cap.setStyleSheet(
            "color:#9fb3c1; font-size:11px;"
        )

        reset_btn = QPushButton("reset")
        reset_btn.setFlat(True)
        reset_btn.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        reset_btn.setStyleSheet(
            "color:#48c9d6; font-size:11px; border:none;"
        )
        reset_btn.clicked.connect(self.reset)

        title_row.addWidget(cap)
        title_row.addStretch(1)
        title_row.addWidget(reset_btn)

        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(0, 0, 0, 0)

        self._slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self._slider.setMinimum(
            int(round(minimum / step))
        )

        self._slider.setMaximum(
            int(round(maximum / step))
        )

        self._spin = QDoubleSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(3)
        self._spin.setMaximumWidth(80)

        ctrl_row.addWidget(self._slider, 1)
        ctrl_row.addWidget(self._spin)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addLayout(title_row)
        outer.addLayout(ctrl_row)

        self._slider.valueChanged.connect(
            self._on_slider
        )

        self._spin.valueChanged.connect(
            self._on_spin
        )

        self._slider.sliderReleased.connect(
            self.committed
        )

        self._spin.editingFinished.connect(
            self.committed
        )

        self.setValue(default)

    def _on_slider(self, ivalue):
        """Slider moved: scale the integer back to a float and mirror it into
        the spin box without letting that echo back here."""
        value = ivalue * self._step

        self._spin.blockSignals(True)
        self._spin.setValue(value)
        self._spin.blockSignals(False)

        self.valueChanged.emit(value)

    def _on_spin(self, value):
        """Spin box edited: mirror into the slider, rounding to the nearest
        step. The displayed float is authoritative — the slider is the lossy
        view, so `value()` reads the spin box rather than the slider."""
        self._slider.blockSignals(True)

        self._slider.setValue(
            int(round(value / self._step))
        )

        self._slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self):
        """Current threshold. Read from the spin box, which holds full
        precision; the slider is quantized to `step`."""
        return self._spin.value()

    def setValue(self, value):
        """Set both widgets. Assigning to the spin box is enough — its
        valueChanged runs _on_spin, which syncs the slider."""
        self._spin.setValue(value)

    def setDefault(self, value):
        """Update the reset value and display the new default.

        Called when the preset lake changes, so 'reset' returns to the
        calibrated threshold for the lake now selected, not the one that was
        selected when the panel was first built.
        """
        self._default = value
        self.setValue(value)

    def reset(self):
        """Restore the default and reclassify.

        Emits `committed` explicitly because setValue only produces
        valueChanged, which nothing reclassifies on — without this line the
        reset link would move the slider but leave the map stale.
        """
        self.setValue(self._default)
        self.committed.emit()


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------

class LakeClassifierDock(QDockWidget):
    """AOI, visualization, index, and classification controls.

    One long-lived instance per QGIS session, created by lake_classifier.py.
    All panel state lives on this object; nothing is stored in the QGIS project,
    so closing and reopening the dock starts fresh (the Earth Engine project ID
    is the one exception — it is persisted through QgsSettings).
    """

    def __init__(self, iface):
        """Initialise all panel state to empty, then build the UI.

        Every attribute is declared here rather than being created on first use,
        so the full state of the dock is readable in one place — the fetch and
        classify handlers all test these for None as their preconditions.
        """
        super().__init__("Saline Lake Classifier")

        # QGIS interface handle: used for the map canvas and the message bar.
        self.iface = iface

        # --- Stage 1: area of interest -------------------------------------
        # aoi_extent is the single source of truth for "where", ALWAYS in
        # EPSG:4326 regardless of project CRS. None means nothing chosen yet,
        # which is what _get_fetch_settings checks first.
        self.aoi_extent = None
        # Set only when the AOI came from a vector file, so the layer can be
        # referenced later; None for preset and drawn AOIs.
        self.aoi_layer = None
        # Built lazily on first use and then reused, so repeated drawing does
        # not accumulate map tools on the canvas.
        self.rectangle_tool = None

        # --- Stage 2: fetched display layers -------------------------------
        # Kept so each fetch can replace its own layer rather than piling up
        # duplicates in the layer tree.
        self.true_color_layer = None
        self.lst_layer = None

        self.ndwi_layer = None
        self.ndvi_layer = None
        self.ndmi_layer = None

        self.index_layer = None
        # Path to the full 3-band index GeoTIFF. This is the handoff from
        # Stage 2 to Stage 3: it is set even for bands the user chose not to
        # display, and its presence is what enables classification.
        self.index_raster_path = None

        # --- Stage 3: classification ---------------------------------------
        # Previous result, replaced on every reclassify; class_path is tracked
        # so the superseded temp file can be deleted.
        self.class_layer = None
        self.class_path = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Build the interface
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Construct the whole panel once, top to bottom.

        Every widget the rest of the class touches is assigned to self here;
        purely decorative widgets are added to the layout and forgotten. The
        panel is wrapped in a QScrollArea at the end because five stages do not
        fit in a docked side panel at typical heights.
        """
        panel = QWidget()

        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # --- Stage 1 ---
        v.addWidget(
            _heading("1 \u00b7 Area of Interest")
        )

        self.lake_combo = QComboBox()
        self.lake_combo.addItems(
            list(PRESETS.keys())
        )
        self.lake_combo.currentIndexChanged.connect(
            self._update_default_thresholds
        )
        v.addWidget(self.lake_combo)

        btn_preset = QPushButton(
            "Use selected lake"
        )
        btn_preset.clicked.connect(
            self._use_preset
        )
        v.addWidget(btn_preset)

        btn_file = QPushButton(
            "Load AOI from file\u2026"
        )
        btn_file.clicked.connect(
            self._load_file
        )
        v.addWidget(btn_file)

        btn_draw = QPushButton(
            "Draw AOI box on map"
        )
        btn_draw.clicked.connect(
            self._start_drawing_aoi
        )
        v.addWidget(btn_draw)

        self.aoi_status = QLabel(
            "No area of interest selected yet."
        )
        self.aoi_status.setWordWrap(True)
        self.aoi_status.setStyleSheet(
            "color:#8ea6b6;"
        )
        v.addWidget(self.aoi_status)

        v.addWidget(_divider())

        # --- Stage 2 ---
        v.addWidget(
            _heading("2 \u00b7 Outputs")
        )

        v.addWidget(
            _caption("Earth Engine project ID:")
        )

        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText(
            "e.g. my-ee-project"
        )

        # Restored from QgsSettings so the ID survives QGIS restarts — it is the
        # only piece of panel state that persists. Re-saved on each successful
        # fetch in _get_fetch_settings.
        self.project_edit.setText(
            QgsSettings().value(
                "lake_classifier/ee_project",
                "",
                str,
            )
        )

        v.addWidget(self.project_edit)

        v.addWidget(
            _caption(
                "Year (summer composite, June\u2013September):"
            )
        )

        self.year_combo = QComboBox()
        self.year_combo.addItems(YEARS)
        v.addWidget(self.year_combo)

        v.addWidget(
            _caption(
                "Select the outputs to fetch "
                "(hover a checkbox for details):"
            )
        )

        self.chk_true_color = _output_checkbox(
            "True Color", "true_color"
        )
        v.addWidget(self.chk_true_color)

        self.chk_lst = _output_checkbox(
            "LST (Land Surface Temperature)", "lst"
        )
        v.addWidget(self.chk_lst)

        self.chk_ndwi = _output_checkbox(
            "NDWI (water index)", "ndwi"
        )
        v.addWidget(self.chk_ndwi)

        self.chk_ndvi = _output_checkbox(
            "NDVI (vegetation index)", "ndvi"
        )
        v.addWidget(self.chk_ndvi)

        self.chk_ndmi = _output_checkbox(
            "NDMI (moisture index)", "ndmi"
        )
        v.addWidget(self.chk_ndmi)

        btn_fetch = QPushButton(
            "Fetch selected outputs"
        )
        btn_fetch.clicked.connect(
            self._fetch_selected
        )
        v.addWidget(btn_fetch)

        self.vis_status = QLabel(
            "Set an AOI and Earth Engine project ID, "
            "check one or more outputs above, then fetch."
        )
        self.vis_status.setWordWrap(True)
        self.vis_status.setStyleSheet(
            "color:#8ea6b6;"
        )
        v.addWidget(self.vis_status)

        self.index_status = QLabel(
            "Fetch NDWI, NDVI, and NDMI above before "
            "running the classification."
        )
        self.index_status.setWordWrap(True)
        self.index_status.setStyleSheet(
            "color:#8ea6b6;"
        )
        v.addWidget(self.index_status)

        v.addWidget(_divider())

        # --- Stage 3 ---
        v.addWidget(
            _heading(
                "3 \u00b7 Threshold-Based Land Classification"
            )
        )

        v.addWidget(
            _caption(
                "Drag or type each cutoff. The classification "
                "combines the thresholds into open water, "
                "wetland, and land."
            )
        )

        # The three sliders span the full -1..1 range of a normalized-difference
        # index at 0.001 resolution. They are seeded with the first preset's
        # values purely because the combo box opens on that entry;
        # _update_default_thresholds overwrites all three whenever the lake
        # changes, so this initial choice is never what the user ends up with.
        # Each connects `committed` (not `valueChanged`) to reclassification, so
        # the map updates when a drag ends rather than on every pixel of travel.
        self.ndwi_threshold = ThresholdControl(
            "NDWI water cutoff",
            -1,
            1,
            DEFAULT_THRESHOLDS["Great Salt Lake, UT"]["ndwi"],
            step=0.001,
        )
        self.ndwi_threshold.committed.connect(
            self._reclassify_if_active
        )
        v.addWidget(self.ndwi_threshold)

        self.ndvi_threshold = ThresholdControl(
            "NDVI vegetation cutoff",
            -1,
            1,
            DEFAULT_THRESHOLDS["Great Salt Lake, UT"]["ndvi"],
            step=0.001,
        )
        self.ndvi_threshold.committed.connect(
            self._reclassify_if_active
        )
        v.addWidget(self.ndvi_threshold)

        self.ndmi_threshold = ThresholdControl(
            "NDMI moisture cutoff",
            -1,
            1,
            DEFAULT_THRESHOLDS["Great Salt Lake, UT"]["ndmi"],
            step=0.001,
        )
        self.ndmi_threshold.committed.connect(
            self._reclassify_if_active
        )
        v.addWidget(self.ndmi_threshold)

        btn_classify = QPushButton(
            "Classify: water / wetland / land"
        )
        btn_classify.clicked.connect(
            self._classify
        )
        v.addWidget(btn_classify)

        self.area_readout = QLabel(
            "Classify to see class areas."
        )
        self.area_readout.setWordWrap(True)
        self.area_readout.setStyleSheet(
            "color:#cdd9e1; "
            "font-size:11px; "
            "font-family:monospace;"
        )
        v.addWidget(self.area_readout)

        # --- Stage 4 ---
        v.addWidget(_divider())
        v.addWidget(_heading("4 · Time Series Analysis"))

        v.addWidget(_caption("Download monthly NDWI / NDVI / NDMI time series for the selected AOI and year."))

        btn_timeseries = QPushButton("Download Monthly Time Series CSV")
        v.addWidget(btn_timeseries)

        # Deliberately NOT connected here. Unlike every other button in this
        # panel, the time-series handler lives in lake_classifier.py, so the
        # button and its status label are exposed as attributes for that module
        # to wire up. If clicking it does nothing, the connection there is the
        # place to look, not this file.
        self.btn_timeseries = btn_timeseries
        self.csv_label = QLabel("CSV output: (none yet)")
        v.addWidget(self.csv_label)
        
        # -- Stage 5 --
        
        v.addWidget(
            _heading("5 \u00b7 OpenET")
        )

        v.addWidget(
            _caption(
                "Uses the AOI, project ID, year "
                "selected above. Coverage is CONUS only."
            )
        )

        btn_openet = QPushButton(
            "Fetch OpenET"
        )
        btn_openet.clicked.connect(
            self._fetch_openet
        )
        v.addWidget(btn_openet)

        self.openet_status = QLabel(
            "Fetch evapotranspiration (ET) for this AOI."
        )
        self.openet_status.setWordWrap(True)
        self.openet_status.setStyleSheet(
            "color:#8ea6b6;"
        )
        v.addWidget(self.openet_status)

        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.setWidget(scroll)

    # ------------------------------------------------------------------
    # Stage 1 — Area of Interest
    # ------------------------------------------------------------------

    def _zoom_to(self, rect_in_src, src_crs):
        """Zoom the canvas to `rect_in_src`, reprojecting into the project CRS.

        Takes the source CRS explicitly rather than assuming WGS84, because
        _load_file passes a rectangle in the file's own CRS — reprojecting the
        already-converted WGS84 copy would round-trip the coordinates twice.
        transformBoundingBox (not transform) is correct here: a reprojected
        rectangle is not generally a rectangle, so this returns the bounding box
        of the transformed shape.
        """
        canvas = self.iface.mapCanvas()
        dst_crs = QgsProject.instance().crs()

        if src_crs != dst_crs:
            transform = QgsCoordinateTransform(
                src_crs,
                dst_crs,
                QgsProject.instance(),
            )

            rect_in_src = (
                transform.transformBoundingBox(
                    rect_in_src
                )
            )

        canvas.setExtent(rect_in_src)
        canvas.refresh()

    def _use_preset(self):
        """Adopt the selected preset lake as the AOI."""
        name = self.lake_combo.currentText()
        rect = PRESETS[name]

        # Copied, not referenced: QgsRectangle is mutable, and handing out the
        # module-level PRESETS object would let later code corrupt the preset
        # for the rest of the session.
        self.aoi_extent = QgsRectangle(rect)
        # Cleared because a preset AOI has no backing vector layer.
        self.aoi_layer = None

        self._zoom_to(
            QgsRectangle(rect),
            WGS84,
        )

        self.aoi_status.setText(
            "AOI set to %s." % name
        )

        self._update_default_thresholds()

    def _update_default_thresholds(self):
        """Apply the calibrated defaults for the selected preset lake.

        Fires on every combo-box change, not just on 'Use selected lake', so
        the sliders track the dropdown even before the AOI is applied.

        setDefault (rather than setValue) also moves each control's reset
        target, so 'reset' afterwards returns to the newly selected lake's
        calibration. A lake with no DEFAULT_THRESHOLDS entry leaves the sliders
        untouched rather than zeroing them.
        """
        lake = self.lake_combo.currentText()
        values = DEFAULT_THRESHOLDS.get(lake)

        if values is None:
            return

        self.ndwi_threshold.setDefault(values["ndwi"])
        self.ndvi_threshold.setDefault(values["ndvi"])
        self.ndmi_threshold.setDefault(values["ndmi"])

    def _load_file(self):
        """Load an AOI from a vector file and adopt its bounding box.

        Only the extent is used — the AOI is always a rectangle downstream, so
        an irregular polygon is reduced to its bounds rather than being used as
        a true clip boundary. The layer itself is added to the project so the
        real shape stays visible over the results.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an AOI polygon",
            "",
            "Vector files (*.geojson *.json *.shp *.gpkg);;"
            "All files (*)",
        )

        if not path:
            return

        layer = QgsVectorLayer(
            path,
            os.path.basename(path),
            "ogr",
        )

        if not layer.isValid():
            self.aoi_status.setText(
                "Couldn't read that file as a vector layer."
            )
            return

        QgsProject.instance().addMapLayer(layer)

        self.aoi_layer = layer

        src_crs = layer.crs()
        extent = layer.extent()

        # Store in WGS84 for Earth Engine, but zoom using the ORIGINAL extent
        # and CRS below — _zoom_to reprojects to the project CRS itself, so
        # passing the converted copy would transform twice.
        if src_crs != WGS84:
            transform = QgsCoordinateTransform(
                src_crs,
                WGS84,
                QgsProject.instance(),
            )

            self.aoi_extent = (
                transform.transformBoundingBox(
                    extent
                )
            )
        else:
            self.aoi_extent = QgsRectangle(extent)

        self._zoom_to(
            extent,
            src_crs,
        )

        self.aoi_status.setText(
            "AOI loaded from %s." % layer.name()
        )

    def _start_drawing_aoi(self):
        """Activate the rectangle tool so the user can drag an AOI on the map.

        The tool is created once and reused; only its state is cleared between
        activations, which keeps the signal connected to exactly one handler
        instead of stacking a new connection per click.
        """
        canvas = self.iface.mapCanvas()

        if self.rectangle_tool is None:
            self.rectangle_tool = RectangleMapTool(
                canvas
            )

            self.rectangle_tool.rectangleDrawn.connect(
                self._finish_drawing_aoi
            )

        self.rectangle_tool.clear()
        canvas.setMapTool(self.rectangle_tool)

        self.aoi_status.setText(
            "Click and drag on the map to draw "
            "a rectangular AOI."
        )

    def _finish_drawing_aoi(self, canvas_rect):
        """Receive the drawn rectangle, convert to WGS84, and release the tool.

        The tool emits in canvas CRS, so this is where the drawn path joins the
        same WGS84 convention as the preset and file paths. Unsetting the map
        tool at the end returns the canvas to normal panning — otherwise the
        next click would begin drawing another box.
        """
        canvas_crs = QgsProject.instance().crs()

        if canvas_crs != WGS84:
            transform = QgsCoordinateTransform(
                canvas_crs,
                WGS84,
                QgsProject.instance(),
            )

            self.aoi_extent = (
                transform.transformBoundingBox(
                    canvas_rect
                )
            )
        else:
            self.aoi_extent = QgsRectangle(
                canvas_rect
            )

        self.aoi_layer = None

        self.aoi_status.setText(
            "AOI set from the box drawn on the map."
        )

        self.iface.mapCanvas().unsetMapTool(
            self.rectangle_tool
        )

    # ------------------------------------------------------------------
    # Shared Earth Engine settings
    # ------------------------------------------------------------------

    def _get_fetch_settings(self, status_label):
        """Validate the AOI, project ID, and year settings.

        The single precondition gate for every Earth Engine call — Stage 2,
        Stage 5, and (via lake_classifier.py) Stage 4 all route through it, so
        the "no AOI" and "no project ID" messages stay identical everywhere.

        Takes the status label to write into, because each stage reports into
        its own label. Returns None when anything is missing (caller must
        abort), otherwise (project, year, start, end).

        The date window is hardcoded to June 1 – September 30: a summer
        composite avoids snow and low sun angles, and matches the season the
        default thresholds were calibrated on. Changing it here silently
        invalidates those calibrations.
        """
        if self.aoi_extent is None:
            status_label.setText(
                "Choose an area of interest first "
                "in Section 1."
            )
            return None

        project = self.project_edit.text().strip()

        if not project:
            status_label.setText(
                "Enter your Earth Engine project ID."
            )
            return None

        # Persisted on every successful validation, so a working project ID is
        # remembered across QGIS restarts without a separate save action.
        QgsSettings().setValue(
            "lake_classifier/ee_project",
            project,
        )

        year = int(
            self.year_combo.currentText()
        )

        # Summer window — see the docstring.
        start = "%d-06-01" % year
        end = "%d-09-30" % year

        return project, year, start, end

    # ------------------------------------------------------------------
    # Stage 2 — Outputs (dispatcher + one helper per output)
    # ------------------------------------------------------------------

    def _fetch_selected(self):
        """Fetch whichever outputs are checked, using one shared settings read.

        Dispatcher for the Stage 2 button. Two things it deliberately does:

        * Validates ONCE via _get_fetch_settings and passes the result to each
          helper, so a three-output fetch cannot half-succeed on a stale AOI.
        * Collects each helper's returned status string and joins them into a
          single multi-line message, which is why the helpers return text
          instead of writing to the label themselves.

        The three indices share one download (_do_fetch_indices) because they
        come from the same composite — checking all three costs one Earth
        Engine round trip, not three.
        """
        want_true_color = self.chk_true_color.isChecked()
        want_lst = self.chk_lst.isChecked()
        want_ndwi = self.chk_ndwi.isChecked()
        want_ndvi = self.chk_ndvi.isChecked()
        want_ndmi = self.chk_ndmi.isChecked()
        want_indices = want_ndwi or want_ndvi or want_ndmi

        if not (want_true_color or want_lst or want_indices):
            self.vis_status.setText(
                "Check at least one output above, then fetch."
            )
            return

        settings = self._get_fetch_settings(self.vis_status)

        if settings is None:
            return

        messages = []

        if want_true_color:
            messages.append(
                self._do_fetch_true_color(settings)
            )

        if want_lst:
            messages.append(
                self._do_fetch_lst(settings)
            )

        if want_indices:
            messages.append(
                self._do_fetch_indices(
                    settings,
                    want_ndwi,
                    want_ndvi,
                    want_ndmi,
                )
            )

        self.vis_status.setText(
            "\n".join(messages)
        )

    def _do_fetch_true_color(self, settings):
        """Fetch and load the True Color composite. Returns a status string.

        Shape shared by all three Stage 2 helpers: set a progress label, pump
        the event loop, call ee_backend, build a styled layer, return a
        one-line result. ImportError is caught separately from other failures
        because a missing earthengine-api is a setup problem with a specific
        remedy, not an Earth Engine error.
        """
        project, year, start, end = settings

        self.vis_status.setText(
            "Fetching True Color from Earth Engine\u2026"
        )
        # Forces the label to paint before the blocking download starts \u2014
        # without it the panel just freezes with no explanation.
        QApplication.processEvents()

        try:
            # Imported inside the handler so the plugin still loads (and can
            # explain itself) on a QGIS without earthengine-api installed.
            from . import ee_backend

            ee = ee_backend.initialize(project)

            image, region = (
                ee_backend.build_true_color_image(
                    ee,
                    self.aoi_extent,
                    start,
                    end,
                )
            )

            path, scale = (
                ee_backend.download_geotiff(
                    image,
                    region,
                    self.aoi_extent,
                    filename="lake_true_color.tif",
                )
            )

        except ImportError:
            return (
                "earthengine-api isn't installed in QGIS yet. "
                "Install the 'Google Earth Engine' plugin, "
                "then try again."
            )

        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "Saline Lake Classifier",
                str(exc),
            )
            return "True Color error: %s" % exc

        layer = _make_true_color_layer(
            path,
            "True Color %d" % year,
        )

        if layer is None:
            return (
                "Downloaded the True Color file, "
                "but QGIS couldn't open it."
            )

        QgsProject.instance().addMapLayer(layer)
        self.true_color_layer = layer

        return (
            "Loaded True Color for %d at approximately "
            "%d m/pixel."
            % (year, round(scale))
        )

    def _do_fetch_lst(self, settings):
        """Fetch and load the LST composite. Returns a status string.

        Same shape as _do_fetch_true_color, but rendered with LST_STOPS, whose
        values are degrees Celsius rather than a -1..1 index — the one ramp in
        this file that is not a normalized-difference index.
        """
        project, year, start, end = settings

        self.vis_status.setText(
            "Fetching land surface temperature "
            "from Earth Engine\u2026"
        )
        QApplication.processEvents()

        try:
            from . import ee_backend

            ee = ee_backend.initialize(project)

            image, region = (
                ee_backend.build_lst_image(
                    ee,
                    self.aoi_extent,
                    start,
                    end,
                )
            )

            path, scale = (
                ee_backend.download_geotiff(
                    image,
                    region,
                    self.aoi_extent,
                    filename="lake_lst.tif",
                )
            )

        except ImportError:
            return (
                "earthengine-api isn't installed in QGIS yet. "
                "Install the 'Google Earth Engine' plugin, "
                "then try again."
            )

        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "Saline Lake Classifier",
                str(exc),
            )
            return "LST error: %s" % exc

        layer = _make_index_layer(
            path,
            1,
            "LST \u00b0C %d" % year,
            LST_STOPS,
        )

        if layer is None:
            return (
                "Downloaded the LST file, "
                "but QGIS couldn't open it."
            )

        QgsProject.instance().addMapLayer(layer)
        self.lst_layer = layer

        return (
            "Loaded land surface temperature for %d "
            "at approximately %d m/pixel."
            % (year, round(scale))
        )

    def _do_fetch_indices(
        self,
        settings,
        want_ndwi,
        want_ndvi,
        want_ndmi,
    ):
        """Fetch NDWI/NDVI/NDMI (one shared download) and load only the
        checked bands as map layers. The full 3-band raster is always kept
        as self.index_raster_path so Stage 3 classification can run even
        if not every band was displayed. Returns a status string.
        """
        project, year, start, end = settings

        self.vis_status.setText(
            "Fetching NDWI, NDVI, and NDMI "
            "from Earth Engine\u2026"
        )
        QApplication.processEvents()

        try:
            from . import ee_backend

            ee = ee_backend.initialize(project)

            image, region = (
                ee_backend.build_index_image(
                    ee,
                    self.aoi_extent,
                    start,
                    end,
                )
            )

            path, scale = (
                ee_backend.download_geotiff(
                    image,
                    region,
                    self.aoi_extent,
                    filename="lake_indices.tif",
                )
            )

        except ImportError:
            msg = (
                "earthengine-api isn't installed in QGIS yet. "
                "Install the 'Google Earth Engine' plugin, "
                "then try again."
            )
            self.index_status.setText(msg)
            return msg

        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "Saline Lake Classifier",
                str(exc),
            )
            msg = "Indices error: %s" % exc
            self.index_status.setText(msg)
            return msg

        # (wanted, band index, short name, layer name, ramp, attribute to set).
        # Band numbers are the contract with ee_backend.build_index_image and
        # with _classify_raster, which reads the same 1/2/3 order.
        bands = (
            (want_ndwi, 1, "NDWI", "NDWI %d" % year, NDWI_STOPS, "ndwi_layer"),
            (want_ndvi, 2, "NDVI", "NDVI %d" % year, NDVI_STOPS, "ndvi_layer"),
            (want_ndmi, 3, "NDMI", "NDMI %d" % year, NDMI_STOPS, "ndmi_layer"),
        )

        loaded_names = []
        failed_names = []

        # Add bottom-to-top so NDWI ends up on top when all three are checked.
        # QGIS inserts each new layer above the previous one, so iterating in
        # reverse leaves the layer tree in the declared NDWI/NDVI/NDMI order.
        for want, band, name, layer_name, stops, attr in reversed(bands):
            if not want:
                continue

            layer = _make_index_layer(path, band, layer_name, stops)

            # One unreadable band does not abort the others; the names are
            # collected and reported together at the end.
            if layer is None:
                failed_names.append(name)
                continue

            QgsProject.instance().addMapLayer(layer)
            setattr(self, attr, layer)
            loaded_names.append(name)

        # The full raster (all 3 bands) is always available for classification,
        # regardless of which bands were checked for display.
        self.index_raster_path = path

        if failed_names and not loaded_names:
            msg = (
                "Downloaded the index file, but QGIS "
                "couldn't open %s." % ", ".join(failed_names)
            )
            self.index_status.setText(msg)
            return msg

        self.index_status.setText(
            "Indices ready for %d at approximately %d m/pixel. "
            "Set the thresholds and classify."
            % (year, round(scale))
        )

        result = (
            "Loaded %s for %d at approximately %d m/pixel."
            % (", ".join(reversed(loaded_names)), year, round(scale))
        )

        if failed_names:
            result += (
                " (QGIS couldn't open %s.)" % ", ".join(failed_names)
            )

        return result
    
    def _fetch_openet(self):
        """Fetch and load an OpenET evapotranspiration raster (Stage 5).

        Coverage is CONUS only — an AOI outside the continental US will fail in
        ee_backend rather than here.

        THREE INCONSISTENCIES WITH THE STAGE 2 HELPERS, noted because they are
        easy to trip over and are behavioural, not stylistic:

        1. This is connected DIRECTLY to its button, so it is the handler, not
           a helper. Its return value is therefore discarded by Qt.
        2. It mixes both reporting styles: the error paths call
           self.openet_status.setText(...), but the two success/failure paths
           at the end `return` a string that nothing displays. The result is
           that a successful fetch leaves the label reading
           "Fetching evapotranspiration…" indefinitely.
        3. On success it assigns the new layer to self.lst_layer, overwriting
           the reference held by _do_fetch_lst. The layer itself stays on the
           map, but the dock loses its handle on the LST layer.

        Left as-is because changing any of them alters behaviour; flagged here
        so the next reader does not mistake it for the pattern to copy.
        """
        settings = self._get_fetch_settings(
            self.openet_status
        )
        if settings is None:
            return
        project, year, start, end = settings
        self.openet_status.setText(
            "Fetching evapotranspiration "
            "from Earth Engine\u2026"
        )
        QApplication.processEvents()
        try:
            from . import ee_backend
            ee = ee_backend.initialize(project)
            image, region = (
                ee_backend.build_openet_image(
                    ee,
                    self.aoi_extent,
                    start,
                    end,
                )
            )
            path, scale = (
                ee_backend.download_geotiff(
                    image,
                    region,
                    self.aoi_extent,
                    filename="lake_openet.tif",
                )
            )
        except ImportError:
            self.openet_status.setText(
                "earthengine-api isn't installed in QGIS yet. "
                "Install the 'Google Earth Engine' plugin, "
                "then try again."
            )
            return
        except Exception as exc:
            self.openet_status.setText(
                "Earth Engine error: %s" % exc
            )
            self.iface.messageBar().pushWarning(
                "Saline Lake Classifier",
                str(exc),
            )
            return
        layer = _make_index_layer(
            path,
            1,
            "ET mm %d" % year,
            OPENET_STOPS,
        )
        if layer is None:
            return (
                "Downloaded the OpenET file, "
                "but QGIS couldn't open it."
            )

        QgsProject.instance().addMapLayer(layer)
        # NOTE: overwrites the LST layer handle — see this method's docstring.
        self.lst_layer = layer

        # NOTE: returned to Qt, which discards it, so this message never
        # reaches openet_status. See point 2 in the docstring.
        return (
            "Loaded OpenET %d "
            "at approximately %d m/pixel."
            % (year, round(scale))
        )

    # ------------------------------------------------------------------
    # Stage 3 — Threshold-Based Land Classification
    # ------------------------------------------------------------------

    def _reclassify_if_active(self):
        """Refresh the classification after a threshold is changed.

        The guard is what makes threshold edits free before the first manual
        classify: moving a slider only recomputes once a classification layer
        already exists, so adjusting thresholds while exploring indices does no
        work and shows no errors.
        """
        if self.class_layer is not None:
            self._classify()

    def _classify(self):
        """Run the threshold classification and swap in the resulting layer.

        Requires a completed Stage 2 index fetch — index_raster_path is the
        precondition, and it is set whenever indices were downloaded, even for
        bands that were never displayed.

        Each run writes a NEW temp file rather than overwriting the old one:
        QGIS holds an open handle on the layer it is displaying, so rewriting
        that path in place would produce a locked file on Windows and a stale
        render elsewhere. The previous layer is removed and its file deleted
        only after the replacement is safely registered.
        """
        if not self.index_raster_path:
            self.index_status.setText(
                "Fetch the Landsat indices first, "
                "then classify."
            )
            return

        ndwi_threshold = (
            self.ndwi_threshold.value()
        )

        ndvi_threshold = (
            self.ndvi_threshold.value()
        )

        ndmi_threshold = (
            self.ndmi_threshold.value()
        )

        year = self.year_combo.currentText()

        fd, out_path = tempfile.mkstemp(
            prefix="lake_class_",
            suffix=".tif",
        )

        os.close(fd)

        try:
            counts, pixel_area_km2 = (
                _classify_raster(
                    self.index_raster_path,
                    ndwi_threshold,
                    ndvi_threshold,
                    ndmi_threshold,
                    out_path,
                )
            )

        except Exception as exc:
            self.index_status.setText(
                "Classification error: %s" % exc
            )

            self.iface.messageBar().pushWarning(
                "Saline Lake Classifier",
                str(exc),
            )
            return

        layer = QgsRasterLayer(
            out_path,
            "Classification %s" % year,
        )

        if not layer.isValid():
            self.index_status.setText(
                "Built the classified file, "
                "but QGIS couldn't load it."
            )
            return

        classes = [
            QgsPalettedRasterRenderer.Class(
                value,
                QColor(color),
                label,
            )
            for value, color, label in CLASS_DEFS
        ]

        renderer = QgsPalettedRasterRenderer(
            layer.dataProvider(),
            1,
            classes,
        )

        layer.setRenderer(renderer)

        # Replace the previous classification layer. Wrapped in try/except
        # because the user may have deleted it from the layer tree by hand, in
        # which case the stored id is already gone — not worth an error dialog.
        if self.class_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(
                    self.class_layer.id()
                )
            except Exception:
                pass

        old_path = self.class_path

        QgsProject.instance().addMapLayer(layer)

        self.class_layer = layer
        self.class_path = out_path

        # Delete the superseded temp file only after the new layer is live, so
        # repeated slider adjustments do not litter the temp directory with one
        # GeoTIFF per reclassification. Failure is ignored: on Windows the file
        # may still be locked by the renderer, and a leftover temp file is a
        # far smaller problem than a raised exception mid-classify.
        if old_path and old_path != out_path:
            try:
                os.remove(old_path)
            except OSError:
                pass

        self._update_area_readout(
            counts,
            pixel_area_km2,
        )

        self.index_status.setText(
            "Classified %s "
            "(NDWI \u2265 %.3f, NDVI \u2265 %.3f, "
            "NDMI \u2265 %.3f)."
            % (
                year,
                ndwi_threshold,
                ndvi_threshold,
                ndmi_threshold,
            )
        )

    def _update_area_readout(
        self,
        counts,
        pixel_area_km2,
    ):
        """Render the three class areas as an aligned km² / percent table.

        Percentages are of CLASSIFIED area, not of the AOI — NoData pixels are
        excluded from `total`, so the three always sum to 100% however much of
        the scene was cloud-masked. The area figures inherit the approximation
        in _classify_raster's pixel_area_km2 and are best read as close
        estimates rather than survey numbers.
        """
        total = (
            counts[1]
            + counts[2]
            + counts[3]
        )

        if total == 0:
            self.area_readout.setText(
                "No classified pixels - check thresholds."
            )
            return

        # %-11s pads the class name to a fixed width; combined with the
        # monospace font set on area_readout in _build_ui, this is what keeps
        # the three rows aligned into columns.
        def line(name, value):
            area = (
                counts[value]
                * pixel_area_km2
            )

            percentage = (
                100.0
                * counts[value]
                / total
            )

            return "%-11s %s km\u00b2  (%2.0f%%)" % (
                name,
                format(area, ",.0f"),
                percentage,
            )

        self.area_readout.setText(
            line("Open water", 1)
            + "\n"
            + line("Wetland", 2)
            + "\n"
            + line("Land", 3)
        )

    # ------------------------------------------------------------------
    # Time Series Plotting
    # ------------------------------------------------------------------

    def plot_timeseries(self, df, year):
        """Plot NDWI, NDVI, NDMI monthly time series.

        Called by lake_classifier.py after it retrieves the monthly table from
        ee_backend.compute_monthly_timeseries — this class never fetches the
        data itself. `df` needs a 'month' column plus one column per index.

        Opens a blocking matplotlib window via plt.show(), outside QGIS's own
        widget system; the panel is unresponsive until it is closed.

        NOTE: month_labels is fixed at four entries matching the June–September
        window in _get_fetch_settings. If that window ever widens, the tick
        labels here will silently mismatch the data — set_xticks uses the real
        month values, but the labels drawn over them come from this list.
        """
        import matplotlib.pyplot as plt

        month_labels = ["Jun", "Jul", "Aug", "Sep"]

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(df["month"], df["NDWI"], label="NDWI", color="blue", marker="o")
        ax.plot(df["month"], df["NDVI"], label="NDVI", color="green", marker="o")
        ax.plot(df["month"], df["NDMI"], label="NDMI", color="brown", marker="o")

        # Value labels on every point: with only four months, the exact numbers
        # matter more than the trend line, and these indices differ at the
        # third decimal place — too fine to read off an axis.
        for _, row in df.iterrows():
            m = row["month"]
            ax.text(m, row["NDWI"], f"{row['NDWI']:.3f}", color="blue", fontsize=8, ha="left", va="bottom")
            ax.text(m, row["NDVI"], f"{row['NDVI']:.3f}", color="green", fontsize=8, ha="left", va="bottom")
            ax.text(m, row["NDMI"], f"{row['NDMI']:.3f}", color="brown", fontsize=8, ha="left", va="bottom")

        ax.set_xticks(df["month"])
        ax.set_xticklabels(month_labels, rotation=45)

        ax.set_xlabel("Month")
        ax.set_ylabel("Index Value")
        ax.set_title(f"Summer NDWI / NDVI/ NDMI - {year} (Median Composite)")
        ax.legend()

        fig.tight_layout()
        plt.show()

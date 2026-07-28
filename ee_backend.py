"""Earth Engine backend for the Saline Lake Classifier.

This file owns every Google Earth Engine operation used by the plugin:
Landsat collection loading, cloud masking, band scaling, compositing, spectral
index calculation, OpenET processing, GeoTIFF download, and monthly time-series
generation.

Keeping these operations separate from dock.py provides two benefits:

* The user interface can import this file lazily, allowing the plugin panel
  to open and display a useful setup message even when earthengine-api is not
  installed in the QGIS Python environment.
* The processing functions can be tested independently from the dock widget
  using the QGIS Python Console.

Quick console test after Earth Engine authentication:

    from lake_classifier import ee_backend
    from qgis.core import QgsRectangle

    ee = ee_backend.initialize("your-project-id")

    extent = QgsRectangle(    # Great Salt Lake 
        -113.1,
        40.6,
        -111.9,
        41.7,
    )

    image, region = ee_backend.build_index_image(
        ee,
        extent,
        "2022-06-01",
        "2022-09-30",
    )

    path, scale = ee_backend.download_geotiff(
        image,
        region,
        extent,
    )

    print(path, scale)
"""

import math
import os
import tempfile
import urllib.request


# ---------------------------------------------------------------------------
# Earth Engine datasets and Landsat mission configuration
# ---------------------------------------------------------------------------

# OpenET monthly ensemble evapotranspiration data. The selected band stores
# monthly ensemble ET values for the continental United States.
OPENET_COLLECTION = "OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"
OPENET_BAND = "et_ensemble_mad"


# Landsat Collection 2, Tier 1, Level 2 collection IDs and mission-specific
# band names.
#
# Landsat missions do not use identical source band numbers. Each dictionary
# identifies which original band represents blue, green, red, near-infrared,
# shortwave-infrared, and thermal data for that mission. _rename_bands later
# converts these mission-specific names into one shared naming convention.
COLLECTIONS = {
    "L5": {
        "id": "LANDSAT/LT05/C02/T1_L2",
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "tirs": "ST_B6",
    },
    "L7": {
        "id": "LANDSAT/LE07/C02/T1_L2",
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "tirs": "ST_B6",
    },
    "L8": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "tirs": "ST_B10",
    },
    "L9": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "tirs": "ST_B10",
    },
}


# Years during which each Landsat mission is eligible for use by the plugin.
# _load_collection checks this table before querying Earth Engine so it does
# not request imagery from a mission that was not operating in the selected
# year.
#
# Python ranges exclude their final number:
#     range(1985, 2013) includes 1985 through 2012.
MISSION_YEARS = {
    "L5": range(1985, 2013),
    "L7": range(1999, 2021),
    "L8": range(2013, 2100),
    "L9": range(2022, 2100),
}


# ---------------------------------------------------------------------------
# Earth Engine initialization
# ---------------------------------------------------------------------------

def initialize(project_id):
    """Initialize Earth Engine using the supplied cloud project ID.

    Assumes the user has already authenticated through the Google Earth Engine
    QGIS plugin or ee.Authenticate(). Authentication establishes the user's
    identity, while the project ID identifies the Google Cloud project used for
    the Earth Engine requests.

    Returns the initialized Earth Engine module so downstream functions do not
    need to import or initialize it independently.
    """
    import ee

    ee.Initialize(project=project_id)

    return ee


# ---------------------------------------------------------------------------
# Per-image Landsat preprocessing helpers
# ---------------------------------------------------------------------------

def _mask_clouds(image):
    """Mask pixels identified as clouds in the Landsat QA_PIXEL band.

    Bit 3 of QA_PIXEL stores the cloud flag. The bitwise operation isolates
    that flag, and eq(0) retains pixels that are not classified as cloudy.

    Cloud-shadow masking is intentionally skipped because dark water surfaces
    can sometimes be confused with shadows by the QA algorithm.
    Masking those pixels could remove valid open-water observations.
    """
    # Shift the binary value 1 to bit position 3, isolate that bit in
    # QA_PIXEL, and retain pixels where the cloud flag equals zero.
    cloud_bit = (
        image.select("QA_PIXEL")
        .bitwiseAnd(1 << 3)
        .eq(0)
    )

    return image.updateMask(cloud_bit)


def _scale_sr(image):
    """Convert stored Landsat surface-reflectance values to reflectance.

    Landsat Collection 2 Level 2 surface-reflectance bands are stored as scaled
    integer Digital Number values. Applying the USGS scale factor and offset
    converts those stored values into dimensionless surface reflectance:

        reflectance = DN * 0.0000275 - 0.2

    Values below zero are clipped to zero by the existing max(0) operation.
    The scaled bands replace the original SR_B* bands in the returned image.
    """
    # The wildcard selects every surface-reflectance band present in the image,
    # regardless of which Landsat mission produced it.
    sr_bands = image.select("SR_B.*")

    scaled = (
        sr_bands
        .multiply(0.0000275)
        .add(-0.2)
        .max(0)
    )

    # Replace the stored integer SR bands with their scaled reflectance values.
    return image.addBands(
        scaled,
        overwrite=True,
    )


def _scale_lst(image):
    """Convert the Landsat surface-temperature band to degrees Celsius.

    Landsat Collection 2 Level 2 surface temperature uses a scale factor and
    offset. The first two operations convert the stored value to Kelvin, and
    subtracting 273.15 converts Kelvin to degrees Celsius.

    The converted values overwrite the original ST_B6 or ST_B10 band.
    """
    # Match the thermal surface-temperature band used by any supported mission.
    st_band = image.select("ST_B.*")

    lst_c = (
        st_band
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
    )

    # Replace the original thermal-band values with land surface temperature
    # expressed in degrees Celsius.
    return image.addBands(
        lst_c,
        overwrite=True,
    )


def _rename_bands(
    image,
    blue,
    green,
    red,
    nir,
    swir1,
    tirs,
):
    """Rename mission-specific Landsat bands to common names.

    Landsat 5 and 7 use different source band numbers from Landsat 8 and 9.
    Standardizing the names here allows every downstream function to refer to
    BLUE, GREEN, RED, NIR, SWIR1, and LST without checking which mission
    produced the image.
    """
    source_bands = [
        blue,
        green,
        red,
        nir,
        swir1,
        tirs,
    ]

    common_names = [
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR1",
        "LST",
    ]

    return image.select(
        source_bands,
        common_names,
    )


def _load_collection(
    ee,
    mission_key,
    region,
    start_date,
    end_date,
):
    """Load and preprocess one Landsat mission's image collection.

    The function first checks whether the mission was operating during the
    requested year. Eligible scenes are then filtered spatially and temporally,
    screened using scene-level cloud metadata, masked at the pixel level,
    scaled, and renamed to the shared band convention.

    Returns:
        An ee.ImageCollection with bands named BLUE, GREEN, RED, NIR, SWIR1,
        and LST, or None when the mission is not eligible for the selected year.
    """
    # The date strings use YYYY-MM-DD format, so the first four characters
    # provide the selected year without requiring a date-parsing dependency.
    year = int(start_date[:4])

    # Skip missions that were not operating during the requested year.
    if year not in MISSION_YEARS[mission_key]:
        return None

    # Load the Earth Engine collection ID and original band names for the
    # selected Landsat mission.
    cfg = COLLECTIONS[mission_key]

    collection = (
        ee.ImageCollection(cfg["id"])

        # Retain scenes whose footprints intersect the selected AOI.
        .filterBounds(region)

        # Retain scenes acquired during the requested date window.
        .filterDate(start_date, end_date)

        # Remove entire scenes with 60 percent or greater reported cloud cover.
        # This is a scene-level filter and is separate from _mask_clouds, which
        # masks individual cloudy pixels in each retained scene.
        .filter(
            ee.Filter.lt(
                "CLOUD_COVER",
                60,
            )
        )

        # Apply the preprocessing helpers to every retained image.
        .map(_mask_clouds)
        .map(_scale_sr)
        .map(_scale_lst)

        # Pass this mission's source band names into the shared renaming helper.
        .map(
            lambda image: _rename_bands(
                image,
                cfg["blue"],
                cfg["green"],
                cfg["red"],
                cfg["nir"],
                cfg["swir1"],
                cfg["tirs"],
            )
        )
    )

    return collection


# ---------------------------------------------------------------------------
# Shared Landsat composite
# ---------------------------------------------------------------------------

def _build_landsat_composite(
    ee,
    extent,
    start_date,
    end_date,
):
    """Build a median composite from every eligible Landsat mission.

    The QGIS AOI rectangle is converted to an Earth Engine geometry, and one
    preprocessed image collection is loaded for each mission operating during
    the requested year. Those collections are merged before a pixel-wise median
    composite is calculated.

    The returned image uses the common bands BLUE, GREEN, RED, NIR, SWIR1,
    and LST.

    Returns:
        (composite, region), where composite is an ee.Image and region is the
        Earth Engine rectangle representing the QGIS AOI.
    """
    # Convert the QGIS QgsRectangle coordinates into an Earth Engine geometry.
    # dock.py stores every AOI in EPSG:4326, so no additional transformation is
    # needed here.
    region = ee.Geometry.Rectangle(
        [
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        ]
    )

    # Store one preprocessed ImageCollection for every eligible mission.
    parts = []

    # Check all supported missions in chronological order.
    for mission_key in ("L5", "L7", "L8", "L9"):
        collection = _load_collection(
            ee,
            mission_key,
            region,
            start_date,
            end_date,
        )

        if collection is not None:
            parts.append(collection)

    if not parts:
        raise ValueError(
            "No Landsat missions were active for %s–%s."
            % (
                start_date,
                end_date,
            )
        )

    if len(parts) == 1:
        # Only one mission is eligible, so no collection merge is required.
        merged = parts[0]

    else:
        # Begin with the first mission and merge each remaining mission into
        # the same combined image collection.
        merged = parts[0]

        for collection in parts[1:]:
            merged = merged.merge(collection)

    # Create a pixel-wise median composite. The median reduces the influence of
    # isolated cloud remnants, sensor noise, and other outlying observations.
    composite = (
        merged
        .median()
        .clip(region)
    )

    return composite, region


# ---------------------------------------------------------------------------
# Public image-building functions
# ---------------------------------------------------------------------------

def build_index_image(
    ee,
    extent,
    start_date,
    end_date,
):
    """Build a three-band NDWI, NDVI, and NDMI Landsat composite.

    Uses all eligible Landsat missions to create one median composite for the
    requested area and date range, then calculates the three normalized-
    difference indices used by the plugin.

    Indices:
        NDWI = (Green - NIR) / (Green + NIR)
        NDVI = (NIR - Red) / (NIR + Red)
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Args:
        ee:
            Initialized Earth Engine module returned by initialize().

        extent:
            QgsRectangle in EPSG:4326 defining the area of interest.

        start_date:
            Beginning of the composite window in YYYY-MM-DD format.

        end_date:
            End of the composite window in YYYY-MM-DD format.

    Returns:
        (image, region), where image contains:

            Band 1 = NDWI
            Band 2 = NDVI
            Band 3 = NDMI
    """
    composite, region = _build_landsat_composite(
        ee,
        extent,
        start_date,
        end_date,
    )

    # normalizedDifference performs (first band - second band) divided by
    # (first band + second band).
    ndwi = composite.normalizedDifference(
        [
            "GREEN",
            "NIR",
        ]
    ).rename("NDWI")

    ndvi = composite.normalizedDifference(
        [
            "NIR",
            "RED",
        ]
    ).rename("NDVI")

    ndmi = composite.normalizedDifference(
        [
            "NIR",
            "SWIR1",
        ]
    ).rename("NDMI")

    # Band order is a shared contract with dock._classify_raster and must remain
    # NDWI, NDVI, NDMI unless the readers in dock.py are updated as well.
    image = (
        ndwi
        .addBands(ndvi)
        .addBands(ndmi)
        .clip(region)
    )

    return image, region


def build_true_color_image(
    ee,
    extent,
    start_date,
    end_date,
):
    """Build an eight-bit true-color Landsat composite.

    The shared Landsat median composite is visualized using the red, green, and
    blue reflectance bands. Earth Engine's visualize operation converts the
    continuous reflectance values into an eight-bit RGB image suitable for
    direct display in QGIS.

    Returns:
        (image, region), where image contains red, green, and blue display bands
        in that order.
    """
    composite, region = _build_landsat_composite(
        ee,
        extent,
        start_date,
        end_date,
    )

    image = composite.visualize(
        bands=[
            "RED",
            "GREEN",
            "BLUE",
        ],
        min=0.0,
        max=0.3,
        gamma=1.2,
    ).clip(region)

    return image, region


def build_lst_image(
    ee,
    extent,
    start_date,
    end_date,
):
    """Build a median land surface temperature image in degrees Celsius.

    Returns the LST band from the shared Landsat median composite and preserves
    the LST band name for use in the downloaded GeoTIFF metadata.
    """
    composite, region = _build_landsat_composite(
        ee,
        extent,
        start_date,
        end_date,
    )

    image = (
        composite
        .select("LST")
        .rename("LST")
        .clip(region)
    )

    return image, region


def build_openet_image(
    ee,
    extent,
    start_date,
    end_date,
):
    """Build total OpenET evapotranspiration for the requested period.

    Loads monthly OpenET ensemble data intersecting the AOI, selects the
    ensemble evapotranspiration band, and sums the monthly values over the
    requested date window.

    OpenET Ensemble coverage is limited to the continental United States.
    An informative ValueError is raised when no matching images are available.

    Returns:
        (image, region), where image contains one band named ET with the summed
        evapotranspiration values.
    """
    # Convert the QGIS QgsRectangle into an Earth Engine rectangle.
    region = ee.Geometry.Rectangle(
        [
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        ]
    )

    collection = (
        ee.ImageCollection(OPENET_COLLECTION)
        .filterBounds(region)
        .filterDate(
            start_date,
            end_date,
        )
    )

    # Transfer only the collection count to local Python. Checking before sum()
    # allows the plugin to provide a clear coverage message instead of failing
    # later while processing an empty collection.
    if collection.size().getInfo() == 0:
        raise ValueError(
            "No OpenET data for %s–%s in this area. "
            "OpenET Ensemble coverage is CONUS only."
            % (
                start_date,
                end_date,
            )
        )

    # Sum the monthly ensemble ET values into one seasonal total.
    image = (
        collection
        .select(OPENET_BAND)
        .sum()
        .rename("ET")
        .clip(region)
    )

    return image, region


# ---------------------------------------------------------------------------
# GeoTIFF download and metadata
# ---------------------------------------------------------------------------

def download_geotiff(
    image,
    region,
    extent,
    out_path=None,
    target_px=1500,
    filename="lake_raster.tif",
):
    """Download an Earth Engine image as a local GeoTIFF.

    The export resolution is selected dynamically so the AOI's longer
    dimension contains approximately target_px pixels. This keeps large AOIs
    within practical Earth Engine download limits while retaining more detail
    for smaller AOIs.

    The calculated scale is constrained to 30–300 meters per pixel:

        30 meters = native Landsat multispectral resolution
        300 meters = the coarsest permitted plugin export resolution

    Args:
        image:
            Earth Engine image containing the bands to download.

        region:
            Earth Engine geometry defining the export region.

        extent:
            QgsRectangle in EPSG:4326 used to estimate the AOI dimensions.

        out_path:
            Optional permanent output location. When omitted, the file is
            written to the operating system's temporary directory.

        target_px:
            Approximate number of pixels along the AOI's longer dimension.

        filename:
            Filename used when out_path is not supplied.

    Returns:
        (path, scale), where path is the local GeoTIFF path and scale is the
        export spatial resolution in meters per pixel.
    """
    # Calculate the AOI's central latitude. The ground distance represented by
    # one degree of longitude decreases toward the poles, so longitude distance
    # must be adjusted using the cosine of latitude.
    lat_center = (
        extent.yMinimum()
        + extent.yMaximum()
    ) / 2.0

    meters_per_degree_longitude = (
        111_000
        * math.cos(
            math.radians(lat_center)
        )
    )

    width_m = (
        abs(
            extent.xMaximum()
            - extent.xMinimum()
        )
        * meters_per_degree_longitude
    )

    # One degree of latitude is approximated as 111 kilometers.
    height_m = (
        abs(
            extent.yMaximum()
            - extent.yMinimum()
        )
        * 111_000
    )

    # Select a scale that produces approximately target_px pixels along the
    # longer side of the requested AOI.
    scale = (
        max(
            width_m,
            height_m,
        )
        / float(target_px)
    )

    # Keep the export between native Landsat resolution and the maximum coarse
    # resolution allowed by this plugin.
    scale = max(
        30.0,
        min(
            300.0,
            scale,
        ),
    )

    # Save to the operating system's temporary directory when no permanent
    # output path was provided.
    if out_path is None:
        out_path = os.path.join(
            tempfile.gettempdir(),
            filename,
        )

    # Ask Earth Engine to create a temporary URL for the requested GeoTIFF.
    url = image.getDownloadURL(
        {
            "region": region,
            "scale": scale,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )

    # Transfer the generated file from Earth Engine to the local path.
    urllib.request.urlretrieve(
        url,
        out_path,
    )

    # Retrieve the Earth Engine band names and write them into the local
    # GeoTIFF so QGIS displays meaningful names such as NDWI, NDVI, or LST.
    band_names = image.bandNames().getInfo()

    set_band_names(
        out_path,
        band_names,
    )

    return out_path, scale


def set_band_names(path, band_names):
    """Write readable band descriptions into a downloaded GeoTIFF.

    Earth Engine downloads may not preserve descriptive band names in a form
    QGIS displays clearly. This function opens the local file with GDAL update
    access, assigns each Earth Engine name to the corresponding raster band,
    and removes RGB interpretations from analytical data bands.

    Args:
        path:
            Local path to the downloaded GeoTIFF.

        band_names:
            Ordered sequence of Earth Engine band names.
    """
    # Imported only when a downloaded file needs to be edited, avoiding an
    # unnecessary GDAL import during Earth Engine-only operations.
    from osgeo import gdal

    dataset = gdal.Open(
        path,
        gdal.GA_Update,
    )

    if dataset is None:
        raise IOError(
            "Could not open %s for editing."
            % path
        )

    # GDAL numbers raster bands starting at 1 rather than 0, so enumeration
    # must also begin at 1.
    for band_number, band_name in enumerate(
        band_names,
        start=1,
    ):
        # Ignore extra names if Earth Engine returns more names than the local
        # raster contains, rather than attempting to access an invalid band.
        if band_number <= dataset.RasterCount:
            band = dataset.GetRasterBand(
                band_number
            )

            # Store a readable label such as NDWI, NDVI, NDMI, LST, or ET.
            band.SetDescription(
                band_name
            )

            # Mark the band as analytical data rather than a color channel.
            # Without this, GIS software may interpret a three-band index file
            # as red, green, and blue imagery.
            band.SetColorInterpretation(
                gdal.GCI_Undefined
            )

    # Write metadata changes to disk before releasing the dataset handle.
    dataset.FlushCache()
    dataset = None


# ---------------------------------------------------------------------------
# Monthly summer time series
# ---------------------------------------------------------------------------

def compute_monthly_timeseries(
    ee,
    extent,
    year,
):
    """Calculate monthly median NDWI, NDVI, and NDMI values for summer.

    Loads every eligible Landsat mission for the selected year, adds the three
    indices to each scene, and creates monthly median composites for June,
    July, August, and September. Each monthly image is then reduced to one
    AOI-wide median value per index.

    Args:
        ee:
            Initialized Earth Engine module returned by initialize().

        extent:
            QgsRectangle in EPSG:4326 defining the area of interest.

        year:
            Four-digit year supplied as either an integer or string.

    Returns:
        pandas.DataFrame with columns month, NDWI, NDVI, and NDMI, sorted in
        chronological month order.
    """
    # Imported only for the final local table conversion. All raster processing
    # remains server-side in Earth Engine.
    import pandas as pd

    # Accept either the combo box's string value or an integer supplied by
    # another caller.
    year = int(year)

    # Convert the QGIS AOI into an Earth Engine rectangle.
    ee_extent = ee.Geometry.Rectangle(
        [
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        ]
    )

    # Load the complete calendar year before grouping individual scenes by
    # acquisition month.
    start = "%d-01-01" % year
    end = "%d-01-01" % (year + 1)

    # Store one preprocessed collection for every eligible Landsat mission.
    parts = []

    for mission_key in ("L5", "L7", "L8", "L9"):
        collection = _load_collection(
            ee,
            mission_key,
            ee_extent,
            start,
            end,
        )

        if collection is not None:
            parts.append(collection)

    if not parts:
        raise ValueError(
            "No Landsat missions were active in %d."
            % year
        )

    # Begin with the first eligible mission, then merge each additional
    # mission into the same image collection.
    merged = parts[0]

    for collection in parts[1:]:
        merged = merged.merge(collection)

    def add_month(image):
        """Attach the acquisition month as an image property.

        system:time_start is stored as a timestamp by Earth Engine. Converting
        it to ee.Date allows the calendar month to be extracted and used for
        filtering later in the workflow.
        """
        month = ee.Date(
            image.get("system:time_start")
        ).get("month")

        return image.set(
            "month",
            month,
        )

    def add_indices(image):
        """Calculate and append NDWI, NDVI, and NDMI to one Landsat scene."""
        ndwi = image.normalizedDifference(
            [
                "GREEN",
                "NIR",
            ]
        ).rename("NDWI")

        ndvi = image.normalizedDifference(
            [
                "NIR",
                "RED",
            ]
        ).rename("NDVI")

        ndmi = image.normalizedDifference(
            [
                "NIR",
                "SWIR1",
            ]
        ).rename("NDMI")

        return image.addBands(
            [
                ndwi,
                ndvi,
                ndmi,
            ]
        )

    # Attach the month property and spectral-index bands to every scene before
    # constructing the monthly composites.
    merged = (
        merged
        .map(add_month)
        .map(add_indices)
    )

    # The plugin's time-series figure covers the same June–September summer
    # period used by the primary imagery workflow.
    summer_months = [
        6,
        7,
        8,
        9,
    ]

    def monthly_composite(month):
        """Create one pixel-wise median composite for a selected month."""
        month = ee.Number(month)

        month_images = merged.filter(
            ee.Filter.eq(
                "month",
                month,
            )
        )

        # Preserve the month on the composite so reduce_month can include it in
        # the exported feature table.
        return (
            month_images
            .median()
            .set(
                "month",
                month,
            )
        )

    monthly_images = ee.ImageCollection(
        ee.List(summer_months).map(
            monthly_composite
        )
    )

    def reduce_month(image):
        """Reduce one monthly composite to AOI-wide median index values."""
        stats = image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=ee_extent,
            scale=30,
            bestEffort=True,
        )

        return ee.Feature(
            None,
            {
                "month": image.get("month"),
                "NDWI": stats.get("NDWI"),
                "NDVI": stats.get("NDVI"),
                "NDMI": stats.get("NDMI"),
            },
        )

    # Execute the small four-row feature table and transfer it from Earth
    # Engine to local Python. Raster imagery remains server-side.
    results = monthly_images.map(
        reduce_month
    ).getInfo()

    # Convert Earth Engine feature dictionaries into ordinary Python rows that
    # pandas can assemble into a DataFrame.
    rows = []

    for feature in results.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        rows.append(
            {
                "month": int(
                    properties["month"]
                ),
                "NDWI": properties.get("NDWI"),
                "NDVI": properties.get("NDVI"),
                "NDMI": properties.get("NDMI"),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("month")
        .reset_index(drop=True)
    )
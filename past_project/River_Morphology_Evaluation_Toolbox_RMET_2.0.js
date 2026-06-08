/*****************************RMET 2.0***********************************
 * Script: RMET 2.0 (updated from RMET 1.0)
 * Original Authors: Margaret Mulhern, Kristen Dennis, Daniel Carver, Nathaniel Penrod, Tim Mayer
 * Modernized: 2024, Tim Mayer
 * Contact: develop.geoinformatics@gmail.com, dennisk.kristen@gmail.com
 *
 * CHANGES FROM RMET 1.0:
 *  - Fusion Tables removed (deprecated 2019). AOIs are now drawn geometries
 *    or user-supplied FeatureCollection asset paths.
 *  - Landsat collections updated from C01 T1_SR → C02 T2_L2.
 *  - Band names updated for C02 surface reflectance (SR_B2, SR_B3 … SR_B7
 *    for LS5/7; SR_B2…SR_B7 for LS8 with same naming). Thermal bands
 *    excluded — spectral indices only.
 *  - QA_PIXEL replaces pixel_qa; cloud/shadow bitmasks updated to C02 values.
 *  - Sentinel-2 collection updated to COPERNICUS/S2_SR_HARMONIZED.
 *  - LS8-to-LS7 harmonization coefficients applied to C02 SR_B* bands.
 *  - Scale factor applied to C02 SR values (multiply by 0.0000275, add -0.2).
 *
 * Usage:
 *  To use your own AOI, paste a GEE FeatureCollection asset path into the
 *  text box (e.g. "projects/my-project/assets/my_aoi") or draw a geometry
 *  directly on the map and select "Drawn geometry" from the AOI dropdown.
 *
 * Areas of interest that previously used Fusion Tables must be re-uploaded
 * as GEE assets (shapefiles → Assets → Upload) or replaced with map drawings.
 **************************************************************************/


///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////

var aoi, areaOfInterest, geometry,
    Fimageyr1, Fimageyr2, Fimageyr3,
    s2Imageyr4, s2Imageyr5, s2Imageyr6,
    yr1, yr2, yr3, yr4, yr5, yr6,
    point, annualIndexCollection;

///////////////////////////////////////////////////////////////////////////////
// VISUALIZATION PARAMETERS
///////////////////////////////////////////////////////////////////////////////

var VEG   = {opacity:1, palette:['2ddc0d'],          min:0,  max:1};
var WATER = {opacity:1, palette:['36abff'],           min:0,  max:1};
var MG    = {opacity:1, palette:['ffffff','085a16'],  min:0,  max:2};
var LG    = {opacity:1, palette:['ff7c0b','ffffff'],  min:-2, max:0};
var LW    = {opacity:1, palette:['ed982f','ffffff'],  min:-2, max:0};
var MW    = {opacity:1, palette:['ffffff','0bb0ff'],  min:0,  max:2};
var FLG   = {opacity:1, palette:['ffffff','ff7c0b'],  min:0,  max:2};
var FLW   = {opacity:1, palette:['ffffff','ed982f'],  min:0,  max:2};
var visNDVI  = {min:-1, max:1, palette:['blue','white','green']};
var visMNDWI = {min:-1, max:1, palette:['yellow','white','blue']};
var sites    = {opacity:1, palette:['ff7c0b']};

///////////////////////////////////////////////////////////////////////////////
// DRAWN GEOMETRY PLACEHOLDER
// Draw your AOI on the map using the geometry tools, then it appears here
// automatically as `geometry`. Alternatively supply an asset path in the
// text box. The dropdown entry "Drawn geometry" uses `Map.drawingTools()`.
///////////////////////////////////////////////////////////////////////////////

// Placeholder — replaced at runtime by the drawing tool or a feature collection.
var drawnGeometry = /* color: #d63000 */ ee.Geometry.Point([0, 0]);

///////////////////////////////////////////////////////////////////////////////
// AOI LABELS
///////////////////////////////////////////////////////////////////////////////

var DRAWN = 'Drawn geometry (draw on map)';
var OWN   = 'Your asset (paste path below)';

///////////////////////////////////////////////////////////////////////////////
// HELPER: C02 SCALE FACTOR
// Landsat C02 L2 SR bands require: reflectance = pixel * 0.0000275 + (-0.2)
// Values outside valid range (7273–43636) should be masked before scaling.
///////////////////////////////////////////////////////////////////////////////

var applyScaleFactorsLS = function(image) {
  var srBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  return image.addBands(srBands, null, true);
};

///////////////////////////////////////////////////////////////////////////////
// HELPER: C02 CLOUD / SHADOW MASK  (QA_PIXEL bitmask)
// Bit 1 = Dilated cloud, Bit 2 = Cirrus, Bit 3 = Cloud,
// Bit 4 = Cloud shadow, Bit 5 = Snow
///////////////////////////////////////////////////////////////////////////////

var maskLS = function(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)  // dilated cloud
               .and(qa.bitwiseAnd(1 << 3).eq(0))  // cloud
               .and(qa.bitwiseAnd(1 << 4).eq(0))  // cloud shadow
               .and(qa.bitwiseAnd(1 << 5).eq(0));  // snow
  return image.updateMask(mask);
};

///////////////////////////////////////////////////////////////////////////////
// SPECTRAL INDEX FUNCTIONS — LANDSAT 5 (C02: SR_B1–SR_B7)
// Band mapping: B1=SR_B1, B2=SR_B2, B3=SR_B3, B4=SR_B4, B5=SR_B5, B7=SR_B7
///////////////////////////////////////////////////////////////////////////////

var NDVI5  = function(img) { return img.normalizedDifference(['SR_B4','SR_B3']).rename('NDVI'); };
var NDWI5  = function(img) { return img.normalizedDifference(['SR_B2','SR_B4']).rename('NDWI'); };
var MNDWI5 = function(img) { return img.normalizedDifference(['SR_B2','SR_B5']).rename('MNDWI'); };
var SAVI5  = function(img) {
  return img.expression('((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
    NIR: img.select('SR_B4'), RED: img.select('SR_B3')
  }).rename('SAVI');
};
var NDMI5  = function(img) { return img.normalizedDifference(['SR_B4','SR_B5']).rename('NDMI'); };

///////////////////////////////////////////////////////////////////////////////
// SPECTRAL INDEX FUNCTIONS — LANDSAT 7 (C02: same band naming as LS5)
///////////////////////////////////////////////////////////////////////////////

var NDVI7  = function(img) { return img.normalizedDifference(['SR_B4','SR_B3']).rename('NDVI'); };
var NDWI7  = function(img) { return img.normalizedDifference(['SR_B2','SR_B4']).rename('NDWI'); };
var MNDWI7 = function(img) { return img.normalizedDifference(['SR_B2','SR_B5']).rename('MNDWI'); };
var SAVI7  = function(img) {
  return img.expression('((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
    NIR: img.select('SR_B4'), RED: img.select('SR_B3')
  }).rename('SAVI');
};
var NDMI7  = function(img) { return img.normalizedDifference(['SR_B4','SR_B5']).rename('NDMI'); };

///////////////////////////////////////////////////////////////////////////////
// SPECTRAL INDEX FUNCTIONS — LANDSAT 8/9 (C02: SR_B2–SR_B7; NIR=SR_B5, SWIR=SR_B6)
///////////////////////////////////////////////////////////////////////////////

var NDVI8  = function(img) { return img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI'); };
var NDWI8  = function(img) { return img.normalizedDifference(['SR_B3','SR_B5']).rename('NDWI'); };
var MNDWI8 = function(img) { return img.normalizedDifference(['SR_B3','SR_B6']).rename('MNDWI'); };
var SAVI8  = function(img) {
  return img.expression('((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
    NIR: img.select('SR_B5'), RED: img.select('SR_B4')
  }).rename('SAVI');
};
var NDMI8  = function(img) { return img.normalizedDifference(['SR_B5','SR_B6']).rename('NDMI'); };

///////////////////////////////////////////////////////////////////////////////
// SPECTRAL INDEX FUNCTIONS — SENTINEL-2 SR HARMONIZED
// NIR=B8, RED=B4, GREEN=B3, SWIR1=B11
///////////////////////////////////////////////////////////////////////////////

var NDVIS  = function(img) { return img.normalizedDifference(['B8','B4']).rename('NDVI'); };
var NDWIS  = function(img) { return img.normalizedDifference(['B3','B11']).rename('NDWI'); };
var SAVIS  = function(img) {
  return img.expression('((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
    NIR: img.select('B8'), RED: img.select('B4')
  }).rename('SAVI');
};

///////////////////////////////////////////////////////////////////////////////
// SENTINEL-2 CLOUD MASK (SCL band — Scene Classification Layer)
// Class 3 = cloud shadow, 8 = medium-prob cloud, 9 = high-prob cloud, 10 = cirrus
///////////////////////////////////////////////////////////////////////////////

var maskS2 = function(image) {
  var scl  = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(mask).divide(10000);
};

///////////////////////////////////////////////////////////////////////////////
// DICTIONARY BUILDERS
///////////////////////////////////////////////////////////////////////////////

/**
 * constructLSDict — builds a dict of per-year median composites using
 * Landsat C02 T2 L2 collections (LS5, LS7, LS8/9).
 */
var constructLSDict = function(geom) {
  var startMonth = '-04-05';
  var endMonth   = '-09-30';

  // Landsat 8 + 9: 2013–present
  var y8  = ee.List.sequence(2013, 2023);
  var ys8 = y8.map(function(y){ return ee.Number(y).format('%1d'); });
  var ls8 = y8.map(function(y){
    return ee.ImageCollection('LANDSAT/LC08/C02/T2_L2')
      .filterDate(ee.String(ee.Number(y).format('%1d')).cat(startMonth),
                  ee.String(ee.Number(y).format('%1d')).cat(endMonth))
      .filterBounds(geom)
      .filter(ee.Filter.lt('CLOUD_COVER', 30))
      .map(maskLS)
      .map(applyScaleFactorsLS)
      .median();
  });
  var ls8_dict = ee.Dictionary.fromLists(ys8, ls8);

  // Landsat 7: 2012
  var y7  = ee.List.sequence(2012, 2012);
  var ys7 = y7.map(function(y){ return ee.Number(y).format('%1d'); });
  var ls7 = y7.map(function(y){
    return ee.ImageCollection('LANDSAT/LE07/C02/T2_L2')
      .filterDate(ee.String(ee.Number(y).format('%1d')).cat(startMonth),
                  ee.String(ee.Number(y).format('%1d')).cat(endMonth))
      .filterBounds(geom)
      .filter(ee.Filter.lt('CLOUD_COVER', 30))
      .map(maskLS)
      .map(applyScaleFactorsLS)
      .median();
  });
  var ls7_dict = ee.Dictionary.fromLists(ys7, ls7);

  // Landsat 5: 1984–2011
  var y5  = ee.List.sequence(1984, 2011);
  var ys5 = y5.map(function(y){ return ee.Number(y).format('%1d'); });
  var ls5 = y5.map(function(y){
    return ee.ImageCollection('LANDSAT/LT05/C02/T2_L2')
      .filterDate(ee.String(ee.Number(y).format('%1d')).cat(startMonth),
                  ee.String(ee.Number(y).format('%1d')).cat(endMonth))
      .filterBounds(geom)
      .filter(ee.Filter.lt('CLOUD_COVER', 30))
      .map(maskLS)
      .map(applyScaleFactorsLS)
      .median();
  });
  var ls5_dict = ee.Dictionary.fromLists(ys5, ls5);

  return ls8_dict.combine(ls7_dict).combine(ls5_dict);
};

/**
 * constructS2Dict — builds a per-year median composite dict using
 * COPERNICUS/S2_SR_HARMONIZED (replaces deprecated COPERNICUS/S2).
 */
var constructS2Dict = function(geom) {
  var startMonth = '-04-05';
  var endMonth   = '-09-30';
  var y_list  = ee.List.sequence(2017, 2023);  // S2 SR available from 2017
  var ys_list = y_list.map(function(y){ return ee.Number(y).format('%1d'); });
  var s2 = y_list.map(function(y){
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate(ee.String(ee.Number(y).format('%1d')).cat(startMonth),
                  ee.String(ee.Number(y).format('%1d')).cat(endMonth))
      .filterBounds(geom)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(maskS2)
      .median();
  });
  return ee.Dictionary.fromLists(ys_list, s2);
};

///////////////////////////////////////////////////////////////////////////////
// BAND SELECTION + INDEX ADDITION
///////////////////////////////////////////////////////////////////////////////

/**
 * selectYearAddBands — pulls a year from the LS dictionary, clips to
 * geometry, and adds spectral index bands.  Sensor is determined by year.
 */
var selectYearAddBands = function(dictionary, year, geom) {
  var imageYear = ee.Image(dictionary.get(year)).clip(geom);
  if (year >= 2013) {
    return imageYear
      .addBands(NDVI8(imageYear))
      .addBands(NDWI8(imageYear))
      .addBands(MNDWI8(imageYear))
      .addBands(SAVI8(imageYear))
      .addBands(NDMI8(imageYear));
  } else if (year === 2012) {
    return imageYear
      .addBands(NDVI7(imageYear))
      .addBands(NDWI7(imageYear))
      .addBands(MNDWI7(imageYear))
      .addBands(SAVI7(imageYear))
      .addBands(NDMI7(imageYear));
  } else {
    return imageYear
      .addBands(NDVI5(imageYear))
      .addBands(NDWI5(imageYear))
      .addBands(MNDWI5(imageYear))
      .addBands(SAVI5(imageYear))
      .addBands(NDMI5(imageYear));
  }
};

/**
 * selectYearAddBandsS2 — pulls a year from the S2 dictionary and adds indices.
 */
var selectYearAddBandsS2 = function(dictionary, year, geom) {
  var imageYear = ee.Image(dictionary.get(year)).clip(geom);
  return imageYear
    .addBands(NDVIS(imageYear))
    .addBands(NDWIS(imageYear))
    .addBands(SAVIS(imageYear));
};

///////////////////////////////////////////////////////////////////////////////
// APPLY FILTER (main data-loading function, called by each map button)
///////////////////////////////////////////////////////////////////////////////

function applyfilter() {

  // --- Set area of interest ---
  aoi = selectAoi.getValue();
  if (aoi === DRAWN) {
    // Use geometry drawn on the map via the drawing tools
    var drawn = Map.drawingTools().layers();
    if (drawn.length() > 0) {
      areaOfInterest = drawn.get(0).toGeometry();
    } else {
      print('No drawn geometry found. Draw a polygon on the map first.');
      return;
    }
  } else if (aoi === OWN) {
    var userPath = inputTextbox.getValue();
    areaOfInterest = ee.FeatureCollection(ee.String(userPath));
  } else {
    print('Select an AOI option.');
    return;
  }

  geometry = areaOfInterest;

  // --- Read year inputs ---
  yr1 = selectYr1.getValue();
  yr2 = selectYr2.getValue();
  yr3 = selectYr3.getValue();
  yr4 = selectYr4.getValue();
  yr5 = selectYr5.getValue();
  yr6 = selectYr6.getValue();

  print('Year 1 (LS binary):', yr1);
  print('Year 2 (LS change start):', yr2);
  print('Year 3 (LS change end):', yr3);
  print('Year 4 (S2 binary):', yr4);
  print('Year 5 (S2 change start):', yr5);
  print('Year 6 (S2 change end):', yr6);

  // --- Build Landsat dictionary and select images ---
  var dict_ls = constructLSDict(geometry);
  Fimageyr1 = selectYearAddBands(dict_ls, yr1, geometry);
  Fimageyr2 = selectYearAddBands(dict_ls, yr2, geometry);
  Fimageyr3 = selectYearAddBands(dict_ls, yr3, geometry);

  // --- Build Sentinel-2 dictionary and select images ---
  var sen2_dict = constructS2Dict(geometry);
  s2Imageyr4 = selectYearAddBandsS2(sen2_dict, yr4, geometry);
  s2Imageyr5 = selectYearAddBandsS2(sen2_dict, yr5, geometry);
  s2Imageyr6 = selectYearAddBandsS2(sen2_dict, yr6, geometry);

  print('Landsat image (yr1):', Fimageyr1);
  print('Sentinel-2 image (yr4):', s2Imageyr4);
}

///////////////////////////////////////////////////////////////////////////////
// BINARY MAP FUNCTIONS — LANDSAT
///////////////////////////////////////////////////////////////////////////////

function NDVIBIN() {
  applyfilter();
  var Nimage = Fimageyr1.expression('NDVI >= thresh', {
    NDVI:   Fimageyr1.select('NDVI').clip(geometry),
    thresh: NDVIslider.getValue()
  });
  Map.centerObject(geometry);
  Map.addLayer(Nimage.updateMask(Nimage.gt(0)), VEG, 'NDVI Landsat Binary (' + yr1 + ')');
}

function MNDWIBIN() {
  applyfilter();
  var MNimage = Fimageyr1.expression('MNDWI >= thresh', {
    MNDWI:  Fimageyr1.select('MNDWI').clip(geometry),
    thresh: MNDWIslider.getValue()
  });
  Map.centerObject(geometry);
  Map.addLayer(MNimage.updateMask(MNimage.gt(0)), WATER, 'MNDWI Landsat Binary (' + yr1 + ')');
}

function SAVIBIN() {
  applyfilter();
  var SAimage = Fimageyr1.expression('SAVI >= thresh', {
    SAVI:   Fimageyr1.select('SAVI').clip(geometry),
    thresh: SAVIslider.getValue()
  });
  Map.centerObject(geometry);
  Map.addLayer(SAimage.updateMask(SAimage.gt(0)), VEG, 'SAVI Landsat Binary (' + yr1 + ')');
}

///////////////////////////////////////////////////////////////////////////////
// BINARY MAP FUNCTIONS — SENTINEL-2
///////////////////////////////////////////////////////////////////////////////

function SNDVIBIN() {
  applyfilter();
  var SNimage = s2Imageyr4.expression('NDVI >= thresh', {
    NDVI:   s2Imageyr4.select('NDVI').clip(geometry),
    thresh: SNDVIslider.getValue()
  });
  Map.centerObject(geometry);
  Map.addLayer(SNimage.updateMask(SNimage.gt(0)), VEG, 'NDVI Sentinel-2 Binary (' + yr4 + ')');
}

function SNDWIBIN() {
  applyfilter();
  var SMNimage = s2Imageyr4.expression('NDWI >= thresh', {
    NDWI:   s2Imageyr4.select('NDWI').clip(geometry),
    thresh: SNDWIslider.getValue()
  });
  Map.centerObject(geometry);
  Map.addLayer(SMNimage.updateMask(SMNimage.gt(0)), WATER, 'NDWI Sentinel-2 Binary (' + yr4 + ')');
}

///////////////////////////////////////////////////////////////////////////////
// CHANGE MAP FUNCTIONS — LANDSAT
///////////////////////////////////////////////////////////////////////////////

function GREENER() {
  applyfilter();
  var d1 = Fimageyr2.select('NDVI');
  var d2 = Fimageyr3.select('NDVI');
  var final = d2.gt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, MG, 'More green (' + yr2 + ')–(' + yr3 + ')');
  Map.addLayer(d1, visNDVI, 'NDVI (' + yr2 + ')', false);
  Map.addLayer(d2, visNDVI, 'NDVI (' + yr3 + ')', false);
}

function LESSGREEN() {
  applyfilter();
  var d1 = Fimageyr2.select('NDVI');
  var d2 = Fimageyr3.select('NDVI');
  var final = d2.lt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, LG, 'Less green (' + yr2 + ')–(' + yr3 + ')');
  Map.addLayer(d1, visNDVI, 'NDVI (' + yr2 + ')', false);
  Map.addLayer(d2, visNDVI, 'NDVI (' + yr3 + ')', false);
}

function WETTER() {
  applyfilter();
  var d1 = Fimageyr2.select('MNDWI');
  var d2 = Fimageyr3.select('MNDWI');
  var final = d2.gt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, MW, 'More wet (' + yr2 + ')–(' + yr3 + ')');
  Map.addLayer(d1, visMNDWI, 'MNDWI (' + yr2 + ')', false);
  Map.addLayer(d2, visMNDWI, 'MNDWI (' + yr3 + ')', false);
}

function LESSWET() {
  applyfilter();
  var d1 = Fimageyr2.select('MNDWI');
  var d2 = Fimageyr3.select('MNDWI');
  var final = d2.lt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, LW, 'Less wet (' + yr2 + ')–(' + yr3 + ')');
  Map.addLayer(d1, visMNDWI, 'MNDWI (' + yr2 + ')', false);
  Map.addLayer(d2, visMNDWI, 'MNDWI (' + yr3 + ')', false);
}

///////////////////////////////////////////////////////////////////////////////
// CHANGE MAP FUNCTIONS — SENTINEL-2
///////////////////////////////////////////////////////////////////////////////

function SGREENER() {
  applyfilter();
  var d1 = s2Imageyr5.select('NDVI');
  var d2 = s2Imageyr6.select('NDVI');
  var final = d2.gt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, MG, 'S2 more green (' + yr5 + ')–(' + yr6 + ')');
  Map.addLayer(d1, visNDVI, 'S2 NDVI (' + yr5 + ')', false);
  Map.addLayer(d2, visNDVI, 'S2 NDVI (' + yr6 + ')', false);
}

function SLESSGREEN() {
  applyfilter();
  var d1 = s2Imageyr5.select('NDVI');
  var d2 = s2Imageyr6.select('NDVI');
  var final = d2.lt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, LG, 'S2 less green (' + yr5 + ')–(' + yr6 + ')');
  Map.addLayer(d1, visNDVI, 'S2 NDVI (' + yr5 + ')', false);
  Map.addLayer(d2, visNDVI, 'S2 NDVI (' + yr6 + ')', false);
}

function SWETTER() {
  applyfilter();
  var d1 = s2Imageyr5.select('NDWI');
  var d2 = s2Imageyr6.select('NDWI');
  var final = d2.gt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, MW, 'S2 more wet (' + yr5 + ')–(' + yr6 + ')');
  Map.addLayer(d1, visMNDWI, 'S2 NDWI (' + yr5 + ')', false);
  Map.addLayer(d2, visMNDWI, 'S2 NDWI (' + yr6 + ')', false);
}

function SLESSWET() {
  applyfilter();
  var d1 = s2Imageyr5.select('NDWI');
  var d2 = s2Imageyr6.select('NDWI');
  var final = d2.lt(d1).multiply(d2.subtract(d1)).clip(geometry);
  Map.centerObject(geometry);
  Map.addLayer(final, LW, 'S2 less wet (' + yr5 + ')–(' + yr6 + ')');
  Map.addLayer(d1, visMNDWI, 'S2 NDWI (' + yr5 + ')', false);
  Map.addLayer(d2, visMNDWI, 'S2 NDWI (' + yr6 + ')', false);
}

///////////////////////////////////////////////////////////////////////////////
// MAP RESET
///////////////////////////////////////////////////////////////////////////////

function reset() { Map.clear(); }

///////////////////////////////////////////////////////////////////////////////
// WIDGET PANEL
///////////////////////////////////////////////////////////////////////////////

var spacer = ui.Label('           ');
var panel  = ui.Panel({style: {width:'33.333%'}});
ui.root.insert(0, panel);

// Header
panel.add(ui.Label('RMET: River Morphology Evaluation Toolbox',
  {fontWeight:'bold', fontSize:'24px', margin:'10px 5px'}));
panel.add(ui.Label(
  'Use 30 m Landsat 5/7/8 (1984–2023) or 10 m Sentinel-2 (2017–2023) imagery ' +
  'to analyse changes in river geomorphology using spectral indices. ' +
  'NOTE: Fusion Tables are no longer supported. Draw your AOI on the map ' +
  'or supply a GEE FeatureCollection asset path.'));

// ---- AOI SELECTION ----
panel.add(ui.Label('Select area of interest',
  {fontSize:'18px', fontWeight:'bold'}));

var selectAoi = ui.Select({
  items: [DRAWN, OWN],
  placeholder: 'Select AOI method',
  value: DRAWN
});
panel.add(selectAoi);

panel.add(ui.Label(
  'Draw a polygon on the map, OR paste your FeatureCollection asset path:'));
var inputTextbox = ui.Textbox({
  style: {width:'250px'},
  placeholder: 'projects/my-project/assets/my_aoi'
});
panel.add(inputTextbox);

panel.add(ui.Label(
  'TIP: Upload your old study-area shapefiles via the Assets panel, then ' +
  'paste the asset path above.',
  {color:'gray', fontSize:'12px'}));

// ---- LANDSAT BINARY ----
panel.add(ui.Label('Select year for Landsat 30 m binary map',
  {fontSize:'18px', fontWeight:'bold'}));
panel.add(ui.Label('1984–2023 (uses LS5, LS7, or LS8/9)',
  {color:'gray', fontSize:'12px'}));

var selectYr1 = ui.Textbox({placeholder:'Year', value:'2020', style:{width:'100px'}});
panel.add(ui.Panel([selectYr1], ui.Panel.Layout.flow('horizontal')));

// NDVI slider + button
var NDVIslider = ui.Slider({min:-1, max:1, value:0.17, step:0.01});
var NDVIBINMAP = ui.Button('Landsat NDVI Map', NDVIBIN);
var colorboxA  = ui.Label({style:{backgroundColor:'#2ddc0d', padding:'15px', margin:'10px'}});
panel.add(ui.Panel([NDVIslider, NDVIBINMAP, colorboxA], ui.Panel.Layout.flow('horizontal')));

// MNDWI slider + button
var MNDWIslider = ui.Slider({min:-1, max:1, value:0.25, step:0.01});
var MNDWIBINMAP = ui.Button('Landsat MNDWI Map', MNDWIBIN);
var colorboxB   = ui.Label({style:{backgroundColor:'#36abff', padding:'15px', margin:'10px'}});
panel.add(ui.Panel([MNDWIslider, MNDWIBINMAP, colorboxB], ui.Panel.Layout.flow('horizontal')));

// SAVI slider + button
var SAVIslider = ui.Slider({min:-1, max:1, value:0.40, step:0.01});
var SAVIBINMAP = ui.Button('Landsat SAVI Map', SAVIBIN);
var colorboxC  = ui.Label({style:{backgroundColor:'#2ddc0d', padding:'15px', margin:'10px'}});
panel.add(ui.Panel([SAVIslider, SAVIBINMAP, colorboxC], ui.Panel.Layout.flow('horizontal')));

// ---- LANDSAT CHANGE ----

function makeLegend(vis, label1, label2) {
  var lon = ee.Image.pixelLonLat().select('longitude');
  var gradient = lon.multiply((vis.max - vis.min) / 100.0).add(vis.min);
  var thumb = ui.Thumbnail({
    image: gradient.visualize(vis),
    params: {bbox:'0,0,100,8', dimensions:'300x15'},
    style: {position:'bottom-center'}
  });
  var text = ui.Panel({
    widgets:[
      ui.Label(label1),
      ui.Label({style:{stretch:'horizontal'}}),
      ui.Label(label2)
    ],
    layout: ui.Panel.Layout.flow('horizontal'),
    style: {padding:'10px', stretch:'horizontal', fontSize:'12px', color:'gray'}
  });
  return ui.Panel({style:{position:'bottom-left'}}).add(text).add(thumb);
}

var MGLEGEND = makeLegend(MG,  'Low',  'High');
var LGLEGEND = makeLegend(FLG, 'Low',  'High');
var MWLEGEND = makeLegend(MW,  'Low',  'High');
var LWLEGEND = makeLegend(FLW, 'Low',  'High');

panel.add(ui.Label('Select years for Landsat 30 m change map',
  {fontSize:'18px', fontWeight:'bold'}));
panel.add(ui.Label('Start year (1984–)        End year (–2023)',
  {color:'gray', fontSize:'12px'}));

var selectYr2 = ui.Textbox({placeholder:'Start', value:'2000', style:{width:'100px'}});
var selectYr3 = ui.Textbox({placeholder:'End',   value:'2020', style:{width:'100px'}});
panel.add(ui.Panel([selectYr2, selectYr3], ui.Panel.Layout.flow('horizontal')));

panel.add(ui.Panel([ui.Button('More Green Trend Map', GREENER),  MGLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button('Less Green Trend Map', LESSGREEN),LGLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button('More Wet Trend Map',   WETTER),   MWLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button('Less Wet Trend Map',   LESSWET),  LWLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Button('Reset Map', reset));

// ---- SENTINEL-2 BINARY ----
panel.add(ui.Label('Select year for Sentinel-2 10 m binary map',
  {fontSize:'18px', fontWeight:'bold'}));
panel.add(ui.Label('2017–2023 (S2_SR_HARMONIZED)', {color:'gray', fontSize:'12px'}));

var selectYr4 = ui.Textbox({placeholder:'Year', value:'2022', style:{width:'100px'}});
panel.add(ui.Panel([selectYr4], ui.Panel.Layout.flow('horizontal')));

// S2 NDVI
var SNDVIslider = ui.Slider({min:-1, max:1, value:0.34, step:0.01});
var SNDVIBINMAP = ui.Button('Sentinel-2 NDVI Map', SNDVIBIN);
var colorboxD   = ui.Label({style:{backgroundColor:'#2ddc0d', padding:'15px', margin:'10px'}});
panel.add(ui.Panel([SNDVIslider, SNDVIBINMAP, colorboxD], ui.Panel.Layout.flow('horizontal')));

// S2 NDWI
var SNDWIslider = ui.Slider({min:-1, max:1, value:0.25, step:0.01});
var NDWIBINMAP  = ui.Button('Sentinel-2 NDWI Map', SNDWIBIN);
var colorboxE   = ui.Label({style:{backgroundColor:'#36abff', padding:'15px', margin:'10px'}});
panel.add(ui.Panel([SNDWIslider, NDWIBINMAP, colorboxE], ui.Panel.Layout.flow('horizontal')));

// ---- SENTINEL-2 CHANGE ----
var SMGLEGEND = makeLegend(MG,  'Low', 'High');
var SLGLEGEND = makeLegend(FLG, 'Low', 'High');
var SMWLEGEND = makeLegend(MW,  'Low', 'High');
var SLWLEGEND = makeLegend(FLW, 'Low', 'High');

panel.add(ui.Label('Select years for Sentinel-2 10 m change map',
  {fontSize:'18px', fontWeight:'bold'}));
panel.add(ui.Label('Start year (2017–)        End year (–2023)',
  {color:'gray', fontSize:'12px'}));

var selectYr5 = ui.Textbox({placeholder:'Start', value:'2018', style:{width:'100px'}});
var selectYr6 = ui.Textbox({placeholder:'End',   value:'2022', style:{width:'100px'}});
panel.add(ui.Panel([selectYr5, selectYr6], ui.Panel.Layout.flow('horizontal')));

panel.add(ui.Panel([ui.Button('More Green Trend Map', SGREENER),  SMGLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button('Less Green Trend Map', SLESSGREEN),SLGLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button('More Wet Trend Map',   SWETTER),   SMWLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button('Less Wet Trend Map',   SLESSWET),  SLWLEGEND], ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Button('Reset Map', reset));

///////////////////////////////////////////////////////////////////////////////
// CHANGE INSPECTOR — click a point to get a 1985–2022 NDVI time series
///////////////////////////////////////////////////////////////////////////////

panel.add(ui.Label('Change Inspector',
  {fontSize:'18px', fontWeight:'bold'}));
panel.add(ui.Label('Click a point on the map to plot an NDVI time series.'));

var lon_label = ui.Label();
var lat_label = ui.Label();
panel.add(ui.Panel([lon_label, lat_label], ui.Panel.Layout.flow('horizontal')));

Map.onClick(function(coords) {
  lon_label.setValue('lon: ' + coords.lon.toFixed(4));
  lat_label.setValue('lat: ' + coords.lat.toFixed(4));

  point = ee.Geometry.Point(coords.lon, coords.lat);
  Map.layers().set(4, ui.Map.Layer(point, {color:'FFFFFF'}));

  // ---- Time series collection (1985–2022) ----
  var startYear = 1985;
  var endYear   = 2022;
  var startDay  = '04-01';
  var endDay    = '09-01';

  // Dummy collection for years with no imagery
  var dummyCollection = ee.ImageCollection([ee.Image([0,0,0,0,0,0]).mask(ee.Image(0))]);

  // C02-aware cloud mask (same as maskLS above but inline for this closure)
  var maskC02 = function(img) {
    var qa   = img.select('QA_PIXEL');
    var mask = qa.bitwiseAnd(1 << 3).eq(0)
                 .and(qa.bitwiseAnd(1 << 4).eq(0));
    return img.updateMask(mask);
  };

  var applyScale = function(img) {
    return img.addBands(img.select('SR_B.').multiply(0.0000275).add(-0.2), null, true);
  };

  // Harmonize LS8/9 C02 SR bands to LS7-equivalent
  // Roy et al. 2016 coefficients — updated for C02 naming convention
  var harmonizeC02 = function(img) {
    var slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]);
    var itcp   = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]);
    // Rename LS8 C02 bands to LS7-style for harmonization
    var y = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
                       ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'])
              .resample('bicubic')
              .subtract(itcp).divide(slopes)
              .set('system:time_start', img.get('system:time_start'));
    return y.toFloat();
  };

  var getSRcollection = function(year, sensor) {
    var coll = ee.ImageCollection('LANDSAT/' + sensor + '/C02/T2_L2')
      .filterBounds(point)
      .filterDate(year + '-' + startDay, year + '-' + endDay)
      .map(maskC02)
      .map(applyScale);
    if (sensor === 'LC08' || sensor === 'LC09') {
      coll = coll.map(harmonizeC02);
    } else {
      coll = coll.map(function(img) {
        return img.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'])
                  .set('system:time_start', img.get('system:time_start'));
      });
    }
    return coll;
  };

  var getCombinedSR = function(year) {
    var lt5 = getSRcollection(year, 'LT05');
    var le7 = getSRcollection(year, 'LE07');
    var lc8 = getSRcollection(year, 'LC08');
    var lc9 = getSRcollection(year, 'LC09');
    return lt5.merge(le7).merge(lc8).merge(lc9);
  };

  var medoidMosaic = function(collection) {
    var count = collection.toList(1).length();
    var finalColl = ee.ImageCollection(
      ee.Algorithms.If(count.gt(0), collection, dummyCollection));
    var median   = finalColl.max();
    var difFromMedian = finalColl.map(function(img) {
      var diff = ee.Image(img).subtract(median).pow(ee.Image.constant(2));
      return diff.reduce('sum').addBands(img);
    });
    return ee.ImageCollection(difFromMedian)
             .reduce(ee.Reducer.min(7))
             .select([1,2,3,4,5,6], ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']);
  };

  var buildAnnualCollection = function() {
    var imgs = [];
    for (var i = startYear; i <= endYear; i++) {
      var coll  = getCombinedSR(i);
      var mosaic = medoidMosaic(coll)
        .set('system:time_start', (new Date(i, 8, 1)).valueOf());
      imgs = imgs.concat(mosaic);
    }
    return ee.ImageCollection(imgs);
  };

  var annualSR = buildAnnualCollection();

  // NDVI from harmonized SR (band 4 = NIR, band 3 = RED in LS7-style naming)
  var distDir = -1;
  annualIndexCollection = annualSR.map(function(img) {
    var ndvi = img.normalizedDifference(['SR_B4', 'SR_B3'])
                  .multiply(1)
                  .select([0], ['NDVI'])
                  .set('system:time_start', img.get('system:time_start'));
    return ndvi.multiply(distDir)
               .set('system:time_start', img.get('system:time_start'));
  });

  var NDVIchart = ui.Chart.image.series(annualIndexCollection, point, ee.Reducer.mean(), 30);
  NDVIchart.setOptions({
    title: 'NDVI time series (1985–2022)',
    vAxis: {title:'NDVI', minValue:-1, maxValue:1},
    hAxis: {title:'Year'}
  });
  panel.add(NDVIchart);
});
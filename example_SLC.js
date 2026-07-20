//======================================================
// T. Mayer 7/20/36 example SLC 
//======================================================
// // Region of interest
// var roi = geometry;

// Region of Interest (ROI)
var roi = ee.Geometry.Polygon([
  [
    [-120.51193638520617, 42.384905429332335],
    [-119.89670201020617, 42.384905429332335],
    [-119.89670201020617, 42.82157808688595],
    [-120.51193638520617, 42.82157808688595],
    [-120.51193638520617, 42.384905429332335]
  ]
]);

Map.centerObject(roi, 10);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// Primary Landsat 7 image (SLC-off)
var primary = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
  .filterBounds(roi)
  .filterDate('2012-06-01', '2012-06-30')
  .median();

Map.centerObject(roi, 10);
Map.addLayer(primary, {bands:['SR_B3','SR_B2','SR_B1'], min:7000, max:18000}, 'Original');




//======================================================
// Landsat 7 SLC-off 3-Month Composite Gap Filling
//======================================================

// Center month
var start = ee.Date('2012-06-01');
var end   = start.advance(3, 'month');

//------------------------------------------------------
// Load Landsat 7 Collection 2 Level 2
//------------------------------------------------------
var collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterBounds(roi)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUD_COVER', 60));

//------------------------------------------------------
// Apply Scale Factors
//------------------------------------------------------
function applyScaleFactors(image) {

  var optical = image.select('SR_B.*')
      .multiply(0.0000275)
      .add(-0.2);

  var thermal = image.select('ST_B6')
      .multiply(0.00341802)
      .add(149.0);

  return image
      .addBands(optical, null, true)
      .addBands(thermal, null, true);
}

//------------------------------------------------------
// Cloud Mask
//------------------------------------------------------
function maskL7(image){

  var qa = image.select('QA_PIXEL');

  var mask = qa.bitwiseAnd(1 << 3).eq(0)   // Cloud
      .and(qa.bitwiseAnd(1 << 4).eq(0))    // Shadow
      .and(qa.bitwiseAnd(1 << 5).eq(0))    // Snow
      .and(qa.bitwiseAnd(1 << 1).eq(0));   // Dilated cloud

  return image.updateMask(mask);
}

//------------------------------------------------------
// Prepare Collection
//------------------------------------------------------
collection = collection
    .map(applyScaleFactors)
    .map(maskL7);

//------------------------------------------------------
// Composite
//------------------------------------------------------

// Median automatically fills SLC gaps where another
// image has valid observations.

var composite = collection
    .select([
      'SR_B1',
      'SR_B2',
      'SR_B3',
      'SR_B4',
      'SR_B5',
      'SR_B7'
    ])
    .median()
    .clip(roi);

//------------------------------------------------------
// Display
//------------------------------------------------------
Map.centerObject(roi,10);

Map.addLayer(composite,{
    bands:['SR_B3','SR_B2','SR_B1'],
    min:0.02,
    max:0.30
},'3-Month Composite');

//------------------------------------------------------
// Count observations
//------------------------------------------------------

var obs = collection.select('SR_B3').count();

Map.addLayer(obs,{
    min:0,
    max:10,
    palette:['red','yellow','green']
},'Observation Count');

//------------------------------------------------------
// Export
//------------------------------------------------------

Export.image.toDrive({
  image: composite,
  description: 'L7_3Month_GapFilled',
  folder: 'GEE',
  region: roi,
  scale: 30,
  maxPixels: 1e13
});

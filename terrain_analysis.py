import argparse
import numpy as np # moved import up 
import geopandas as gpd # moved import up
import pandas as pd
def convert_to_rasterio(raster_data, template_raster):
    """
    Converts NumPy array into in-memory raster file 
    using settings from template raster. 
    
    This helps work with data like a normal .tif map 
    """
    
    import rasterio 
    from rasterio.io import MemoryFile 
    
    profile = template_raster.profile.copy()
    profile.update({ 
        "height": raster_data.shape[0], 
        "width": raster_data.shape[1],
        "count": 1, 
        "dtype": raster_data.dtype
    })
    
    memfile = MemoryFile()
    dataset = memfile.open(**profile) 
    dataset.write(raster_data, 1) 
    
    #Keep reference to memfile alive 
    dataset.memfile = memfile 
    
    return dataset 
    

def extract_values_from_raster(raster, shape_object):
    """Return raster values at the location of each point geometry."""
    
    band=raster.read(1)
    values = [] 
    rows, cols = band.shape
    
    for p in shape_object: 
        try: 
            row,col = raster.index(p.x, p.y) 
            if 0 <= row < rows and 0 <= col < cols: 
                values.append(band[row,col]) 
            else: 
                values.append(np.nan) 
                
        except: 
            values.append(np.nan) 
            
    return values 
        




def make_classifier(x, y, verbose=False):
    """ 
    Train Random Forest classifier on the input features and labels.
    
    Args: 
        x: DataFrame of feature columns 
        y: Series or list of labels (1 = landslide, 0 = not)
        verbose: If True, print training accuracy 
        
    Returns: 
        Trained RandomForestClassifier
    """

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x,y)

    if verbose: 
        print("Training accuracy:", clf.score(x,y)) 
    
    return clf 

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    
    """ 
    Create raster of landslide probabilities using classifier. 
    
    Args: 
        topo, geo, lc, dist_fault, slope: rasterio datset objects
        classifier: trained RandomForestClassifier 
        
    Returns: 
        2D NumPy arrary of predicted landslide probabilities (values 0 to 1)
    """ 
    
    
    # 1. Read rasters into 2D arrays 
    elev = topo.read(1) 
    geol = geo.read(1) 
    landcover = lc.read(1) 
    fault = dist_fault.read(1) 
    slope_data = slope.read(1) 
    
    # 2. Flatten arrays into 1D
    elev_flat = elev.flatten() 
    geol_flat = geol.flatten() 
    landcover_flat = landcover.flatten() 
    fault_flat = fault.flatten() 
    slope_flat = slope_data.flatten()
    
    # 3. Stack into 2D matrix 
    X = np.stack([elev_flat, fault_flat, slope_flat, landcover_flat, geol_flat], axis=1)
    
    # 4. Mask invalid data 
    mask = np.all (~np.isnan(X), axis=1) 
    
    # 5. Create ouput array and fill 
    prob_flat = np.full(elev_flat.shape, np.nan) 
    
    # 6. Predict only where data is valid 
    if np.any(mask): 
        probs = classifier.predict_proba(X[mask]) 
        prob_flat[mask] = probs[:,1] # Column 1 = probability of Landslide (class = 1)
    
    # 7. Reshape to 2D 
    prob_map = prob_flat.reshape(elev.shape)
    
    return prob_map 
    
def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    """ 
    Create Geographic DataFrame containing raster values at given 
        geometries, and label whether it's a landslide. 
        
        Args: 
            topo, geo, lc, dist_falt, slope: rasterio datasets
            shape: list of shapely Point geometries
            landslides: 1 for landslide points, 0 for non-landslide points
            
        Returns: 
            geopandas.GeoDataFrame with columns: elev, fault, slope, LC, Geol, ls
    """ 
    
    
    #Create dataframe with only expected test columns (exclude geometry)
        
    df = gpd.GeoDataFrame({ 
            "elev": extract_values_from_raster(topo, shape),
            "fault": extract_values_from_raster(dist_fault,shape), 
            "slope": extract_values_from_raster(slope,shape), 
            "LC": extract_values_from_raster(lc,shape),
            "Geol": extract_values_from_raster(geo,shape), 
            "ls":[landslides] * len(shape),
    }) 
    
    #Test expecting 6 columns therefore, attach geometry later or test always fails
     
    

    return df

def calculate_slope(dem_array, transform): 
    """Calculate slope using basic gradient estimation.""" 
    from scipy.ndimage import sobel 
    
    dzdx = sobel(dem_array, axis=1) 
    dzdy = sobel(dem_array, axis=0) 
    
    xres = transform.a 
    yres = -transform.e 
    
    slope = np.sqrt((dzdx / (8*xres))**2 + (dzdy / (8 * yres))**2) 
    
    return slope 

def main():
    import argparse 
    import rasterio 
    import geopandas as gpd 
    import numpy as np 


    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    
    parser.add_argument('--topography', 
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    
    # 1. Open raster files 
    topo = rasterio.open(args.topography) 
    geo = rasterio.open(args.geology) 
    lc = rasterio.open(args.landcover) 
    
    # 1b. Calculate slope and wrap as rasterio object
    
    elevation_array = topo.read(1) 
    slope_array = calculate_slope(elevation_array, topo.transform) 
    slope_raster = convert_to_rasterio(slope_array, topo) 
    
    # 2. Load fault shapefile and rasterize distance-from-fault
    from rasterio.features import rasterize 
    from proximity import proximity 
    
    faults = gpd.read_file(args.faults) 
    
    #Turned falt lines into raster grid 
    fault_array = rasterize( 
        [(geom, 1) for geom in faults.geometry], 
        out_shape=topo.shape, 
        transform=topo.transform,
        fill=0,
        dtype= 'uint8'
    ) 
    
    #Measure distance from pixel to nearest fault line
    dist_array = proximity(topo, fault_array, 1)
    dist_fault = convert_to_rasterio(proximity(topo, fault_array, 1), topo) 
    print("dist_fault type:", type(dist_fault))
    
    # 3. Load and Prepare landslide shapefile 
    landslides = gpd.read_file(args.landslides)
    
    #(bug fix) Turned each landslide into point using its centroid
    ls_geom = [geom.centroid for geom in landslides.geometry]
    
    # 4. Sample non-landslide (negative) points 
    import random 
    from shapely.geometry import Point 
    np.random.seed(42) 
    rows, cols = topo.read(1).shape 
    neg_geom = [] 
    attempts = 0 
    max_attempts = len(ls_geom) * 20 #Allow 20 tries 
    
    while len(neg_geom) < len(ls_geom) and attempts < max_attempts: 
        col = random.randint(0, cols -1) 
        row = random.randint(0, rows -1) 
        x,y = topo.xy(row,col) 
        point = Point(x,y) 
        
        
        if point not in ls_geom: 
            neg_geom.append(point) 
            
            
        attempts +=1 
        
    if len(neg_geom) < len(ls_geom): 
       print("Warning: Only generated", len(neg_geom), "negative points.")
        
        
            
        
    # 5. Create dataframes for landslide and non-landslide 
    df_landslides = create_dataframe(topo, geo, lc, dist_fault, slope_raster, ls_geom, 1) 
    df_negatives = create_dataframe(topo, geo, lc, dist_fault, slope_raster, neg_geom, 0)
    
    data = pd.concat([df_landslides, df_negatives], ignore_index=True)
    
    # 6. Train classifier 
    from sklearn.model_selection import train_test_split 
    
    x = data.drop(columns = ["ls"]) 
    y = data["ls"] 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = make_classifier(x_train, y_train, verbose = args.verbose)
    
    if args.verbose: 
        print("Model accuracy", clf.score(x_test, y_test)) 
    
    # 7. Predict probabilities raster    
    prob_map = make_prob_raster_data(topo, geo, lc, dist_fault, slope_raster, clf)
    
    # 8. Save to file 
    profile = topo.profile 
    profile.update(dtype=rasterio.float32, count=1) 
    
    with rasterio.open(args.output, 'w', **profile) as dst: 
        dst.write(prob_map.astype(np.float32), 1)


if __name__ == '__main__':
    main()

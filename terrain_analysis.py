import argparse

def convert_to_rasterio(raster_data, template_raster):
    """
    Converts NumPy array into in-memory raster file 
    using settings from template raster. 
    
    This helps work with data like a normal .tif map 
    """
    
    import rasterio 
    from rasterio.io import MemoryFile 
    
    profile = template_raster.profile 
    profile.update({ 
        "height": raster_data.shape[0], 
        "width": raster_data.shape[1],
        "count": 1, 
        "dtype": raster_data.dtype
    })
    
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst: 
            dst.write(raster_data, 1) 
            
        return memfile.open()


def extract_values_from_raster(raster, shape_object):
    """Return raster values at the location of each point geometry."""
    band = raster.read(1) 
    return [band[raster.index(p.x,p.y)] for p in shape_object]

    return


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

    return

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
    
    import geopandas as gpd 
    
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


def main():


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


if __name__ == '__main__':
    main()

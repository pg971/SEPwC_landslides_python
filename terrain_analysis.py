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

    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    return


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

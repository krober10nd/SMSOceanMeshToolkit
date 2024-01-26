import sys, logging

import SMSOceanMeshToolkit as smsom

import geopandas as gpd
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"

def test_signed_distance_functions_fully_enclosed_polygon(): 
    '''
    Test the signed distance function creation for a bounding polygon that completely encloses the shoreline polygons
    In this case the mainland polygon is equivalent to the largest inner polygon (by area) in the shoreline polygons  
    '''
    my_file = 'data/polys.shp'
    min_mesh_size = 1000.0 / 111e3
    c = gpd.read_file(my_file)
    bounds = c.bounds 
    bounds = (bounds.minx.min(), bounds.maxx.max(), bounds.miny.min(), bounds.maxy.max())
    print(bounds)

    shore_line = smsom.CoastalGeometry(my_file, bounds, min_mesh_size)
    sdf = smsom.signed_distance_function(shore_line)
    print(sdf)
    sdf.plot()
    plt.savefig('sdf.png')
    # create the raster
    #sdf.to_raster(out_path, cell_size)

    
def test_signed_distance_functions(): 
    bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)
    for minimum_mesh_size in (1000.0 / 111e3, 500.0 / 111e3, 100.0 / 111e3, 50.0 / 111e3):
        shoreline = smsom.CoastalGeometry(
            vector_data, bounding_box, minimum_mesh_size, crs="EPSG:4326"
        )
        print(shoreline)
        sdf = smsom.signed_distance_function(shoreline)
        sdf.plot()

def test_signed_distance_functions_from_gpkg(): 
    bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)
    for minimum_mesh_size in (1000.0 / 111e3, 500.0 / 111e3, 100.0 / 111e3, 50.0 / 111e3):
        shoreline = smsom.CoastalGeometry(
            vector_data, bounding_box, minimum_mesh_size, crs="EPSG:4326"
        )
        gdf=shoreline.to_geodataframe()
        print(gdf)
        sdf = smsom.signed_distance_function(gdf)
        ds = sdf.to_xarray(sdf)
        ds.to_netcdf("sdf.nc")
        ax=sdf.plot()
        shoreline.plot(ax=ax)

if __name__ == "__main__":
    # for debugging
    #test_signed_distance_functions()
    #test_signed_distance_functions_from_gpkg()
    test_signed_distance_functions_fully_enclosed_polygon()
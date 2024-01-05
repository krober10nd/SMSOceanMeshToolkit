import sys, logging
import matplotlib.pyplot as plt
import geopandas as gpd

import SMSOceanMeshToolkit as smsom

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"

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
    test_signed_distance_functions_from_gpkg()
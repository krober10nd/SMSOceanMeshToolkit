import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

import SMSOceanMeshToolkit as smsom
import logging
import sys
import geopandas as gpd

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"

def test_bounding_box_with_different_mesh_sizes():
    bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)
    for minimum_mesh_size in (1000.0 / 111e3, 500.0 / 111e3, 100.0 / 111e3, 50.0 / 111e3):
        shoreline = smsom.CoastalGeometry(
            vector_data, bounding_box, minimum_mesh_size, crs="EPSG:4326"
        )
        shoreline.plot()
        gdf = shoreline.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        plt.close()

def test_bounding_polygon_with_different_mesh_sizes():
    bounding_box = "data/my_test_bounding_polygon.shp"
    for minimum_mesh_size in (1000.0 / 111e3, 500.0 / 111e3, 100.0 / 111e3, 50.0 / 111e3):
        shoreline = smsom.CoastalGeometry(
            vector_data, bounding_box, minimum_mesh_size, crs="EPSG:4326"
        )
        shoreline.plot()
        gdf = shoreline.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        plt.close()

def test_bounding_polygon_with_moving_window_smoothing():
    bounding_box = "data/my_test_bounding_polygon.shp"
    for minimum_mesh_size in (1000.0 / 111e3, 500.0 / 111e3, 100.0 / 111e3, 50.0 / 111e3):
        shoreline = smsom.CoastalGeometry(
            vector_data,
            bounding_box,
            minimum_mesh_size,
            crs="EPSG:4326",
            smooth_shoreline=True,
            smoothing_approach="moving_window",
            smoothing_window=5,
        )
        shoreline.plot()
        gdf = shoreline.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        plt.close()

if __name__ == "__main__":
    test_bounding_box_with_different_mesh_sizes()
    #test_bounding_polygon_with_different_mesh_sizes()
    #test_bounding_polygon_with_moving_window_smoothing()
import SMSOceanMeshToolkit as smsom
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"

# Test 1: for tuple bounding box as tuple with 4 different minimum mesh sizes
# # area near Toledo, OH in geographic coordinates
# bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)

# for minimum_mesh_size in (1000.0/111e3, 500.0/111e3, 100.0/111e3, 50.0/111e3):
#     shoreline = smsom.CoastalGeometry(vector_data, bounding_box, minimum_mesh_size, crs='EPSG:4326')
#     # Check the repr
#     print(shoreline)
#     # Check the conversion to a geopandas dataframe
#     gdf = shoreline.to_geodataframe()
#     gdf.to_file(f"test1_{minimum_mesh_size}.gpkg", driver="GPKG")
#     # Read again but turn off shoreline smoothing & refinements to see difference
#     shoreline = smsom.CoastalGeometry(
#         vector_data,
#         bounding_box,
#         minimum_mesh_size,
#         smooth_shoreline=False,
#         refinements=0,
#         crs='EPSG:4326',
#     )
#     print(shoreline)
#     gdf = shoreline.to_geodataframe()
#     gdf.to_file(f"test1_wo_smooth_{minimum_mesh_size}_tuple_box.gpkg", driver="GPKG")

# Test 2: for bounding box defined by a polygon with 4 different minimum mesh sizes
bounding_box = "data/my_test_bounding_polygon.shp"

for minimum_mesh_size in (1000.0/111e3, 500.0/111e3, 100.0/111e3, 50.0/111e3):
    shoreline = smsom.CoastalGeometry(vector_data, bounding_box, minimum_mesh_size, crs='EPSG:4326')
    shoreline.plot()
    # Check the repr
    print(shoreline)
    # Check the conversion to a geopandas dataframe
    gdf = shoreline.to_geodataframe()
    gdf.to_file(f"test1_{minimum_mesh_size}_polygon_box.gpkg", driver="GPKG")

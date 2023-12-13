import SMSOceanMeshToolkit as smsom
import numpy as np

vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"

# Test 1: for tuple bounding box as tuple with 4 different minimum mesh sizes
# area near Toledo, OH in geographic coordinates
bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)
for minimum_mesh_size in (1000.0, 500.0, 100.0, 50.0):
    minimum_mesh_size /= 111e3  # rough meters per degree at equator
    shoreline = smsom.CoastalGeometry(vector_data, bounding_box, minimum_mesh_size)
    print(shoreline)
    gdf = shoreline.to_geodataframe()
    gdf.to_file(f"test1_{minimum_mesh_size}.gpkg", driver="GPKG")

    # Read again but turn off shoreline smoothing & refinements to see difference
    shoreline = smsom.CoastalGeometry(
        vector_data,
        bounding_box,
        minimum_mesh_size,
        smooth_shoreline=False,
        refinements=0,
    )
    print(shoreline)
    gdf = shoreline.to_geodataframe()
    gdf.to_file(f"test1_wo_smooth_{minimum_mesh_size}.gpkg", driver="GPKG")


# Test 2: for bounding box defined by a polygon with 4 different minimum mesh sizes

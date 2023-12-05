from SMSOceanMeshToolkit import CoastalBoundary 

vector_data = 'data/Lk_erie_Lk_st_clair_shoreline_polygons.shp'

# Test 1: for tuple bounding box as tuple with 4 different minimum mesh sizes
# area near Toledo, OH
bounding_box =  (-82.73483139,-82.57544578, 41.40461165,41.50774352)
for minimum_mesh_size in (1000, 500, 100, 50):
    shoreline = CoastalBoundary(vector_data, bounding_box, minimum_mesh_size)
    gdf = shoreline.to_geodataframe()
    print(gdf.head())

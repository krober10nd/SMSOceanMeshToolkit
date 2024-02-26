"""
Tests for mesh sizing functions
"""
import SMSOceanMeshToolkit as smsom
import matplotlib.pyplot as plt
import geopandas as gpd

def test_wave_sizing_function():
    # set up the region and grid
    # region =  
    # grid = 
    # read in the dem 
    # szfx = smsom.wavelength_sizing_function(grid, dem, wave_height, wave_period, water_depth)
    pass 

def test_feature_sizing_function():
    minimum_mesh_size = 100.0 / 111e3
    vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"
    bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)

    region = smsom.Region(bounding_box, crs="EPSG:4326")
    grid = smsom.Grid(region, dx=50.0 / 111e3)

    shoreline = smsom.CoastalGeometry(
        vector_data, bounding_box, minimum_mesh_size, crs="EPSG:4326"
    )
    szfx = smsom.feature_sizing_function(grid, shoreline, number_of_elements_per_width=3) 
    print(szfx)
    
    ax =szfx.plot(plot_colorbar=True, cbarlabel=f"Feature size ({grid.units})", holding=True)
    shoreline.plot(ax=ax)
    plt.savefig("feature_sizing_function.png")
    plt.show()
    
    
def test_distance_sizing_function():
    minimum_mesh_size = 100.0 / 111e3
    vector_data = "data/Lk_erie_Lk_st_clair_shoreline_polygons.shp"
    bounding_box = (-83.22137037, -82.5754578, 41.32025057, 41.89550398)

    region = smsom.Region(bounding_box, crs="EPSG:4326")
    grid = smsom.Grid(region, dx=50.0 / 111e3)

    shoreline = smsom.CoastalGeometry(
        vector_data, bounding_box, minimum_mesh_size, crs="EPSG:4326"
    )
    szfx  = smsom.distance_sizing_function(grid, shoreline, rate=0.10, max_edge_length=1000.0 / 111e3)
    ax,_=szfx.plot(plot_colorbar=True, cbarlabel=f"Distance from shoreline ({grid.units})", holding=True)
    shoreline.plot(ax=ax)
    plt.savefig("distance_sizing_function.png")

def test_distance_form_linestring():
    bbox = (-72.96163620, -72.89554781, 41.21484433, 41.24162214)
    linestrings = "data/my_linestrings.shp"
    region = smsom.Region(bbox, crs="EPSG:4326")
    grid = smsom.Grid(region, dx=50.0 / 111e3)
    min_edge_length = 10.0 / 111e3
    szfx = smsom.distance_sizing_from_linestring_function(
        grid, linestrings, min_edge_length
    )
    gdf = gpd.read_file(linestrings)
    ax, _ = szfx.plot(
        holding=True,
        plot_colorbar=True,
        cbarlabel=f"Distance from linestring ({grid.units})",
    )
    gdf.plot(ax=ax, color="red")
    plt.show()

def test_distance_from_points(): 
    bbox = (-72.96163620, -72.89554781, 41.21484433, 41.24162214)
    points = "data/my_points.shp"
    region = smsom.Region(bbox, crs="EPSG:4326")
    grid = smsom.Grid(region, dx=100.0 / 111e3)
    min_edge_length = 10.0 / 111e3
    szfx = smsom.distance_sizing_from_point_function(
        grid, points, min_edge_length, max_edge_length=100.0 / 111e3
    )
    gdf = gpd.read_file(points)
    fig, ax, _ = szfx.plot(
        holding=True,
        plot_colorbar=True,
        cbarlabel=f"Distance from points ({grid.units})",
    )
    # set the aspect ratio to 1
    ax.set_aspect('equal')
    gdf.plot(ax=ax, color="red")


if __name__ == "__main__":
    test_wave_sizing_function()
    #test_feature_sizing_function()
    #test_distance_sizing_function()
    #test_distance_form_linestring()
    #test_distance_from_points()

"""
Tests for mesh sizing functions
"""
import SMSOceanMeshToolkit as smsom
import matplotlib.pyplot as plt
import geopandas as gpd


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
    _, ax, _ = szfx.plot(
        holding=True,
        plot_colorbar=True,
        cbarlabel=f"Distance from linestring ({grid.units})",
    )
    gdf.plot(ax=ax, color="red")
    plt.show()


if __name__ == "__main__":
    test_distance_form_linestring()

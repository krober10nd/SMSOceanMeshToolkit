'''
Mesh sizing functions
'''
import logging

import geopandas as gpd
import numpy as np
from inpoly import inpoly2
from shapely.geometry import LineString

import skfmm # fast marching method
from .edges import get_poly_edges
from .Grid import Grid

logger = logging.getLogger(__name__)

__all__ = [
    "distance_sizing_function",
    "distance_sizing_from_point_function",
    "distance_sizing_from_linestring_function",
]

def _line_to_points_array(line):
    """Convert a shapely LineString to a numpy array of points"""
    return np.array(line.coords)

def _resample_line(row, min_edge_length):
    """Resample a line to a minimum mesh size length"""
    line = row["geometry"]
    resampled_points = []
    distance = 0
    while distance < line.length:
        resampled_points.append(line.interpolate(distance))
        distance += min_edge_length / 2
    resampled_line = LineString(resampled_points)
    row["geometry"] = resampled_line
    return row


def distance_sizing_from_linestring_function(
    grid,
    line_file,
    min_edge_length,
    rate=0.05,
    max_edge_length=np.inf,
):
    """
    Mesh sizes that vary linearly at `rate` from a LineString(s)

    Parameters
    ----------
    grid: class:`Grid`
        A grid object that will contain the distance sizing function
    line_file: str or Path
        Path to a vector file containing LineString(s)
    min_edge_length: float
        The minimum edge length desired in the mesh. Must be in the units 
        of the grid's crs
    rate: float
        The decimal mesh expansion rate from the line(s).
    max_edge_length: float, optional
        The maximum edge length of the mesh

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field populated with distance sizing
    """
    logger.info("Building a distance mesh sizing function from a LineString(s)...")
    # check a grid is provided
    assert isinstance(grid, Grid), "A grid object must be provided"
    line_geodataframe = gpd.read_file(line_file)
    # check all the geometries are linestrings
    assert all(
        line_geodataframe.geometry.geom_type == "LineString"
    ), "All geometries in line_file must be linestrings"
    # check the crs and reproject if necessary
    if line_geodataframe.crs != grid.crs:
        # add a logging message
        logger.info(
            f"Reprojecting the line geodataframe from {line_geodataframe.crs} to {grid.crs}"
        )
        line_geodataframe = line_geodataframe.to_crs(grid.crs)

    # Resample the spacing along the lines so that the minimum edge length is met
    line_geodataframe = line_geodataframe.apply(
        _resample_line, axis=1, min_edge_length=min_edge_length
    )
    # Get the coordinates of the linestrings from the geodataframe
    # Convert all the LineStrings in the dataframe to arrays of points
    points_list = [
        _line_to_points_array(line) for line in line_geodataframe["geometry"]
    ]
    points = np.concatenate(points_list)

    # create phi (-1 where point(s) intersect grid points -1 elsewhere 0)
    phi = np.ones(shape=(grid.nx, grid.ny))
    xg, yg = grid.create_grid()
    # find location of points on grid
    indices = grid.find_indices(points, xg, yg)
    phi[indices] = -1.0
    try:
        dis = np.abs(skfmm.distance(phi, [grid._dx, grid._dy]))
    except ValueError:
        logger.info("0-level set not found in domain or grid malformed")
        dis = np.zeros((grid.nx, grid.ny)) + 999.
    tmp = min_edge_length + dis * rate
    if max_edge_length is not np.inf:
        tmp[tmp > max_edge_length] = max_edge_length
    grid.values = np.ma.array(tmp)
    grid.build_interpolant()
    return grid

def distance_sizing_from_point_function(
    grid, 
    point_file,
    min_edge_length,
    rate=0.15,
    max_edge_length=np.inf,
):
    '''
    Mesh sizes that vary linearly at `rate` from a point or points
    contained within a dataframe.

     Parameters
    ----------
    grid: class:`Grid`
        A grid object that will contain the distance sizing function
    point_file: str or Path
        Path to a vector file containing point(s)
    min_edge_length: float
        The minimum edge length of the mesh
    rate: float
        The decimal percent mesh expansion rate from the point(s)

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field populated with distance sizing

    '''

    logger.info("Building a distance sizing from point(s) function...")
    assert isinstance(grid, Grid), "A grid object must be provided"
    point_geodataframe = gpd.read_file(point_file)
    assert all(
        point_geodataframe.geometry.geom_type == "Point"
    ), "All geometries must be points"
    if point_geodataframe.crs != grid.crs:
        # add a logging message
        logger.info(
            f"Reprojecting the point geodataframe from {point_geodataframe.crs} to {grid.crs}"
        )
        point_geodataframe = point_geodataframe.to_crs(grid.crs)

    # Get the coordinates of the points from the geodataframe
    points = np.array(point_geodataframe.geometry.apply(lambda x: (x.x, x.y)).tolist())
    # create phi (-1 where point(s) intersect grid points -1 elsewhere 0)
    phi = np.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    # find location of points on grid
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    try:
        dis = np.abs(skfmm.distance(phi, [grid._dx, grid._dy]))
    except ValueError:
        logger.info("0-level set not found in domain or grid malformed")
        dis = np.zeros((grid.nx, grid.ny)) + 999
    tmp = min_edge_length + dis * rate
    if max_edge_length is not None:
        tmp[tmp > max_edge_length] = max_edge_length
    grid.values = np.ma.array(tmp)
    grid.build_interpolant()
    return grid


def distance_sizing_function(
    grid, 
    coastal_geometry,
    rate=0.15,
    max_edge_length=np.inf,
):
    """
    Mesh sizes that vary linearly at `rate` from coordinates in `obj`:CoastalGeometry
    
    Parameters
    ----------
    grid: class:`Grid`
        A grid object that will contain the distance sizing function
    coastal_geometry: :class:`CoastalGeometry`
        Vector data processed
    rate: float, optional
        The rate of expansion in decimal percent from the shoreline.
    max_edge_length: float, optional
        The maximum allowable edge length

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value
    """
    logger.info("Building a distance mesh sizing function...")

    # create phi (-1 where coastal vector intersects grid points 1 elsewhere)
    phi = np.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = np.vstack((coastal_geometry.inner, coastal_geometry.mainland))
    # remove shoreline components outside the shoreline.boubox
    boubox = np.nan_to_num(coastal_geometry.region_polygon)  # remove nan for inpoly2
    e_box = get_poly_edges(coastal_geometry.region_polygon)
    mask = np.ones((grid.nx, grid.ny), dtype=bool)
    if len(points) > 0:
        try:
            in_boubox, _ = inpoly2(points, boubox, e_box)
            points = points[in_boubox]

            qpts = np.column_stack((lon.flatten(), lat.flatten()))
            in_boubox, _ = inpoly2(qpts, boubox, e_box)
            mask_indices = grid.find_indices(qpts[in_boubox, :], lon, lat)
            mask[mask_indices] = False
        except Exception as e:
            logger.error(e)
            ...

    # find location of points on grid
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    try:
        dis = np.abs(skfmm.distance(phi, [grid.dx, grid.dy]))
    except ValueError:
        logger.info("0-level set not found in domain or grid malformed")
        dis = np.zeros((grid.nx, grid.ny)) + 99999
    tmp = coastal_geometry.minimum_mesh_size + dis * rate
    if max_edge_length is not np.inf:
        tmp[tmp > max_edge_length] = max_edge_length
    grid.values = np.ma.array(tmp, mask=mask)
    grid.build_interpolant()
    return grid
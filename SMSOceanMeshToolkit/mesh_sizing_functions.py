'''
Mesh sizing functions
'''
# stdlib
import logging

# tpl
import geopandas as gpd
import numpy as np
from inpoly import inpoly2
from shapely.geometry import LineString
import numpy as np
import scipy.spatial
from skimage.morphology import medial_axis
import skfmm # fast marching method

# from SMSOceanMeshToolkit/local
from .edges import get_poly_edges
from .Region import Region
from .Grid import Grid
from .signed_distance_function import signed_distance_function

from .libs._HamiltonJacobi import gradient_limit

logger = logging.getLogger(__name__)

__all__ = [
    "distance_sizing_function",
    "distance_sizing_from_point_function",
    "distance_sizing_from_linestring_function",
    "feature_sizing_function",
    "enforce_mesh_gradation",
    "wavelength_sizing_function",
]


def wavelength_sizing_function(
    grid, 
    dem,
    wl=10, 
    max_edge_length=np.inf,
    period=12.42 * 3600,  # M2 period in seconds
    gravity=9.81,  # m/s^2
    crs="EPSG:4326",
):
    """
    Mesh sizes that vary proportional to an estimate of the wavelength
    of a period (default M2-period on Earth is 12.42 hours) and the
    acceleration due to gravity.

    Parameters
    ----------
    grid: :class:`Grid`
        A grid object that will contain the wavelength mesh sizing function
    dem:  :class:`DEM`
        Data processed from :class:`DEM`.
    wl: integer, optional
        The number of desired elements per wavelength of the M2 constituent
    max_edge_length: float, optional
        The maximum edge length in meters in the domain.
    period: float, optional
        The wavelength is estimated with shallow water theory and this period
        in seconds
    gravity: float, optional
        The acceleration due to gravity in m/s^2


    Returns
    -------
    :class:`Grid` objectd
        A sizing function that takes a point and returns a value

    """
    logger.info("Building a wavelength sizing function...")

    x, y = grid.create_vectors()
    # Interpolate the DEM onto the grid points
    tmpz = dem.da.interp(x=x, y=y)

    if crs == "EPSG:4326" or crs == 4326:
        mean_latitude = np.mean(grid.bbox[2:])
        meters_per_degree = (
            111132.92
            - 559.82 * np.cos(2 * mean_latitude)
            + 1.175 * np.cos(4 * mean_latitude)
            - 0.0023 * np.cos(6 * mean_latitude)
        )
    tmpz = tmpz.values.T
    tmpz[np.abs(tmpz) < 1] # avoid division by zero
    # Calculate the wavelength of a wave with period `period` and
    # acceleration due to gravity `gravity`  
    grid.values = period * np.sqrt(gravity * np.abs(tmpz)) / wl

    # Convert back to degrees from meters (if geographic)
    if crs == "EPSG:4326" or crs == 4326:
        grid.values /= meters_per_degree

    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.build_interpolant()
    return grid


def enforce_mesh_gradation(grid, gradation=0.05):
    """
    Enforce a mesh size gradation bound `gradation` on a :class:`grid`

    Parameters
    ----------
    grid: :class:`Grid`
        A grid object with its values field populated
    gradation: float
        The decimal percent mesh size gradation rate to-be-enforced.

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field gradient limited

    """
    if gradation <= 0:
        raise ValueError("Parameter `gradation` must be > 0.0")
    if gradation >= 0.30:
        logger.warning("Parameter `gradation` is set excessively high (> 0.30)")

    logger.info(f"Enforcing mesh size gradation of {gradation} decimal percent...")

    elen = grid.dx
    # TODO: add support for unequal grid spaces
    assert (
        grid.dx == grid.dy
    ), "Structured grids with unequal grid spaces not yet supported"
    cell_size = grid.values.copy()
    sz = cell_size.shape
    sz = (sz[0], sz[1], 1)
    cell_size = cell_size.flatten("F")
    MAXITER = 10000 # maximum number of iterations in cpp code 
    # NB: cell_size is in units of the grid's crs
    tmp = gradient_limit([*sz], elen, gradation, MAXITER, cell_size)
    tmp = np.reshape(tmp, (sz[0], sz[1]), "F")
    # Replace the grid's values with the gradient limited values
    grid.values = tmp
    # Rebuild the interpolant
    grid.build_interpolant()
    return grid


def feature_sizing_function(
    grid, 
    coastal_geometry,
    number_of_elements_per_width=3,
    max_edge_length=np.inf,
    gradation=0.05,
):
    """
    Mesh sizes vary proportional to the width or "thickness" of the shoreline

    Parameters
    ----------
    grid: class:`Grid`
        A grid object that will contain the feature sizing function
    coastal_geometry: :class:`CoastalGeometry`
        Vector data processed
    number_of_elements_per_width: integer, optional
        The number of elements per estimated width of the vector data
    max_edge_length: float, optional
        The maximum edge length of the mesh in the units of the grid's crs
    gradation: float, optional
        The decimal percent mesh size gradation rate to-be-enforced.

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field populated with feature sizing

    """

    logger.info("Building a feature sizing function...")
    assert number_of_elements_per_width > 0, "local feature size  must be greater than 0"
    
    # create a Region 
    region = Region(coastal_geometry.bbox, grid.crs)
    
    # form a signed distance function from the coastal geometry
    my_signed_distance_function = signed_distance_function(coastal_geometry)

    min_edge_length = coastal_geometry.minimum_mesh_size
    # The medial axis calculation requires a finer grid than the final grid by a factor of 2x
    grid_calc = Grid(
        region, 
        dx=min_edge_length / 2.0,  # dx is half that of the original shoreline spacing
        values=0.0,
        extrapolate=True,
    )
    x, y = grid_calc.create_grid()
    qpts = np.column_stack((x.flatten(), y.flatten()))
    phi = my_signed_distance_function.eval(qpts)
    # outside 
    phi[phi > 0] = 999
    # inside and on the boundary
    phi[phi <= 0] = 1.0
    # n/a values
    phi[phi == 999] = 0.0
    phi = np.reshape(phi, grid_calc.values.shape)

    # calculate the medial axis points
    skel = medial_axis(phi, return_distance=False)

    indicies_medial_points = skel == 1
    medial_points_x = x[indicies_medial_points]
    medial_points_y = y[indicies_medial_points]
    medial_points = np.column_stack((medial_points_x, medial_points_y))

    phi2 = np.ones(shape=(grid_calc.nx, grid_calc.ny))
    points = np.vstack((coastal_geometry.inner, coastal_geometry.mainland))
    # find location of points on grid
    indices = grid_calc.find_indices(points, x, y)
    phi2[indices] = -1.0
    dis = np.abs(skfmm.distance(phi2, [grid_calc.dx, grid_calc.dy]))

    # calculate the distance to medial axis
    tree = scipy.spatial.cKDTree(medial_points)
    try:
        dMA, _ = tree.query(qpts, k=1, workers=-1)
    except (Exception,):
        dMA, _ = tree.query(qpts, k=1, n_jobs=-1)
    dMA = dMA.reshape(*dis.shape)
    W = dMA + np.abs(dis)
    feature_size = (2 * W) / number_of_elements_per_width

    grid_calc.values = feature_size
    grid_calc.build_interpolant()
    # interpolate the finer grid used for calculations to the final coarser grid
    grid = grid_calc.interpolate_onto(grid)
    if min_edge_length is not None:
        grid.values[grid.values < min_edge_length] = min_edge_length
    if max_edge_length is not np.inf:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.extrapolate = True
    grid.build_interpolant()
    
    # enforce mesh gradation
    grid = enforce_mesh_gradation(grid, gradation=gradation)
    
    return grid


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
    #grid.values = np.ma.array(tmp, mask=mask)
    grid.values = np.array(tmp)
    grid.build_interpolant()
    return grid
'''
Mesh sizing functions
'''
import logging
import math
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from inpoly import inpoly2
from shapely.geometry import LineString

import skfmm # fast marching method

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
    point_file,
    bbox,
    min_edge_length,
    rate=0.15,
    max_edge_length=None,
    coarsen=1,
    crs="EPSG:4326",
):
    '''Mesh sizes that vary linearly at `rate` from a point or points
    contained within a geopandas dataframe.

     Parameters
    ----------
    point_geodataframe: str
        Path to a vector file containing Points
    bbox: list or tuple
        A list or tuple of the form [xmin, xmax, ymin, ymax] denoting the bounding box of the
        domain
    min_edge_length: float
        The minimum edge length of the mesh
    rate: float
        The decimal percent mesh expansion rate from the point(s)
    coarsen: int
        The coarsening factor of the background grid
    crs: A Python int, dict, or str, optional
        The coordinate reference system

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field gradient limited

    """


    '''
    logger.info("Building a distance sizing from point function...")
    point_geodataframe = gpd.read_file(point_file)
    assert (
        point_geodataframe.crs == crs
    ), "The crs of the point geodataframe must match the crs of the grid"
    # check all the geometries are points
    assert all(
        point_geodataframe.geometry.geom_type == "Point"
    ), "All geometries must be points"
    # Get the coordinates of the points from the geodataframe
    points = np.array(point_geodataframe.geometry.apply(lambda x: (x.x, x.y)).tolist())
    # Create a mesh size function grid
    grid = Grid(
        bbox=bbox,
        dx=min_edge_length * coarsen,
        hmin=min_edge_length,
        extrapolate=True,
        values=0.0,
        crs=crs,
    )
    # create phi (-1 where point(s) intersect grid points -1 elsewhere 0)
    phi = np.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    # find location of points on grid
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    try:
        dis = np.abs(skfmm.distance(phi, [grid.dx, grid.dy]))
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
    shoreline,
    rate=0.15,
    max_edge_length=None,
    coarsen=1.0,
    crs="EPSG:4326",
):
    """
    Mesh sizes that vary linearly at `rate` from coordinates in `obj`:CoastalGeometry
    
    Parameters
    ----------
    shoreline: :class:`Shoreline`
        Data processed from :class:`Shoreline`.
    rate: float, optional
        The rate of expansion in decimal percent from the shoreline.
    max_edge_length: float, optional
        The maximum allowable edge length
    coarsen: integer, optional
        Downsample the grid by a constant factor in x and y axes
    crs: A Python int, dict, or str, optional
        The coordinate reference system

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value
    """
    logger.info("Building a distance mesh sizing function...")

    grid = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0 * coarsen,
        hmin=shoreline.h0,
        extrapolate=True,
        values=0.0,
        crs=crs,
    )
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = np.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = np.vstack((shoreline.inner, shoreline.mainland))
    # remove shoreline components outside the shoreline.boubox
    boubox = np.nan_to_num(shoreline.boubox)  # remove nan for inpoly2
    e_box = edges.get_poly_edges(shoreline.boubox)
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
        dis = np.zeros((grid.nx, grid.ny)) + 999
    tmp = shoreline.h0 + dis * rate
    if max_edge_length is not None:
        tmp[tmp > max_edge_length] = max_edge_length
    grid.values = np.ma.array(tmp, mask=mask)
    grid.build_interpolant()
    return grid


def bathymetric_gradient_sizing_function(
    dem,
    slope_parameter=20,
    filter_quotient=50,
    min_edge_length=None,
    max_edge_length=None,
    min_elevation_cutoff=-50.0,
    type_of_filter="lowpass",
    filter_cutoffs=1000,
    crs="EPSG:4326",
):
    """Mesh sizes that vary proportional to the bathymetryic gradient.
       Bathymetry is filtered by default using a fraction of the
       barotropic Rossby radius however there are several options for
       filtering the bathymetric data (see the Parameters below).

    Parameters
    ----------
    dem:  :class:`DEM`
        Data processed from :class:`DEM`.
    filter_quotient: float, optional
        The filter length equal to Rossby radius divided by fl
    slope_parameter: integer, optional
        The number of nodes to resolve bathymetryic gradients
    min_edge_length: float, optional
        The minimum allowable edge length in CRS units in the domain.
    max_edge_length: float, optional
        The maximum allowable edge length in CRS units in the domain.
    min_elevation_cutoff: float, optional
        abs(elevation) < this value the sizing function is not calculated.
    type_of_filter: str, optional
        Use the barotropic, baroclinic Rossby radius to lowpass filter bathymetry
        prior to calculating the sizing function. In addition,
        bandpass, lowpass, highpass can also be utilized.
    filter_cutoff: list, optional
        If filter is bandpass/lowpass/highpass/bandstop, then contains the lower and upper
        bounds for the filter (depends on the filter)
    crs: A Python int, dict, or str, optional
        The coordinate reference system

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """

    logger.info("Building a slope length sizing function...")

    xg, yg = dem.create_grid()
    tmpz = dem.eval((xg, yg))

    grid = Grid(
        bbox=dem.bbox,
        dx=min_edge_length / 2.0,
        dy=min_edge_length / 2.0,
        extrapolate=True,
        values=0.0,
        crs=crs,
        hmin=min_edge_length,
    )
    logger.info(f"Enforcing a minimum elevation cutoff of {min_elevation_cutoff}")
    tmpz[tmpz >= min_elevation_cutoff] = min_elevation_cutoff

    dx, dy = dem.dx, dem.dy  # for gradient function
    nx, ny = dem.nx, dem.ny
    coords = (xg, yg)

    if crs == "EPSG:4326" or crs == 4326:
        mean_latitude = np.mean(dem.bbox[2:])
        meters_per_degree = (
            111132.92
            - 559.82 * np.cos(2 * mean_latitude)
            + 1.175 * np.cos(4 * mean_latitude)
            - 0.0023 * np.cos(6 * mean_latitude)
        )
        dy *= meters_per_degree
        dx *= meters_per_degree
    grid_details = (nx, ny, dx, dy)

    if type_of_filter == "barotropic" and filter_quotient > 0:
        # Temporary fix for barotropic filter not implemented correctly yet
        assert ValueError("Barotropic filter not implemented yet")

        logger.info("Baroptropic Rossby radius calculation...")
        bs, time_taken = rossby_radius_filter(
            tmpz, dem.bbox, grid_details, coords, filter_quotient, True
        )

    elif type_of_filter == "baroclinic" and filter_quotient > 0:
        # Temporary fix for barotropic filter not implemented correctly yet
        assert ValueError("Baroclinic filter not implemented yet")

        logger.info("Baroclinic Rossby radius calculation...")
        bs, time_taken = rossby_radius_filter(
            tmpz, dem.bbox, grid_details, coords, filter_quotient, False
        )
    elif "pass" in type_of_filter:
        logger.info(f"Using a {type_of_filter} filter...")
        tmpzs = filt2(tmpz, dy, filter_cutoffs, type_of_filter)
        by, bx = _earth_gradient(tmpzs, dy, dx)
        bs = np.sqrt(bx**2 + by**2)  # get overall slope
    else:
        msg = f"The type_of_filter {type_of_filter} is not known and remains off"
        logger.info(msg)
        by, bx = _earth_gradient(tmpz, dy, dx)  # get slope in x and y directions
        bs = np.sqrt(bx**2 + by**2)  # get overall slope

    # Calculating the slope function
    eps = 1e-10  # small number to approximate derivative
    dp = np.clip(tmpz, None, -1)
    grid.values = (2 * np.pi / slope_parameter) * np.abs(dp) / (bs + eps)

    # Convert back to degrees from meters (if geographic)
    if crs == "EPSG:4326" or crs == 4326:
        grid.values /= meters_per_degree
        grid.dx = dem.dx
        grid.dy = dem.dy

    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    if min_edge_length is None:
        min_edge_length = grid.dx

    grid.values[grid.values < min_edge_length] = min_edge_length
    grid.build_interpolant()

    return grid


def rossby_radius_filter(tmpz, bbox, grid_details, coords, rbfilt, barot):
    """
    Performs the Rossby radius filtering

    Parameters
    ----------
    tmpz : numpy.ndarray
        Contains the bathymetric data across the grid formed by coordinate
        arrays (xg, yg).
    bbox : tuple
        Describes the boundary box of our domain.
    grid_details : tuple
        Contains the information regarding normals and grid resolutions,
        (nx, ny, dx, dy).
    coords : tuple np.ndarray
        A tuple of two numpy.ndarray describing the longitude and latitude
        coordinate system of our grid.
    rbfilt : float
        Describes the corresponding rossby radius to filter out
    barot : bool
        If True, the function uses the barotropic Rossby radius of deformation.

    Returns
    -------
    bs : numpy.ndarray
        This is essentially grad(h) squared after performing the bandpass
        filtering on the Rossby radius of deformation.
    time_taken : float
        the time taken to prform the filtering process.

    """

    x0, xN, y0, yN = bbox

    nx, ny, dx, dy = grid_details
    xg, yg = coords

    start = time.perf_counter()
    bs = np.empty(tmpz.shape)
    bs[:] = np.nan

    # Break into 10 deg latitude chunks or less if higher resolution
    div = math.ceil(min(1e7 / nx, 10 * ny / (yN - y0)))
    grav, Rre = 9.807, 7.29e-5  # Gravity and Rotation rate of Earth in radians
    # per second
    number_of_blocks = math.ceil(ny / div)
    n2s = 0

    for jj in range(number_of_blocks):
        n2e = min(ny, n2s + div)
        # Rossby radius of deformation filter
        # See Shelton, D. B., et al. (1998): Geographical variability of the
        # first-baroclinic Rossby radius of deformation. J. Phys. Oceanogr.,
        # 28, 433-460.
        ygg = yg[:, n2s:n2e]
        dxx = np.mean(np.diff(xg[n2s:n2e, 0]))
        f = 2 * Rre * abs(np.sin(ygg * np.pi / 180))
        if barot:
            # Barotropic case
            c = np.sqrt(grav * np.maximum(1, -tmpz[:, n2s:n2e]))

        else:
            # Baroclinic case (estimate Nm to be 2.5e-3)
            Nm = 2.5e-3  # Δz x N, where N is Brunt-Vaisala frequency,
            # sqrt(-g/ρ0 * dρ/dz), giving sqrt(-g * (Δρ/ρ0) * Δz)
            c = Nm * np.maximum(1, -tmpz[:, n2s:n2e]) / np.pi

        rosb = c / f
        # Update for equatorial regions
        indices = abs(ygg) < 5
        Re = 6.371e6  # Earth radius at equator in SI units of metres
        twobeta = 4 * Rre * np.cos(ygg[indices] * np.pi / 180) / Re
        rosb[indices] = np.sqrt(c[indices] / twobeta)
        # limit rossby radius to 10,000 km for practical purposes
        rosb[rosb > 1e7] = 1e7
        # Keep lengthscales rbfilt * barotropic
        # radius of deformation
        rosb = np.minimum(10, np.maximum(0, np.floor(np.log2(rosb / dy / rbfilt))))
        edges = np.unique(np.copy(rosb))
        bst = rosb * 0
        for i in range(len(edges)):
            if edges[i] > 0:
                mult = 2 ** edges[i]
                xl, xu = 1, nx
                if ((np.max(xg) > 179 and np.min(xg) < -179)) or (
                    np.max(xg) > 359 and np.min(xg) < 1
                ):
                    # wraps around
                    logger.info("wrapping around")
                    xr = np.concatenate(
                        [
                            np.arange(nx - mult / 2, nx, 1),
                            np.arange(xl, xu),
                            np.arange(1, mult / 2),
                        ],
                        dtype=int,
                    )
                else:
                    xr = np.arange(xl - 1, xu, dtype=int)

                yl, yu = max(1, n2s - mult / 2), min(ny, n2e + mult / 2)
                if np.max(yg) > 89 and yu == ny:
                    # create mirror around pole
                    yr = np.concatenate(
                        [
                            np.arange(yl, yu),
                            np.arange(yu - 1, 2 * ny - n2e - mult / 2, -1),
                        ],
                        dtype=int,
                    )
                else:
                    yr = np.arange(yl - 1, yu, dtype=int)

                xr, yr = xr[:, None], yr[None, :]

                if mult == 2:
                    tmpz_ft = filt2(tmpz[xr, yr], min([dxx, dy]), dy * 2.01, "lowpass")
                else:
                    tmpz_ft = filt2(tmpz[xr, yr], min([dxx, dy]), dy * mult, "lowpass")

                # delete the padded region
                tmpz_ft[: np.where(xr == 0)[0][0], :] = 0
                tmpz_ft[nx:, :] = 0
                tmpz_ft[:, : np.where(yr == n2s)[0][0]] = 0
                tmpz_ft[:, n2e - n2s + 2 :] = 0
                tmpz_ft = tmpz[:, n2s:n2e]

            else:
                tmpz_ft = tmpz[:, n2s:n2e]

            by, bx = _earth_gradient(
                tmpz_ft, dy, dx
            )  # [n2s:n2e]) # get slope in x and y directions
            tempbs = np.sqrt(bx**2 + by**2)  # get overall slope

            bst[rosb == edges[i]] = tempbs[rosb == edges[i]]

        bs[:, n2s:n2e] = bst
        n2s = n2e

    time_taken = time.perf_counter() - start

    return bs, time_taken


def feature_sizing_function(
    shoreline,
    signed_distance_function,
    r=3,
    min_edge_length=None,
    max_edge_length=None,
    plot=False,
    crs="EPSG:4326",
):
    """Mesh sizes vary proportional to the width or "thickness" of the shoreline

    Parameters
    ----------
    shoreline: :class:`Shoreline`
        Data processed from :class:`Shoreline`.
    signed_distance_function: a function
        A `signed_distance_function` object
    r: float, optional
        The number of times to divide the shoreline thickness/width to calculate
        the local element size.
    min_edge_length: float, optional
        The minimum allowable edge length in meters in the domain.
    max_edge_length: float, optional
        The maximum allowable edge length in meters in the domain.
    plot: boolean, optional
        Visualize the medial points ontop of the shoreline
    crs: A Python int, dict, or str, optional
        The coordinate reference system

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """

    logger.info("Building a feature sizing function...")

    assert r > 0, "local feature size "
    grid_calc = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0 / 2,  # dx is half that of the original shoreline spacing
        hmin=shoreline.h0,
        values=0.0,
        extrapolate=True,
        crs=crs,
    )
    grid = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0,
        hmin=shoreline.h0,
        values=0.0,
        extrapolate=True,
        crs=crs,
    )
    lon, lat = grid_calc.create_grid()
    qpts = np.column_stack((lon.flatten(), lat.flatten()))
    phi = signed_distance_function.eval(qpts)
    phi[phi > 0] = 999
    phi[phi <= 0] = 1.0
    phi[phi == 999] = 0.0
    phi = np.reshape(phi, grid_calc.values.shape)

    skel = medial_axis(phi, return_distance=False)

    indicies_medial_points = skel == 1
    medial_points_x = lon[indicies_medial_points]
    medial_points_y = lat[indicies_medial_points]
    medial_points = np.column_stack((medial_points_x, medial_points_y))

    phi2 = np.ones(shape=(grid_calc.nx, grid_calc.ny))
    points = np.vstack((shoreline.inner, shoreline.mainland))
    # find location of points on grid
    indices = grid_calc.find_indices(points, lon, lat)
    phi2[indices] = -1.0
    dis = np.abs(skfmm.distance(phi2, [grid_calc.dx, grid_calc.dy]))

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.pcolor(lon, lat, skel, cmap=plt.cm.gray)
        ax1.axis("off")
        ax2.pcolor(lon, lat, skel)
        ax2.contour(lon, lat, phi, [0.5], colors="w")
        ax2.axis("off")

        fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
        plt.show()

    # calculate distance to medial axis
    tree = scipy.spatial.cKDTree(medial_points)
    try:
        dMA, _ = tree.query(qpts, k=1, workers=-1)
    except (Exception,):
        dMA, _ = tree.query(qpts, k=1, n_jobs=-1)
    dMA = dMA.reshape(*dis.shape)
    W = dMA + np.abs(dis)
    feature_size = (2 * W) / r

    grid_calc.values = feature_size
    grid_calc.build_interpolant()
    # interpolate the finer grid used for calculations to the final coarser grid
    grid = grid_calc.interpolate_to(grid)
    if min_edge_length is not None:
        grid.values[grid.values < min_edge_length] = min_edge_length
    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.hmin = shoreline.h0

    grid.extrapolate = True
    grid.build_interpolant()
    return grid


def wavelength_sizing_function(
    dem,
    wl=10,
    min_edgelength=None,
    max_edge_length=None,
    period=12.42 * 3600,  # M2 period in seconds
    gravity=9.81,  # m/s^2
    crs="EPSG:4326",
):
    """Mesh sizes that vary proportional to an estimate of the wavelength
       of a period (default M2-period)

    Parameters
    ----------
    dem:  :class:`Dem`
        Data processed from :class:`Dem`.
    wl: integer, optional
        The number of desired elements per wavelength of the M2 constituent
    min_edgelength: float, optional
        The minimum edge length in meters in the domain. If None, the min
        of the edgelength function is used.
    max_edge_length: float, optional
        The maximum edge length in meters in the domain.
    period: float, optional
        The wavelength is estimated with shallow water theory and this period
        in seconds
    gravity: float, optional
        The acceleration due to gravity in m/s^2
    crs: A Python int, dict, or str, optional
        The coordinate reference system


    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """
    logger.info("Building a wavelength sizing function...")

    lon, lat = dem.create_grid()
    tmpz = dem.eval((lon, lat))

    dx, dy = dem.dx, dem.dy  # for gradient function

    if crs == "EPSG:4326" or crs == 4326:
        mean_latitude = np.mean(dem.bbox[2:])
        meters_per_degree = (
            111132.92
            - 559.82 * np.cos(2 * mean_latitude)
            + 1.175 * np.cos(4 * mean_latitude)
            - 0.0023 * np.cos(6 * mean_latitude)
        )
        dy *= meters_per_degree
        dx *= meters_per_degree
    grid = Grid(
        bbox=dem.bbox, dx=dem.dx, dy=dem.dy, extrapolate=True, values=0.0, crs=crs
    )
    tmpz[np.abs(tmpz) < 1] = 1
    grid.values = period * np.sqrt(gravity * np.abs(tmpz)) / wl

    # Convert back to degrees from meters (if geographic)
    if crs == "EPSG:4326" or crs == 4326:
        grid.values /= meters_per_degree
        grid.dx = dem.dx
        grid.dy = dem.dy

    if min_edgelength is None:
        min_edgelength = np.amin(grid.values)
    else:
        grid.values[grid.values < min_edgelength] = min_edgelength

    grid.hmin = min_edgelength

    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.build_interpolant()
    return grid


def multiscale_sizing_function(
    list_of_grids,
    p=3,
    nnear=28,
    blend_width=1000,
):
    """Given a list of mesh size functions in a hierarchy
    w.r.t. to minimum mesh size (largest -> smallest),
    create a so-called multiscale mesh size function

    Parameters
    ----------
    list_of_grids: a Python list
        A list containing grids with resolution in decreasing order.
    p: int, optional
        use 1 / distance**p weights nearer points more, farther points less.
    nnear: int, optional
        how many nearest neighbors should one take to perform IDW interp?
    blend_width: float, optional
        The width of the blending zone between nests in meters

    Returns
    -------
    func: a function
        The global sizing funcion defined over the union of all domains
    new_list_of_grids: a list of  function
        A list of sizing function that takes a point and returns a value

    """
    err = "grid objects must appear in order of descending dx spacing"
    for i, grid in enumerate(list_of_grids[:-1]):
        assert grid.dx >= list_of_grids[i + 1].dx, err

    new_list_of_grids = []
    # loop through remaining sizing functions
    for idx1, new_coarse in enumerate(list_of_grids[:-1]):
        logger.info(f"For sizing function #{idx1}")

        # interpolate all finer nests onto coarse func and enforce gradation rate
        for k, finer in enumerate(list_of_grids[idx1 + 1 :]):
            logger.info(
                f"  Interpolating sizing function #{idx1 + 1 + k} onto sizing function #{idx1}"
            )
            _wkt = finer.crs.to_dict()
            if "units" in _wkt:
                _dx = finer.dx
            else:
                # it must be degrees?
                _dx = finer.dx * 111e3
            _blend_width = int(np.floor(blend_width / _dx))
            finer.extrapolate = False
            new_coarse = finer.blend_into(
                new_coarse, blend_width=_blend_width, p=p, nnear=nnear
            )
            new_coarse.extrapolate = True
            new_coarse.build_interpolant()
        # append it to list
        new_list_of_grids.append(new_coarse)

    list_of_grids[-1].extrapolate = True

    # retain the finest
    new_list_of_grids.append(list_of_grids[-1])

    # return the mesh size function to query during genertaion
    # NB: only keep the minimum value over all grids
    def func(qpts):
        hmin = np.array([999999] * len(qpts))
        for i, grid in enumerate(new_list_of_grids):
            if i == 0:
                grid.extrapolate = True
            else:
                grid.extrapolate = False
            grid.build_interpolant()
            _hmin = grid.eval(qpts)
            hmin = np.min(np.column_stack([_hmin, hmin]), axis=1)
        return hmin

    return func, new_list_of_grids


def _earth_gradient(F, dx, dy):
    """
    earth_gradient(F,HX,HY), where F is 2-D, uses the spacing
    specified by HX and HY. HX and HY can either be scalars to specify
    the spacing between coordinates or vectors to specify the
    coordinates of the points.  If HX and HY are vectors, their length
    must match the corresponding dimension of F.
    """
    Fy, Fx = np.zeros(F.shape), np.zeros(F.shape)

    # Forward diferences on edges
    Fx[:, 0] = (F[:, 1] - F[:, 0]) / dx
    Fx[:, -1] = (F[:, -1] - F[:, -2]) / dx
    Fy[0, :] = (F[1, :] - F[0, :]) / dy
    Fy[-1, :] = (F[-1, :] - F[-2, :]) / dy

    # Central Differences on interior
    Fx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dx)
    Fy[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * dy)

    return Fy, Fx
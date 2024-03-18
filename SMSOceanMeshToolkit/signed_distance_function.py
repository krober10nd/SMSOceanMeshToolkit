import logging

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import scipy.spatial
import xarray as xr
from inpoly import inpoly2
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .edges import get_poly_edges
from .geospatial_data import CoastalGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "signed_distance_function",
    "SDFDomain",
]

nan = np.nan


# def get_poly_edges(poly):
#     """
#     Given a winded polygon represented as a set of line segments
#     with separated polygons indicated by a row of nans, this function returns
#     the edges of the polygon such that each edge contains an index to the start and end
#     coordinates.

#     Parameters
#     ----------
#     poly: array-like, float
#         A 2D array of point coordinates with features sepearated by NaNs

#     Returns
#     -------
#     edges: array-like, int
#         A 2D array of integers containing indexes into the `poly` array.

#     """
#     ix = np.argwhere(np.isnan(poly[:, 0])).ravel()
#     ix = np.insert(ix, 0, -1)

#     edges = []
#     for s in range(len(ix) - 1):
#         ix_start = ix[s] + 1
#         ix_end = ix[s + 1] - 1
#         col1 = np.arange(ix_start, ix_end - 1)
#         col2 = np.arange(ix_start + 1, ix_end)
#         tmp = np.vstack((col1, col2)).T
#         tmp = np.append(tmp, [[ix_end, ix_start]], axis=0)
#         edges.append(tmp)
#     return np.concatenate(edges, axis=0)


def _plot(geo, grid_size=100, show=False):
    # Assuming _generate_samples and geo.eval are defined elsewhere
    # Grid for hatching
    x_min, x_max, y_min, y_max = geo.bbox
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    # Evaluate SDF on grid and determine interior points
    signed_distance = geo.eval(np.vstack((x_mesh.ravel(), y_mesh.ravel())).T).reshape(
        x_mesh.shape
    )
    interior_mask = signed_distance <= 0
    # Plotting
    fig, ax = plt.subplots()

    # Create hatched patches for the interior
    patches = []
    colors = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            if interior_mask[j, i]:
                rect = Rectangle(
                    (x_grid[i], y_grid[j]),
                    x_grid[i + 1] - x_grid[i],
                    y_grid[j + 1] - y_grid[j],
                )
                patches.append(rect)
                colors.append(signed_distance[j, i])

    # Discrete colormap
    num_colors = 10  # Number of colors in the discrete colormap
    cmap = plt.cm.inferno  # Inferno colormap
    bounds = np.linspace(np.min(colors), np.max(colors), num_colors + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create PatchCollection with discrete colors
    p = PatchCollection(patches, cmap=cmap, norm=norm)  # , alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)

    # Adding a colorbar for the signed distance
    cb = plt.colorbar(p, ax=ax, boundaries=bounds)
    # set the x and y limit
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    cb.set_label("Signed Distance")

    plt.title("Signed Distance Function \n $\Omega$ is shaded")

    return ax


class SDFDomain:
    """
    Simple class to represent a signed distance function domain
    """

    def __init__(self, bbox, func):
        self.bbox = bbox
        self.domain = func

    def eval(self, x):
        return self.domain(x)

    def plot(self, grid_size=100, show=True):
        ax = _plot(self, grid_size=grid_size, show=show)
        return ax

    @staticmethod
    def to_xarray(self, crs, grid_size=100):
        """
        Evaluate the signed distance function on a grid and return an xarray dataset
        """
        x_min, x_max, y_min, y_max = self.bbox
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        # Evaluate SDF on grid and determine interior points
        signed_distance = self.eval(
            np.vstack((x_mesh.ravel(), y_mesh.ravel())).T
        ).reshape(x_mesh.shape)
        # package the data into an xarray dataset
        ds = xr.Dataset(
            {
                "signed_distance": (["y", "x"], signed_distance),
            },
            coords={
                "x": x_grid,
                "y": y_grid,
            },
        )
        # add the crs
        ds.rio.write_crs(crs, inplace=True)

        return ds


def polygons_to_numpy(gdf):
    coords_list = []
    for polygon in gdf.geometry:
        if polygon.is_empty:
            continue
        # Extract only exterior coordinates
        x, y = polygon.exterior.coords.xy
        coords_list.extend(list(zip(x, y)))
        # Add NaNs to separate different polygons
        coords_list.append((np.nan, np.nan))

    return np.array(coords_list)


def signed_distance_function(coastal_geometry, invert=False):
    """
    Takes a :class:`CoastalGeometry` object containing labeled polygons representing meshing boundaries
    and creates a callable signed distance function that is used during mesh generation.
    The signed distance function returns a negative value if the point is inside the domain
    and a positive value if the point is outside the domain. Consequently, points with
    0 value are on the boundary of the domain.

    The returned function `func` becomes a method of the :class:`Domain` and is queried during
    mesh generation several times per meshing iteration.

    Parameters
    ----------
    coastal_geometry: a :class:`CoastalGeometry` object
        The processed data from :class:`CoastalGeometry` either in the
        form of a Python class or a GeoDataFrame.
    invert: boolean, optional
        Invert the definition of the domain. Can be useful
        if the region meshed appears to be the water body instead of the land.

    Returns
    -------
    domain: a :class:`Domain` object
        Contains a callback to the signed distance function along with an extent `bbox`

    """
    logger.info("Building a signed distance function...")

    assert isinstance(
        coastal_geometry, (CoastalGeometry, gpd.GeoDataFrame)
    ), "coastal_geometry is not a CoastGeometry or GeoDataFrame object"

    if isinstance(coastal_geometry, gpd.GeoDataFrame):
        bbox = coastal_geometry.total_bounds
        # change the order of the bbox since it's minx maxx miny maxy
        bbox = (bbox[0], bbox[2], bbox[1], bbox[3])
        # group all the polygons with the label inner into a nan separated array
        inner = coastal_geometry[coastal_geometry["labels"] == "inner"]
        inner = polygons_to_numpy(inner)

        region_polygon = coastal_geometry[coastal_geometry["labels"] == "outer"]
        region_polygon = polygons_to_numpy(region_polygon)

        boubox = coastal_geometry[coastal_geometry["labels"] == "boubox"]
        boubox = polygons_to_numpy(boubox)

    elif isinstance(coastal_geometry, CoastalGeometry):
        bbox = coastal_geometry.bbox
        boubox = coastal_geometry.boubox
        inner = coastal_geometry.inner
        region_polygon = coastal_geometry.region_polygon

    # make sure the first row and the last row are the same
    if not np.all(boubox[0, :] == boubox[-1, :]):
        boubox = np.vstack((boubox, boubox[0, :]))
    boubox = np.asarray(boubox)

    # combine the inner and region polygons (if they exist)
    if inner is None:
        poly = region_polygon
    else:
        # region_polygon is outer boundary
        poly = np.vstack((inner, region_polygon))

    # create a kdtree for nearest neighbor search for SDF
    tree = scipy.spatial.cKDTree(poly[~np.isnan(poly[:, 0]), :])
    # edges of the polygon
    e = get_poly_edges(poly)
    # vend = e[-1:, 1]
    # append a row to e
    # e = np.vstack((e, [vend[0], e[0, 0]]))

    def func(x):
        # Initialize d with some positive number larger than geps
        dist = np.zeros(len(x)) + 999.0
        # all points are assumed to be outside the domain
        inside = np.zeros(len(x), dtype=bool)
        # are points inside the polygon (boubox) indicating the domain?
        in_boubox, _ = inpoly2(x, boubox[:-1])
        # points outside the boubox are not inside the domain
        inside[in_boubox] = True
        # Of the points that are inside the boubox, determine which are also
        # inside the multi-polygon representing the domain
        in_shoreline, _ = inpoly2(x[in_boubox], poly, e)
        if invert:
            in_shoreline = ~in_shoreline
        inside[in_boubox] = in_shoreline
        # d is signed negative if inside the
        # compute nearest distance to multi poly (operate in parallel workers=-1)
        dist, _ = tree.query(x, k=1, workers=-1)
        # sign the distance
        dist = (-1) ** (inside) * dist
        return dist

    return SDFDomain(bbox, func)

import logging
 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
 
import numpy as np
import scipy.spatial
from inpoly import inpoly2
import xarray as xr
import rioxarray
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from .geospatial_data import CoastalGeometry, _create_boubox
import geopandas as gpd

logger = logging.getLogger(__name__)
 
__all__ = [
    "signed_distance_function",
    "SDFDomain",
]
 
nan = np.nan
 
 
def get_poly_edges(poly):
    """
    Given a winded polygon represented as a set of line segments
    with separated polygons indicated by a row of nans, this function returns
    the edges of the polygon such that each edge contains an index to the start and end
    coordinates.
 
    Parameters
    ----------
    poly: array-like, float
        A 2D array of point coordinates with features sepearated by NaNs
 
    Returns
    -------
    edges: array-like, int
        A 2D array of integers containing indexes into the `poly` array.
 
    """
    ix = np.argwhere(np.isnan(poly[:, 0])).ravel()
    ix = np.insert(ix, 0, -1)
 
    edges = []
    for s in range(len(ix) - 1):
        ix_start = ix[s] + 1
        ix_end = ix[s + 1] - 1
        col1 = np.arange(ix_start, ix_end - 1)
        col2 = np.arange(ix_start + 1, ix_end)
        tmp = np.vstack((col1, col2)).T
        tmp = np.append(tmp, [[ix_end, ix_start]], axis=0)
        edges.append(tmp)
    return np.concatenate(edges, axis=0)

 
def _plot(geo, grid_size=100):
    # Assuming _generate_samples and geo.eval are defined elsewhere
    # Grid for hatching
    x_min, x_max, y_min, y_max = geo.bbox
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    # Evaluate SDF on grid and determine interior points
    signed_distance = geo.eval(np.vstack((x_mesh.ravel(), y_mesh.ravel())).T).reshape(x_mesh.shape) 
    interior_mask = signed_distance <= 0
    # Plotting
    fig, ax = plt.subplots()
 
    # Create hatched patches for the interior
    patches = []
    colors = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            if interior_mask[j, i]:
                rect = Rectangle((x_grid[i], y_grid[j]), 
                                 x_grid[i + 1] - x_grid[i], 
                                 y_grid[j + 1] - y_grid[j])
                patches.append(rect)
                colors.append(signed_distance[j, i])
 
    # Discrete colormap
    num_colors = 10  # Number of colors in the discrete colormap
    cmap = plt.cm.inferno  # Inferno colormap
    bounds = np.linspace(np.min(colors), np.max(colors), num_colors + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
 
    # Create PatchCollection with discrete colors
    p = PatchCollection(patches, cmap=cmap, norm=norm) #, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
 
    # Adding a colorbar for the signed distance
    cb = plt.colorbar(p, ax=ax, boundaries=bounds)
    # set the x and y limi 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    cb.set_label("Signed Distance")
 
    plt.title("Signed Distance Function \n $\Omega$ is hatched")
 
    return ax
 
 
class SDFDomain:
    def __init__(self, bbox, func, covering=None):
        self.bbox = bbox
        self.domain = func
        self.covering = covering
 
    def eval(self, x):
        return self.domain(x)
 
    def plot(self, grid_size=100):
        ax = _plot(self, grid_size=grid_size)
        return ax

    @staticmethod
    def to_xarray(self, crs, grid_size=100): 
        '''
        Evaluate the signed distance function on a grid and return an xarray dataset
        '''
        x_min, x_max, y_min, y_max = self.bbox
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        # Evaluate SDF on grid and determine interior points
        signed_distance = self.eval(np.vstack((x_mesh.ravel(), y_mesh.ravel())).T).reshape(x_mesh.shape) 
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
    and a positive value if the point is outside the domain. 
    The returned function `func` becomes a method of the :class:`Domain` and is queried during
    mesh generation several times per meshing iteration.
 
    Parameters
    ----------
    coastal_geometry: a :class:`CoastalGeometry` object
        The processed data from :class:`CoastalGeometry` either in the 
        form of a Python class or a GeoDataFrame.
    invert: boolean, optional
        Invert the definition of the domain.
 
    Returns
    -------
    domain: a :class:`Domain` object
        Contains a callback to the signed distance function along with an extent `bbox`
 
    """
    logger.info("Building a signed distance function...")
 
    assert isinstance(coastal_geometry, (CoastalGeometry, gpd.GeoDataFrame)), "coastal_geometry is not a CoastGeometry or GeoDataFrame object"
    
    if isinstance(coastal_geometry, gpd.GeoDataFrame):
        bbox = coastal_geometry.total_bounds
        # change the order of the bbox since it's minx maxx miny maxy
        bbox = (bbox[0], bbox[2], bbox[1], bbox[3])
        # group all the polygons with the label inner into a nan separated array
        inner = coastal_geometry[coastal_geometry["labels"] == "inner"]
        inner = polygons_to_numpy(inner)
        
        region_polygon = coastal_geometry[coastal_geometry['labels']=='outer']
        region_polygon = polygons_to_numpy(region_polygon)
    
    elif isinstance(coastal_geometry, CoastalGeometry):
        bbox = coastal_geometry.bbox
        inner = coastal_geometry.inner
        region_polygon = coastal_geometry.region_polygon
    
    # create a bounding box for the domain
    boubox = _create_boubox(bbox)
    # add a row of nans to separate the polygons
    boubox = np.vstack((boubox, np.array([nan, nan])))
    boubox = np.asarray(boubox)
    e_box = get_poly_edges(boubox)
 
    # combine the inner and mainland polygons 
    # handle case of no inner 
    if inner is None:
        poly = region_polygon
    else:
        # region_polygon is outer boundary
        poly = np.vstack((inner, region_polygon))
    # create a kdtree for fast nearest neighbor search
    tree = scipy.spatial.cKDTree(
        poly[~np.isnan(poly[:, 0]), :], balanced_tree=False, leafsize=50
    )
    # edges of the polygon
    e = get_poly_edges(poly)
 
    def func(x):
        # Initialize d with some positive number larger than geps
        dist = np.zeros(len(x)) + 1.0
        # are points inside the polygon (boubox) indicating the domain?
        in_boubox, _ = inpoly2(x, boubox, e_box)
        # are points inside any of the polygons representing the shoreline?
        in_shoreline, _ = inpoly2(x, np.nan_to_num(poly), e)
        # compute nearest distance to shoreline (operate in parallel workers=-1)
        d, _ = tree.query(x, k=1, workers=-1)
        # d is signed negative if inside the
        # intersection of two areas and vice versa.
        cond = np.logical_and(in_shoreline, in_boubox)
        dist = (-1) ** (cond) * d
        if invert:
            dist *= -1
        return dist
 
    poly2 = boubox
    # TODO: investigate performance by modifying the leafsize & balanced_tree
    tree2 = scipy.spatial.cKDTree(
        poly2[~np.isnan(poly2[:, 0]), :], #balanced_tree=False, leafsize=50
    )
    # this callback is used for multiscale meshing 
    def func_covering(x):
        # Initialize d with some positive number larger than geps
        dist = np.zeros(len(x)) + 1.0
        # are points inside the boubox?
        in_boubox, _ = inpoly2(x, boubox, e_box)
        # compute dist to shoreline
        d, _ = tree2.query(x, k=1, workers=-1)
        # d is signed negative if inside the
        # intersection of two areas and vice versa.
        dist = (-1) ** (in_boubox) * d
        return dist
 
    return SDFDomain(bbox, func, covering=func_covering)

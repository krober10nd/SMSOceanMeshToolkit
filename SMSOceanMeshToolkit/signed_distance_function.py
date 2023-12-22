import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from inpoly import inpoly2

from .geospatial_data import CoastalGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "signed_distance_function",
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



def _generate_samples(bbox, N):
    N = int(N)
    points = []
    _xrange = (bbox[0] - 0.01, bbox[1] + 0.01)
    _yrange = (bbox[2] - 0.01, bbox[3] + 0.01)
    points.append(
        [
            (
                random.uniform(*_xrange),
                random.uniform(*_yrange),
            )
            for i in range(N)
        ]
    )
    points = np.asarray(points)
    points = points.reshape(-1, 2)
    return points


def _plot(geo, filename=None, samples=100000):
    p = _generate_samples(geo.bbox, N=samples)
    d = geo.eval(p)
    ix = np.logical_and(d > -0.0001, d < 0.0001)

    fig = plt.figure()
    ax = fig.add_subplot(111)  
    im = ax.scatter(p[ix, 0], p[ix, 1], c=d[ix], marker=".", s=5.0)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    plt.title("Approximate 0-level set")
    fig.colorbar(im, ax=ax)
    im.set_clim(-0.001, 0.001)
    ax.set_aspect("auto")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


class Domain:
    def __init__(self, bbox, func, covering=None):
        self.bbox = bbox
        self.domain = func
        self.covering = covering

    def eval(self, x):
        return self.domain(x)

    def plot(self, filename='my_sdf.png', samples=10000):
        _plot(self, filename=filename, samples=samples)


def signed_distance_function(coastal_geometry, invert=False):
    """
    Takes a :class:`CoastalGeometry` object containing polygons representing meshing boundaries
    and creates a callable signed distance function.
    
    The returned function `func` becomes a bound method of the :class:`Domain` and is queried during
    mesh generation several times per iteration.

    Parameters
    ----------
    coastal_geometry: a :class:`CoastalGeometry` object
        The processed shapefile data from :class:`Geodata`
    invert: boolean, optional
        Invert the definition of the domain.

    Returns
    -------
    domain: a :class:`Domain` object
        Contains a signed distance function along with an extent `bbox`

    """
    logger.info("Building a signed distance function...")

    assert isinstance(coastal_geometry, CoastalGeometry), "coastal_geometry is not a CoastGeometry object"
    # Note to self what happens if no islands? 
    poly = np.vstack((coastal_geometry.inner, coastal_geometry.boubox))
    # create a kdtree for fast nearest neighbor search
    tree = scipy.spatial.cKDTree(
        poly[~np.isnan(poly[:, 0]), :], balanced_tree=False, leafsize=50
    )
    e = get_poly_edges(poly)

    boubox = np.nan_to_num(coastal_geometry.region_polygon)
    e_box = get_poly_edges(coastal_geometry.region_polygon)

    def func(x):
        # Initialize d with some positive number larger than geps
        dist = np.zeros(len(x)) + 1.0
        # are points inside the polygon indicating the domain?
        in_boubox, _ = inpoly2(x, boubox, e_box)
        # are points inside any of the polygons?
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

    poly2 = coastal_geometry.boubox
    # TODO: investigate performance by modifying the leafsize & balanced_tree
    tree2 = scipy.spatial.cKDTree(
        poly2[~np.isnan(poly2[:, 0]), :], balanced_tree=False, leafsize=50
    )

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

    return Domain(coastal_geometry.bbox, func, covering=func_covering)

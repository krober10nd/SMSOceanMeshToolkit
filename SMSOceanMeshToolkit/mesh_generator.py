# stdlib
import logging
import warnings
import time

# tpl
import numpy as np
import scipy.sparse as spsparse

# local
from .signed_distance_function import signed_distance_function as Domain
from .Grid import Grid
from .edges import unique_edges
from _delaunay_class import DelaunayTriangulation as DT


logger = logging.getLogger(__name__)

__all__ = ["generate_mesh", "fix_mesh", "unique_rows", "simp_vol", "simp_qual"]



def simp_qual(p, t):
    """
    Simplex quality radius-to-edge ratio
    
    :param p: vertex coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]
    :return: signed mesh quality: signed mesh quality (1.0 is perfect)
    :rtype: numpy.ndarray[`float` x 1]
    """
    assert p.ndim == 2 and t.ndim == 2 and p.shape[1] + 1 == t.shape[1]

    def length(p1):
        return np.sqrt((p1**2).sum(1))

    a = length(p[t[:, 1]] - p[t[:, 0]])
    b = length(p[t[:, 2]] - p[t[:, 0]])
    c = length(p[t[:, 2]] - p[t[:, 1]])
    # Suppress Runtime warnings here because we know that mult1/denom1 can be negative
    # as the mesh is being cleaned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mult1 = (b + c - a) * (c + a - b) * (a + b - c) / (a + b + c)
        denom1 = np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
        r = 0.5 * mult1
        R = a * b * c / denom1
        return 2 * r / R


def fix_mesh(p, t, ptol=2e-13, dim=2, delete_unused=False):
    """
    Remove duplicated/unused vertices and entities and
    ensure orientation of entities is CCW.
    
    :param p: point coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]
    :param ptol: point tolerance to detect duplicates
    :type ptol: `float`, optional
    :param dim: dimension of mesh
    :type dim: `int`, optional
    :param delete_unused: flag to delete disjoint vertices.
    :type delete_unused: `boolean`, optional
    :return: p: updated point coordinates of mesh
    :rtype: numpy.ndarray[`float` x dim]
    :return: t: updated mesh connectivity
    :rtype: numpy.ndarray[`int` x (dim+1)]
    """

    # duplicate vertices
    snap = (p.max(0) - p.min(0)).max() * ptol
    _, ix, jx = unique_rows(np.round(p / snap) * snap, True, True)

    p = p[ix]
    t = jx[t]

    # duplicate entities
    t = np.sort(t, axis=1)
    t = unique_rows(t)

    # delete disjoint vertices
    if delete_unused:
        pix, _, jx = np.unique(t, return_index=True, return_inverse=True)
        t = np.reshape(jx, (t.shape))
        p = p[pix, :]

    # entity orientation is CCW
    flip = simp_vol(p, t) < 0
    t[flip, :2] = t[flip, 1::-1]

    return p, t, jx


def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]
    
    :param  A: array of data
    :type A: numpy.ndarray[`int`/`float` x N]
    :param return_index: whether to return the indices of unique data
    :type return_index: `boolean`, optional
    :param return_inverse: whether to return the inverse mapping back to A from B.
    :type return_inverse: `boolean`, optional
    :return: B: array of data with duplicates removed
    :rtype: numpy.ndarray[`int`/`float` x N]
    :return: I: array of indices to unique data B.
    :rtype: numpy.ndarray[`int` x 1]
    :return: J: array of indices to A from B.
    :rtype: numpy.ndarray[`int` x 1]
    """
    A = np.require(A, requirements="C")
    assert A.ndim == 2, "array must be 2-dim'l"

    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize * ncolumns))
    B, I, J = np.unique(A.view(dtype), return_index=True, return_inverse=True)

    B = B.view(orig_dtype).reshape((-1, ncolumns), order="C")

    # There must be a better way to do this:
    if return_index:
        if return_inverse:
            return B, I, J
        else:
            return B, I
    else:
        if return_inverse:
            return B, J
        else:
            return B


def simp_vol(p, t):
    """
    Signed volumes of the simplex elements in the mesh.
    :param p: point coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]
    :return: volume: signed volume of entity/simplex.
    :rtype: numpy.ndarray[`float` x 1]
    """

    dim = p.shape[1]
    if dim == 1:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        return d01
    elif dim == 2:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        d02 = p[t[:, 2]] - p[t[:, 0]]
        return (d01[:, 0] * d02[:, 1] - d01[:, 1] * d02[:, 0]) / 2
    elif dim == 3:
        d01 = p[t[:, 1], :] - p[t[:, 0], :]
        d02 = p[t[:, 2], :] - p[t[:, 0], :]
        d03 = p[t[:, 3], :] - p[t[:, 0], :]
        return np.einsum("ij,ij->i", np.cross(d01, d02), d03) / 6
    else:
        raise NotImplementedError


def _parse_kwargs(kwargs):
    for key in kwargs:
        if key in {
            "nscreen",
            "max_iter",
            "seed",
            "pfix",
            "points",
            "domain",
            "edge_length",
            "bbox",
            "min_edge_length",
            "plot",
            "pseudo_dt",
        }:
            pass
        else:
            raise ValueError(
                "Option %s with parameter %s not recognized " % (key, kwargs[key])
            )


def _check_bbox(bbox):
    assert isinstance(bbox, tuple), "`bbox` must be a tuple"
    assert int(len(bbox) / 2), "`dim` must be 2"


def generate_mesh(domain, edge_length, **kwargs):
    """
    Generate a 2D triangular mesh using callbacks to a
    mesh sizing function `edge_length` and signed distance function `domain`

    Parameters
    ----------
    domain: A function object.
        A function that takes a point and returns the signed nearest distance to the domain boundary Ω.
    edge_length: A function object.
        A function that can evalulate a point and return a mesh size.
    \**kwargs:
        See below

    :Keyword Arguments:
        * *bbox* (``tuple``) --
            Bounding box containing domain extents. REQUIRED IF NOT USING :class:`edge_length`
        * *max_iter* (``float``) --
            Maximum number of meshing iterations. (default==50)
        * *seed* (``float`` or ``int``) --
            Pseudo-random seed to initialize meshing points. (default==0)
        * *pfix* (`array-like`) --
            An array of points to constrain in the mesh. (default==None)
        * *min_edge_length* (``float``) --
            The minimum element size in the domain. REQUIRED IF NOT USING :class:`edge_length`
        * *plot* (``int``) --
            The mesh is visualized every `plot` meshing iterations.
        * *pseudo_dt* (``float``) --
            The pseudo time step for the meshing algorithm. (default==0.2)

    Returns
    -------
    points: array-like
        vertex coordinates of mesh
    t: array-like
        mesh connectivity table.

    """
    _DIM = 2
    opts = {
        "max_iter": 50,
        "seed": 0,
        "pfix": None,
        "points": None,
        "min_edge_length": None,
        "plot": 999999,
        "pseudo_dt": 0.2,
    }
    opts.update(kwargs)
    _parse_kwargs(kwargs)

    fd, bbox = _unpack_domain(domain, opts)
    fh, min_edge_length = _unpack_sizing(edge_length, opts)

    _check_bbox(bbox)
    bbox = np.array(bbox).reshape(-1, 2)

    assert min_edge_length > 0, "`min_edge_length` must be > 0"

    assert opts["max_iter"] > 0, "`max_iter` must be > 0"
    max_iter = opts["max_iter"]

    np.random.seed(opts["seed"])

    L0mult = 1 + 0.4 / 2 ** (_DIM - 1)
    delta_t = opts["pseudo_dt"]
    geps = 1e-3 * np.amin(min_edge_length)
    deps = np.sqrt(np.finfo(np.double).eps)  # * np.amin(min_edge_length)

    pfix, nfix = _unpack_pfix(_DIM, opts)

    if opts["points"] is None:
        p = _generate_initial_points(
            min_edge_length,
            geps,
            bbox,
            fh,
            fd,
            pfix,
        )
    else:
        p = opts["points"]

    N = p.shape[0]

    assert N > 0, "No vertices to mesh with!"

    logger.info(
        f"Commencing mesh generation with {N} vertices will perform {max_iter} iterations."
    )
    for count in range(max_iter):
        start = time.time()

        # (Re)-triangulation by the Delaunay algorithm
        dt = DT()
        dt.insert(p.ravel().tolist())

        # Get the current topology of the triangulation
        p, t = _get_topology(dt)

        ifix = []
        # Find where pfix went
        if nfix > 0:
            for fix in pfix:
                ind = _closest_node(fix, p)
                ifix.append(ind)
                p[ind] = fix

        # Remove points outside the domain
        t = _remove_triangles_outside(p, t, fd, geps)

        # Number of iterations reached, stop.
        if count == (max_iter - 1):
            p, t, _ = fix_mesh(p, t, dim=_DIM, delete_unused=True)
            logger.info("Termination reached...maximum number of iterations.")
            return p, t

        # Compute the forces on the bars
        Ftot = _compute_forces(p, t, fh, min_edge_length, L0mult, opts)

        # Force = 0 at fixed points
        Ftot[:nfix] = 0

        # Update positions
        p += delta_t * Ftot

        # Bring outside points back to the boundary
        p = _project_points_back(p, fd, deps)

        # Show the user some progress so they know something is happening
        maxdp = delta_t * np.sqrt((Ftot**2).sum(1)).max()

        logger.info(
            f"Iteration #{count + 1}, max movement is {maxdp}, there are {len(p)} vertices and {len(t)}"
        )

        end = time.time()
        logger.info(f"Elapsed wall-clock time {end - start} seconds")


def _unpack_sizing(edge_length, opts):
    if isinstance(edge_length, Grid):
        fh = edge_length.eval
        min_edge_length = edge_length.hmin
    elif callable(edge_length):
        fh = edge_length
        min_edge_length = opts["min_edge_length"]
    else:
        raise ValueError(
            "`edge_length` must either be a function or a `edge_length` object"
        )
    return fh, min_edge_length


def _unpack_domain(domain, opts):
    if isinstance(domain, Domain):
        bbox = domain.bbox
        fd = domain.eval
    elif callable(domain):
        bbox = opts["bbox"]
        fd = domain
    else:
        raise ValueError(
            "`domain` must be a function or a :class:`signed_distance_function object"
        )
    return fd, bbox


def _get_bars(t):
    """Describe each bar by a unique pair of nodes"""
    bars = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return unique_edges(bars)


# Persson-Strang
def _compute_forces(p, t, fh, min_edge_length, L0mult, opts):
    """Compute the forces on each edge based on the sizing function"""
    N = p.shape[0]
    bars = _get_bars(t)
    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
    L = np.sqrt((barvec**2).sum(1))  # L = Bar lengths
    L[L == 0] = np.finfo(float).eps
    hbars = fh(p[bars].sum(1) / 2)
    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
    F = L0 - L
    F[F < 0] = 0  # Bar forces (scalars)
    Fvec = (
        F[:, None] / L[:, None].dot(np.ones((1, 2))) * barvec
    )  # Bar forces (x,y components)
    Ftot = _dense(
        bars[:, [0] * 2 + [1] * 2],
        np.repeat([list(range(2)) * 2], len(F), axis=0),
        np.hstack((Fvec, -Fvec)),
        shape=(N, 2),
    )
    return Ftot


# Bossen-Heckbert
# def _compute_forces(p, t, fh, min_edge_length, L0mult):
#    """Compute the forces on each edge based on the sizing function"""
#    N = p.shape[0]
#    bars = _get_bars(t)
#    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
#    L = np.sqrt((barvec ** 2).sum(1))  # L = Bar lengths
#    L[L == 0] = np.finfo(float).eps
#    hbars = fh(p[bars].sum(1) / 2)
#    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
#    LN = L / L0
#    F = (1 - LN ** 4) * np.exp(-(LN ** 4)) / LN
#    Fvec = (
#        F[:, None] / LN[:, None].dot(np.ones((1, 2))) * barvec
#    )  # Bar forces (x,y components)
#    Ftot = _dense(
#        bars[:, [0] * 2 + [1] * 2],
#        np.repeat([list(range(2)) * 2], len(F), axis=0),
#        np.hstack((Fvec, -Fvec)),
#        shape=(N, 2),
#    )
#    return Ftot


def _dense(Ix, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...), but instead returning a
    dense array.
    """

    # Advanced usage: allow J and S to be scalars.
    if np.isscalar(J):
        x = J
        J = np.empty(Ix.shape, dtype=int)
        J.fill(x)
    if np.isscalar(S):
        x = S
        S = np.empty(Ix.shape)
        S.fill(x)

    # Turn these into 1-d arrays for processing.
    S = S.flat
    II = Ix.flat
    J = J.flat
    return spsparse.coo_matrix((S, (II, J)), shape, dtype).toarray()


def _remove_triangles_outside(p, t, fd, geps):
    """Remove vertices outside the domain"""
    pmid = p[t].sum(1) / 3  # Compute centroids
    return t[fd(pmid) < -geps]  # Keep interior triangles


def _project_points_back(p, fd, deps):
    """Project points outsidt the domain back within"""
    d = fd(p)
    ix = d > 0  # Find points outside (d>0)
    if ix.any():

        def _deps_vec(i):
            a = [0] * 2
            a[i] = deps
            return a

        try:
            dgrads = [
                (fd(p[ix] + _deps_vec(i)) - d[ix]) / deps for i in range(2)
            ]  # old method
        except ValueError:  # an error is thrown if all points in fd are outside
            # bbox domain, so instead calulate all fd and then
            # take the solely ones outside domain
            dgrads = [(fd(p + _deps_vec(i)) - d) / deps for i in range(2)]
            dgrads = list(np.array(dgrads)[:, ix])
        dgrad2 = sum(dgrad**2 for dgrad in dgrads)
        dgrad2 = np.where(dgrad2 < deps, deps, dgrad2)
        p[ix] -= (d[ix] * np.vstack(dgrads) / dgrad2).T  # Project
    return p


def _generate_initial_points(min_edge_length, geps, bbox, fh, fd, pfix, stereo=False):
    """Create initial distribution in bounding box (equilateral triangles)"""
    if stereo:
        bbox = np.array([[-180, 180], [-89, 89]])
    p = np.mgrid[
        tuple(slice(min, max + min_edge_length, min_edge_length) for min, max in bbox)
    ].astype(float)
    p = p.reshape(2, -1).T
    r0 = fh(p)
    r0m = np.min(r0[r0 >= min_edge_length])
    p = p[np.random.rand(p.shape[0]) < r0m**2 / r0**2]
    p = p[fd(p) < geps]  # Keep only d<0 points
    return np.vstack(
        (
            pfix,
            p,
        )
    )


def _dist(p1, p2):
    """Euclidean distance between two sets of points"""
    return np.sqrt(((p1 - p2) ** 2).sum(1))


def _unpack_pfix(dim, opts):
    """Unpack fixed points"""
    pfix = np.empty((0, dim))
    nfix = 0
    if opts["pfix"] is not None:
        pfix = np.array(opts["pfix"], dtype="d")
        nfix = len(pfix)
        logger.info(f"Constraining {nfix} fixed points..")
    return pfix, nfix


def _get_topology(dt):
    """Get points and entities from :clas:`CGAL:DelaunayTriangulation2/3` object"""
    return dt.get_finite_vertices(), dt.get_finite_cells()


def _closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)

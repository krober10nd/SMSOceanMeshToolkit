import copy
import logging

import numpy as np
import scipy.sparse as spsparse

import warnings

from . import edges

logger = logging.getLogger(__name__)

__all__ = [
    "make_mesh_boundaries_traversable",
    "delete_interior_faces",
    "delete_exterior_faces",
    "delete_faces_connected_to_one_face",
    "delete_boundary_faces",
    "laplacian2",
    "mesh_clean",
    "unique_rows",
    "simp_vol",
    "simp_qual",
    "fix_mesh",
]


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

def simp_qual(p, t):
    """Simplex quality radius-to-edge ratio

    :param p: vertex coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]

    :return: signed mesh quality: signed mesh quality (1.0 is perfect)
    :rtype: numpy.ndarray[`float` x 1]

    """
    assert p.ndim == 2 and t.ndim == 2 and p.shape[1] + 1 == t.shape[1]

    def length(p1):
        return np.sqrt((p1 ** 2).sum(1))

    a = length(p[t[:, 1]] - p[t[:, 0]])
    b = length(p[t[:, 2]] - p[t[:, 0]])
    c = length(p[t[:, 2]] - p[t[:, 1]])
    r = 0.5 * np.sqrt((b + c - a) * (c + a - b) * (a + b - c) / (a + b + c))
    # suppress any divide by 0 warnings 
    warnings.filterwarnings("ignore")
    R = a * b * c / np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
    return 2 * r / R




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


def mesh_clean(
    points,
    cells,
    min_element_qual=0.01,
    min_percent_disconnected_area=0.05,
    max_iter=20,
    tol=0.01,
    pfix=None,
):
    """Clean a mesh by removing bad quality elements and boundary faces.

    Parameters
    ----------
    points : array-like
        Mesh vertices.
    cells : array-like
        Mesh connectivity table.
    min_element_qual : float, optional
        Enforce Mmnimum quality of elements through deletion. The default is 0.01.
    min_percent_disconnected_area : float, optional
        Delete disconnected areas smaller than this threshold. The default is 0.05 x
        the total area of the mesh.
    max_iter : int, optional
        Maximum number of iterations for the Laplacian smoothing. The default is 20.
    tol : float, optional
        Tolerance for the Laplacian smoothing. The default is 0.01. Laplacian terminates
        after max_iter or when the maximum change in any vertex is less than tol.
    pfix : array-like, optional
        Indices of points to fix during the Laplacian smoothing. The default is None.

    Returns
    -------
    points : array-like
        Mesh vertices cleaned.
    cells : array-like
        Mesh connectivity table cleaned.

    """
    points, cells = make_mesh_boundaries_traversable(
        points, cells, min_disconnected_area=min_percent_disconnected_area
    )
    if min_element_qual > 0:
        points, cells = delete_boundary_faces(points, cells, min_qual=min_element_qual)
    points, cells = delete_faces_connected_to_one_face(points, cells)
    if max_iter > 0:
        points, cells = laplacian2(points, cells, max_iter=max_iter, tol=tol, pfix=pfix)

    points, cells = make_mesh_boundaries_traversable(
        points, cells, min_disconnected_area=min_percent_disconnected_area
    )
    points, cells, _ = fix_mesh(points, cells, delete_unused=True)
    return points, cells


def _arg_sortrows(arr):
    """Before a multi column sort like MATLAB's sortrows"""
    i = arr[:, 1].argsort()  # First sort doesn't need to be stable.
    j = arr[i, 0].argsort(kind="mergesort")
    return i[j]


def _face_to_face(t):
    """Face to face connectivity table.
        Face `i` is connected to faces `ftof[ix[i]:ix[i+1], 1]`
        By connected, I mean shares a mutual edge.

    Parameters
    ----------
    t: array-like
        Mesh connectivity table.

    Returns
    -------
    ftof: array-like
        Face numbers connected to faces.
    ix: array-like
        indices into `ftof`

    """
    nt = len(t)
    t = np.sort(t, axis=1)
    e = t[:, [[0, 1], [0, 2], [1, 2]]].reshape((nt * 3, 2))
    trinum = np.repeat(np.arange(nt), 3)
    j = _arg_sortrows(e)
    e = e[j, :]
    trinum = trinum[j]
    k = np.argwhere(~np.diff(e, axis=0).any(axis=1))
    ftof = np.concatenate((trinum[k], trinum[k + 1]), axis=1)
    dmy1 = ftof[:, 0].argsort()
    dmy2 = ftof[:, 1].argsort()
    tmp = np.vstack(
        (
            ftof[dmy1, :],
            np.fliplr(ftof[dmy2]),
            np.column_stack((np.arange(nt), np.arange(nt))),
        )
    )
    j = _arg_sortrows(tmp)
    ftof = tmp[j, :]
    ix = np.argwhere(np.diff(ftof[:, 0])) + 1
    ix = np.insert(ix, 0, 0)
    ix = np.append(ix, len(ftof))
    return ftof, ix


def _vertex_to_face(vertices, faces):
    """Determine which faces are connected to which vertices.

     Parameters
    ----------
    vertices: array-like
        Vertices of the mesh.
    faces: array-like
        Mesh connectivity table.

    Returns
    -------
    vtoc: array-like
        face numbers connected to vertices.
    ix: array-like
        indices into `vtoc`

    """
    num_faces = len(faces)

    ext = np.tile(np.arange(0, num_faces), (3, 1)).reshape(-1, order="F")
    ve = np.reshape(faces, (-1,))
    ve = np.vstack((ve, ext)).T
    ve = ve[ve[:, 0].argsort(), :]

    idx = np.insert(np.diff(ve[:, 0]), 0, 0)
    vtoc_pointer = np.argwhere(idx)
    vtoc_pointer = np.insert(vtoc_pointer, 0, 0)
    vtoc_pointer = np.append(vtoc_pointer, num_faces * 3)

    vtoc = ve[:, 1]

    return vtoc, vtoc_pointer


def make_mesh_boundaries_traversable(vertices, faces, min_disconnected_area=0.05):
    """
    A mesh described by vertices and faces is  "cleaned" and returned.
    Alternates between checking "interior" and "exterior" portions
    of the mesh until convergence is obtained. Convergence is defined as:
    having no vertices connected to more than two boundary edges.

    Parameters
    ----------
    vertices: array-like
        The vertices of the "uncleaned" mesh.
    faces: array-like
        The "uncleaned" mesh connectivity.
    min_disconnected_area: float, optional
        A decimal percentage (max 1.0) used to decide whether to keep or remove
        disconnected portions of the meshing domain.


    Returns
    -------
    vertices: array-like
        The vertices of the "cleaned" mesh.

    faces: array-like
        The "cleaned" mesh connectivity.

    Notes
    -----

    Interior Check: Deletes faces that are within the interior of the
    mesh so that no vertices are connected to more than two boundary edges. For
    example, a barrier island could become very thin in a middle portion so that you
    have a vertex connected to two faces but four boundary edges, in a
    bow-tie type formation.

    This code will delete one of those connecting
    faces to ensure the spit is `clean` in the sense that two boundary edges
    are connected to that vertex. In the case of a choice between faces to
    delete, the one with the lowest quality is chosen.

    Exterior Check: Finds small disjoint portions of the mesh and removes
    them using a depth-first search. The individual disjoint portions are
    removed based on `min_disconnected_area` which is a decimal representing a fractional
    threshold component of the total mesh.

    """

    boundary_edges, boundary_vertices = _external_topology(vertices, faces)

    logger.info("Performing mesh cleaning operations...")
    # NB: when this inequality is not met, the mesh boundary is  not valid and non-manifold
    while len(boundary_edges) > len(boundary_vertices):
        faces = delete_exterior_faces(vertices, faces, min_disconnected_area)
        vertices, faces, _ = fix_mesh(vertices, faces, delete_unused=True)

        faces, _ = delete_interior_faces(vertices, faces)
        vertices, faces, _ = fix_mesh(vertices, faces, delete_unused=True)

        boundary_edges, boundary_vertices = _external_topology(vertices, faces)

    return vertices, faces


def _external_topology(vertices, faces):
    """Get edges and vertices that make up the boundary of the mesh"""
    boundary_edges = edges.get_boundary_edges(faces)
    boundary_vertices = vertices[np.unique(boundary_edges.reshape(-1))]
    return boundary_edges, boundary_vertices


def delete_exterior_faces(vertices, faces, min_disconnected_area):
    """Deletes portions of the mesh that are "outside" or not
    connected to the majority which represent a fractional
    area less than `min_disconnected_area`.
    """
    t1 = copy.deepcopy(faces)
    t = np.array([])
    # Calculate the total area of the patch
    A = np.sum(simp_vol(vertices, faces))
    An = A
    # Based on area proportion
    while (An / A) > min_disconnected_area:
        # Perform the depth-First-Search to get `nflag`
        nflag = _depth_first_search(t1)

        # Get new triangulation and its area
        t2 = t1[nflag == 1, :]
        An = np.sum(simp_vol(vertices, t2))

        # If large enough, retain this component
        if (An / A) > min_disconnected_area:
            if len(t) == 0:
                t = t2
            else:
                t = np.concatenate((t, t2))

        # Delete where nflag == 1 from tmp t1 mesh
        t1 = np.delete(t1, nflag == 1, axis=0)
        logger.info(
            f"ACCEPTED: Deleting {int(np.sum(nflag == 0))} faces outside the main mesh"
        )

        # Calculate the remaining area
        An = np.sum(simp_vol(vertices, t1))

    return t


def delete_interior_faces(vertices, faces):
    """Delete interior faces that have vertices with more than
    two vertices declared as boundary vertices
    """
    # Get updated boundary topology
    boundary_edges, boundary_vertices = _external_topology(vertices, faces)
    etbv = boundary_edges.reshape(-1)
    # Count how many edges a vertex appears in.
    uebtv, count = np.unique(etbv, return_counts=True)
    # Get the faces connected to the vertices
    vtoc, nne = _vertex_to_face(vertices, faces)
    # Vertices which appear more than twice (implying they are shared by
    # more than two boundary edges)
    del_face_idx = []
    for ix in uebtv[count > 2]:
        conn_faces = vtoc[nne[ix] : nne[ix + 1]]
        del_face = []
        for conn_face in conn_faces:
            II = etbv == faces[conn_face, 0]
            JJ = etbv == faces[conn_face, 1]
            KK = etbv == faces[conn_face, 2]
            if np.any(II) and np.any(JJ) and np.any(KK):
                del_face.append(conn_face)

        if len(del_face) == 1:
            del_face_idx.append(del_face[0])
        elif len(del_face) > 1:
            # Delete worst quality qualifying face.
            qual = simp_qual(vertices, faces[del_face])
            idx = np.argmin(qual)
            del_face_idx.append(del_face[idx])
        else:
            # No connected faces have all vertices on boundary edge so we
            # select the worst quality connecting face.
            qual = simp_qual(vertices, faces[conn_faces])
            idx = np.argmin(qual)
            del_face_idx.append(conn_faces[idx])

    logger.info(f"ACCEPTED: Deleting {len(del_face_idx)} faces inside the main mesh")
    faces = np.delete(faces, del_face_idx, 0)

    return faces, del_face_idx


def _depth_first_search(faces):
    """Depth-First-Search (DFS) across the triangulation"""

    # Get graph connectivity.
    ftof, idx = _face_to_face(faces)

    nt = len(faces)

    # select a random face
    selected = np.random.randint(0, nt, 1)

    nflag = np.zeros(nt)

    searching = True

    visited = []
    visited.append(*selected)

    # Traverse through connected mesh
    while searching:
        searching = False
        for c in visited:
            # Flag the current face as visited
            nflag[c] = 1
            # Search connected faces
            neis = [nei for nei in ftof[idx[c] : idx[c + 1], 1]]
            # Flag connected faces as visited
            for nei in neis:
                if nflag[nei] == 0:
                    nflag[nei] = 1
                    # Append visited cells to a list
                    visited.append(nei)
                    searching = True
    return nflag


def delete_faces_connected_to_one_face(vertices, faces, max_iter=np.inf):
    """Iteratively deletes faces connected to one face.

    Parameters
    ----------
    vertices: array-like
        The vertices of the "uncleaned" mesh.
    faces: array-like
        The "uncleaned" mesh connectivity.
    max_iter: float, optional
        The number of iterations to repeatedly delete faces connected to one face
        If the mesh is well-formed, this will converge in a finite
        number of iterations. Default is np.inf.

    Returns
    -------
    vertices: array-like
        The vertices of the "cleaned" mesh.

    faces: array-like
        The "cleaned" mesh connectivity.

    """
    assert max_iter > 0, "max_iter set too low"

    count = 0
    start_len = len(faces)
    while count < max_iter:
        _, idx = _face_to_face(faces)
        nn = np.diff(idx, 1)
        delete = np.argwhere(nn == 2)
        if len(delete) > 0:
            logger.info(f"ACCEPTED: Deleting {int(len(delete))} faces")
            faces = np.delete(faces, delete, axis=0)
            vertices, faces, _ = fix_mesh(vertices, faces, delete_unused=True)
            count += 1
        else:
            break
        logger.info(
            f"Deleted {int(start_len - len(faces))} faces after {int(count)} iterations"
        )
    return vertices, faces


def _sparse(Ix, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...)
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
    return spsparse.coo_matrix((S, (II, J)), shape, dtype)


def laplacian2(vertices, entities, max_iter=20, tol=0.01, pfix=None):
    """Move vertices to the average position of their connected neighbors
    with the goal to hopefully improve geometric entity quality.
    :param vertices: vertex coordinates of mesh
    :type vertices: numpy.ndarray[`float` x dim]
    :param entities: the mesh connectivity
    :type entities: numpy.ndarray[`int` x (dim+1)]
    :param max_iter: maximum number of iterations to perform
    :type max_iter: `int`, optional
    :param tol: iterations will cease when movement < tol
    :type tol: `float`, optional
    :param pfix: coordinates that you don't wish to move
    :type pfix: array-like
    :return vertices: updated vertices of mesh
    :rtype: numpy.ndarray[`float` x dim]
    :return: entities: updated mesh connectivity
    :rtype: numpy.ndarray[`int` x (dim+1)]
    """
    if vertices.ndim != 2:
        raise NotImplementedError("Laplacian smoothing only works in 2D for now")

    def _closest_node(node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum("ij,ij->i", deltas, deltas)
        return np.argmin(dist_2)

    eps = np.finfo(float).eps

    n = len(vertices)

    S = _sparse(
        entities[:, [0, 0, 1, 1, 2, 2]],
        entities[:, [1, 2, 0, 2, 0, 1]],
        1,
        shape=(n, n),
    )
    # bnd = get_boundary_vertices(entities)
    edge = edges.get_edges(entities)
    boundary_edges, _ = _external_topology(vertices, entities)
    bnd = np.unique(boundary_edges.reshape(-1))
    if pfix is not None:
        ifix = []
        for fix in pfix:
            ifix.append(_closest_node(fix, vertices))
        ifix = np.asarray(ifix)
        bnd = np.concatenate((bnd, ifix))

    W = np.sum(S, 1)
    if np.any(W == 0):
        print("Invalid mesh. Disjoint vertices found. Returning", flush=True)
        print(np.argwhere(W == 0), flush=True)
        return vertices, entities

    L = np.sqrt(
        np.sum(np.square(vertices[edge[:, 0], :] - vertices[edge[:, 1], :]), axis=1)
    )
    L[L < eps] = eps
    L = L[:, None]
    for it in range(max_iter):
        pnew = np.divide(S * np.matrix(vertices), np.hstack((W, W)))
        pnew[bnd, :] = vertices[bnd, :]
        vertices = pnew
        Lnew = np.sqrt(
            np.sum(np.square(vertices[edge[:, 0], :] - vertices[edge[:, 1], :]), axis=1)
        )
        Lnew[Lnew < eps] = eps
        move = np.amax(np.divide((Lnew - L), Lnew))
        if move < tol:
            logger.info(f"Movement tolerance reached after {it} iterations..exiting")
            break
        L = Lnew
    vertices = np.array(vertices)
    return vertices, entities


def get_boundary_entities(vertices, entities, dim=2):
    """Determine the entities that lie on the boundary of the mesh.
    :param vertices: vertex coordinates of mesh
    :type vertices: numpy.ndarray[`float` x dim]
    :param entities: the mesh connectivity
    :type entities: numpy.ndarray[`int` x (dim+1)]
    :param dim: dimension of the mesh
    :type dim: `int`, optional
    :return: bele: indices of entities on the boundary of the mesh.
    :rtype: numpy.ndarray[`int` x 1]
    """
    boundary_edges, _ = _external_topology(vertices, entities)
    boundary_vertices = np.unique(boundary_edges.reshape(-1))
    vtoe, ptr = _vertex_to_face(vertices, entities)
    bele = np.array([], dtype=int)
    for vertex in boundary_vertices:
        for ele in zip(vtoe[ptr[vertex] : ptr[vertex + 1]]):
            bele = np.append(bele, ele)
    bele = np.unique(bele)
    return bele


def delete_boundary_faces(vertices, entities, dim=2, min_qual=0.10, verbose=1):
    """Delete boundary faces with poor geometric quality (i.e., < min. quality)
    :param vertices: vertex coordinates of mesh
    :type vertices: numpy.ndarray[`float` x dim]
    :param entities: the mesh connectivity
    :type entities: numpy.ndarray[`int` x (dim+1)]
    :param dim: dimension of the mesh
    :type dim: `int`, optional
    :param min_qual: minimum geometric quality to consider "poor" quality
    :type min_qual: `float`, optional
    :return: vertices: updated vertex array of mesh
    :rtype: numpy.ndarray[`int` x dim]
    :return: entities: update mesh connectivity
    :rtype: numpy.ndarray[`int` x (dim+1)]
    """
    qual = simp_qual(vertices, entities)
    bele, _ = _external_topology(vertices, entities)
    bele = get_boundary_entities(vertices, entities, dim=dim)
    qualBou = qual[bele]
    delete = qualBou < min_qual
    logger.info(f"Deleting {np.sum(delete)} poor quality boundary entities...")
    delete = np.argwhere(delete == 1)
    entities = np.delete(entities, bele[delete], axis=0)
    vertices, entities, _ = fix_mesh(vertices, entities, delete_unused=True, dim=dim)
    return vertices, entities

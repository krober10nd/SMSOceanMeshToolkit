import numpy as np

__all__ = ["get_poly_edges"]

nan = np.nan


def get_poly_edges(poly):
    """Given a winded polygon represented as a set of ascending line segments
    with separated features indicated by nans, this function calculates
    the edges of the polygon such that each edge indexes the start and end
    coordinates of each line segment of the polygon.

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

"""
Routines to prepare vector data for use in oceanmesh
"""
import errno
import logging
import os
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import shapely.geometry
import shapely.validation
from pyproj import CRS

from .Region import Region

nan = np.nan
fiona_version = fiona.__version__

logger = logging.getLogger(__name__)

__all__ = ["CoastalGeometry"]


def _convert_to_array(lst):
    """Converts a list of numpy arrays to a np array"""
    return np.concatenate(lst, axis=0)


def _convert_to_list(arr):
    """Converts a nan-delimited numpy array to a list of numpy arrays"""
    a = np.insert(arr, 0, [[nan, nan]], axis=0)
    tmp = [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a[:, 0]))]
    return [np.append(a, [[nan, nan]], axis=0) for a in tmp]


def _create_boubox(bbox):
    """Create a bounding box from domain extents `bbox`. Path orientation will be CCW."""
    if isinstance(bbox, tuple):
        xmin, xmax, ymin, ymax = bbox
        return [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]
    return bbox


def _create_ranges(start, stop, N, endpoint=True):
    """Vectorized alternative to numpy.linspace
    https://stackoverflow.com/questions/40624409/vectorized-np-linspace-for-multiple-start-and-stop-values
    """
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]


def _densify(poly, maxdiff, bbox, radius=0.0):
    """Fills in any gaps in latitude or longitude arrays
    that are greater than a `maxdiff` (degrees) apart
    """
    logger.debug("Entering:_densify")

    boubox = _create_boubox(bbox)
    path = mpltPath.Path(boubox, closed=True)
    inside = path.contains_points(poly, radius=radius)  # add a small radius
    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = np.abs(lat[1:] - lat[:-1])
    dlon = np.abs(lon[1:] - lon[:-1])
    nin = np.ceil(np.maximum(dlat, dlon) / maxdiff) - 1
    nin[~inside[1:]] = 0  # no need to densify outside of bbox please
    # handle negative values
    nin[nin < 0] = 0
    sumnin = np.nansum(nin)
    if sumnin == 0:
        return np.hstack((lon[:, None], lat[:, None]))
    nout = sumnin + nx

    lonout = np.full((int(nout)), nan, dtype=float)
    latout = np.full((int(nout)), nan, dtype=float)

    n = 0
    for i in range(nx - 1):
        ni = nin[i]
        if ni == 0 or np.isnan(ni):
            latout[n] = lat[i]
            lonout[n] = lon[i]
            nstep = 1
        else:
            ni = int(ni)
            icoords = _create_ranges(
                np.array([lat[i], lon[i]]),
                np.array([lat[i + 1], lon[i + 1]]),
                ni + 2,
            )
            latout[n : n + ni + 1] = icoords[0, : ni + 1]
            lonout[n : n + ni + 1] = icoords[1, : ni + 1]
            nstep = ni + 1
        n += nstep

    latout[-1] = lat[-1]
    lonout[-1] = lon[-1]

    logger.debug("Exiting:_densify")

    return np.hstack((lonout[:, None], latout[:, None]))


def _classify_shoreline(bbox, boubox, polys, h0, minimum_area_mult):
    """Classify segments in numpy.array `polys` as either `inner` or `mainland`.
    (1) The `mainland` category contains segments that are not totally enclosed inside the `bbox`.
    (2) The `inner` (i.e., islands) category contains segments totally enclosed inside the `bbox`.
        NB: Removes `inner` geometry with area < `minimum_area_mult`*`h0`**2
    (3) `boubox` polygon array will be clipped by segments contained by `mainland`.
    """
    logger.debug("Entering:_classify_shoreline")

    _AREAMIN = minimum_area_mult * h0**2

    if len(boubox) == 0:
        # if it's empty, create a boubox from the bbox
        boubox = _create_boubox(bbox)
        boubox = np.asarray(boubox)
    elif not _is_path_ccw(boubox):
        boubox = np.flipud(boubox)

    # Densify boubox to ensure that the minimum spacing along it is <= `h0` / 2
    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)

    # Remove nan's (append again at end)
    isNaN = np.sum(np.isnan(boubox), axis=1) > 0
    if any(isNaN):
        boubox = np.delete(boubox, isNaN, axis=0)
    del isNaN

    inner = np.empty(shape=(0, 2))
    inner[:] = nan
    mainland = np.empty(shape=(0, 2))
    mainland[:] = nan

    polyL = _convert_to_list(polys)
    bSGP = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        pSGP = shapely.geometry.Polygon(poly[:-2, :])
        if bSGP.contains(pSGP):
            area = pSGP.area
            if area >= _AREAMIN:
                inner = np.append(inner, poly, axis=0)
        elif pSGP.overlaps(bSGP):
            bSGP = bSGP.difference(pSGP)
            # Append polygon segment to mainland
            mainland = np.vstack((mainland, poly))
            # Clip polygon segment from boubox and regenerate path

    out = np.empty(shape=(0, 2))

    if bSGP.geom_type == "Polygon":
        # Convert to `MultiPolygon`
        bSGP = shapely.geometry.MultiPolygon([bSGP])

    # MultiPolygon members can be accessed via iterator protocol using `in` loop.
    for b in bSGP.geoms:
        xy = np.asarray(b.exterior.coords)
        xy = np.vstack((xy, xy[0]))
        out = np.vstack((out, xy, [nan, nan]))

    logger.debug("Exiting:classify_shoreline")

    return inner, mainland, out


def _chaikins_corner_cutting(coords, refinements=5):
    """
    Apply Chaikin's corner cutting algorithm to `coords`.

    Reference:
    ----------
    Chaikin, G. An algorithm for high speed curve generation. Computer Graphics and Image Processing 3 (1974), 346–349

    """
    logger.debug("Entering:_chaikins_corner_cutting")
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    logger.debug("Exiting:_chaikins_corner_cutting")
    return coords


def _smooth_vector_data(polys, N):
    """Smoothes the shoreline segment-by-segment using
    a `N` refinement Chaikins Corner cutting algorithm.
    """
    logger.debug("Entering:_smooth_vector_data")

    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        tmp = _chaikins_corner_cutting(poly[:-1], refinements=N)
        tmp = np.append(tmp, [[nan, nan]], axis=0)
        out.append(tmp)

    logger.debug("Exiting:_smooth_vector_data")

    return _convert_to_array(out)


def _clip_polys(polys, bbox, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    Dependencies: shapely.geometry and numpy
    """

    logger.debug("Entering:_clip_polys")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = np.asarray(_create_boubox(bbox))
    polyL = _convert_to_list(polys)

    out = np.empty(shape=(0, 2))

    b = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        mp = shapely.geometry.Polygon(poly[:-2, :])
        if not mp.is_valid:
            logger.warning(
                "Shapely.geometry.Polygon "
                + f"{shapely.validation.explain_validity(mp)}."
                + " Applying tiny buffer to make valid."
            )
            mp = mp.buffer(1.0e-6)  # ~0.1m
            if mp.geom_type == "Polygon":
                mp = shapely.geometry.MultiPolygon([mp])
        else:
            mp = shapely.geometry.MultiPolygon([mp])

        for p in mp.geoms:
            pi = p.intersection(b)
            if b.contains(p):
                out = np.vstack((out, poly))
            elif not pi.is_empty:
                # assert(pi.geom_type,'MultiPolygon')
                if pi.geom_type == "Polygon":
                    pi = shapely.geometry.MultiPolygon([pi])

                for ppi in pi.geoms:
                    xy = np.asarray(ppi.exterior.coords)
                    xy = np.vstack((xy, xy[0]))
                    out = np.vstack((out, xy, [nan, nan]))

                del (ppi, xy)
            del pi
        del (p, mp)

    logger.debug("Exiting:_clip_polys")

    return out


def _is_path_ccw(_p):
    """Compute curve orientation from first two line segment of a polygon.
    Source: https://en.wikipedia.org/wiki/Curve_orientation
    """
    detO = 0.0
    O3 = np.ones((3, 3))

    i = 0
    while (i + 3 < _p.shape[0]) and np.isclose(detO, 0.0):
        # Colinear vectors detected. Try again with next 3 indices.
        O3[:, 1:] = _p[i : (i + 3), :]
        detO = np.linalg.det(O3)
        i += 1

    if np.isclose(detO, 0.0):
        raise RuntimeError("Cannot determine orientation from colinear path.")

    return detO > 0.0


def _is_overlapping(bbox1, bbox2):
    """Determines if two axis-aligned boxes intersect"""
    x1min, x1max, y1min, y1max = bbox1
    x2min, x2max, y2min, y2max = bbox2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def remove_dup(arr: np.ndarray):
    """Remove duplicate element from np.ndarray"""
    result = np.concatenate((arr[np.nonzero(np.diff(arr))[0]], [arr[-1]]))
    return result

def convert_to_list_of_lists(data_list):
    # Split the list into sublists at NaNs and filter out empty lists
    tmp =  [list(group) for group in np.split(data_list, np.where(np.isnan(data_list))[0]) if len(group) > 0]
    tmp = [np.vstack(d) for d in tmp]
    # drop the last NaN
    tmp = tmp[:-1]
    return tmp


class CoastalGeometry(Region):
    """
    Class for reading in vector data from a vector file
    and preparing it for use in oceanmesh.

    Parameters
    ----------
    vector_data : str or pathlib.Path
        Path to vector file containing vector data typically representing
        a shoreline or coastal boundary. Accepts all formats supported by geopandas.
    bounding_box : tuple
        Bounding box of the region of interest. The format is:
        (xmin, xmax, ymin, ymax).
    minimum_mesh_size : float
        Minimum mesh size spacing in the coordinate reference system units.
    crs : str, optional
        Coordinate reference system (crs) of the vector file.
        Default is 'EPSG:4326'.
    smooth_shoreline : bool, optional
        Smooth the shoreline using a corner cutting algorithm. Default is True.
        See the following argument `refinements` for more details.
    refinements : int, optional
        Number of refinements to apply to the vector data. Default is 1.
    minimum_area_mult : float, optional
        Minimum area multiplier. Features with an area less than
        minimum_mesh_size*minimum_area_mult are removed.
    """

    def __init__(
        self,
        vector_data,
        bounding_box,
        minimum_mesh_size,
        crs="EPSG:4326",
        smooth_shoreline=True,
        refinements=1,
        minimum_area_mult=4.0,
    ):
        if isinstance(vector_data, str):
            vector_data = Path(vector_data)
        # check if the vector data exists
        if not vector_data.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), vector_data
            )

        if isinstance(bounding_box, tuple):
            # form a bounding box polygon
            _boubox = np.asarray(_create_boubox(bounding_box))
        else:
            # assume the bounding box is a polygon
            _boubox = np.asarray(bounding_box)

            # ensure the polygon is ccw
            if not _is_path_ccw(_boubox):
                _boubox = np.flipud(_boubox)
            # ensure the polygon is closed
            bounding_box = (
                np.nanmin(_boubox[:, 0]),
                np.nanmax(_boubox[:, 0]),
                np.nanmin(_boubox[:, 1]),
                np.nanmax(_boubox[:, 1]),
            )
        # initialize the Region class
        super().__init__(bounding_box, crs)

        self.vector_data = vector_data
        self.minimum_mesh_size = minimum_mesh_size

        self.boubox = _boubox
        self.refinements = refinements
        self.minimum_area_mult = minimum_area_mult

        # Initialize empty lists to store the processed vector data
        self.inner = []
        self.outer = []
        self.mainland = []

        # read in the vector data
        polys = self._read()

        # optionally smooth the shoreline by cutting corners
        if smooth_shoreline:
            polys = _smooth_vector_data(polys, self.refinements)

        # densify the vector data so that the minimum spacing along it is <= `minimum_mesh_size` / 2
        polys = _densify(polys, self.minimum_mesh_size, self.bbox)

        # clip the vector data to the bounding box (which comes from the Region class)
        polys = _clip_polys(polys, self.bbox)

        # classify the vector data as inner, outer, or mainland based on how much
        # of the bounding box it intersects
        self.inner, self.mainland, self.boubox = _classify_shoreline(
            self.bbox,
            self.boubox,
            polys,
            self.minimum_mesh_size / 2,
            self.minimum_area_mult,
        )

    def __repr__(self):
        outputs = []
        outputs.append("\nCoastalGeometry object")
        outputs.append(f"vector_data: {self.vector_data}")
        outputs.append(f"bbox: {self.bbox}")
        outputs.append(f"minimum_mesh_size: {self.minimum_mesh_size}")
        outputs.append(f"minimum_area_mult: {self.minimum_area_mult}")
        outputs.append(f"refinements: {self.refinements}")
        # list the number of classified segments
        outputs.append(f"inner: {len(self.inner)} nodes")
        outputs.append(f"outer: {len(self.outer)} nodes")
        outputs.append(f"mainland: {len(self.mainland)} nodes")
        outputs.append(f"crs: {self.crs}")
        return "\n".join(outputs)

    @property
    def vector_data(self):
        return self.__vector_data

    @vector_data.setter
    def vector_data(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        self.__vector_data = filename

    @property
    def refinements(self):
        return self.__refinements

    @refinements.setter
    def refinements(self, value):
        if value < 0:
            raise ValueError("Refinements must be > 0")
        self.__refinements = value

    @property
    def minimum_area_mult(self):
        return self.__minimum_area_mult

    @minimum_area_mult.setter
    def minimum_area_mult(self, value):
        if value <= 0.0:
            raise ValueError(
                "Minimum area multiplier * minimum_mesh_size**2 to "
                " prune polygons must be > 0"
            )
        self.__minimum_area_mult = value

    @property
    def minimum_mesh_size(self):
        return self.__minimum_mesh_size

    @minimum_mesh_size.setter
    def minimum_mesh_size(self, value):
        if value <= 0:
            raise ValueError("minimum_mesh_size must be > 0")
        self.__minimum_mesh_size = value

    @staticmethod
    def transform_to(gdf, dst_crs):
        """
        Transform geodataframe ``gdf`` representing
        a shoreline to dst_crs
        """
        dst_crs = CRS.from_user_input(dst_crs)
        if not gdf.crs.equals(dst_crs):
            logger.info(f"Reprojecting vector data from {gdf.crs} to {dst_crs}")
            gdf = gdf.to_crs(dst_crs)
        return gdf

    def to_geodataframe(self):
        """
        Convert the processed vector data to a vector file
        """
        # Package up the inner, outer, and mainland data into a geodataframe
        # raise an error if the data is not processed yet
        #if len(self.outer) == 0:
        #    raise ValueError("Vector data has not been processed yet")
        # where mainland has a row of nans separating each polygon
        # into a new sublist 
        mainland = convert_to_list_of_lists(self.mainland)
        inner = convert_to_list_of_lists(self.inner)
        outer = convert_to_list_of_lists(self.outer)

        _tmp = []
        labels = []
        for _inner in inner:
            _tmp.append(shapely.geometry.Polygon(_inner))
            labels.append("inner")
        for _mainland in mainland:
            _tmp.append(shapely.geometry.Polygon(_mainland))
            labels.append("mainland")
        for _outer in outer:
            _tmp.append(shapely.geometry.Polygon(_outer))
            labels.append("outer")
        # Create a geodataframe
        gdf = gpd.GeoDataFrame(geometry=_tmp, crs=self.crs)
        gdf["labels"] = labels
        return gdf

    def _read(self):
        """Reads a vector file from `filename` ∩ `bbox`
        using geopandas and returns a numpy array of
        the coordinates of the polygons in the file.
        """
        logger.debug("Entering: _read")

        _bbox = self.bbox

        msg = f"Reading in vector file {self.vector_data}"
        logger.info(msg)

        # transform if necessary
        s = self.transform_to(gpd.read_file(self.vector_data), self.crs)

        # Explode to remove multipolygons or multi-linestrings (if present)
        s = s.explode(index_parts=True)

        polys = []  # store polygons

        delimiter = np.empty((1, 2))
        delimiter[:] = np.nan
        re = numpy.array([0, 2, 1, 3], dtype=int)

        for g in s.geometry:
            # extent of geometry
            bbox2 = [g.bounds[r] for r in re]
            if _is_overlapping(_bbox, bbox2):
                if g.geom_type == "LineString":
                    poly = np.asarray(g.coords)
                elif g.geom_type == "Polygon":  # a polygon
                    poly = np.asarray(g.exterior.coords.xy).T
                else:
                    raise ValueError(f"Unsupported geometry type: {g.geom_type}")

                poly = remove_dup(poly)
                polys.append(np.row_stack((poly, delimiter)))

        if len(polys) == 0:
            raise ValueError("Vector data does not intersect with bbox")

        logger.debug("Exiting: _read")

        return _convert_to_array(polys)

    def plot(
        self,
        ax=None,
        xlabel=None,
        ylabel=None,
        title=None,
        file_name=None,
        show=True,
    ):
        """
        Visualize the content in the classified vector fields of
        CoastalGeometry object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, otherwise uses current axes.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        title : str, optional
            Title for the plot.
        file_name : str, optional
            File name to save the figure to.
        show : bool, optional
            Show the figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot.
        """
        flg1, flg2 = False, False

        if ax is None:
            fig, ax = plt.subplots()
            ax.axis("equal")

        if len(self.mainland) != 0:
            (line1,) = ax.plot(self.mainland[:, 0], self.mainland[:, 1], "k-")
            flg1 = True

        if len(self.inner) != 0:
            (line2,) = ax.plot(self.inner[:, 0], self.inner[:, 1], "r-")
            flg2 = True

        # Note that the boubox has to exist
        (line3,) = ax.plot(self.boubox[:, 0], self.boubox[:, 1], "g--")

        xmin, xmax, ymin, ymax = self.bbox
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=None,
            hatch="////",
            alpha=0.2,
            label="bounding box",
        )

        border = 0.10 * (xmax - xmin)
        if ax is None:
            plt.xlim(xmin - border, xmax + border)
            plt.ylim(ymin - border, ymax + border)

        ax.add_patch(rect)

        if flg1 and flg2:
            ax.legend((line1, line2, line3), ("mainland", "inner", "outer"))
        elif flg1 and not flg2:
            ax.legend((line1, line3), ("mainland", "outer"))
        elif flg2 and not flg1:
            ax.legend((line2, line3), ("inner", "outer"))

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        ax.set_aspect("equal", adjustable="box")

        if show:
            plt.show()

        if file_name is not None:
            plt.savefig(file_name, dpi=300, bbox_inches="tight")

        return ax

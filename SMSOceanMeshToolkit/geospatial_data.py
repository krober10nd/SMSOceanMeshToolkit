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
    """
    Classify segments in numpy.array `polys` as either `inner` or `mainland`.

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
    boxSGP = shapely.geometry.Polygon(boubox)

    inner_polygons = [] 
    for poly in polyL:
        pSGP = shapely.geometry.Polygon(poly[:-2, :])
        if boxSGP.contains(pSGP):
            # fully contained within the domain so an inner island
            area = pSGP.area
            if area >= _AREAMIN:
                inner = np.append(inner, poly, axis=0)
                inner_polygons.append(pSGP)
        elif pSGP.overlaps(boxSGP):
            # polygon partially enclosed by the domain so a mainland polygon. 
            # only keep the part hat is inside the domain
            bSGP = boxSGP.intersection(pSGP)
            # Append polygon segment to mainland
            mainland = np.vstack((mainland, poly))
    
    # if bSGP is empty, then it must be equal to the largest inner polygon by area 
    # check if bSGP exists
    if 'bSGP' not in locals():
        # determine the largest inner polygon by area
        largest_inner_polygon = max(inner_polygons, key=lambda a: a.area)
        # determine the index of the largest inner polygon
        index_of_largest_inner_polygon = inner_polygons.index(largest_inner_polygon)
        # remove the largest inner polygon from the inner polygons list
        inner_polygons.pop(index_of_largest_inner_polygon)
        # recreate the inner polygons as a nan-delimited numpy array
        inner = np.empty(shape=(0, 2))
        inner[:] = nan
        for inner_polygon in inner_polygons:
            xy = np.asarray(inner_polygon.exterior.coords)
            xy = np.vstack((xy, xy[0]))
            inner = np.vstack((inner, xy, [nan, nan]))
        # this now becomes the bSGP 
        bSGP = largest_inner_polygon
        # get coordinates of this polygon
        xy = np.asarray(bSGP.exterior.coords)
        # set equal to mainland 
        mainland = xy
        # append a row of nans 
        mainland = np.vstack((mainland, [np.nan, np.nan]))
        
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


#def _classify_shoreline(bbox, boubox, polys, h0, minimum_area_mult):
#    """Classify segments in numpy.array `polys` as either `inner` or `mainland`.
#    (1) The `mainland` category contains segments that are not totally enclosed inside the `bbox`.
#    (2) The `inner` (i.e., islands) category contains segments totally enclosed inside the `bbox`.
#        NB: Removes `inner` geometry with area < `minimum_area_mult`*`h0`**2
#    (3) `boubox` polygon array will be clipped by segments contained by `mainland`.
#    """
#    logger.debug("Entering:_classify_shoreline")
#        
#    _AREAMIN = minimum_area_mult * h0**2
#
#    if len(boubox) == 0:
#        # if it's empty, create a boubox from the bbox
#        boubox = _create_boubox(bbox)
#        boubox = np.asarray(boubox)
#    elif not _is_path_ccw(boubox):
#        boubox = np.flipud(boubox)
#
#    # Densify boubox to ensure that the minimum spacing along it is <= `h0` / 2
#    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)
#
#    # Remove nan's (append again at end)
#    isNaN = np.sum(np.isnan(boubox), axis=1) > 0
#    if any(isNaN):
#        boubox = np.delete(boubox, isNaN, axis=0)
#    del isNaN
#
#    inner = np.empty(shape=(0, 2))
#    inner[:] = nan
#    mainland = np.empty(shape=(0, 2))
#    mainland[:] = nan
#
#    polyL = _convert_to_list(polys)
#    boxSGP = shapely.geometry.Polygon(boubox)
#
#    for poly in polyL:
#        pSGP = shapely.geometry.Polygon(poly[:-2, :])
#        if boxSGP.contains(pSGP):
#            area = pSGP.area
#            if area >= _AREAMIN:
#                inner = np.append(inner, poly, axis=0)
#        elif pSGP.overlaps(boxSGP):
#            bSGP = boxSGP.intersection(pSGP)
#            #bSGP = boxSGP.symmetric_difference(pSGP)
#            # Append polygon segment to mainland
#            mainland = np.vstack((mainland, poly))
#            # Clip polygon segment from boubox and regenerate path
#
#    out = np.empty(shape=(0, 2))
#
#    if bSGP.geom_type == "Polygon":
#        # Convert to `MultiPolygon`
#        bSGP = shapely.geometry.MultiPolygon([bSGP])
#
#    # MultiPolygon members can be accessed via iterator protocol using `in` loop.
#    for b in bSGP.geoms:
#        xy = np.asarray(b.exterior.coords)
#        xy = np.vstack((xy, xy[0]))
#        out = np.vstack((out, xy, [nan, nan]))
#
#    logger.debug("Exiting:classify_shoreline")
#
#    return inner, mainland, out


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
    """
    Clip segments in `polys` that intersect with `bbox`.
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
    # if the number of nans in data_list is 2 then don't add 1 to the index
    if np.sum(np.isnan(data_list)) == 2:
        tmp =  [list(group) for group in np.split(data_list, np.unique(np.where(np.isnan(data_list))[0])) if len(group) > 0]
    else:
        tmp =  [list(group) for group in np.split(data_list, np.unique(np.where(np.isnan(data_list))[0]+1)) if len(group) > 0]
    tmp = [np.vstack(d)[:-1] for d in tmp]
    return tmp

def _smooth_vector_data_moving_avg(polygons, window_size):
    ''' 
    Move each coordinate in the polygon to the average of its neigbhors 
    +- window_size.
    '''
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
   
    if not isinstance(polygons, list):
        polygons = convert_to_list_of_lists(polygons)


    out = []
    for polygon in polygons: 

        exterior_coords = polygon 
    
        # Initialize smoothed_coords with the original start and end points
        smoothed_coords = exterior_coords[:window_size // 2].tolist()

        # Applying moving average to the interior points
        for i in range(window_size // 2, len(exterior_coords) - window_size // 2):
            window = exterior_coords[i - window_size // 2:i + window_size // 2 + 1]
            mean_coord = window.mean(axis=0)
            smoothed_coords.append(mean_coord.tolist())

        # Append the original end points
        smoothed_coords += exterior_coords[-(window_size // 2):].tolist()
    
        # Creating a new smoothed polygon
        smoothed_coords = np.vstack(smoothed_coords)
        # append a row of nans to the end of the smoothed_coords
        smoothed_coords = np.vstack((smoothed_coords, [np.nan, np.nan]))
        out.append(smoothed_coords) 
   
    smoothed_polygons =  _convert_to_array(out)
    return smoothed_polygons


class CoastalGeometry(Region):
    """
    Class for processing vector data from a file for ocean meshing, representing
    coastal boundaries or shorelines.

    Parameters
    ----------
    vector_data : str or pathlib.Path
        Path to the vector file containing coastal boundary data. 
        Supports formats compatible with geopandas.
    region_boundary : tuple or str or np.ndarray
        Defines the region of interest. Can be a bounding box (tuple: xmin, xmax, ymin, ymax),
        a shapefile path (str), or a polygon (numpy array).
    minimum_mesh_size : float
        The smallest allowable mesh size in the specified coordinate system units.
    crs : str, optional
        Coordinate reference system of the vector file, default 'EPSG:4326'.
    smooth_shoreline : bool, optional
        If True, apply a corner cutting algorithm to smooth the shoreline. Default is True.
    smoothing_approach: str, optional
        Approach to use for smoothing the shoreline. Default is 'chaikin' but can also 'moving_window'
        The chaikin approach smoothes out corners in the shoreline by applying Chaikin's corner cutting
        The moving_window approach smoothes out the shoreline by applying a moving window average.
    smoothing_window: int, optional
        Number of points to use for the moving window approach. Default is 5. Must be odd. 
    refinements : int, optional
        Number of iterations for application of Chaikin's algorithm. Default is 1.
    minimum_area_mult : float, optional
        Factor for filtering small features; those smaller than 
        minimum_mesh_size * minimum_area_mult are removed. 
    """

    def __init__(self, vector_data, region_boundary, minimum_mesh_size, crs="EPSG:4326",
                 smooth_shoreline=True, smoothing_approach='chaikin', smoothing_window=0, 
                 refinements=1, minimum_area_mult=4.0):
        
        if isinstance(vector_data, str):
            vector_data = Path(vector_data)
        if not vector_data.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vector_data)

        if isinstance(region_boundary, tuple):
            _region_polygon = np.asarray(_create_boubox(region_boundary))
        elif isinstance(region_boundary, np.ndarray):
            _region_polygon = np.asarray(region_boundary)
        elif isinstance(region_boundary, str):  # then assume shapefile
            try:
                _region_polygon = gpd.read_file(region_boundary)
                _region_polygon = _region_polygon.iloc[0].geometry
                if _region_polygon.geom_type != 'Polygon':
                    raise ValueError(f"Region must be a polygon. Got {_region_polygon.geom_type} instead.")
                _region_polygon = np.asarray(_region_polygon.exterior.coords)
            except Exception as e:
                raise ValueError(f"Could not read vector data from {region_boundary}. Got {e} instead.") 
        else: 
            raise ValueError(f"region_boundary must be a tuple, numpy array, or shapefile. Got {type(region_boundary)} instead.")
        
        if not _is_path_ccw(_region_polygon):
            _region_polygon = np.flipud(_region_polygon)

        region_bbox = (
            np.nanmin(_region_polygon[:, 0]),
            np.nanmax(_region_polygon[:, 0]),
            np.nanmin(_region_polygon[:, 1]),
            np.nanmax(_region_polygon[:, 1]),
        )

        super().__init__(region_bbox, crs)

        self.vector_data = vector_data
        self.minimum_mesh_size = minimum_mesh_size
        self.region_polygon = _region_polygon
        self.smoothing_approach = smoothing_approach
        self.smoothing_window = smoothing_window
        self.refinements = refinements
        self.minimum_area_mult = minimum_area_mult

        # Initialize empty lists to store the processed vector data
        self.inner = []
        self.region_polygon = []
        self.mainland = []

        polys = self._read()

        polys = _densify(polys, self.minimum_mesh_size, region_bbox)

        if smooth_shoreline and smoothing_approach == 'chaikin':
            polys = _smooth_vector_data(polys, self.refinements)
        elif smooth_shoreline and smoothing_approach == 'moving_window':
            polys = _smooth_vector_data_moving_avg(polys, self.smoothing_window)
        elif smooth_shoreline and smoothing_approach not in ('chaikin', 'moving_window','none'):
            raise ValueError(f"Unknown smoothing approach {self.smoothing_approach}. Must be 'chaikin' or 'moving_window' or 'none'.")

        polys = _clip_polys(polys, region_bbox)

        self.inner, self.mainland, self.region_polygon = _classify_shoreline(
            region_bbox, self.region_polygon, polys, 
            self.minimum_mesh_size / 2, self.minimum_area_mult
        )

    def save_meta_data(self, filename):
        ''' 
        Write the meta data about the processed vector data to a text file
        '''
        with open(filename, 'w') as f:
            f.write(self.__repr__())
        
    def __repr__(self):
        # count the number of polygons 
        if len(self.mainland) != 0:
            number_of_mainland = len(convert_to_list_of_lists(self.mainland))
            
        if len(self.inner) != 0:
            number_of_inner = _inner = len(convert_to_list_of_lists(self.inner))
        
        outputs = [
            "\nCoastalGeometry object",
            f"Vector Data Path: {self.vector_data}",
            f"Coordinate Reference System: {self.crs}",
            f"Units: {self.units}",
            f"Region Bounding Box: {self.bbox}",
            f"Minimum Mesh Size: {self.minimum_mesh_size}",
            f"Minimum Area Multiplier: {self.minimum_area_mult}",
            f"Minimum Area: {self.minimum_mesh_size * self.minimum_area_mult} sq. {self.units}",
            f"Smoothing Approach: {self.smoothing_approach}",
            f"Smoothing Window: {self.smoothing_window}",
            f"# of Corner Cuts: {self.refinements}",
            f"# of Inner Nodes: {len(self.inner)}",
            f"# of Inner Polygons: {number_of_inner}",
            f"# of Mainland Nodes: {len(self.mainland)}",
            f"# of Mainland Polygons: {number_of_mainland}",
        ]
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
                "Minimum area multiplier * minimum_mesh_size**2 to prune polygons must be > 0"
            )
        self.__minimum_area_mult = value

    @property
    def minimum_mesh_size(self):
        return self.__minimum_mesh_size

    @minimum_mesh_size.setter
    def minimum_mesh_size(self, value):
        if value <= 0:
            raise ValueError("Minimum mesh size must be > 0")
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
        logger.debug("Entering: to_geodataframe")
        # into a new sublist 
        mainland = convert_to_list_of_lists(self.mainland)
        inner = convert_to_list_of_lists(self.inner)
        outer = convert_to_list_of_lists(self.region_polygon)

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
        logger.debug("Exiting: to_geodataframe")
        return gdf

    def _read(self):
        """
        Reads a vector file from `filename` ∩ `bbox`
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
            raise ValueError("Vector data does not intersect with region boundary")

        logger.debug("Exiting: _read")

        return _convert_to_array(polys)

    def plot(self, ax=None, xlabel=None, ylabel=None, title=None, filename=None, show=True):
        """
        Visualize the content in the classified vector fields of
        CoastalGeometry object.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If not provided, a new figure will be created.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        title : str, optional
            Title for the plot.
        filename : str, optional
            Path to save the figure.
        show : bool, optional
        
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.axis("equal")

        # Plotting mainland, inner, and region_polygon
        if len(self.mainland) != 0:
            ax.plot(self.mainland[:, 0], self.mainland[:, 1], "k-", label='Mainland')
            
        if len(self.inner) != 0:
            ax.plot(self.inner[:, 0], self.inner[:, 1], "r-", label='Inner')
        
        # TODO: this will be hatched by evalulating the signed distance function
        #ax.plot(self.region_polygon[:, 0], self.region_polygon[:, 1], "g--", label='Meshing Domain')

        # Creating a polygon from region_polygon
        region_poly = shapely.geometry.Polygon(self.region_polygon[:-1])

        # combine mainland & any inner polygons into one multipolygon
        if len(self.mainland) != 0:
            _mainland = convert_to_list_of_lists(self.mainland)
            
        if len(self.inner) != 0:
            _inner = convert_to_list_of_lists(self.inner)
        
        # TODO:  use a signed distance funciton
        #outer_poly = [] 
        #if len(self.mainland) != 0: 
        #    for _main in _mainland:
        #        outer_poly.append(shapely.geometry.Polygon(_main[:-1]))
        #        
        #inner_poly = []
        #if len(self.inner) != 0:
        #    for _inn in _inner:
        #        inner_poly.append(shapely.geometry.Polygon(_inn[:-1]))
        #        
        ## Creating a polygon from outer_poly
        #outer_poly = shapely.geometry.MultiPolygon(outer_poly)
        
        #inner_poly = shapely.geometry.MultiPolygon(inner_poly)
        
        #intersection_poly = region_poly.intersection(outer_poly)

        ## Plotting the intersection area with hatching
        #if not intersection_poly.is_empty:
        #    if intersection_poly.geom_type == "Polygon": 
        #        x, y = intersection_poly.exterior.xy
        #        ax.fill(x, y, alpha=0.2, hatch='////', label="Meshing Domain")
        #    elif intersection_poly.geom_type == "MultiPolygon":
        #        print("MultiPolygon")
        #        for p in intersection_poly.geoms:
        #            x, y = p.exterior.xy
        #            ax.fill(x, y, alpha=0.2, hatch='////', label="Meshing Domain")

        # Setting plot boundaries
        xmin, xmax, ymin, ymax = self.bbox
        border = 0.10 * (xmax - xmin)
        ax.set_xlim(xmin - border, xmax + border)
        ax.set_ylim(ymin - border, ymax + border)

        # Adding labels, title, and legend
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.legend()

        ax.set_aspect("equal", adjustable="box")

        # Displaying or saving the plot
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        return ax

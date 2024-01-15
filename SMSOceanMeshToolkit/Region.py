import numpy as np
from pyproj import Transformer
from pyproj import CRS

def get_crs_units(crs_input):
    crs = CRS(crs_input)
    return crs.axis_info[0].unit_name

__all__ = ["Region", "warp_coordinates"]

def warp_coordinates(points, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return np.column_stack(transformer.transform(points[:, 0], points[:, 1]))

class Region:
    def __init__(self, extent, crs):
        '''
        Define a region of interest (ROI) by its bounding box or polygon.
        '''
        self.bbox = extent
        self._crs = crs
        # given the crs determine the units 
        self._units = get_crs_units(crs)

    @property
    def units(self):
        return self._units
        
    @property
    def crs(self):
        return self._crs

    @property
    def bbox(self):
        return self.__bbox

    @property
    def total_bounds(self):
        if isinstance(self.bbox, tuple):
            return self.bbox
        else:
            return np.array([self.bbox[:, 0].min(), self.bbox[:, 0].max(),
                             self.bbox[:, 1].min(), self.bbox[:, 1].max()])

    @bbox.setter
    def bbox(self, value):
        if isinstance(value, tuple):
            if len(value) != 4:
                raise ValueError("bbox must have four values.")
            if any(value[i] >= value[i + 1] for i in (0, 2)):
                raise ValueError("bbox values are not in the correct order.")
        elif isinstance(value, np.ndarray):
            pass 
        self.__bbox = value

    def transform_to(self, dst_crs):
        if not self._crs == dst_crs:
            transformer = Transformer.from_crs(self.crs, dst_crs, always_xy=True)
            if isinstance(self.bbox, tuple):
                self.bbox = transformer.transform(*self.bbox[:2]) + transformer.transform(*self.bbox[2:])
            else:
                self.bbox = np.column_stack(transformer.transform(self.bbox[:, 0], self.bbox[:, 1]))
            self._crs = dst_crs
        return self

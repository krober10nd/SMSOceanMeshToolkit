import logging
from pathlib import Path

import rioxarray as rxr
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


__all__ = ["DEM"]


class DEM:
    """
    Digitial elevation model read in from a tif or NetCDF file
    """

    def __init__(self, dem_filename, ll_ur=None, minimum_resolution=None):
        """
        Read in a digitial elevation model for later use
        in developing mesh sizing functions.

        Parameters
        ----------
        dem_filename : str or pathlib.Path
            Path to the DEM file
        llr_ur : tuple, optional
            Lower left and upper right coordinates of the region of interest
            (default is None, which implies the entire domain)
        minimum_resolution : float, optional
            Desired minimum resolution DEM shall be used for mesh generation
            (default is None, which implies no downsampling).
        """

        if isinstance(dem_filename, str):
            dem_filename = Path(dem_filename)
        # check if file exists
        if not dem_filename.exists():
            raise FileNotFoundError(f"File {dem_filename} does not exist")
        self.da = rxr.open_rasterio(dem_filename, masked=True).squeeze().drop("band")
        # check for the crs
        if "crs" not in self.da.attrs:
            # assign geographic crs
            self.da.rio.write_crs("EPSG:4326", inplace=True)

        # TODO: catch other cases
        # if lon is dimension rename to longitude
        if "lat" in self.da.dims:
            self.da = self.ds.rename({"lat": "y"})
        if "latitude" in self.da.dims:
            self.da = self.ds.rename({"latitude": "y"})
        if "lon" in self.da.dims:
            self.da = self.ds.rename({"lon": "x"})
        if "longitude" in self.da.dims:
            self.da = self.ds.rename({"longitude": "x"})

        # clip the data to the region of interest
        if ll_ur is not None:
            self = self.clip(ll_ur)

        if minimum_resolution is not None:
            self = self.downsample(minimum_resolution)

    def downsample(self, r_specified, desired_ratio=3):
        """_summary_

        Args:
            minimum_resolution (_type_): _description_
        """
        # determine ratio of current resolution to minimum resolution
        r_current = self.da.rio.resolution()[0]
        # Calculate the target resolution to be `desired_ratio` finer than the specified resolution
        r_target = r_specified / desired_ratio
        # Calculate the downsample factor
        downsample_factor = int(r_target / r_current)
        logger.info(f"Downsampling DEM by a factor of {downsample_factor}")
        self.da = self.da.coarsen(
            x=downsample_factor, y=downsample_factor, boundary="trim"
        ).mean()
        return self

    def eval(self, query_points):
        """Evaluate the data array at a set of grid points"""
        interpolator = RegularGridInterpolator(
            (self.da.y, self.da.x), self.da.values, fill_value=0.0, bounds_error=False
        )
        interpolated_values = interpolator(query_points)
        return interpolated_values

    def clip(self, ll_ur):
        """
        Clip the DEM to the region of interest
        """
        logger.info(f"Clipping DEM to {ll_ur}")
        min_x, max_x, min_y, max_y = ll_ur
        # verify min_x < max_x and min_y < max_y
        if min_x > max_x:
            raise ValueError("min_x must be less than max_x")
        if min_y > max_y:
            raise ValueError("min_y must be less than max_y")
        # assumes the data has latitude and longitude dimensions
        self.da = self.da.rio.clip_box(min_x, min_y, max_x, max_y)
        return self

    # plot method
    def plot(self, ax=None, **kwargs):
        """
        Plot the DEM
        """
        x = self.da.plot(ax=ax, **kwargs)
        return x.axes

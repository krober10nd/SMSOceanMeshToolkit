from pathlib import Path

import rioxarray as rxr
import xarray as xr

__all__ = ["DEM"]


class DEM:
    """
    Digitial elevation model read in from a tif or NetCDF file
    """

    def __init__(self, dem_filename, ll_ur=None, band="data"):
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
        band : str, optional
            Name of the band to read in (default is 'data')
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

    def clip(self, ll_ur):
        """
        Clip the DEM to the region of interest
        """
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
        return self.da.plot(ax=ax, **kwargs)

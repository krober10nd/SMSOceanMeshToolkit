import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy.interpolate import RegularGridInterpolator
import xarray as xr 
import rioxarray


from .Region import Region

logger = logging.getLogger(__name__)

__all__ = ["Grid"]


class Grid(Region):
    """
    A structured grid along with
    primitive operations (e.g., nterpolate_to, etc.) and
    stores data `values` defined at each grid point.

    Parameters
    ----------
    region: :obj:`Region`
        A :obj:`Region` object that defines the grid's extent and crs
    dx: float
        approximate spacing between grid points along x-axis in crs units
    dy: float, optional
        approximate spacing grid grid points along y-axis in crs units
    values: scalar or array-like
        values at grid points. If scalar, then an array of
        the value is created matching the extent.
    extrapolate: boolean, optional
        Whether the grid can extrapolate outside its bbox

    Attributes
    ----------
        x0y0: tuple
            bottom left corner coordinate
        nx: int
            number of grid points in x-direction
        ny: int
            number of grid points in y-direction
        eval: func
            A function that takes a vector of points
            and returns a vector of values
    """

    def __init__(
        self,
        region,
        dx,
        dy=None,
        values=np.nan,
        extrapolate=False,
        eval=None
    ):
        super().__init__(region.bbox, region.crs)
        self.dx = (
            dx  # Set dx first to ensure it's positive before using in calculations
        )
        self.dy = dy if dy is not None else dx  # Simplified assignment
        self.x0y0 = (self.bbox[0], self.bbox[2])  # bottom left corner
        self.extrapolate = extrapolate


        # NOTE: We must ensure that the entire region is covered by the grid object
        # and sometimes this is not possible for arbitrary dx and dy. So, we recompute the dx and dy
        # thus the user specified dx and dy may be different than the actual dx and dy
        _xvec, _yvec = self.create_vectors()
        
        self._dx = _xvec[1] - _xvec[0]
        self._dy = _yvec[1] - _yvec[0]

        self.nx = len(_xvec)
        self.ny = len(_yvec)
        
        self.values = values # see setter method 

        self.eval = eval

    def __repr__(self):
       return f"bbox={self.bbox}, dx={self.dx}, dy={self.dy}, crs={self.crs}), units={self.units}"

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, value):
        if value <= 0:
            raise ValueError("Grid spacing (dx) must be >= 0.0")
        self.__dx = value

    @property
    def dy(self):
        return self.__dy

    @dy.setter
    def dy(self, value):
        if value <= 0:
            raise ValueError("Grid spacing (dy) must be >= 0.0")
        self.__dy = value

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, data):
        if np.isscalar(data):
            data = np.full((self.nx, self.ny), data)
        elif data.shape != (self.nx, self.ny):
            raise ValueError(f"Data shape does not match grid dimensions: {data.shape} != ({self.nx}, {self.ny})")
        # print a warning if there are NaNs in the data
        if np.isnan(data).any() and not np.isnan(data).all():
            logger.warning("WARNING: NaNs found in data contained in grid object's values. Interpolation may be inaccurate.")
        self.__values = data

    @staticmethod
    def get_border(self, arr):
        """
        Get the border values of a 2D array
        """
        return np.concatenate(
            [arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]], axis=0
        )

    def create_vectors(self):
        """
        Build coordinate vectors used to form the grid

        Parameters
        ----------
            None

        Returns
        -------
        x: ndarray
            1D array contain data with `float` type of x-coordinates.
        y: ndarray
            1D array contain data with `float` type of y-coordinates.

        """
        # estimate nx and ny based on user supplied dx and dy
        _nx = int(np.ceil((self.bbox[1] - self.bbox[0]) / self.dx)) + 1
        _ny = int(np.ceil((self.bbox[3] - self.bbox[2]) / self.dy)) + 1
        
        x = np.linspace(self.bbox[0], self.bbox[1], _nx)
        y = np.linspace(self.bbox[2], self.bbox[3], _ny)
        return x, y

    def create_grid(self):
        """
        Build a structured grid with the 'ij' indexing convention

        Parameters
        ----------
            None

        Returns
        -------
        xg: ndarray
            2D array contain data with `float` type.
        yg: ndarray
            2D array contain data with `float` type.

        """
        x, y = self.create_vectors()
        xg, yg = np.meshgrid(x, y, indexing="ij")
        return xg, yg

    def find_indices(self, points, x, y, tree=None, k=1):
        """
        Find linear indices `indices` into a 2D array such that they
        return the closest k point(s) in the structured grid defined by coordinates
        `x` and `y` to query `points`.

        Parameters
        ----------
        points: ndarray
            Query points. 2D array of (x,y) coordinates with `float` type.
        x: ndarray
            Grid points in x-dimension. 2D array with `float` type.
        y: ndarray
            Grid points in y-dimension. 2D array with `float` type.
        tree: :obj:`scipy.spatial.ckdtree`, optional
            A KDtree with coordinates from :class:`CoastalGeometry`
        k: int, optional
            Number of closest points to return

        Returns
        -------
        indices: ndarray
            Indicies into an array. 1D array with `int` type.

        """
        points = points[~np.isnan(points[:, 0]), :]
        if tree is None:
            xy = np.column_stack((x.ravel(), y.ravel()))
            tree = scipy.spatial.cKDTree(xy)
        try:
            _, idx = tree.query(points, k=k, workers=-1)
        except (Exception,):
            _, idx = tree.query(points, k=k, n_jobs=-1)
        return np.unravel_index(idx, x.shape)

    def interpolate_onto(self, grid2, method="nearest"):
        """
        Interpolates self.values onto :class`Grid` grid2 forming a new
        :class:`Grid` object grid3.

        Note
        ----
        In other words, in areas of overlap, grid1 values
        take precedence elsewhere grid2 values are retained.

        Grid3 has dx & dy spacings following the resolution of grid2.
        
        Parameters
        ----------
        grid2: :obj:`Grid`
            A :obj:`Grid` with `values`.
        method: str, optional
            Way to interpolate data between grids

        Returns
        -------
        grid3: :obj:`Grid`
            A new `obj`:`Grid` with projected `values`.
        """
        assert isinstance(grid2, Grid), "Object to interpolate to must be class:Grid."
        # make sure their crs are the same
        assert self.crs == grid2.crs, "Grid objects must have the same crs."
        # check if they overlap
        x1min, x1max, y1min, y1max = self.bbox
        x2min, x2max, y2min, y2max = grid2.bbox
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        assert overlap, "Grid objects do not overlap."
        x1, y1 = self.create_vectors()
        x2, y2 = grid2.create_vectors()
        if self.extrapolate:
            _FILL = None
        else:
            _FILL = -999
        # take data from grid1 --> grid2
        fp = RegularGridInterpolator(
            (x1, y1),
            self.values,
            method=method,
            bounds_error=False,
            fill_value=_FILL,
        )
        xg2, yg2 = np.meshgrid(x2, y2, indexing="ij")
        new_values = fp((xg2, yg2))
        # where fill replace with grid2 values
        new_values[new_values == _FILL] = grid2.values[new_values == _FILL]
        _region2 = Region(grid2.bbox, grid2.crs)
        return Grid(
            _region2,
            dx=grid2.dx,
            dy=grid2.dy,
            values=new_values,
        )

    def plot(
        self,
        ax=None,
        xlabel="X-coordinate",
        ylabel="Y-coordinate",
        title="Grid Values",
        holding=False,
        coarsen=1,
        plot_colorbar=False,
        cbarlabel=None,
        xlim=None,
        ylim=None,
        filename=None,
        **kwargs,
    ):
        """
        Visualize the values in :obj:`Grid`

        Parameters
        ----------
        holding: boolean, optional
            Whether to create a new plot axis.

        Returns
        -------
        fig:
        ax: handle to axis of plot
            handle to axis of plot.

        """
        _xg, _yg = self.create_grid()

        if ax is None:
            fig, ax = plt.subplots()
            ax.axis("equal")
        pc = ax.pcolor(
            _xg[::coarsen, ::coarsen],
            _yg[::coarsen, ::coarsen],
            self.values[::coarsen, ::coarsen],
            **kwargs,
        )
        # in the top right corner plot the dx, dy, and x0y0 values
        ax.text(
            0.95,
            0.95,
            f"dx={self.dx:.4f}, dy={self.dy:.4f}, x0y0={self.x0y0}, units={self.units}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=6, 
            bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='none'),
        )
        # if all values are nan then put a text box in red in the middle 
        # of the figure saying no data 
        if np.isnan(self.values).all():
            _xvec, _yvec = self.create_vectors()
            xmid = np.mean(_xvec)
            ymid = np.mean(_yvec)
            print(xmid, ymid)
            ax.text(xmid, ymid, 'NO DATA', horizontalalignment='center', verticalalignment='center', color='red', fontsize=20)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if cbarlabel is not None:
            plot_colorbar = True
            
        if plot_colorbar or cbarlabel:
            cbar = fig.colorbar(pc)
            cbar.set_label(cbarlabel)

        if filename is not None:
            plt.savefig(filename)
        if holding is False:
            plt.show()
            
        return fig, ax, pc

    def build_interpolant(self):
        """
        Construct a RegularGriddedInterpolant sizing function and store it as
        the `eval` field of the grid:class

        Parameters
        ----------
        values: array-like
            An an array of values that form the gridded interpolant:w

        """
        x, y = self.create_vectors()

        if self.extrapolate:
            _FILL = None
        else:
            _FILL = 999999

        fp = RegularGridInterpolator(
            (x, y),
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=_FILL,
        )

        def sizing_function(x):
            return fp(x)

        self.eval = sizing_function

        return self

    def to_xarray(self):
        """
        Convert a :obj:`Grid` object to an xarray dataset

        Parameters
        ----------
        None

        Returns
        -------
        ds: :obj:`xarray.Dataset`
            An xarray dataset with the grid values

        """
        x, y = self.create_vectors()
        #xg, yg = self.create_grid()
        ds = xr.Dataset(
            {
                "values": (["y", "x"], self.values.T),
            },
            coords={
                "x": x,
                "y": y,
            },
        )
        ds.rio.write_crs(self.crs, inplace=True)

        return ds
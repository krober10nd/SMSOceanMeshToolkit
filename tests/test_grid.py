'''Testing Grid class'''
from SMSOceanMeshToolkit import Grid, Region


def test_irregular_grid():
    '''
    Builds a Grid object for testing
    '''
    region = Region((-20.352, 10., -10.2, 10.), 'EPSG:4326')
    grid = Grid(region, dx=1.0, dy=2.0, values=1.2)
    print(grid)
    ds = grid.to_xarray()
    print(ds)
    grid.plot(filename='test_irregular_grid.png', plot_colorbar=True)

def test_interpolate_onto(): 
    '''
    Given two grids that are overlapping, interpolate the values from one grid onto the other
    '''
    region1 = Region((-20.352, 10., -10.2, 10.), 'EPSG:4326')
    region2 = Region((-10., 5., -5., 5.), 'EPSG:4326')
    grid1 = Grid(region1, dx=1.0, dy=2.0, values=1.2)
    grid2 = Grid(region2, dx=0.5, dy=0.5, values=2.2)
    # will put grid2 onto grid1
    grid3 = grid2.interpolate_onto(grid1)
    grid3.plot(filename='test_interpolate_onto.png', plot_colorbar=True)
    

def test_regular_grid(): 
    '''Builds a Grid object for testing'''
    region = Region((-10., 10., -10., 10.), 'EPSG:4326')
    grid = Grid(region, dx=1.0)
    print(grid)
    # assert that x0y0 is -10, -10
    assert grid.x0y0 == (-10., -10.)
    assert grid.dx == 1.0
    assert grid.dy == 1.0
    # create a dummy grid and verify its properties
    xg, yg = grid.create_grid()
    assert xg.shape == yg.shape
    assert xg.shape == (21, 21)
    assert xg[0, 0] == -10.
    assert yg[0, 0] == -10.
    assert xg[-1, -1] == 10.
    assert yg[-1, -1] == 10.
    # test the grid values
    assert grid.values.shape == (21, 21)

     
if __name__ == '__main__':
    #test_regular_grid()
    #test_irregular_grid()
    test_interpolate_onto()
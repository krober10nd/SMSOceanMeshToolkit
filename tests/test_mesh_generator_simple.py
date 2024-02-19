import numpy as np

from SMSOceanMeshToolkit import generate_mesh, simp_vol
# make a plot of the triangle mesh
import matplotlib.pyplot as plt
# add logging 
import sys 
import os 
import logging

# set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def test_mesh_generator_rectangle():
    min_edge_length = 0.1
    bbox = (0.0, 1.0, 0.0, 1.0)

    def drectangle(p, x1, x2, y1, y2):
        min = np.minimum
        return -min(min(min(-y1 + p[:, 1], y2 - p[:, 1]), -x1 + p[:, 0]), x2 - p[:, 0])

    def domain(x):
        return drectangle(x, *bbox)

    def edge_length(p):
        return np.array([0.1] * len(p))

    points, cells = generate_mesh(
        domain=domain,
        edge_length=edge_length,
        min_edge_length=min_edge_length,
        bbox=bbox,
    )
    #fig, ax = plt.subplots()
    #ax.triplot(points[:, 0], points[:, 1], cells)
    #ax.set_aspect("equal")
    #plt.show()
    assert np.isclose(np.sum(simp_vol(points, cells)), 1.0, 0.01)

    
if __name__ == "__main__":
    test_mesh_generator_rectangle()
    #print("Everything passed")
# encoding: utf-8
"""Plotting routines."""

#-----------------------------------------------------------------------------
#  Copyright (C) 2024- Keith Roberts
#  Copyright (C) 2004-2012 Per-Olof Persson
#  Copyright (C) 2012 Bradley Froehle

#  Distributed under the terms of the GNU General Public License. You should
#  have received a copy of the license along with this program. If not,
#  see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

__all__ = ['SimplexCollection', 'simpplot']


import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from tqdm import tqdm
from .clean import simp_qual

__all__ = ['SimplexCollection', 'simpplot']

#-----------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------

class SimplexCollection(PathCollection):
    """A collection of triangles."""
    def __init__(self, simplices=None, **kwargs):
        # Make a 11 color discrete colormap spanning 0 to 1 with red to green
        cmap = plt.cm.get_cmap('RdYlGn', 8)
        
        kwargs.setdefault('linewidths', 0.5)
        kwargs.setdefault('edgecolors', 'k')
        kwargs.setdefault('facecolors', (0.8, 0.9, 1.0))
        color_by_quality = kwargs.pop('color_by_quality', False)
        PathCollection.__init__(self, [], **kwargs)
        self.set_cmap(cmap)
        # set bounds to colormap 
        self.set_clim(0, 0.8)
        if simplices is not None:
            self.set_simplices(simplices)
            if color_by_quality: 
                self.set_array(simp_qual(simplices[0], simplices[1]))

    def set_simplices(self, simplices, color_by_quality=False):
        """Usage: set_simplices((p, t))"""
        p, t = simplices
        code = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        self.set_paths([Path(edge, code) for edge in p[t[:,[0,1,2,0]]]])
        if color_by_quality: 
            qual = simp_qual(p, t)
            # set the pathcollections colormap to be red to green 
            self.set_array(qual)

#-----------------------------------------------------------------------------
# 2D Plotting
#-----------------------------------------------------------------------------

def axes_simpplot2d(ax, p, t, nodes=False, annotate='', **kwargs):
    """Plot a triangulation

    Parameters
    ----------
    p : array, shape (np, 2)
        nodes
    t : array, shape (nt, 3)
        simplices
    nodes : bool, optional
        draw a marker at each node
    annotate : str, optional
        'p' : annotate nodes
        't' : annotate simplices
    **kwargs : dict
        additional arguments to pass to SimplexCollection
    """
    scalex = kwargs.pop('scalex', True)
    scaley = kwargs.pop('scaley', True)
    #if not ax._hold: ax.cla()
    colorbar = kwargs.pop('colorbar', False)

    assert p.shape[1] == 2

    c = SimplexCollection((p, t), **kwargs)
    ax.add_collection(c)
    if nodes:
        ax.plot(p[:,0], p[:,1], '.k', markersize=16)
    if 'p' in annotate:
        for i in range(len(p)):
            ax.annotate(str(i), p[i], ha='center', va='center')
    if 't' in annotate:
        for it in range(len(t)):
            pmid = p[t[it]].mean(0)
            ax.annotate(str(it), pmid, ha='center', va='center')
    if 'qual' in annotate: 
        qual = simp_qual(p, t)
        for it in tqdm(range(len(t)), desc='Annotating Triangles', total=len(t)):
            pmid = p[t[it]].mean(0)
            qual_str = f'{qual[it]:.2f}'
            ax.annotate(qual_str, pmid, ha='center', va='center')
    if colorbar: 
        # add a colorbar to the plot
        mappable = ax.collections[0]   
        # reduce the size of the colorbar
        cb= plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.5)
        # label it 
        cb.set_label('Mesh Quality')
        # set the title 
    # add a title indicating the number of nodes and triangles with commas 
    ax.set_title(f'{len(p):,} Nodes, {len(t):,} Triangles')
    # annotate the minimum and mean mesh quality 
    qual = simp_qual(p, t)
    ax.annotate(f'Min Quality: {min(qual):.2f} \n Mean Quality: {sum(qual)/len(qual):.2f}', (0.05, 0.05), xycoords='axes fraction', bbox=dict(facecolor='white', alpha=0.5))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.autoscale_view(scalex=scalex, scaley=scaley)
    return ax


#-----------------------------------------------------------------------------
# pyplot interface
#-----------------------------------------------------------------------------

def simpplot(p, t, *args, **kwargs):
    """Plot a simplicial mesh

    Parameters
    ----------
    p : array, shape (np, dim)
        nodes
    t : array, shape (nt, dim+1)
        elements

    Additional 2D parameters
    ------------------------
    nodes : bool, optional
        draw a marker at each node
    annotate : str, optional
        'p' : annotate nodes
        't' : annotate simplices

    Additional 3D parameters
    ------------------------
    pmask : callable or bool array of shape (np,)
    """
    fig, ax = plt.subplots()
    dim = p.shape[1]
    if dim == 2:
        ret = axes_simpplot2d(ax, p, t, *args, **kwargs)
    else:
        raise NotImplementedError("Unknown dimension.")
    #plt.draw_if_interactive()
    return ax 
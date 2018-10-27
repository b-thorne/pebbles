import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pebbles
from itertools import product
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import ScalarFormatter, EngFormatter, LogFormatter
from matplotlib.projections.geo import GeoAxes
from os.path import expandvars

class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        """Shifts labelling by pi
        Shifts labelling from -180,180 to 0-360"""
        def __call__(self, x, pos=None):
            if x != 0:
                x *= -1
            if x < 0:
                x += 2*np.pi
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)

def cm2inch(cm):
    """Centimeters to inches"""
    return cm *0.393701

        
if __name__ == '__main__':
    nside = 256 
    nbins= [2, 4, 6, 10]
    mask_funcs = [pebbles.configurations.fittingmodels.comm_betad_wrap(nbin) for nbin in nbins]
    # using directly matplotlib instead of mollview has higher
    # quality output, I plan to merge this into healpy

    # ratio is always 1/2
    xsize = 1000
    ysize = xsize/2.
    linthresh = 0.1
    
    # this is the mollview min and max
    vmin = 0
    vmax = nbins[-1]

    theta = np.linspace(np.pi, 0, ysize)
    phi   = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))

    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)    

    width = 24
    cmap = plt.cm.RdYlBu
    colormaptag = "colombi1_"


    data = [mask_func(nside) for mask_func in mask_funcs]

    fig = plt.figure(figsize=(cm2inch(width), cm2inch(width)*.8))

    figure_rows, figure_columns = 2, 2
    for i, (submap, nbin) in enumerate(zip(data, nbins)):
        # matplotlib is doing the mollveide projection
        submap = pebbles.plotting.apply_so_mask(submap)
        ax = plt.subplot(figure_rows, figure_columns, i+1, projection='mollweide')        
        # rasterized makes the map bitmap while the labels remain vectorial
        # flip longitude to the astro convention
        plt.title(r"$N_{\rm bin}$ ="+"{:d}".format(nbin), fontsize='small')
        image = plt.pcolormesh(longitude[::-1], latitude, submap[grid_pix], vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
        # remove tick labels
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # remove grid
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])


    # colorbar
    #cax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
    #cb = fig.colorbar(image, cax=cax, orientation='horizontal')

    # workaround for issue with viewers, see colorbar docstring
    #cb.solids.set_edgecolor("face")
    
    plt.subplots_adjust(left=0, right=1, top=.9, wspace=.1, hspace=.01, bottom=.14)
    plt.savefig(expandvars("$HOME/Projects/simonsobs/mapspace/plots/sed_pix_inds.png"),
                    bbox_inches='tight', pad_inches=0.02)
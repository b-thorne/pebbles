import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pebbles
from itertools import product
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import ScalarFormatter, EngFormatter, LogFormatter
plt.style.use('idl')


def cm2inch(cm):
    """Centimeters to inches"""
    return cm *0.393701


if __name__ == '__main__':
    
    nside = 256 
    m = pebbles.pebbles.get_nhits(nside)
    # using directly matplotlib instead of mollview has higher
    # quality output, I plan to merge this into healpy

    # ratio is always 1/2
    xsize = 1000
    ysize = xsize/2.
    linthresh = 0.1
    unit = r"$\mu{\rm K}$"
    
    # this is the mollview min and max
    vmin = -100
    vmax = 100

    theta = np.linspace(np.pi, 0, ysize)
    phi   = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))

    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)
    
    from matplotlib.projections.geo import GeoAxes

    class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
        """Shifts labelling by pi
        Shifts labelling from -180,180 to 0-360"""
        def __call__(self, x, pos=None):
            if x != 0:
                x *= -1
            if x < 0:
                x += 2*np.pi
            return GeoAxes.ThetaFormatter.__call__(self, x, pos)

    width = 24
    cmap = plt.cm.RdYlBu
    colormaptag = "colombi1_"

    peb = pebbles.Pebbles(nside, 'simset1', 'planck2015_AL0p5', 'SO_21', nmc=2)
    (data) = np.moveaxis(peb.compute_simulated_data()[0, 0, 0, ...], 0, 1)

    fig = plt.figure(figsize=(cm2inch(width), cm2inch(width)*.6))

    norm = SymLogNorm(linthresh, 1.)
    figure_rows, figure_columns = 2, 3
    subplot_titles = ["27 GHz", "39 GHz", "93 GHz", "145 GHz", "225 GHz", "280 GHz"]
    for i, (submap, s_title) in enumerate(zip(data, subplot_titles)):
        # matplotlib is doing the mollveide projection
        submap = pebbles.plotting.apply_so_mask(submap)
        ax = plt.subplot(figure_rows, figure_columns, i+1, projection='mollweide')        
        # rasterized makes the map bitmap while the labels remain vectorial
        # flip longitude to the astro convention
        
        image = plt.pcolormesh(longitude[::-1], latitude, submap[grid_pix], vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap, norm=norm)
        
        plt.title(s_title, fontsize='small')
        # remove tick labels
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # remove grid
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])


    # colorbar
    cax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
    fmtr = LogFormatter(linthresh=linthresh, labelOnlyBase=False)
    #fmtr = ScalarFormatter()
    cb = fig.colorbar(image, cax=cax, #ticks=[vmin, -10, -1, -0.1, 0.1, 1, 10, vmax],
                      orientation='horizontal',
                      norm=norm)
                      
    cb.ax.xaxis.set_label_text(unit)
    #cb.ax.xaxis.labelpad = -8
    # workaround for issue with viewers, see colorbar docstring
    cb.solids.set_edgecolor("face")
    
    plt.subplots_adjust(left=0, right=1, top=.9, wspace=.1, hspace=.01, bottom=.14)
    plt.savefig("/home/ben/Dropbox/Papers/soforecast/example_multifreq_obs_placeholder.png",
                bbox_inches='tight', pad_inches=0.02)
    

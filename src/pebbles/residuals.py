import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from .plotting import apply_so_mask
from .pebbles import Pebbles

class Residuals(Pebbles):
    def __init__(self, fitting_model, *args, **kwargs):
        """ Class for the calculation of residuals in Pebbles.
        """
        Pebbles.__init__(self, *args, **kwargs)
        self.fitting_model = fitting_model
        # load true cmb map in thermo units
        return

    def calc_residual_cmb_amp_map(self, nmc, fwhm=None):
        """ Method to calculate the stacked residual map of a set of Nmc
        cleaned CMB maps, removing the true CMB.

        Parameters
        ----------
        nmc: int
            Number of MC realizations to use.

        Returns
        -------
        ndarray
            Array containing map of residuals.
        """
        cmb = self.calc_stacked_amp_map('cmb', nmc)
        if fwhm is not None:
            cmb_true = hp.smoothing(self.cmb_true, pol=True,
                                    fwhm=np.pi/180.*fwhm/60.)
        return cmb - cmb_true[1:]

    def calc_stacked_amp_map(self, component):
        """ Method to calculate the stacked residual map of a set of Nmc
        cleaned CMB maps, removing the true CMB.

        Parameters
        ----------
        nmc: int
            Number of MC realizations to use.

        Returns
        -------
        ndarray
            Array containing map of residuals.
        """
        out = np.zeros((2, hp.nside2npix(self.nside)))
        for i in range(self.nmc):
            out[0] += self.load_cleaned_amp_maps(self.fitting_model,
                                                 component, 'q', i)
            out[1] += self.load_cleaned_amp_maps(self.fitting_model,
                                                 component, 'u', i)
        out /= float(self.nmc)
        return out

    def calc_stacked_spec_map(self, nmc):
        """ Method to calculate the stacked residual map of a set of Nmc
        cleaned CMB maps, removing the true CMB.

        Parameters
        ----------
        nmc: int
            Number of MC realizations to use.

        Returns
        -------
        ndarray
            Array containing map of residuals.
        """
        out = np.zeros((2, hp.nside2npix(self.nside_spec)))
        for i in range(nmc):
            out[0] += self.load_cleaned_spec_maps('ml', 2 * i)
            out[1] += self.load_cleaned_spec_maps('ml', 2 * i + 1)
        out /= float(nmc)
        return out

    def plot_stacked_component_map(self, component):
        component_maps = self.calc_stacked_amp_map(component)

        map1 = apply_so_mask(component_maps[0])
        map2 = apply_so_mask(component_maps[1])

        if component == 'cmb':
            vmin = -3
            vmax = 3
        else:
            vmin = -50
            vmax = 50
        
        fig = plt.figure(1, figsize=(5, 8))
        hp.mollview(map1, fig=1, sub=(2, 1, 1), coord=['G', 'C'], min=vmin, max=vmax)
        hp.mollview(map2, fig=1, sub=(2, 1, 2), coord=['G', 'C'], min=vmin, max=vmax)
        fig.savefig(self.meta.stacked_amp_map_fpath(component, self.fitting_model),
                    bbox_inches='tight')
        plt.close('all')
        return

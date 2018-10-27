from scipy.stats import skew, skewtest, normaltest, kurtosis
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from .bfore import MapLike, SkyModel, InstrumentModel
from .pebbles import Pebbles
from .configurations.masking import so_mask_fitting, so_mask_hits
from .configurations.fittingmodels import fittings

class Nongaussianity(Pebbles):
    def __init__(self, fitting_model, *args, **kwargs):
        """ Class for the calculation of power spectra in Pebbles.
        """
        Pebbles.__init__(self, *args, **kwargs)
        self.fitting_model = fitting_model

    def scaled_noise_maps(self, data, params):
        """ Function to calculate the projection of the CMB and noise onto the estimate CMB
        amplitude map.
        """
        sky = SkyModel(fittings[self.fitting_model]['bfore_components'])
        ins = InstrumentModel(self.bps)
        (obs, noiseivar) = data
        self.maplike = MapLike(
            {
                **fittings[self.fitting_model]['bfore_params'],
                'data': obs,
                'noisevar': noiseivar,
            },
            sky,
            ins)
        # reshape the output to un-flatten the polarization dimensions, and select only
        # the CMB maps from the output component maps (index 0 of the final dimension).
        amp_maps = self.maplike.get_amplitude_mean(params).reshape((2, self.npix, 3))[..., 0]
        return amp_maps
    
    def cmb_and_noise_simulations(self):
        """ Method to produce simulations of the posterior CMB map, including
        a noise contribution.
        """
        # since we are trying to detect the non-gaussianity in map space we
        # would actually have a factor of root(2) less noise in each pixel
        # to contend with.
        cmb = np.array([c[:, 1:, :] for c in self.compute_cmb()])
        noise = np.array([nos_map[:, 1:, :] for nos_map, _ in list(self.compute_noise())])
        var = np.array([var_map[:, :, :] for _, var_map in list(self.compute_noise())]) / 2.
        obs =  cmb + noise / np.sqrt(2.)
        hm_combined = np.moveaxis(obs[: int(self.nmc / 2)], 1, -1)
        var = np.moveaxis(var[: int(self.nmc / 2)], 1, -1)
        return (hm_combined, var)
    
def stats(arr):
    """ Function to calculate the non gaussian statistics of an array.
    """
    return (skew(arr), kurtosis(arr))

def apply_mask(amp_maps, mask):
    """ Function to apply a given mask to a pair of Q and U maps.
    """
    ma1 = np.ma.masked_array(data=amp_maps[0], mask=np.logical_not(mask), fill_value=hp.UNSEEN)
    ma2 = np.ma.masked_array(data=amp_maps[1], mask=np.logical_not(mask), fill_value=hp.UNSEEN)
    return np.concatenate((ma1.compressed(), ma2.compressed()))

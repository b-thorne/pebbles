from .configurations.masking import so_mask_fitting, so_mask_hits
import healpy as hp
import numpy as np

def apply_so_mask(hpix_map, weight=True):
    nside = hp.get_nside(hpix_map)
    cut = np.logical_not(so_mask_fitting(nside))
    hits = np.sqrt(so_mask_hits(nside))
    masked = np.ma.masked_array(data=hpix_map, mask=cut, fill_value=hp.UNSEEN)
    return masked

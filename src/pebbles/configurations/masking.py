import healpy as hp
import numpy as np
import pymaster as nmt
from ..metadata import _SO_HIT_MAP, _GAL_MASK

def so_mask_fitting(nside, threshold=None):
    """ Function to return a binary mask based on the SO proposed scan strategy.
    We allow a threshold to be imposed, which will mask the region of sky with
    a statistical weight less than some percentage of the highest sensitivity
    region.

    Parameters
    ----------
    nside: int
        Resolution parameter at which to produce the mask.
    threshold: float
        Threshold below which to mask. This is expressed as a fraction of the
        highest sensitivity region.

    Returns
    -------
    ndarray
        Array containing the binary mask in RING ordering.
    """
    # get the SO hits map
    mask = get_mask(nside)
    # impose threshold if requested
    if threshold is not None:
        mask[mask < threshold] = 0.
    # round up to 1. all the non-zero pixels
    return np.ceil(mask)

def so_mask_binary(nside, aposcale=20, gal_mask=None):
    return nmt.mask_apodization(np.ceil(so_mask_fitting(nside)), aposcale, 'C2')
    
    
def so_mask_hits(nside, aposcale=20, gal_mask=None):
    """ Function to return mask on which power spectra will be calculated. This
    combines the SO hits map (passed as the hits map to each field, not square root)
    with a Galactic mask from Planck (between 20 and 80 % of the sky unmasked) by
    simply multiplying the two masks. The resulting mask then has an additional
    tapering applied to ensure the second derivative is zero at the boundary.

    Parameters
    ----------
    nside: int
        Resolution at which to calculate the mask.
    aposcale: float
        Parameter relevant to the additional tapering applied. This is the length
        scale of the taper, and is defined in the NaMaster documentation. Note that
        we use the 'C2' apodization scheme, which is outlined in Grain et al. 2009.
    gal_mask: int
        Field of the Planck mask fits file to use. This can be 0 - 6. Increasing
        numbers refer to less masking.

    Returns
    -------
    ndarray(float)
        Scalar mask at resolution given by `nside`.
    """
    mask = get_nhits(nside)
    if gal_mask is not None:
        # read planck mask at a given nside, and make binary by applying the
        # numpy.ceil function.
        mask *= np.ceil(hp.ud_grade(hp.read_map(_GAL_MASK, field=gal_mask, verbose=False),
                                    nside_out=nside))
    # apply an additional C2 apodization to the nhits mask + galactic mask
    return nmt.mask_apodization(mask, aposcale, apotype="C2")

def get_nhits(nside_out=64):
    """ Generates an Nhits map in Galactic coordinates.

    Parameters
    ----------
    nside_out : int
        Output resolution.

    Returns
    -------
    ndarray
        Hits map.
    """
    try:
        # Read in a hits map that has already been calculated
        # in galactic coordinates
        mp_G = hp.read_map(_SO_HIT_MAP, verbose=False)
    except FileNotFoundError:
        # if reading in galactic map fails, read in
        # celestial coordinates map and rotate it.
        mp_C = hp.read_map(_SO_HIT_MAP, verbose=False)
        nside_l = hp.get_nside(mp_C)
        nside_h = 512
        ipixG = np.arange(hp.nside2npix(nside_h))
        thG,phiG = hp.pix2ang(nside_h, ipixG)
        r = hp.Rotator(coord=['G','C'])
        thC, phiC = r(thG, phiG)
        ipixC = hp.ang2pix(nside_l, thC, phiC)
        mp_G = hp.ud_grade(mp_C[ipixC], nside_out=nside_l)
    return hp.ud_grade(mp_G, nside_out=nside_out)

def get_mask(nside_out=512):
    """ Generates inverse-variance mask from Nhits map.

    Parameters
    ----------
    nside_out: int 
        Output resolution.

    Returns
    -------
    ndarray
        Mask at given resolution. Unobserved pixels are
        assigned the value 10^-6.
    """
    zer0 = 1E-6
    nhits = get_nhits(nside_out=nside_out)
    nhits /= np.amax(nhits)
    msk = np.zeros(len(nhits))
    not0 = np.where(nhits>zer0)[0]
    msk[not0] = nhits[not0]
    return msk


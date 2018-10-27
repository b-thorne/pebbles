import scipy.constants as constants
import numpy as np
import healpy as hp
import yaml
from operator import itemgetter
from itertools import product

def read_config_file(conf_file):
    """ Given a set of keywords and a configuration dictionary this function
    fetches all combinations of the keyword values to be submitted as jobs.

    Parameters
    ----------
    conf_file: string or Path object
        Path to the configuration file to be run.
    
    Returns
    -------
    function
        Function which will return a list of tuples, each tuple containing
        a unique combination of settings (will only have more than one element
        when at least one option in the config file has a list of settings).
    """
    # read in the specified configuration file as a dictionary
    with open(conf_file, 'r') as file_obj:
        config = yaml.load(file_obj)
    # define a function that will return the product of the lists contained
    # as items in the above dictionary. This allows the configuration file
    # to contain a list of settings for each option, and for these to then
    # be iterated over when running this script.
    def job_combos(*keys):
        return list(product(*itemgetter(*keys)(config)))
    # returning a function
    return job_combos


def nside2npix(nside):
    """ Small function to wrap healpy's nside2npix. This deals with the
    case of nside = 0, which we want to use to mean one pixel.

    Parameters
    ----------
    nside: int
        Resolution of map for which we want the number of pixels.

    Returns
    -------
    int
        Number of pixels.
    """
    if nside == 0:
        return 1
    return hp.nside2npix(nside)

def blackbody(nu, temp):
    """ Function to calculate the black body function.

    Parameters
    ----------
    nu: float
        Frequency in GHz at which to calculate the black body function.
    temp: float
        Temperature in Kelvin of the black body.

    Returns
    -------
    float
        Black body function.
    """
    x = constants.h * nu * 1.e9 / constants.k / temp
    return 2. * constants.h * (nu * 1.e9) ** 3 / constants.c ** 2 / np.expm1(x)


def blackbody_der(nu, temp):
    """ Function to calculate the black body derivative function. This
    converts between CMB anisotropy temperature and flux units.

    Parameters
    ----------
    nu: float
        Frequency in GHz at which to convert.
    temp: float
        Temperature anisotropy in Kelvin.

    Returns
    -------
    float
        Black body derivative function.
    """
    x = constants.h * nu * 1.e9 / constants.k / temp
    return blackbody(nu, temp) / temp * x * np.exp(x) / np.expm1(x)


def kcmb2rj(nu):
    """ Function to calculate the conversion factor between thermodynamic
    and Rayleigh-Jeans units.

    Parameters
    ----------
    nu: float
        Frequency in GHz at which to calculate the conversion.

    Returns
    -------
    float
        The conversion factor.
    """
    rj2flux = 1. / (2. * constants.k) * (constants.c / nu / 1.e9) ** 2
    return rj2flux * blackbody_der(nu, 2.7225)

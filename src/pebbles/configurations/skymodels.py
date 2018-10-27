""" This module contains configurations for the components of the
sky model.
"""

try:
    from pysm.nominal import models
except ModuleNotFoundError:
    pass
import healpy as hp
from ..metadata import _HIGH_RES_BETA

def simset0(nside):
    """ Simset 0 is a model with spatially constant spectral
    parameters.
    """
    config = {
        'dust': models('d0', nside),
        'synchrotron': models('s0', nside)
    }
    return config


def simset1(nside):
    """ Simset 1 is the nominal PySM model with spatially
    varying dust and synchrotron spectral parameters.
    """
    config = {
        'dust': models('d1', nside),
        'synchrotron': models('s1', nside)
    }
    return config


def simset2(nside):
    """ Simset 2 updates the nominal PySM model with a
    synchrotron index that varies more on sub degree scales,
    produced by Nicoletta Krachmalnicoff.
    """
    s1 = models('s1', nside)
    beta_high_res = hp.read_map(_HIGH_RES_BETA) - 3.
    s1[0].update({'spectral_index': beta_high_res})
    config = {
        'dust': models('d1', nside),
        'synchrotron': s1
    }
    return config


def simset3(nside):
    """ Simset 2 updates the nominal PySM model with a
    synchrotron index that varies more on sub degree scales,
    produced by Nicoletta Krachmalnicoff.
    """
    config = {
        'dust': models('d1', nside),
        'synchrotron': models('s0', nside)
    }
    return config


def simset4(nside):
    """ Simset 2 updates the nominal PySM model with a
    synchrotron index that varies more on sub degree scales,
    produced by Nicoletta Krachmalnicoff.
    """
    config = {
        'dust': models('d0', nside),
        'synchrotron': models('s1', nside)
    }
    return config


skymodels = {
    'simset0': {
        'config': simset0,
    },
    'simset1': {
        'config': simset1,
    },
    'simset2': {
        'config': simset2,
    },
    'simset3': {
        'config': simset3,
    },
    'simset4': {
        'config': simset4,
    },
}

cosmologies = {
    'planck2015_AL1': {
        'tag': 'plk15_AL1',
        'params': {
            'a_lens': 1.,
            'output': 'tCl lCl pCl',
            'l_max_scalars': 5000,
            'lensing': 'yes',
            'A_s': 2.2e-9,
            'n_s': 0.97,
            'h': 0.67,
            'omega_b': 0.022,
            'omega_cdm': 0.12
        },
    },

    'planck2015_AL0p5': {
        'tag': 'plk15_AL0p5',
        'params': {
            'a_lens': 0.5,
            'output': 'tCl lCl pCl',
            'l_max_scalars': 5000,
            'lensing': 'yes',
            'A_s': 2.2e-9,
            'n_s': 0.97,
            'h': 0.67,
            'omega_b': 0.022,
            'omega_cdm': 0.12
        },
    },

    'planck2015_AL0p5_r0p1': {
        'tag': 'plk15_AL0p5',
        'params': {
            'a_lens': 0.5,
            'modes': 's,t',
            'output': 'tCl lCl pCl',
            'l_max_scalars': 5000,
            'l_max_tensors': 5000,
            'lensing': 'yes',
            'A_s': 2.2e-9,
            'n_s': 0.97,
            'h': 0.67,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'r': 0.1,
        },
    },
}

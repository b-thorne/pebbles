import healpy as hp
import numpy as np
from .masking import so_mask_fitting
from ..plotting import apply_so_mask
from os.path import abspath, dirname, join


def npix4(nside):
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    theta = np.pi / 2. - theta
    ind1 = (phi < 5.) * (phi > 2.) * (theta > 0)
    ind2 = np.logical_not(ind1) * (theta > 0)
    ind3 = (phi < 5.) * (phi > 2.) * (theta <= 0)
    ind4 = np.logical_not(ind3) * (theta <= 0)
    mask = np.zeros_like(theta)
    mask[ind1] = 0
    mask[ind2] = 1
    mask[ind3] = 2
    mask[ind4] = 3
    return mask

def healpix_ns0(nside):
    return np.zeros(hp.nside2npix(nside))

def healpix_ns1(nside):
    return hp.ud_grade(np.arange(hp.nside2npix(1)), nside_out=nside)

def healpix_ns2(nside):
    return hp.ud_grade(np.arange(hp.nside2npix(2)), nside_out=nside)

def healpix_ns16(nside):
    return hp.ud_grade(np.arange(hp.nside2npix(16)), nside_out=nside)

def mamd8_wrap(nbins):
    def wrapper(nside):
        return mamd8_binned(nside, nbins)
    return wrapper

def mamd8_binned(nside, nbins=3):
    from pysm.nominal import models
    beta = models('s1', nside)[0]['spectral_index']
    masked_beta = apply_so_mask(beta)
    sorted_masked_beta = np.sort(masked_beta)
    stride = int(len(sorted_masked_beta.compressed()) / nbins)
    bounds = sorted_masked_beta.compressed()[stride::stride]
    conds = []
    conds.append((beta < bounds[0]))
    for i in range(0, nbins - 2):
        conds.append((beta > bounds[i]) * (beta < bounds[i + 1]))
    conds.append((beta > bounds[-1]))
    zeros = np.zeros_like(beta)
    for i in range(nbins): 
        zeros[conds[i]] = i
    return zeros

def comm_betad_wrap(nbins):
    def wrap(nside):
        return comm_betad_binned(nside, nbins)
    return wrap

def comm_betad_binned(nside, nbins=3):
    from pysm.nominal import models
    beta = models('d1', nside)[0]['spectral_index']
    masked_beta = apply_so_mask(beta)
    sorted_masked_beta = np.sort(masked_beta)
    stride = int(len(sorted_masked_beta.compressed()) / nbins)
    bounds = sorted_masked_beta.compressed()[stride::stride]
    conds = []
    conds.append((beta < bounds[0]))
    for i in range(0, nbins - 2):
        conds.append((beta > bounds[i]) * (beta < bounds[i + 1]))
    conds.append((beta > bounds[-1]))
    zeros = np.zeros_like(beta)
    for i in range(nbins): 
        zeros[conds[i]] = i
    return zeros

fittings = {

    'betad_only_ns0': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns0,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "beta_s":-3.1, "T_d":20.},
            "var_pars": ["beta_d"],
            "var_prior_mean": [1.5],
            "var_prior_width": [1.],
            "var_prior_type": ['gauss'],
            },
    },

    
    'betad_only_ns2': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns2,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23.},
            "var_pars": ["beta_s", "beta_d", "T_d"],
            "var_prior_mean": [-3.1, 1.55, 20.],
            "var_prior_width": [0.4, 0.2, 4],
            "var_prior_type": ['gauss', 'gauss', 'gauss'],
            },
    },
 
    'betad_only_ns16': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns16,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23.},
            "var_pars": ["beta_s", "beta_d", "T_d"],
            "var_prior_mean": [-3.1, 1.55, 20.],
            "var_prior_width": [0.4, 0.2, 4],
            "var_prior_type": ['gauss', 'gauss', 'gauss'],
            },
    },
    
    'BdBs_ns1': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns1,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
            },
    },

    'BdBs_ns2': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns2,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
            },
    },

    'BdBs_ns0': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns0,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
            },
    },

    'BdBsTd_ns1': {
        'mask': so_mask_fitting,
        'spec_index_map': healpix_ns1,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23.},
            "var_pars": ["beta_s", "beta_d", "T_d"],
            "var_prior_mean": [-3., 1.6, 20.],
            "var_prior_width": [0.5, 0.5, 4.],
            "var_prior_type": ['gauss', 'gauss', 'gauss'],
            },
    },

    'BdBs_npix4': {
        'mask': so_mask_fitting,
        'spec_index_map': npix4,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    'BdBs_mamd8binned_npix02': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(2),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    
   'BdBs_mamd8binned_npix04': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(4),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    'BdBs_mamd8binned_npix06': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(6),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },


    'BdBs_mamd8binned_npix08': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(8),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    'BdBs_mamd8binned_npix10': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(10),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
    
    
    'BdBs_mamd8binned_npix12': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(12),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d":20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    
    'BdBsTd_mamd8binned': {
        'mask': so_mask_fitting,
        'spec_index_map': mamd8_wrap(4),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23.},
            "var_pars": ["beta_s", "beta_d", "T_d"],
            "var_prior_mean": [-3., 1.6, 20.],
            "var_prior_width": [0.5, 0.5, 5.],
            "var_prior_type": ['gauss', 'gauss', 'gauss'],
        },
    },

    'BdBs_comm_betad_binned_npix02': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(2),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
    
    'BdBs_comm_betad_binned_npix03': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(3),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
    'BdBs_comm_betad_binned_npix04': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(4),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    'BdBs_comm_betad_binned_npix06': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(6),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
     'BdBs_comm_betad_binned_npix08': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(8),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
   
    'BdBs_comm_betad_binned_npix10': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(10),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
    
    'BdBs_comm_betad_binned_npix12': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(12),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    'BdBs_comm_betad_binned_npix50': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(50),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },

    'BdBs_comm_betad_binned_npix100': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(100),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
            "var_pars": ["beta_s", "beta_d"],
            "var_prior_mean": [-3., 1.6],
            "var_prior_width": [0.5, 0.5],
            "var_prior_type": ['gauss', 'gauss'],
        },
    },
    
    'BdBsTd_comm_betad_binned_npix12': {
        'mask': so_mask_fitting,
        'spec_index_map': comm_betad_wrap(12),
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23.},
            "var_pars": ["beta_s", "beta_d", "T_d"],
            "var_prior_mean": [-3., 1.6, 20.],
            "var_prior_width": [0.5, 0.5, 5.],
            "var_prior_type": ['gauss', 'gauss', 'gauss'],
        },
    },
    
    'BdBsTd_ns0': {
        'mask': so_mask_fitting,
        'nside_spec': 1,
        'bfore_components': ['cmb', 'syncpl', 'dustmbb'],
        'bfore_params': {
            "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23.},
            "var_pars": ["beta_s", "beta_d", "T_d"],
            "var_prior_mean": [-3., 1.6, 20.],
            "var_prior_width": [0.5, 0.5, 4.],
            "var_prior_type": ['gauss', 'gauss', 'gauss'],
            },
    },
}

""" This submodule takes care of the bookeeping for the module
and its resulting products. The `Metadata` class is used to create
directories in which products are kept if they do not already exist,
and to contain paths to the data products.

This module relies on the $PEBBLES environment variable existing.
This variable is the home directory for all the synthetic data
products, all the maps, and the output power spectra, and plots.
This is kept as an environment variable as it is desirable to keep
large data sets in scratch.

Notes:
- Implement Path object from pathlib instead of using os.path.join.
"""
from os import environ
from os.path import expandvars
from pathlib import Path

_PEBBLES = Path(expandvars('$PEBBLES'))
_SO_HIT_MAP = str(_PEBBLES / 'data' / 'hit' / 'norm_nHits_SA_35FOV_G.fits')
_GAL_MASK = str(_PEBBLES / 'data' / 'masks' / 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits')
_HIGH_RES_BETA = str(_PEBBLES / 'data' / 'beta_synch_NKsims_512_0.fits')

class Metadata():
    """ Class containing metadata about runs of the code. This
    object calculates all the paths to various maps and products
    that are to be saved.
    """
    def __init__(self, skymodel, cosmology, instrument, nside, nmc):
        try:
            assert 'PEBBLES' in environ
        except AssertionError:
            raise OSError(r"""Set PEBBLES environment variable to directory
            "in which products can be saved.""")
        # get paths to the $PEBBLES directory, and to the directory
        # in which the module source code resides.
        self.products_dir = Path(expandvars('$PEBBLES')).resolve()
        self.module_dir = Path(__file__).resolve()
        # make a bunch of subdirectories to store simulations and
        # results in the $PEBBLES directory.
        for dirname in ['maps', 'data', 'chains', 'plots', 'res']:
            attr = "_{:s}_dir".format(dirname)
            setattr(self, attr, self.products_dir / dirname)
            getattr(self, attr).mkdir(parents=True, exist_ok=True)

        for dirname in ['gal', 'nos', 'cmb', 'sky', 'tot']:
            attr = "_{:s}_dir".format(dirname)
            setattr(self, attr, self._maps_dir / dirname)
            getattr(self, attr).mkdir(parents=True, exist_ok=True)

        self.cosmology = cosmology
        self.instrument = instrument
        self.skymodel = skymodel
        self.ns = nside
        self.nmc = nmc
        self.simulation_tag = '_'.join([skymodel, cosmology, instrument,
                                        'nb{:04d}'.format(nside),
                                        'nmc{:03d}'.format(nmc)])
        return

    def cmb_thermo_fpath(self):
        fmt = (self.ns, self.nmc, self.cosmology)
        fname = "ns{:04d}_mc{:03d}_cmb_thermo_{:s}.fits".format(*fmt)
        return self._cmb_dir / fname

    def amp_maps_fpath(self, fitting):
        fmt = (self.simulation_tag, fitting)
        return self._res_dir / u"{:s}_{:s}_amps.h5".format(*fmt)

    def spec_maps_fpath(self, fitting):
        fmt = (self.simulation_tag, fitting)
        return self._res_dir / "{:s}_{:s}_spec.fits".format(*fmt)

    def data_fpath(self):
        return self._tot_dir / "{:s}_sims.npy".format(self.simulation_tag)

    def noise_fpath(self, imc):
        return self._nos_dir / "{:s}_imc{:d}_sims.npy".format(self.simulation_tag, imc)
    
    def cls_fpath(self, nlb, imc, fitting_model, power):
        fmt = (self.simulation_tag, nlb, fitting_model, power, imc)
        return self._res_dir / "{:s}_nlb{:02d}_{:s}_{:s}_nmc{:04d}_cls.txt".format(*fmt)

    def wsp_mcm_fpath(self, nlb, fitting_model, power):
        fmt = (self.simulation_tag, nlb, fitting_model, power)
        return self._res_dir / "{:s}_nlb{:02d}_{:s}_{:s}_wsp_mcm.txt".format(*fmt)
    
    def cls_instrument_fpath(self, nlb, imc, instrument, power):
        fmt = (self.simulation_tag, nlb, instrument, power, imc)
        return self._res_dir / "{:s}_nlb{:02d}_{:s}_{:s}_nmc{:04d}_cls.txt".format(*fmt)

    def posterior_samples(self, nlb, fitting_model, power, likelihood):
        fmt = (self.simulation_tag, nlb, power, fitting_model, likelihood)
        return self._res_dir / "{:s}_nlb{:02d}_{:s}_{:s}_{:s}_samples".format(*fmt)

    def corner_posterior(self, fitting_model, power, likelihood):
        return self._plots_dir / "corner_posterior_{:s}_{:s}_{:s}_{:s}.pdf".format(self.simulation_tag, power, fitting_model, likelihood)

    def marginalized_r_plot_fpath(self, fitting_model, likelihood):
        return self._plots_dir / "marginalized_r_posterior_{:s}_{:s}_{:s}.pdf".format(fitting_model, self.simulation_tag, likelihood)

    def power_spectra_plot_fpath(self, fitting_model, power, likelihood):
        return self._plots_dir / "power_spectra_{:s}_{:s}_{:s}_{:s}.pdf".format(fitting_model, power, self.simulation_tag, likelihood)

    def power_spectra_uncertainty_plot_fpath(self, fitting_model, power, likelihood):
        return self._plots_dir / "power_spectra_uncertainty_{:s}_{:s}_{:s}_{:s}.pdf".format(fitting_model, power, self.simulation_tag, likelihood)

    def power_spectrum_mask_plot_fpath(self):
        return self._plots_dir / "power_spectrum_mask_{:s}.pdf".format(self.simulation_tag)

    def stacked_amp_map_fpath(self, component, fitting_model):
        return self._plots_dir / "stacked_component_{:s}_{:s}_{:s}.pdf".format(component, fitting_model, self.simulation_tag)

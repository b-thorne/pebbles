""" This submodule contains the class used for controlling the computation of power spectra.
"""
import numpy as np
import pymaster as nmt
from tqdm import tqdm
from .pebbles import Pebbles
from .configurations import powerspectras

class PowerSpectra(Pebbles):
    def __init__(self, power, *args, **kwargs):
        """ Class for the calculation of power spectra in Pebbles.
        """
        Pebbles.__init__(self, *args, **kwargs)
        self.power_conf = power
        self.power = powerspectras[power]
        self.nlb = self.power['nlb']
        self.fwhm = self.instrument['beam_fwhm']
        self.aposcale = self.power['aposcale']
        try:
            self.gal_mask = self.power['gal_mask']
        except KeyError:
            self.gal_mask = None
            
        try:
            deproject_dust = self.power['deproject_dust']
            self.deproject = True
            print("Deprojecting dust template.")
            import pysm
            d1 = pysm.nominal.models('d1', self.nside)[0]
            self.templates = [[d1['A_Q'], d1['A_U']]]
        except KeyError:
            print("Not deprojecting dust template.")
            self.deproject = False
            self.templates = None
        self.mask = self.power['mask'](self.nside, self.aposcale, self.gal_mask)
        self.nmtbin = nmt.NmtBin(self.nside, nlb=self.nlb)
        self.purify_b = self.power['purify_b']
        print('Binning is: ', self.nlb)

    def get_nmt_fields(self, fitting_model=None, instrument=None):
        """ Method to create NaMaster field objects from the
        cleaned CMB amplitude maps.

        Parameters
        ----------
        mask: ndarray
            Array containing the mask to be applied to the maps.
        fwhm: float
            FWHM (arcmin) of the beam to be deconvolved from the
            map at the power spectrum level. This assumes a Gaussian
           symmetric beam.

        Returns
        -------
        tuple
            Tuple containing the field objects corresponding to the
            QU maps from the first and second half-mission
            realizaitons, and the difference map.
        """
        if fitting_model is not None:
            beam = beam_profile(self.nside, self.fwhm)
            for imc in tqdm(range(int(self.nmc / 2))):
                qu1 = list((self.load_cleaned_amp_maps(fitting_model, 'cmb', p, imc)
                            for p in ['q', 'u']))
                qu2 = list((self.load_cleaned_amp_maps(fitting_model, 'cmb', p,
                                                       self.nmc - imc - 1)
                            for p in ['q', 'u']))
                if self.deproject:
                    field1 = nmt.NmtField(self.mask, qu1, purify_b=self.purify_b, 
                                      templates=self.templates, beam=beam)
                    field2 = nmt.NmtField(self.mask, qu2, purify_b=self.purify_b, 
                                      templates=self.templates, beam=beam)
                else:
                    field1 = nmt.NmtField(self.mask, qu1, purify_b=self.purify_b, beam=beam)
                    field2 = nmt.NmtField(self.mask, qu2, purify_b=self.purify_b, beam=beam)
                try:
                    assert 'wsp' in locals()
                except AssertionError:
                    wsp = nmt.NmtWorkspace()
                    wsp.compute_coupling_matrix(field1, field2, self.nmtbin)
                    wsp.write_to(str(self.meta.wsp_mcm_fpath(self.nlb, fitting_model,
                                                             self.power_conf)))
                yield (field1, field2, wsp)
        elif instrument is not None:
            for imc in range(self.nmc):
                qus = self.load_simulated_noise(imc)[:, 1:, :]

                for qu in qus:
                    field1 = nmt.NmtField(self.mask, qu, purify_b=self.purify_b)
                    field2 = field1
                    try:
                        assert 'wsp' in locals()
                    except AssertionError:
                        wsp = nmt.NmtWorkspace()
                        wsp.compute_coupling_matrix(field1, field2, self.nmtbin)
                        wsp.write_to(str(self.meta.wsp_mcm_fpath(self.nlb, fitting_model,
                                                                 self.power_conf)))
                    yield (field1, field2, wsp)

    def calc_mc_power(self, fitting_model=None, instrument=None):
        """ Method to calculate the cross power spectra between pairs of
        MC noise simulations.

        Parameters
        ----------
        method: str
            Method by which the maps were cleaned. This tells which
            set of maps to read in.
        mask: ndarray
            Array containing weights of pixels to be used in calculating
            power spectrum. Will be scaled to range 0 - 1, must not
            containg negative values.
        fwhm: float
            FWHM of Gaussian beam to be deconvolved.
        nlb: int
            Width of binning to be applied to power spectrum.

        Returns
        -------
        list(ndarray)
            List of power spectra.
        """
        if self.verbose_mode:
            print(
                'Calculating power spectra of Monte Carlo noise \n'
                'realizations using: \n'
                '\t nlb: {:d} \n'
                '\t fwhm: {:f} \n'
                ''.format(self.nlb, self.fwhm)
            )
        try:
            assert not self.nmc % 2
        except AssertionError:
            print(
                'Odd number of MC simulations, can not compute pairs of '
                'cross-spectra. Exiting.')
            raise
        if self.verbose_mode:
            print(
                '{:d} realizations leads to {:d} pairs.'
                ''.format(self.nmc, int(self.nmc / 2))
            )
        cls = [compute_pseudo_and_decouple(*nama)
               for nama in self.get_nmt_fields(fitting_model, instrument)]
        for imc, cl in enumerate(cls):
            self.save_spectra(cl, imc, fitting_model, instrument)

    def save_spectra(self, cls, imc, fitting_model=None, instrument=None):
        """ Method to save spectra computed from the imc^th simulation
        to file. Must also specify the binning scheme of this spectrum
        and the model used to fit the synthetic data.
        """
        if fitting_model is not None:
            return np.savetxt(self.meta.cls_fpath(self.nlb, imc, fitting_model, self.power_conf),
                              cls.T)
        return np.savetxt(self.meta.cls_instrument_fpath(self.nlb, imc, instrument,
                                                         self.power_conf), cls.T)

    def load_spectra(self, imc, fitting_model=None, instrument=None, freq=None):
        """ Method to load the BB spectra computed from the imc^th simulation.
        Must also specify the binning width, nlb, as this has been used in the
        file name, and must specify fitting_model.
        """
        if fitting_model is not None:
            return np.loadtxt(self.meta.cls_fpath(self.nlb, imc, fitting_model, self.power_conf),
                              unpack=True)[3]

        return np.loadtxt(self.meta.cls_instrument_fpath(self.nlb, imc, instrument,
                                                         self.power_conf), unpack=True)[3]

    def load_wsp(self, fitting_model):
        """ Method to load an NmtWorkspace object from file instead of recomputing it.
        """
        wsp = nmt.NmtWorkspace()
        return wsp.read_from(str(self.meta.wsp_mcm_fpath(self.nlb, fitting_model, self.power_conf)))

def compute_pseudo_and_decouple(field1, field2, wsp):
    """ Function to calculate pseudospectrum using namaster
    and decouple using a provided mode coupling matrix.
    """
    cls_coupled = nmt.compute_coupled_cell(field1, field2)
    cls = wsp.decouple_cell(cls_coupled)
    return cls


def beam_profile(nside, fwhm):
    """ Function to calculate the instrumental beam. This is
    required by NaMaster to have 3*nside elements ranging
    from multipole 0 to 3*nside-1.

    Parameters
    ----------
    fwhm: float
        Full-width at half-maximum of instrumental beam in arcminutes.
    nside: int
        Nside at which to produce the beam array.
    """
    ells = np.arange(0, 3 * nside)
    fwhm_rad = fwhm / 60. * np.pi / 180.
    sigma = fwhm_rad / np.sqrt(8. * np.log(2.))
    return np.exp(-0.5 * ells ** 2 * sigma ** 2)

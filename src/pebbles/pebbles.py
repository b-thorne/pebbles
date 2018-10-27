""" This module contains the central object pebbles.Pebbles, which controls the
running of the various simulations components as well as the cleaning process.

Objects
-------

- Pebbles

Functions
---------
- get_bps
- make_fitting_index_map
"""
from itertools import product
import healpy as hp
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .configurations import instruments, skymodels, cosmologies, fittings
from .metadata import Metadata
from classy import Class
try:
    import pysm
except ModuleNotFoundError:
    pass
from . import V3calc as v3
from .bfore import MapLike, SkyModel, InstrumentModel
from .bfore.sampling import run_minimize
from .utils import *
from .configurations.masking import get_nhits

class Pebbles(object):
    def __init__(self, nside, skymodel, cosmology, instrument, verbose=True,
                 nmc=50):
        """ Class to govern the running of the code. This class contains
        methods to run the various parts of the pipeline, or to read in
        the results of a previous run for further analsyis.

        Parameters:
        -----------
        cnfg_name: str
            This is the name of the configuration to load. This name
            must already be present as an item in the  in
            `configurations.run` dictionary.

        Attributes:
        -----------
        method: str
            String specifying the method used to determine the spectral
            parameters. This can be either 'sampling', or 'ml', denoting
            either a full sampling of the likelihood, or maximum likelihood.
        nside: int
            Nside resolution of the base amplitudes of maps in this simulation.
        nside_spec: int
            Nside of spectral parameters to be fitted to the maps.
        gal_tag: str
            Identification tag for the galaxy model used.
        ins_tag: str
            Identification tag for the instrument model used.
        cos_tag: str
            Identification tag for the cosmology model used.
        tag_name: str
            Identification tag for this run, which combines all the above tags.
        mask: array_like(float)
        """
        self.nmc = nmc
        self.nside = nside
        self.npix = nside2npix(self.nside)
        self.skymodel = skymodels[skymodel]
        self.instrument = instruments[instrument]
        self.cosmology = cosmologies[cosmology]
        # build some strings to contain name of the mock observations to be used
        # when saving files.
        self.meta = Metadata(skymodel, cosmology, instrument, nside, nmc)
        # we have to use bandpass instead of frequencies to interface with BFoRe.
        self.bps = get_bps(self.instrument['nus'])
        # print debug information if requested.
        self.verbose_mode = verbose
        return

    def compute_simulated_data(self):
        """ Method to run a computation of the simulated data.

        Parameters:
        -----------
        noise_seed: int
            Random seed for the noise realizations.

        Returns:
        -------
        list(array_like(float))
            List containing the two sets of data (half-mission simulations with
            different noise realizatoins). The shape of the arrays is:
            (Npol, Npix, Nfreq), which is the order assumed by the `bfore.MapLike`.
            object.
        """
        # compute galaxy simulation and save to disk this array has shape
        # (Nfreqs, Npol, Npix)
        galaxy = self.compute_galaxy()
        # apply beams to galaxy. CMB has beams applied separately during call
        # to synfast.
        galaxy = self.apply_beams(galaxy)
        # compute cmb realizations and save to disk this array  has shape
        # (Nmc, Nfreqs, Npol, Npix)
        # But the imc and nmc - imc -1 realizations are the same, as these are
        # used to calculate the power spectra.
        cmb = self.compute_cmb()
        # combine galaxy and CMB in order to get array of shape
        # (Nmc, Nfeqs, Npol, Npix)
        sky = cmb + galaxy[None, ...]
        # compute noise realizations and save to disk
        nus = self.instrument['nus']
        observations = np.empty((self.nmc, 2, len(nus), 2, self.npix))
        for i, (noise, var) in enumerate(self.compute_noise()):
            print("Adding noise map number ", i, " of ", self.nmc)
            self.save_simulated_noise(noise, i)
            observations[i, 0, ...] = sky[i, :, 1:, :] + noise[:, 1:, :]
            observations[i, 1, ...] = var
        # change to nmc, 2, npol, npix, nfreq
        observations = np.moveaxis(observations, 2, -1)
        self.save_simulated_data(observations)
        return observations

    def apply_beams(self, skies):
        """ Method to apply instrument beams to sky.
        """
        amin2rad = np.pi / 180. / 60.
        fwhm = self.instrument['beam_fwhm']
        return np.array([hp.smoothing(sky, fwhm=fwhm * amin2rad, verbose=False)
                         for sky in skies])

    def compute_galaxy(self):
        """ Method to compute the Galactic foregrounds for the given model.
        This essentially just calls pysm for the given set of component
        models.
        """
        # get the PySM Sky object
        gal = pysm.Sky(self.skymodel['config'](self.nside))
        # observe the sky at the given set of frequencies
        return gal.signal()(self.instrument['nus'])

    def compute_theory_cls(self):
        """ Method to run CLASS to compute theory spectra.
        """
        return class_spectrum(self.cosmology['params'])

    def compute_cmb(self):
        """ Function to calculate a set of CMB Q and U maps from a given
        cosmology and instrument parameters. We generate a CMB realization
        from a run of CLASS for the given cosmology. This is then scaled
        to a the requested set of frequencies specified in the instrument
        configuration.
        """
        try:
            assert not self.nmc % 2
        except AssertionError:
            print(
                'Odd number of MC simulations, can not compute pairs of '
                'cross-spectra. Exiting.')
            raise

        nus = np.array(self.instrument['nus'])
        nfreqs = len(nus)
        fwhm = self.instrument['beam_fwhm']
        # claculate the power spectrum for this cosmology
        cmb_cls = self.compute_theory_cls()
        # set the random seed and iterate over the number of mc realizations
        # of the CMB requested. for now this will always be one, but may be
        # changed in the future.
        np.random.seed(None)
        # define array containing CMB. This is of shape (Nmc, Nfreq, Npol, Npix)
        # The same realization is placed at imc, and Nmc - imc - 1, as
        # we will use these maps to calculate cross spectra later.
        cmb = np.empty((self.nmc, nfreqs, 3, self.npix))
        print("Generating CMB realizations")
        for imc in tqdm(range(int(self.nmc / 2))):
            print("Computing CMB map ", imc, " of ", self.nmc / 2)
            cmb_real = np.array(hp.synfast(cmb_cls, self.nside, new=True, pol=True,
                                           verbose=False, fwhm=np.pi/180. * fwhm / 60.))
            cmb_real_ukrj = kcmb2rj(nus)[..., None, None] * cmb_real
            cmb[imc, :, :, :] = cmb_real_ukrj
            cmb[self.nmc - imc - 1, :, :, :] = cmb_real_ukrj
        # scale the CMB map from the CMB units to uK_RJ.
        return cmb

    def compute_noise(self):
        """ Method to compute the nmc noise realizations for the given
        noise model.
        """
        nus = np.array(self.instrument['nus'])
        print("Computing noise realizations")
        for _ in tqdm(range(self.nmc)):
            # add break here for white noise simulations
            np.random.seed(None)
            nos_maps, var_maps = get_noise_sim(**self.instrument['noise_params'],
                                               nside_out=self.nside)
            thermo_nos_maps = nos_maps * kcmb2rj(nus)[:, None, None]
            thermo_var_maps = var_maps * kcmb2rj(nus)[:, None, None] ** 2
            yield (thermo_nos_maps, thermo_var_maps)


    def clean_simulated_data(self, data, fitting_model, pool):
        """ Method to run the cleaning of the data. This first sets up the
        tasks by chunking the data, and then distributes it using the
        `multiprocessing.Pool` object.

        Parameters
        ----------
        data: array_like(float)
            Data to be cleaned, of shape (Npol, Npix, Nfreq).
        pool: `multiprocessing.Pool`
            Instance of the `multiprocessing.Pool` object used to distribute
            tasks.

        Returns
        -------
        tuple(array_like(float))
            Tuple containing two arrays, one being the best fit spectral
            parameter maps, and one being the corresponding amplitude maps.
        """
        if self.verbose_mode:
            print('fitting:', fitting_model)
        mask = fittings[fitting_model]['mask'](self.nside)
        # get a map at the same resolution as the amplitudes, with pixels
        # of integer values between 1 and N_spec where N_spec is the number
        # of different patches on which to fit spectral indices.
        #index_map = fittings[fitting_model]['spec_index_map'](self.nside)
        # split the data up into a list of tasks, where each task is a fit
        # to be performed (Nmc * N_spec tasks)
        tasks = self._prepare_simulated_data(data, fitting_model, mask)
        # perform the fitting over the list of tasks.
        results = pool.map(ml_pixel, tqdm(tasks))
        # from the list of fitting jobs reform the results into maps
        (spec_maps, amp_maps) = self._assemble_maps(results, fitting_model)
        self.save_cleaned_amp_maps(amp_maps, fitting_model)
        self.save_cleaned_spec_maps(spec_maps, fitting_model)
        return

    def save_simulated_data(self, data):
        """ Method to save to disk the simulated observations as a binary.
        """
        return np.save(self.meta.data_fpath(), data)

    def save_simulated_noise(self, noise, imc):
        """ Method to save to disk the simulated observations as a binary.
        """
        return np.save(self.meta.noise_fpath(imc), noise)

    def load_simulated_noise(self, imc):
        """ Method to save to disk the simulated observations as a binary.
        """
        return np.load(self.meta.noise_fpath(imc))


    def save_cleaned_amp_maps(self, amp_maps, fitting_model):
        """ Method to save cleaned amplitude maps to file.
        The maps are layed out with component changing fastest,
        then polarization then mc index.
        """
        with h5py.File(self.meta.amp_maps_fpath(fitting_model), 'a') as hp5:
            dset_name = ''.join([fitting_model, u'cleaned_amp_maps'])
            try:
                dset = hp5.create_dataset(dset_name, data=amp_maps)
            except RuntimeError:
                dset = hp5[dset_name]
                dset[...] = amp_maps
        return

    def save_cleaned_spec_maps(self, spec_maps, fitting_model):
        """ Method to save template maps to file.
        """
        if spec_maps.shape[-1] == 1:
            spec_maps = [spec_map * np.ones(12) for spec_map in spec_maps]
        return hp.write_map(self.meta.spec_maps_fpath(fitting_model), spec_maps,
                            overwrite=True)

    def load_simulated_data(self):
        """ Load simulated data as numpy array from binary.
        """
        return np.load(self.meta.data_fpath())

    def load_cleaned_amp_maps(self, fitting_model, component, pol, imc):
        """ Function to load the amplitude maps for a given component,
        MC realization and polarization field

        For example, the maps are layed out within the fits file in the
        following order:

        cmb_q_mc0 sync_q_mc0 dust_q_mc0 cmb_u_mc0 sync_u_mc0 dust_u_mc
        cmb_q_mc1 ...

        I.e. component changes fastest, then polarization field, then mc.
        The order of the components depends on the order in which they
        are specified in the fitting model configuration dictionary, which
        is not ideal.

        Parameters
        ----------
        method: str
            Method used to produce the cleaned map from simulations. Can either
            be 'sampling' or 'ml'.
        component: str
            Component of the sky model to be retrieved, can be 'cmb', 'synch',
            or 'dust'.
        pol: str
            Polarization field to retrieve, can be either 'q' or 'u'.
        imc: int
            Index of the MC noise realizations to retrive.

        Returns
        -------
        ndarray
            Array containing the map.
        """
        try:
            assert(component in ['cmb', 'synch', 'dust'])
        except AssertionError:
            print("Component must be 'cmb', 'synch', or 'dust'")
            raise
        ncomp = len(fittings[fitting_model]['bfore_components'])
        if component == 'cmb':
            ic = 0
        if component == 'synch':
            ic = 1
        if component == 'dust':
            ic = 2
        inf = 0 if pol == 'q' else ncomp
        field = imc * (2 * ncomp) + ic + inf

        with h5py.File(self.meta.amp_maps_fpath(fitting_model), 'r') as f:
            dset_name = ''.join([fitting_model, u'cleaned_amp_maps'])
            dset = f[dset_name]
            out = dset[field, ...]
        return out

    def save_stacked_cleaned_amp_maps(self, fit, comp, maps):
        """ Method to write stacked posterior amplitude maps to file.
        """
        hp.write_map(self.meta.stacked_amp_map_fpath(comp, fit), maps, overwrite=True)

    def load_stacked_cleaned_amp_maps(self, fit, comp):
        """ Method to read stacked posterior amplitude maps from file.
        """
        return hp.read_map(str(self.meta.stacked_amp_map_fpath(comp, fit)), field=(0, 1))

    def load_cleaned_spec_maps(self, fitting_model, field):
        """ Method to load ML spectral parameter maps from file.
        """
        return hp.read_map(str(self.meta.spec_maps_fpath(fitting_model)),
                           field=field, verbose=False)

    def _assemble_maps(self, results, fitting_model):
        """ This function takes the results of the cleaning process, which is
        a list of the noise mc index, spectral pixel index, and the corresponding
        amplitudes for the different components, and value of the spectral index,
        and construct from them the maps of amplitudes and spectral indices.

        Parameters
        ----------
        results: list(tuple)
            List of tuples, each tuple contains the MC iteration number,
            the spectral parameter pixel index, the correpsonding amplitude
            indices, the expected spectral parameter values and the expected
            amplitude values.

        Returns
        -------
        tuple(ndarray)
            Tuple containing the spectral parameter maps and the amplitude maps.
        """
        # get the number of spectral parameters
        nvar = len(fittings[fitting_model]['bfore_params']['var_pars'])
        ncomp = len(fittings[fitting_model]['bfore_components'])
        # Here we check the number of spectral parameters we fit and create the arrays
        # containing them. This is set by an input mask at the same resolution as
        # nside, with pixels of value between 1 and N_spec. spec_pars_exp is going to
        # be a numpy array of length N_spec, which does not necessarily correspond to
        # a healpix array. Therefore it might make most sense to compare this to the
        # fitting_index_mask to assign values.
        # npix_spec, spec_maps = parse_spec_pars_map
        
        #nside_spec = fittings[fitting_model]['nside_spec']
        #npix_spec = nside2npix(nside_spec)
        # make the arrays into which we put the data
        spec_maps = np.zeros((self.nmc, self.npix, nvar))
        # shape of the amplitudes is
        amp_maps = np.zeros((self.nmc, self.npix, 2, ncomp))
        # iterate through the results of cleaning each pixel, and assign
        # the expectation values of the spectral parameters and amplitudes to
        # the above arrays
        for (imc, ispec, iamp, spec_pars_exp, amps_exp) in results:
            spec_maps[(imc, iamp)] = spec_pars_exp
            # we have to reshuffle the ordering of the amplitude data due
            # to the flattening and reshaping done by BFoRe.
            amp_maps[(imc, iamp)] = np.moveaxis(amps_exp.reshape(2, len(iamp), ncomp), 1, 0)
        # reorganize axes so that the pixel dimension is last. Flatten all
        # preceding dimensions to save to file. The order of maps is then
        # mc realization changes fastest, then q/u, then component.
        spec_maps = np.moveaxis(spec_maps, 1, -1).reshape(-1, self.npix)
        amp_maps = np.moveaxis(amp_maps, 1, -1).reshape(-1, self.npix)
        return (spec_maps, amp_maps)

    def _prepare_simulated_data(self, data, fitting_model, mask):
        """ Method to make sure the data has been generated, and prepare the
        tasks to be distributed to different processes.

        Parameters
        ----------
        data: list(array_like(float))
            List of the two noise realizations to be cleaned.

        Returns
        -------
        generator
            Generator object which yields tasks to be distributed.
        """
        bfore_components = fittings[fitting_model]['bfore_components']
        bfore_params = fittings[fitting_model]['bfore_params']
        spec_index_map = fittings[fitting_model]['spec_index_map'](self.nside)
        #nside_spec = fittings[fitting_model]['nside_spec']
        pos0 = bfore_params['var_prior_mean']
        # defing a function to generate `bfore.MapLike` configuration dict
        # as a function of amplitude pixel indices.
        lkl_cfg = lambda imc, ispec, iamp: {
            **bfore_params,
            'data': data[imc][0][:, iamp, :],
            'noisevar': data[imc][1][:, iamp, :],
        }
        # for each of the large pixels create a task which is a spectral index
        # pixel index and the corresponding `maplike` configuration.
        # get a list of pixels that exist within the mask chosen. This is used to
        # only fit the model in pixels we care about.
        pixels = self._get_pixel_indices(mask, spec_index_map)
        tasks = [(pixel, lkl_cfg(*pixel)) for pixel in pixels]
        # get the configurations for the sky and instrument, which are the same
        # for all pixels.
        sky = SkyModel(bfore_components)
        self.bps = get_bps(self.instrument['nus'])
        ins = InstrumentModel(self.bps)
        # initialize the maplike objects.
        tasks = [(imc, ispec, iamp, MapLike(cnfg, sky, ins), pos0)
                 for ((imc, ispec, iamp), cnfg) in tasks]
        return tasks

    def _get_pixel_indices(self, mask, spec_index_map):
        """ Method to compute the indices of the amplitude maps that correspond
        to each pixel in the spectral parameter maps (nside_spec < nside), over
        a given binary mask.

        Parameters
        ----------
        mask: ndarray
            Healpix map containing a map of zeros and ones. Ones correspond to
            pixels we would like to include in the fitting, and ones will be
            ignored.
        index_map: ndarray
            Array at the same resolution as `mask` and the base amplitude maps.
            Each pixel has an integer value between 1 and N_spec, denoting the
            fitting pixels to which it belongs.

        Returns
        -------
        list(tuple)
            List of tuples, each list contains (imc, ispec, iamp). This set of
            indices defines the set of data to be used in each run of MapLike.
        """
        # we will now start from having the fitting index map and replace these
        # two lines.
        spec_inds = np.unique(spec_index_map)
        n_spec = len(spec_inds)
        #npix_spec = nside2npix(nside_spec)
        #ipix_base = make_fitting_index_map(self.nside, nside_spec)
        # get array of boolean values where True means we want to fit.
        mask_iamp = np.where(mask == 1)[0]
        #calculate where the two maps intersect, in regions allowed by the mask
        in_mask = lambda iamp: np.intersect1d(mask_iamp, np.where(spec_index_map == iamp)[0])
        pixels = ((imc, i, in_mask(i)) for imc, i in product(range(self.nmc),
                                                             range(n_spec)))
        return filter(lambda x: len(x[2]) > 0, pixels)

def get_bps(frequencies):
    """ Method to calculate and return a delta function bandpass in the
    correct form for `bfore.MapLike` from an array of frequencies.

    Parameters
    ----------
    frequencies: array_like(float)
        Array of frequencies at which to center the delta bandpasses.

    Returns
    -------
    list(dict)
        List of dictionaries, each dictionary contains two keys,
        'nu', an array specifying samples within the bandpass, and
        'bps', containing a correspondin weight for each sample in
        frequency.
    """
    bps = np.array([
        {
            'nu': np.array([freq - 0.1, freq + 0.1]),
            'bps': np.array([1])
        } for freq in frequencies])
    return bps


def make_fitting_index_map(nside_amp, nside_spec):
    """ Function calculating a map at the amplitude resolution where each
    pixel contains the index it corresponds to at spectral index resolution.

    So if nside_amp = 256, and nside_spec = 1, the map values will range from
    1 to 12, and there will be 256 ** 2 pixels with each group arranged in the
    shape of the nside 1 pixels.

    Parameters
    ----------
    nside_amp: int
        Resolution at which amplitudes are fit.
    nside_spec: int
        Resolution at which spectral indices are fit.

    Returns
    -------
        ndarray(int)
    """
    npix_spec = nside2npix(nside_spec)
    npix_amp = nside2npix(nside_amp)
    if npix_spec == 1:
        # in the case of nside = 0, we want only one pixel on the sky, and
        # healpix does not account for this in their usual routines.
        ipix_base = np.zeros(npix_amp)
    else:
        # in the rest of the cases we can do this.
        spec_pixels = np.arange(npix_spec)
        ipix_base = hp.ud_grade(spec_pixels, nside_out=nside_amp)
    return ipix_base

def ml_pixel(pixel_info):
    """ Function to implement the cleaning of an individual pixel using
    `bfore`. This is the core function for running the minimization of
    the likelihood.

    First the maximum likelihood spectral parameters are found from the
    marginalized likelihood, and then these are used to compute analytically
    the ML amplitude parameters.

    Parameters
    ----------
    pixel_info: tuple
        Tuple containing the MC iteration number, the pixel index of the
        spectral parameter being inferred, the pixel indices of the
        corresponding amplitudes being inferred, the likelihood function
        for the data here, and the initial guess of the spectral parameters.

    Returns
    -------
    tuple(int, int, ndarray(int), float, ndarray(float))
        Tuple containing the mc iteration, the spectral parameter pixel index,
        the corresponding amplitude pixel indices, the inferred value of the
        spectral parameter, and the corresponding inferred values of the
        amplitudes.
    """
    (imc, ispec, iamp, lkl, pos0) = pixel_info
    # get the index of this pixel in the spectral resolution, and in the
    # amplitude resolution.
    res = run_minimize(lkl.marginal_spectral_likelihood, pos0)
    # save chain to disk
    # get expected spectral parameters
    spec_pars_exp = res['params_ML']
    #if not isinstance(spec_pars_exp, list):
    #    spec_pars_exp = [spec_pars_exp]
    amps_exp = lkl.get_amplitude_mean(spec_pars_exp)
    return (imc, ispec, iamp, spec_pars_exp, amps_exp)

def get_white_noise(sigmas, nside):
    """ Function to generate white noise maps for a list of pixel noise levels and
    a given resolution.

    Parameters
    ----------
    sigmas: list(float)
        List of pixel noise levels, in uKamin.
    nside: int
        Resolution parameter of the maps to which these resolutions correspond.

    Returns
    -------
    list(array_like(float)), list(array_like(float))
        List of noise maps, and list of the corresponding variance maps.

    Notes
    -----
    Assuming white noise, the pixel noise levels given in uKamin can be
    converted to a noice map. We first calculate the noise power spectrum
    according to Knox 1995:

    ..math::
        N_\ell = 4 \pi \frac{\sigma_{\rm amin}^2}{4 \pi (\frac{10800}{\pi})^2}

    We then make a realization of this power spectrum and return it, along
    with a variance map. We calculate the pixel variance as:

    ..math::
       \sigma^2_{\rm pix} / N_{\rm pix} = \sigma^2_{\rm amin} / (4\pi\frac{10800}{\pi})^2
    """
    n_ells_p = [sigma_amin_to_cell(sigma, nside) for sigma in sigmas]
    n_ells_t = [sigma_amin_to_cell(sigma / np.sqrt(2.), nside) for sigma in sigmas]
    # sigmas in uKamin
    np.random.seed(None)
    noise_maps = np.array([np.array(hp.synfast([n_t, n_p, n_p, np.zeros_like(n_p)], nside,
                                               new=True, pol=True, verbose=False))
                           for n_t, n_p in zip(n_ells_t, n_ells_p)])
    #noise_maps = np.array([sigma_amin_p_to_maps(sigma, nside) for sigma in sigmas])
    npix = nside2npix(nside)
    amin2perpix = 4. * np.pi / float(npix) * (10800. / np.pi) ** 2
    var_maps = [np.ones((2, npix)) * sigma ** 2 / amin2perpix for sigma in sigmas]
    return noise_maps, var_maps


def sigma_amin_to_cell(sigma, nside, fsky=1):
    """ Function to calculate the power spectrum for a given instrument
    sensitivity in uK_amin.

    Parameters
    ----------
    sigma: float
        Instrument sensitivity in uKamin.
    nside: int
        Resolution parameter, determines length of output cell array.

    Returns
    -------
    ndarray
        Array containing the white noise power spectrum.

    Notes
    -----
    We apply the simple formula (e.g. Tegmark 1997)

    ..math::
        C_\ell^{\rm noise} = \Omega \frac{\sigma^2_{\rm pix}}{N_{\rm pix}}

    where $\Omega$ is the sky fraction of the survey ($f_{\rm sky} 4 \pi$).
    """
    # calculate the number square arcminutes in full sky
    amin2 = 4. * np.pi * (10800./ np.pi) ** 2
    # calculate white noise power spectrum on whole sky
    wn_ps = 4. * np.pi * sigma ** 2 / amin2
    # rescale to sky fraction actually observerd
    wn_ps *= fsky
    #return as array
    return wn_ps * np.ones(3 * nside)

def class_spectrum(params):
    """ Function to generate the theoretical CMB power spectrum for a
    given set of parameters.

    Parameters
    ----------
    params: dict
        dict containing the parameters expected by CLASS and their values.

    Returns
    -------
    array_like(float)
        Array of shape (4, lmax + 1) containing the TT, EE, BB, TE power
        spectra.

    """
    # This line is crucial to prevent pop from removing the lensing efficieny
    # from future runs in the same script.
    class_pars = {**params}
    try:
        # if lensing amplitude is set, remove it from dictionary that will
        # be passed to CLASS.
        a_lens = class_pars.pop('a_lens')
    except KeyError:
        # if not set in params dictionary just set to 1.
        a_lens = 1
    # generate the CMB realizations.
    print("Running CLASS with lensing efficiency: ", a_lens)
    cos = Class()
    cos.set(class_pars)
    cos.compute()
    # returns CLASS format, 0 to lmax, dimensionless, Cls.
    # multiply by (2.7225e6) ** 2 to get in uK_CMB ^ 2
    lensed_cls = cos.lensed_cl()
    raw_bb = cos.raw_cl()['bb'][:class_pars['l_max_scalars'] + 1]
    # calculate the lensing contribution to BB and rescale by
    # the lensing amplitude factor.
    lensing_bb = a_lens * (lensed_cls['bb'] - raw_bb)
    cos.struct_cleanup()
    cos.empty()
    synfast_cls = np.zeros((4, class_pars['l_max_scalars'] + 1))
    synfast_cls[0, :] = lensed_cls['tt']
    synfast_cls[1, :] = lensed_cls['ee']
    synfast_cls[2, :] = lensing_bb + raw_bb
    synfast_cls[3, :] = lensed_cls['te']
    return synfast_cls * (2.7225e6) ** 2

def get_noise_sim(sensitivity=2, knee_mode=1, ny_lf=1., nside_out=512,
                  beam_corrected=False, **kwargs):
    """ Generates noise simulation
    sensitivity : choice of sensitivity model for SAC's V3
    knee_mode : choice of ell_knee model for SAC's V3
    ny_lf : number of years with an LF tube
    nside_out : output resolution
    """
    nh = get_nhits(nside_out=nside_out)
    fsky = np.mean(nh ** 2) ** 2 / np.mean(nh ** 4)
    # calculate scaling map to rescale white noise realization
    nh[np.where(nh<1E-6)[0]] = np.amax(nh) #zer0
    hits_scaling = np.sqrt(nh / np.amax(nh))
    _, nll, sigma_amin = v3.so_V3_SA_noise(sensitivity, knee_mode, ny_lf,
                                           fsky, 3 * nside_out,
                                           beam_corrected=beam_corrected,
                                           **kwargs)
    # convert the white noise levels to an inverse variance weight
    # map for each frequency. sigma_amin is in uK.
    npix = nside2npix(nside_out)
    amin2perpix = 4. * np.pi * (10800. / np.pi) ** 2 / float(npix)
    ivar_maps = [np.ones((2, npix)) * hits_scaling ** 2 * amin2perpix / s ** 2
                 for s in sigma_amin]
    # add leading zeros for ell = 0, 1
    nll = [np.concatenate((np.array([nl[0]]*2), nl)) for nl in nll]
    # realize and rescale each noise level (list of different frequencies)
    noise_maps = np.array([noise_realization(nside_out, nl, hits_scaling)
                           for nl in nll])
    var_maps = 1./np.array(ivar_maps)
    return noise_maps, var_maps

def noise_realization(nside, nll, nh_sqrt):
    """ Realize a single noise spectrum and multiply by an nhits map.
    """
    # nll is the polarization noise
    nll_teb = np.array([f * nll for f in [0.5, 1., 1., 0., 0., 0.]])
    np.random.seed(None)
    TQU = 1. / nh_sqrt * np.array(hp.synfast(nll_teb, nside, pol=True, new=True,
                                             verbose=False))
    return TQU

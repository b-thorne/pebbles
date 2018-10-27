import numpy as np
import healpy as hp
import emcee
import corner
import pymaster as nmt
from classy import Class
import matplotlib.pyplot as plt
from .powerspectra import PowerSpectra
from .configurations import likelihoods, cosmologies

class Posterior(PowerSpectra):
    def __init__(self, likelihood, fitting_model, *args, **kwargs):
        PowerSpectra.__init__(self, *args, **kwargs)
        self.fitting_model = fitting_model
        self.likelihood_name = likelihood
        self.likelihood = likelihoods[likelihood]
        lmin = self.likelihood['lmin']
        lmax = self.likelihood['lmax']
        self.marg = self.likelihood['marg']
        self.fsky = self.likelihood['fsky']
        template_cosmology = self.likelihood['template_cosmology']
        (self.nmtbin, self.ellb, self.inds) = self.get_model_domain(lmin, lmax,
                                                                    self.nlb)
        #(self.cls, self.cls_var) = self.load_data()
        (self.cls, self.cls_inv_covar) = self.load_data()
        self.wsp = nmt.NmtWorkspace()
        self.wsp.read_from(str(self.meta.wsp_mcm_fpath(self.nlb, self.fitting_model,
                                                         self.power_conf)))
        self.templates = self.do_model_setup({**cosmologies[template_cosmology]['params']})
        self.samples = None
        return

    def load_data(self):
        """ Method to load power spectrum data and compute its mean
        and variance.

        Parameters
        ----------
        nlb: int
            Binning parameter. Load spectra calculated with this bin width.

        Returns
        -------
        tuple(ndarray)
            Tuple containing arrays corresonding to the mean of the
            MC simulations, and their variance.
        """
        monte_carlo_mcs = np.array([self.load_spectra(imc, fitting_model=self.fitting_model)
                                    for imc in range(int(self.nmc / 2))])
        cl_mean = np.mean(monte_carlo_mcs, axis=0)
        cl_covar = np.cov(monte_carlo_mcs, rowvar=False)
        cl_inv_covar = np.linalg.inv(cl_covar)
        return cl_mean, cl_inv_covar

    def get_model_domain(self, lmin, lmax, nlb):
        """ Method to define the binning scheme of the model, and to
        return the multipole range, and the binned multipole range
        that the model is defined over.

        Parameters
        ----------
        lmin: int
            Minimum multipole to consider. Excludes the bin in which
            this falls.
        lmax: int
            Maximum multipole to consider. Excludes the bin in which
            this falls.
        nlb: int
            Width of binning scheme to apply.

        Returns
        -------
        tuple(object, ndarray)
            Binning operator used in applying the model, and array of
            effective multipoles.
        """
        nmtbin = setup_nmt_bin(self.nside, nlb)
        ellb = nmtbin(np.arange(0, 3 * self.nside))
        # sort out the slice of the arrays we will use if imposing
        # a minimum or maximum multipole.
        if lmin is not None:
            for i, lb in enumerate(ellb):
                if (lb - nlb/2. < lmin) and (lmin < lb + nlb/2.):
                    ibin_min = i + 1
                else:
                    ibin_min = 0
        else:
            ibin_min = 0
        if lmax is not None:
            for i, lb in enumerate(ellb[::-1]):
                if (lb - nlb/2. < lmax) and (lmax < lb + nlb/2.):
                    ibin_max = i
                else:
                    ibin_max = -1
        else:
            ibin_max = -1
        inds = slice(ibin_min, ibin_max)
        inds = slice(3, 30)
        print("Using indices after imposing limits: ", inds)
        return (nmtbin, ellb, inds)

    def do_model_setup(self, params):
        """ Method to calculate the power spectrum primordial and
        lensing templates for a given set of cosmological parameters.

        This computation requires that the lmax be set much higher
        that the lmax required in the final analys.s

        Parameters
        ----------
        params: dict
            Dictionary of cosmological parameters to be sent to CLASS.

        Returns
        -------
        tuple(array_like(float))
            Tuple containing the BB primordial and lensing templates.
        """
        try:
            params.pop('a_lens')
        except:
            pass
        params.update({
            'output': 'tCl pCl lCl',
            'l_max_scalars': 5000,
            'l_max_tensors': 2000,
            'modes': 's, t',
            'r': 1,
            'lensing': 'yes',
        })
        cosm = Class()
        cosm.set(params)
        cosm.compute()
        # get the lensed and raw power spectra up to the maximum
        # multipole used in the likelihood analysis. Multiply by
        # T_CMB ^ 2 to get from dimensionless to uK^2 units.
        lensed_cls = cosm.lensed_cl(3 * self.nside - 1)['bb'] *(2.7225e6) ** 2
        raw_cls = cosm.raw_cl(3 * self.nside - 1)['bb'] * (2.7225e6) ** 2
        # get ells, used in the calculation of the foreground model
        # over the same range.
        ells = cosm.raw_cl(3 * self.nside - 1)['ell']
        # do the house keeping for the CLASS code.
        cosm.struct_cleanup()
        cosm.empty()
        # calculate the lensing-only template
        lens_template = self.apply_coupling(lensed_cls - raw_cls)
        raw_cls = self.apply_coupling(raw_cls)
        # now separately do the foreground template setup.
        if self.marg:
            fg_template = np.zeros(3 * self.nside)
            fg_template[1:] = (ells[1:] / 80.) ** -2.4
            fg_template = self.apply_coupling(fg_template)
            return (raw_cls, lens_template, fg_template)
        return (raw_cls, lens_template)

    def apply_coupling(self, cl):
        if cl.ndim == 1:
            cls = np.zeros((4, len(cl)))
            cls[3] = cl
        return self.wsp.decouple_cell(self.wsp.couple_cell(cls))[3]
    
    def binned_model(self, params):
        """ Method to calculate the binned model vector for a given set of
        parameters.

        Parameters
        ----------
        params: tuple(float)
            Tuple containing the tensor to scalar ratio and lensing amplitude.

        Returns
        -------
        ndarray
            Array containing the binned model power spectrum for the given
            params.
        """
        mod = params[0] * self.templates[0] + params[1] * self.templates[1]
        if self.marg:
            mod += params[2] * self.templates[2]
        return mod

    def loglkl(self, params):
        """ Method to calculate the log likelihood of a given tensor
        to scalar ratio and lensing amplitude, given the input data.
        """
        # get model binned with the correct operator, and calcualte residuals compared to
        # the data
        res = self.binned_model(params) - self.cls
        # calculate the Gaussian log-likelihood - 1/2 (C_l - C_l_model)^T Cov^-1 (C_l - C_l_model)
        return - 0.5 * np.dot(res[self.inds].T, np.dot(self.cls_inv_covar[self.inds, self.inds],
                                                       res[self.inds]))

    def sample(self, nwalkers=100, nsamps=2000, discard=500):
        """ Given an instance of the Posterior object above, this function
        samples the posterior using emcee and returns an MCMC chain with
        a certain length (nwalkers * nsamps - discard), where a burn-in length
        removed.
        """
        if self.marg:
            pos0 = [0., 1., 0.]
        else:
            pos0 = [0., 1.]
        ndim = len(pos0)
        pos = [pos0 + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.loglkl)

        for _ in sampler.sample(pos, iterations=nsamps, progress=True):
            continue

        self.samples = sampler.get_chain(discard=discard, flat=True)
        self.save_samples()
        return

    def save_samples(self):
        """ Method to save sampels of posterior to disk.
        """
        np.savetxt(self.meta.posterior_samples(self.nlb, self.fitting_model,
                                               self.power_conf, self.likelihood_name),
                   self.samples)

    def load_samples(self):
        """ Method to load samples of posterior from disk.
        """
        samples = np.loadtxt(self.meta.posterior_samples(self.nlb,
                                                         self.fitting_model,
                                                         self.power_conf,
                                                         self.likelihood_name))
        if self.samples is None:
            self.samples = samples
        return samples

    def get_uncertainties(self):
        rpost = self.get_marginalized_r()
        apost = self.get_marginalized_AL()
        return (np.std(rpost), np.std(apost))

    def get_marginalized_r(self):
        self.load_samples()
        return self.samples[:, 0]

    def get_marginalized_AL(self):
        self.load_samples()
        return self.samples[:, 1]

    def setup_posterior_r_plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"$P(r)$")
        ax.set_xlabel(r"$r$")
        ax.set_xlim(-0.006, 0.006)
        return fig, ax

    def add_marginalized_posterior_r(self, ax):
        r = self.get_marginalized_r()
        ax.hist(r, bins=50, density=True, histtype='step')
        return 

    def plot_marginalized_posterior_r(self):
        fig, ax = self.setup_posterior_r_plot()
        self.add_marginalized_posterior_r(ax)
        fig.savefig(self.meta.marginalized_r_plot_fpath(self.fitting_model, self.likelihood_name),
                    bbox_inches='tight')
        return
    
    def plot_posterior(self):
        if self.marg:
            labels = [r'$r$', r'$A_L$', r'$A_{\rm fg}$']
        else:
            labels = [r'$r$', r'$A_L$']
        corner.corner(self.samples, show_titles=True, quantiles=(0.22, 0.84),
                      title_fmt=".5f", labels=labels, max_n_ticks=3,
                      bins=50)
        fig = plt.gcf()
        fig.savefig(self.meta.corner_posterior(self.fitting_model, self.power_conf,
                                               self.likelihood_name), bbox_inches='tight')
        plt.close('all')
        return 

    def fig_setup(self, plot_theory=True):
        fig, ax = plt.subplots(1, 1)
        if plot_theory:
            cmb_cls = self.compute_theory_cls()
            ells = np.arange(len(cmb_cls[0]))
            dl = ells * (ells + 1) / 2. / np.pi
            ax.semilogy(ells, dl * cmb_cls[2], color='k', linestyle='--',
                        label=r"${\rm Theory} \ \mathcal{D}_\ell^{\rm BB}$")
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_xlim(2, 600)
        ax.set_ylim(5e-5, 1e-1)
        ax.set_xlabel(r"${\rm Multipole} \ \ell_b$")
        ax.set_ylabel(r"$\mathcal{D}_{\ell_b} \ [{\rm \mu K}^2]$")
        return fig, ax

    def add_power_spectrum_lines(self, ax, signal=True, noise=True):
        dlb = self.ellb * (self.ellb + 1) / 2. / np.pi
        if signal:
            ax.plot(self.ellb, dlb * np.sqrt(1./np.diag(self.cls_inv_covar)),
                    label=r'$\sigma(\mathcal{D}_\ell)$', color='0.5')
        if noise:
            ax.errorbar(self.ellb, dlb * self.cls, yerr=dlb * np.sqrt(1./np.diag(self.cls_inv_covar)),
                        fmt='d', label=r'$\hat \mathcal{D}_{\ell_b}$', fillstyle='none', color='0.2')


    def plot_power_spectra(self):
        fig, ax = self.fig_setup()
        self.add_power_spectrum_lines(ax)
        ax.legend(loc=4, ncol=2)
        fig.savefig(self.meta.power_spectra_plot_fpath(self.fitting_model, self.power_conf, self.likelihood_name),
                    bbox_to_inches='tight')

    def plot_power_spectra_uncertainty(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(30, 300)
        #ax.set_ylim(5e-5, 1e-1)
        ax.set_xlabel(r"${\rm Multipole} \ \ell_b$")
        ax.set_ylabel(r"$\sqrt{\ell + 1/2}\sigma(C)_{\ell_b} \ [{\rm \mu K_{\rm CMB}}^2]$")
        c = np.sqrt(self.ellb + 0.5)
        ax.plot(self.ellb, c * np.sqrt(1./np.diag(self.cls_inv_covar)), label=r'$\hat C_{\ell_b}$')
        ax.legend(loc=4, ncol=2)
        fig.savefig(self.meta.power_spectra_uncertainty_plot_fpath(self.fitting_model, self.power_conf, self.likelihood_name),
                    bbox_to_inches='tight')

    
    def plot_power_spectrum_mask(self):
        fig = plt.figure(1)
        hp.mollview(self.mask, cbar=None, coord=['G', 'C'], fig=1)
        fig.savefig(self.meta.power_spectrum_mask_plot_fpath(),
                    bbox_inches='tight')
        plt.close('all')

def setup_nmt_bin(nside, nlb, bpws=None, weights=None, ells=None):
    """ Function to return a function which bins input power spectra
    according to some weighting scheme.

    cls_out = W * cls_in

    Parameters
    ----------
    nside: int
        Resolution parameter for the maps, the power spectra of which
        are being analyzed. This is used to set the default maximum
        multipole in NaMaster, which is lmax = 3 * nside - 1.
    nlb: int
        Width of bandpower bins, assuming the same bin width for all
        bins.
    bpws: array_like(int)
        Array containing an assignment for each multipole of which
        bin (e.g. [-1, -1, -1, 0, 0, 0, 1, 1, 1]  etc.) it is placed
        in. If not specified, is set to uniform binning for the given
        nlb (optional, default=None).
    weights: array_like(float)
        Array of weights. These are normalized to sum to one in each
        bandpower. If not set, set to uniform weighting (optional,
        default=None).
    ells: array_like(int)
        Array of multipoles over which `bpws` and `weights` are
        defined. If not specified, set to range from 0 to lmax
        (optional, default=None).

    Returns
    -------
    Function
        Function that accepts an input power spectrum, and returns
        the binned power spectrum.
    """
    if ells is None:
        ells = np.arange(0, 3 * nside)
    if bpws is None:
        bpws = np.zeros_like(ells) - 1
        i = 0
        # exclude ell = 0, 1, and then divide the multipoles > 1
        # into bins of width nlb. The last bin might be shorter
        # than this, of course.
        while i * nlb < len(bpws):
            bpws[2 + i * nlb : 2 + (i + 1) * nlb] = i
            i += 1
    # if weights are not specified, assume uniform weighting.
    if weights is None:
        weights = np.ones_like(bpws)
    # get the bin lables as a unique set for easier manipulation
    ibins = set(bpws)
    # check whether the last bin is too short and if so remove it
    if sum(1 for b in bpws if b == max(ibins)) < nlb:
        ibins.remove(max(ibins))
    # get list of bandpower labels, and remove those labelled as
    # -1, as we want to set these to zero always.
    try:
        ibins.remove(-1)
    except KeyError:
        pass
    # define an operator, `bin_opr` which is applied to a power
    # spectrum to do the binning.
    bin_opr = np.zeros((len(ibins), len(ells)))
    # the weights for each bin correspond to a row of this operator.
    # for each bin, get the corresponding weights by checking the
    # where the `bpws` array has the correct value, and accessing
    # the same part of the weights array. Assign this to the binning
    # operator after normalizing to sum to one.
    for ibin in ibins:
        bin_weights = weights[bpws == ibin]
        bin_opr[ibin, bpws == ibin] = bin_weights / np.sum(bin_weights)
    # now make the function that will be called when binning a spectrum
    def nmt_bin(cls_in):
        # check if the input is a single array with shape (3nside),
        # and if so bin this array.
        if cls_in.ndim == 1:
            if len(cls_in) != 3*nside:
                logging.warning('Incorrect length of power spectrum for'
                                ' binning')
                raise ValueError
            return np.dot(bin_opr, cls_in)
        # otherwise, check if shape (ncls, 3*nside), and binn all the
        # power spectra by iterating over the first dimension.
        elif cls_in.ndim == 2:
            if any([len(cl) != 3 * nside for cl in cls_in]):
                logging.warning('Incorrect length of power spectrum for'
                                ' binning')
                raise ValueError
            return np.array([np.dot(bin_opr, cl) for cl in cls_in])
        else:
            logging.warning("Too many dimensions for binning")
            raise ValueError
    return nmt_bin

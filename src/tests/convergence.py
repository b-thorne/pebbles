""" In this script we do some tests for the convergence of BFoRe. We
are looking to determine the correct choice of emcee hyperparmaters
nwalkers, nsamples, and nburn.
"""
import sys
import os
import pebbles
import pysm
import numpy as np
import healpy as hp
import emcee
import matplotlib.pyplot as plt
import multiprocessing
import corner
import h5py
plt.style.use('blt_paper')

if __name__ == '__main__':
    # First let's generate some fake data.
    ipix = 10
    nside = 256
    nside_spec = 8
    npix = hp.nside2npix(nside)
    npix_spec = hp.nside2npix(nside_spec)
    mask = np.zeros(npix_spec)
    mask[ipix] = 1
    mask = hp.ud_grade(mask, nside_out=nside)
    simset = 'simset0'

    if simset == 'simset1':
        s1 = pysm.nominal.models("s1", nside_spec)
        d1 = pysm.nominal.models("d1", nside_spec)
        beta_d = d1[0]['spectral_index'][ipix]
        T_d = d1[0]['temp'][ipix]
        beta_s = s1[0]['spectral_index'][ipix]
    else:
        beta_d = 1.54
        T_d = 20.
        beta_s = -3

    if sys.argv[1] == 'sample':
        peb = pebbles.Pebbles(nside=nside, nside_spec=nside_spec,
                              gal_tag=simset, ins_tag='sofidV3_wn_smo',
                              cos_tag='planck2015', mask=mask)
        try:
            data = peb.load_simulated_data()
        except FileNotFoundError:
            data = peb.compute_simulated_data()
        tasks = peb._prepare_simulated_data(data, [beta_s, beta_d, T_d])
        # get hm1 settings for sampling this pixel
        _, _, _, maplike, pos0, mc_fpath, _ = tasks[0]
        nwalkers = 20
        ndim = 3
        max_n = 7000
        nburn = 2500
        # initial positions of the walkers
        pos = [pos0 + 0.01 * np.random.randn(ndim) for i in range(nwalkers)]
        # initiate emcee sample
        if 'dev' in  emcee.__version__:
            filename = "tutorial.h5"
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)

            with multiprocessing.Pool(processes=8) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                maplike.
                                                marginal_spectral_likelihood,
                                                pool=pool, backend=backend)
                # This will be useful to testing convergence
                old_tau = np.inf
                autocorr = []
                # Now we'll sample for up to max_n steps
                for p, log_prob, rstat in sampler.sample(pos, iterations=max_n, progress=True):
                    # Only check convergence every 100 steps
                    if sampler.iteration % 100:
                        continue
                    # Compute the autocorrelation time so far
                    # Using tol=0 means that we'll always get an estimate even
                    # if it isn't trustworthy
                    tau = sampler.get_autocorr_time(tol=0)
                    autocorr.append((sampler.iteration, np.mean(tau)))
                    # Check convergence
                    converged = np.all(tau * 100. < sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                    if converged:
                        break
                    old_tau = tau
                autocorr = np.array(autocorr)
                tau = sampler.get_autocorr_time()
                samples = sampler.get_chain(discard=int(2. * np.max(tau)),
                                            thin=int(0.5 * np.min(tau)),
                                            flat=True)
                np.savetxt("samples.txt", samples)
                np.savetxt("autocorr.txt", autocorr.T)
        else:
            with multiprocessing.Pool(processes=8) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                maplike.
                                                marginal_spectral_likelihood,
                                                pool=pool)
                # This will be useful to testing convergence
                old_tau = np.inf
                autocorr = []
                # Now we'll sample for up to max_n steps
                for i, sample in enumerate(sampler.sample(pos,
                                                          iterations=max_n)):
                    # Only check convergence every 100 steps
                    if i % 100:
                        continue
                    print(i)
                samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
                np.savetxt("samples.txt", samples)

    elif sys.argv[1] == 'ml':
        peb = pebbles.Pebbles(nside=nside, nside_spec=nside_spec,
                              gal_tag=simset, ins_tag='sofidV3_wn_smo',
                              cos_tag='planck2015', mask=mask)
        try:
            data = peb.load_simulated_data()
        except FileNotFoundError:
            data = peb.compute_simulated_data()
        tasks = peb._prepare_simulated_data(data, [beta_s, beta_d, T_d])
        # get hm1 settings for sampling this pixel
        _, _, _, maplike, pos0, mc_fpath, _ = tasks[0]
    else:
        samples = np.loadtxt('samples.txt')
        autocorr = np.loadtxt("autocorr.txt", unpack=True)

    fig = corner.corner(samples, range=[(-3.1, -2.9), (1.4, 1.7), (10, 27)],
                        truths=(beta_s, beta_d, T_d), truth_color="C1",
                        plot_contours=True, show_titles=True,
                        quantiles=(0.22, 0.84))
    fig.savefig("/home/ben/Projects/simonsobs/mapspace/plots/triangle{:s}.png".format(simset))

    fig, ax = plt.subplots(1, 1)
    ax.plot(autocorr[:, 0], autocorr[:, 1])
    ax.set_xlabel(r"$n_{\rm iteration}$")
    ax.set_ylabel(r"$\langle \tau_f \rangle$")
    fig.savefig("/home/ben/Projects/simonsobs/mapspace/plots/autocorr{:s}.png".format(simset))
    plt.show()

""" Script to investigate the non-Gaussianity of foreground
residuals in cleaned CMB maps. This will hopefully provide
a metric to define masks on which cosmological power spectra
can be calculated to avoid having to marginalize over
residuals.

Notes:
- Compute mu3 and mu4 for foregrounds at 150GHz and calculate
the percentage residual required to bias the kurtosis and
skewness to a significant enough level for detection.
"""

from scipy.stats import skew, skewtest, normaltest, kurtosis
import scipy.stats as stats
import numpy as np
import corner
import pebbles
import healpy as hp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from functools import reduce
import pysm

def gal(nside, fwhm=None):
    """ Create image of Galaxy in synchrotron and dust at
    150 GHz.
    """
    s1 = pysm.nominal.models('s1', nside)
    d1 = pysm.nominal.models('d1', nside)
    sky = pysm.Sky({
        'synchrotron': s1,
        'dust': d1})
    c = pysm.common.K_RJ2Jysr(150.) / pysm.common.K_CMB2Jysr(150.)
    return hp.smoothing(sky.signal()(150.), fwhm=fwhm) * c

def gal_mask(image, threshold):
    """ For a given map impose a threshold on it to define
    a new mask. All pixels with values above the threshold
    are masked.
    """
    mask = np.ones_like(image)
    mask[image > threshold] = 0
    return mask

if __name__ == '__main__':
    """ This script performs skewness tests on residual maps from
    foreground cleaning.
    """
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help='Name of experiment setup.')
    parser.add_argument('--cmb_gen', action='store_true', default=False,
                        help='If supplied, generate CMB realizations.')
    parser.add_argument('--cmb_mc_stats', action='store_true', default=False)
    ARGS = parser.parse_args()
    nmc = 400
    fwhm = 30.

    # now use a galactic mask.
    # create Galactic mask
    (galT, galQ, galU) = gal(256, np.pi/180.*2.)
    galP = np.sqrt(galQ ** 2 + galU ** 2)
    mask = gal_mask(galP, 2.)
    hp.mollview(galP)
    hp.mollview(mask)
    plt.show()

    # load results of cleaning
    config_dict = pebbles.configurations.run[ARGS.name]
    res = pebbles.Residuals(**config_dict)

    if ARGS.cmb_gen:
        cl_config = pebbles.configurations.cos['planck2015']['params']
        class_cls = pebbles.pebbles.class_spectrum(cl_config)
        for i in range(nmc):
            synth_cmb = np.array(hp.synfast(class_cls, res.nside, pol=True,
                                            new=True, verbose=False,
                                            fwhm=np.pi/180.*fwhm/60.))
            fname = "maps/nongaussian/cmb{:04d}_fwhm{:02d}amin.fits".format(i, int(fwhm))
            hp.write_map(fname, synth_cmb, overwrite=True)

    if ARGS.cmb_mc_stats:
        synth_map = np.empty((nmc, 2 * hp.nside2npix(res.nside)))
        for i in range(nmc):
            fname = "maps/nongaussian/cmb{:04d}_fwhm{:02d}amin.fits".format(i, int(fwhm))
            synth_map[i] = hp.read_map(fname, verbose=False,
                                       field=(1, 2)).ravel()

        skews = skew(synth_map, axis=1)
        kurts = kurtosis(synth_map, axis=1)
        stats = np.concatenate((skews[None, :], kurts[None, :]))
        fname = 'maps/stats_cmb_nmc{:04d}_fwhm{:04d}amin.txt'.format(nmc, int(fwhm))
        np.savetxt(fname, stats.T)

        fig, ax1 = plt.subplots(1, 1)
        pkwargs = {
            'bins': 20,
            'range': None,
            'histtype': 'step',
            'density': True,
            'color': 'k',
        }
        if fwhm == 90.:
            pkwargs.update({'range': (-0.2, 0.2)})
        if fwhm == 5.:
            pkwargs.update({'range': (-0.02, 0.02)})
        ax1.set_title("Skew and kurt from {:d} simulations of CMB \n"
                      "with FWHM {:02d} amin".format(nmc, int(fwhm)))
        ax1.hist(skews, linestyle='--', label='Skewness', **pkwargs)
        ax1.hist(kurts, linestyle='-', label='Kurtosis', **pkwargs)
        ax1.set_xlabel(r"$\mu_3$ or $\mu_4$")
        ax1.set_ylabel(r"frequency")
        ax1.legend()
        fig.savefig('plots/nongaussian/cmb_mc_stats_nmc'
                    '{:04d}_fwhm{:02d}amin.png'.format(nmc, int(fwhm)),
                    bbox_inches='tight')
        plt.close('all')

    # calculate a stack of nmc cleaned realizations, and a version
    # of the stack in which the true CMB has been removed.
    cleaned_cmb_stack = res.calc_stacked_amp_map('cmb', 50)
    true_cmb = hp.smoothing(res.cmb_true, fwhm=np.pi/180.*fwhm/60.,
                            verbose=False)[1:]

    # now make a mask from the SO hits count. Restrict map to this region
    # and recheck the skewness.
    mask = 1. - pebbles.configurations.fitting_mask(res.nside, thresh=0.1)

    masked_stack = np.ma.masked_array(data=cleaned_cmb_stack[0],
                                      mask=mask,
                                      fill_value=hp.UNSEEN)
    masked_true = np.ma.masked_array(data=true_cmb[0],
                                     mask=mask,
                                     fill_value=hp.UNSEEN)
    # plot histograms
    fig, ax = plt.subplots(1, 1)
    pkwargs = {
        'bins': 100,
        'range': (-2, 2),
        'histtype': 'step',
        'density': True,
    }
    ax.hist(true_cmb.ravel(), label='true', **pkwargs)
    ax.hist(cleaned_cmb_stack.ravel(), label='cleaned stack', **pkwargs)
    ax.legend()
    fig.savefig("plots/nongaussian/historgram.png", bbox_inches='tight')
    plt.close()

    # plot the maps
    if fwhm == 90.:
        kwargs = {'min': -2, 'max': 2}
    else:
        kwargs = {'min': -10, 'max': 10}
    fig = plt.figure(1, figsize=(10, 9))
    hp.mollview(cleaned_cmb_stack[0], fig=1, sub=(221), title="Q Stack",
                unit=r'${\rm \mu K}$', **kwargs)
    hp.mollview(masked_stack, fig=1, sub=(222), title="Q Stack",
                unit=r'${\rm \mu K}$', **kwargs)
    hp.mollview(true_cmb[0], fig=1, sub=(223), title="Q True",
                unit=r'${\rm \mu K}$', **kwargs)
    hp.mollview(masked_true, fig=1, sub=(224), title="Q True",
                unit=r'${\rm \mu K}$', **kwargs)
    fig.savefig('plots/nongaussian/{:s}_stack_res_maps_fwhm'
                '{:02d}amin.png'.format(ARGS.name, int(fwhm)),
                bbox_inches='tight')
    plt.close(1)

    # now we calculate the skewness statistic and the p-value for these
    # maps. If residuals were due only to noise, we would expect these
    # skewness statistics to be consistent with zero.
    print("True CMB skew: ", skew(true_cmb.ravel()))
    print("Stacked CMB skew: ", skew(cleaned_cmb_stack.ravel()))
    nthresh = 20
    threshs = np.linspace(0.1, 0.9, nthresh)
    nrow = np.ceil(nthresh / 2.)
    ncol = 2
    fig = plt.figure(1, figsize=(ncol * 5, nrow * 2.8))
    fskies = []
    for j, i in enumerate(threshs):
        mask = 1. - pebbles.configurations.fitting_mask(res.nside,
                                                        thresh=float(i))
        fsky = 1 - np.mean(mask)
        fskies.append(fsky)
        demo_map = np.ma.masked_array(data=cleaned_cmb_stack[0],
                                      mask=mask,
                                      fill_value=hp.UNSEEN)
        #hp.mollview(demo_map, fig=1, sub=(nrow, ncol, j + 1), min=-3, max=3)
    #fig.savefig("plots/nongaussian/threshold_masks.png", bbox_inches='tight')
    plt.close(1)

    statistics = []
    for i in threshs:
        mask = 1. - pebbles.configurations.fitting_mask(res.nside, thresh=i)
        masked_stack = np.ma.masked_array(data=cleaned_cmb_stack[0],
                                          mask=mask,
                                          fill_value=hp.UNSEEN)
        masked_true = np.ma.masked_array(data=true_cmb[0],
                                     mask=mask,
                                     fill_value=hp.UNSEEN)
        # calculate the 3rd and 4th moments over the CMB MC simulations on this mask.
        true_cmb_skew = []
        true_cmb_kurt = []
        for j in range(nmc):
            fname = "maps/nongaussian/cmb{:04d}_fwhm{:02d}amin.fits".format(j, int(fwhm))
            q, u = hp.read_map(fname, verbose=False, field=(1, 2))
            maskq = np.ma.masked_array(data=q, mask=mask,
                                       fill_value=hp.UNSEEN)
            masku = np.ma.masked_array(data=u, mask=mask,
                                       fill_value=hp.UNSEEN)
            mask_cmb = np.concatenate((maskq.compressed(), masku.compressed()))
            true_cmb_kurt.append(kurtosis(mask_cmb))
            true_cmb_skew.append(skew(mask_cmb))

        true_cmb_skew_std = np.std(np.array(true_cmb_skew))
        true_cmb_kurt_std = np.std(np.array(true_cmb_kurt))

        statistics.append(
            np.array(
                [
                    skew(masked_stack.compressed()),
                    kurtosis(masked_stack.compressed()),
                    skew(masked_true.compressed()),
                    kurtosis(masked_true.compressed()),
                    true_cmb_skew_std,
                    true_cmb_kurt_std
                ]
            )
        )
    statistics = np.array(statistics)
    np.savetxt("maps/mask_var_statistics.txt", statistics, fmt="%.4f")

    # read in stats of skew and kurt for later comparison
    fname = "maps/stats_cmb_nmc{:04d}_fwhm{:04d}amin.txt".format(nmc, int(fwhm))
    skews, kurts = np.loadtxt(fname, unpack=True)
    avg_skew = np.mean(skews) * np.ones_like(statistics[:, 0])
    std_skew = np.std(skews) * np.ones_like(statistics[:, 0])

    avg_kurt = np.mean(kurts) * np.ones_like(statistics[:, 0])
    std_kurt = np.std(kurts) * np.ones_like(statistics[:, 0])

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Skewness comparison")
    ax.plot(fskies, statistics[:, 0], 'k-', label='Cleaned skew')
    ax.plot(fskies, statistics[:, 2], 'C0-', label='True CMB skew')
    ax.fill_between(fskies, - 1 * statistics[:, 4], 1 * statistics[:, 4],
                    color='k', alpha=0.3, label=r'$1-\sigma$ CMB MC spread')
    #ax.fill_between(fskies, avg_skew - std_skew, avg_skew + std_skew, color='k',
    #                alpha=0.5, label=r'$1-\sigma$ CMB MC spread')
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel(r"$f_{\rm sky}$")
    ax.set_ylabel(r"$\mu_3$")
    ax.legend(loc=3, ncol=2)
    fig.savefig('plots/nongaussian/{:s}_skew_threshold_'
                'masks.png'.format(ARGS.name),
                bbox_inches='tight')

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Kurtosis comparison")
    ax.plot(fskies, statistics[:, 1], 'k-', label='Cleaned kurtosis')
    ax.plot(fskies, statistics[:, 3], 'C0-', label='True CMB kurtosis')
    ax.fill_between(fskies, - 1 * statistics[:, 5], 1 * statistics[:, 5],
                    color='k', alpha=0.3, label=r'$1-\sigma$ CMB MC spread')
    #ax.fill_between(fskies, avg_kurt - 3 * std_kurt, avg_kurt + 3 * std_kurt,
    #                color='k', alpha=0.3, label=r'$3-\sigma$ CMB MC spread')
    # 
    # ax.fill_between(fskies, avg_kurt - std_kurt, avg_kurt + std_kurt,
    #color='k', alpha=0.5, label=r'$1-\sigma$ CMB MC spread')
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel(r"$f_{\rm sky}$")
    ax.set_ylabel(r"$\mu_4$")
    ax.legend(loc=3, ncol=2)
    fig.savefig('plots/nongaussian/{:s}_kurt_threshold_'
                'masks.png'.format(ARGS.name),
                bbox_inches='tight')


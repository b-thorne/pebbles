""" This script compares the estimated BB power spectra for various settings.
"""
import pebbles
import numpy as np
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('supermongo')
import argparse
import os

def setup_run():
    """ Function to deal with the arguments passed to the script.
    """
    arp = argparse.ArgumentParser()
    arp.add_argument('configuration', type=str)
    arp.add_argument('-nside', type=int, default=256)
    arp.add_argument('-nmc', type=int, default=200)
    args = arp.parse_args()
    # this function combines all the lists of settings for each option to
    # produce an iterable over all the combinations of settings to be run.
    print(args.configuration)
    jobs = pebbles.read_config_file(args.configuration)
    # if running in a parallel environment set the cpu count.
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    confname = os.path.basename(args.configuration)[:-5]
    return args.nside, args.nmc, jobs, confname

if __name__ == '__main__':

    NSIDE, NMC, JOBS, CONFNAME = setup_run()
    # Plot a comparison of the power spectra.
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel(r"$\mathcal{D}_{\ell_b} \ [{\rm \mu k^2}]$")
    ax.set_xlabel(r"Multipole $\ell_b$")

    counter = 0
    for sim, cos, ins, fit, pwr, lkl in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                             'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
        posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
        cl_mean, cl_inv_covar = posterior.load_data()
        dlb = posterior.ellb * (posterior.ellb + 1) / 2. / np.pi
        dlb = 1
        yerr = dlb * np.sqrt(1. / np.diag(cl_inv_covar))
        ax.errorbar(posterior.ellb + counter, dlb * cl_mean, yerr=yerr, fmt='d', fillstyle='none',
                    label=ins+fit+pwr)
        counter += 1

    ax.set_xlim(30, 100)
    ax.set_xscale('log')
    lgd = ax.legend(bbox_to_anchor=(1., 1.), loc='upper left')
    fig.savefig(os.path.expandvars("$PEBBLES/plots/spectra_{:s}.pdf".format(CONFNAME)),
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


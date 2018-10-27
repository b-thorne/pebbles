#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6
""" This script compares the estimated BB power spectra for various settings.
"""
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['xtick.minor.size'] = 0
mpl.rcParams['xtick.minor.width'] = 0
import matplotlib.pyplot as plt
import pebbles
import numpy as np
import multiprocessing
import yaml
plt.style.use('supermongo')
import argparse
import os
from cycler import cycler
from pathlib import Path
from matplotlib import ticker

def setup_comparison_plot(rinput=0, ylim=(None, None)):
    """ Function to set up figure and axes for r constraint
    comparison plots.
    """
    fig = plt.figure(figsize=(3.3, 3.))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'$10^3 \hat r$')
    ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
    ax.minorticks_on()
    if rinput is not None:
        ax.axhline(y=rinput, color='k', linestyle='--')
    return fig, ax

_plot_dir = Path(os.path.expandvars('$PEBBLES_PLOTS')) / 'paper' / 'r_comparison'

if __name__ == '__main__':
    NSIDE = 256
    NMC = 200
    JOBS_SO21 = pebbles.read_config_file(os.path.expandvars("$HOME/Projects/simonsobs/mapspace/src/paper_configurations/simset1_so_21_betad_pix.yml"))
    JOBS_SO11 = pebbles.read_config_file(os.path.expandvars("$HOME/Projects/simonsobs/mapspace/src/paper_configurations/simset1_so_11_betad_pix.yml"))
    # Plot a comparison of the power spectra.
    y_bias_11 = []
    y_std_11 = []
    for sim, cos, ins, fit, pwr, lkl in JOBS_SO11('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                             'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
        posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
        samples = posterior.load_samples()
        y_bias_11.append(np.mean(samples, axis=0)[0] * 1e3)
        y_std_11.append(np.std(samples, axis=0)[0] * 1e3)

    y_bias_21 = []
    y_std_21 = []
    for sim, cos, ins, fit, pwr, lkl in JOBS_SO21('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                             'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
        posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
        samples = posterior.load_samples()
        y_bias_21.append(np.mean(samples, axis=0)[0] * 1e3)
        y_std_21.append(np.std(samples, axis=0)[0] * 1e3)

    x = np.array([1, 2, 3])
    fig, ax = setup_comparison_plot(rinput=0)
    ax.set_title("simset 1")
    ax.errorbar(x-0.1, y_bias_21, yerr=y_std_21, capsize=6, fmt='d', fillstyle='none', label='opt. goal')
    ax.errorbar(x + 0.1, y_bias_11, yerr=y_std_11, capsize=6, fmt='s', fillstyle='none', label='opt. baseline')
    ax.set_ylim(-2.5, 5)
    ax.set_xticks(x)
    ax.set_xticklabels([1, 4, 12])
    ax.set_xlabel(r"$N_{\rm spec}$")
    ax.legend()
    fig.savefig(_plot_dir / 'compare_so11_so21_betad_pix.pdf', bbox_inches='tight')
   
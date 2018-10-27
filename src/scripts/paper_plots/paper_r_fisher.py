#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6

""" This script compares the Fisher estimate of the tensor
to scalar ratio with the forecasts of the simset0 case for
varying instrument parameters: SO11, SO21, SO10, SO20.
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
    JOBS = pebbles.read_config_file(os.path.expandvars('$HOME/Projects/simonsobs/mapspace/src/paper_configurations/simset0_fisher.yml'))
    # Plot a comparison of the power spectra.
    y_bias = []
    y_std = []
    for sim, cos, ins, fit, pwr, lkl in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                             'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
        posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
        samples = posterior.load_samples()
        y_bias.append(np.mean(samples, axis=0)[0] * 1e3)
        y_std.append(np.std(samples, axis=0)[0] * 1e3)
    fisher = [1.8, 1.4, 1.2, 0.9]    
    data = np.array([1, 2, 3, 4])
    fig, ax = setup_comparison_plot(rinput=0)
    ax.errorbar(data + 0.1, y_bias, yerr=y_std, capsize=6, fmt='d', fillstyle='none', label='simset0')
    ax.errorbar(data - 0.1, np.zeros_like(data), yerr=fisher, capsize=6, fmt='d', fillstyle='none', label='Fisher')
    ax.set_xticks(data)
    ax.set_xticklabels(["pess. \n baseline", "opt. \n baseline", "pess. \n goal", "opt. \n goal"])
    ax.set_ylim(-2.5, 5)
    fig.savefig(_plot_dir / 'fisher_comparison.pdf', bbox_inches='tight')
   
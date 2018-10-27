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

def setup_run():
    """ Function to deal with the arguments passed to the script.
    """
    arp = argparse.ArgumentParser()
    arp.add_argument('configuration', type=str)
    arp.add_argument('-nside', type=int, default=256)
    arp.add_argument('-nmc', type=int, default=200)
    arp.add_argument('-label', type=str, default='sim,fit,pwr')
    args = arp.parse_args()
    # this function combines all the lists of settings for each option to
    # produce an iterable over all the combinations of settings to be run.
    labels = args.label.split(',')
    jobs = pebbles.read_config_file(args.configuration)
    # if running in a parallel environment set the cpu count.
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    confname = os.path.basename(args.configuration)[:-4]
    
    with open(args.configuration, 'r') as file_obj:
        config = yaml.load(file_obj)
    title = config['title']
    labels = config['labels']
    return args.nside, args.nmc, jobs, confname, title, labels

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
    NSIDE, NMC, JOBS, CONFNAME, TITLE, XLABELS = setup_run()
    # Plot a comparison of the power spectra.
    y_bias = []
    y_std = []
    for sim, cos, ins, fit, pwr, lkl in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                             'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
        posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
        samples = posterior.load_samples()
        y_bias.append(np.mean(samples, axis=0)[0] * 1e3)
        y_std.append(np.std(samples, axis=0)[0] * 1e3)
    
    fig, ax = setup_comparison_plot(rinput=0)
    ax.set_title(TITLE)
    ax.set_prop_cycle(cycler(marker=['o', 'd', 's', 'D', '*']))
    ax.errorbar(XLABELS, y_bias, yerr=y_std, capsize=6, fmt='d', fillstyle='none')
    ax.set_ylim(-2.5, 5)
    fig.savefig(_plot_dir / '{:s}.pdf'.format(CONFNAME), bbox_inches='tight')
   
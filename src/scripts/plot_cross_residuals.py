#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6
""" 
Description
-----------
Script to plot the result of cross correlating the best-fit
CMB amplitude maps with tempaltes of dust and synchrotron emission.

The power spectra to be plotted should be previously calculated by
`postprocessing.py`.

Usage
-----
Requires one argument, the path to the configuration file containing
the run settings to be plotted.
"""
import matplotlib
matplotlib.use('Agg')
import multiprocessing
import os
from pathlib import Path
import argparse
import pebbles
import numpy as np
import pysm
import pymaster as nmt
import matplotlib.pyplot as plt
from tqdm import tqdm

def setup_run():
    """ Function to deal with the arguments passed to the script.
    """
    arp = argparse.ArgumentParser()
    arp.add_argument('configuration', type=str)
    args = arp.parse_args()
    # this function combines all the lists of settings for each option to
    # produce an iterable over all the combinations of settings to be run.
    jobs = pebbles.read_config_file(args.configuration)
    # if running in a parallel environment set the cpu count.
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    return args.nside, args.nmc, jobs, args

_plot_dir = Path(os.path.expandvars('$PEBBLES_PLOTS'))

if __name__ == '__main__':
    # parse the arguments to this script and set environment variables.
    NSIDE, NMC, JOBS, STEPS = setup_run()
    # Loop through the cleaned simulations and calculate the cross spectrum with
    # foreground templates.
    for sim, cos, ins, fit, pwr, lkl in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                                 'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
        (ellb, cl_d, cl_s) = np.load(_plot_dir / "{:s}_{:s}_{:s}_xdust_xsync_cls.npy".format(sim, ins, fit))
        
        cl_d_mean = np.mean(cl_d, axis=0)[3]
        cl_d_cov = np.cov(cl_d[:, 3, :], rowvar=False)
        cl_d_inv_cov = np.linalg.inv(cl_d_cov)
        cl_d_err = 1. / np.sqrt(np.diag(cl_d_inv_cov))
        cl_s_mean = np.mean(cl_s, axis=0)[3]
        cl_s_cov = np.cov(cl_s[:, 3, :], rowvar=False)
        cl_s_inv_cov = np.linalg.inv(cl_s_cov)
        cl_s_err = 1. / np.sqrt(np.diag(cl_s_inv_cov))
        
        dlb = ellb * (ellb + 1) / 2. / np.pi
        
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(ellb, dlb * cl_d_mean, yerr=dlb * cl_d_err, fmt='d', fillstyle='none', color='k', label='Dust BB')
        ax.errorbar(ellb+5, dlb * cl_s_mean, yerr=dlb * cl_s_err, fmt='o', fillstyle='none', color='k', label='Sync BB')
        ax.legend()
        ax.set_xlabel(r"${\rm Multipole} \ \ell$")
        ax.set_ylabel(r"$\mathcal{D}_{\ell_b} \ [{\rm \mu K^2}]$")
        ax.set_xlim(30, 300)
        ax.set_yscale('log')
        #ax.set_yscale('symlog', linthresh=1e-6)
        fig.savefig(_plot_dir / "residuals" / "{:s}_{:s}_{:s}_res.png".format(sim, ins, fit))
#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6

#SBATCH -N 1
#SBATCH -t 05:30:00
#SBATCH -q premium
#SBATCH -L SCRATCH
#SBATCH -C haswell
""" 
Description
-----------
Script to calculate summaries of various parts of the run, and reduce the
amount of data for plotting.

- Stack CMB maps to isolate foreground residuals (just average of all Monte Carlo
noise and CMB realizations.)
- Stack maximum likelihood spectral parameter maps in the same way.
- Calculate summary statistics of the emcee chains calculated when sampling the
final power spectruml likelihood.
- Calculate the mean and covariance of the power spectrum band powers.
- Calculate the cross power with synchrotron and dust templates.

Use
---

This script has one positional argument, the path to the configuration file
to be run. There are several optional arguments:

- nside
- nmc
- stack
- stats_rpost
- cross
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
    arp.add_argument('-nside', type=int, default=256)
    arp.add_argument('-nmc', type=int, default=200)
    arp.add_argument('-stack', action='store_true')
    arp.add_argument('-stats_rpost', action='store_true')
    arp.add_argument('-cross', action='store_true')
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

    if STEPS.stack:
        # Loop through the cleaned simulations and stack the cleaned CMB
        # Q and U maps.
        print("Stacking cleaned Monte Carlo simulations.")
        for sim, cos, ins, fit in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS', 'FITTING_MODELS'):
            for comp in ['dust', 'synch', 'cmb']:
                qu = np.zeros((2, 12 * NSIDE ** 2))
                print("\t Working on {:s}".format(comp))
                for imc in tqdm(range(NMC), ncols=82):
                    peb = pebbles.Pebbles(NSIDE, sim, cos, ins, nmc=NMC)
                    qu[0] += peb.load_cleaned_amp_maps(fit, comp, 'q', imc)
                    qu[1] += peb.load_cleaned_amp_maps(fit, comp, 'u', imc)
                qu /= float(NMC)
                peb.save_stacked_cleaned_amp_maps(fit, comp, qu)

        # Loop through best-fit spectral parameters and stack them. Plot the stacked best-fit
        # values.
        for sim, cos, ins, fit in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS', 'FITTING_MODELS'):
            peb = pebbles.Pebbles(NSIDE, sim, cos, ins, nmc=NMC)
            for field in [0, 1]:
                par = peb.load_cleaned_spec_maps(fit, field)
        
    if STEPS.stats_rpost:
        # Loop through the sampled posteriors for r, A_L, and possibly A_fg. Compute the
        # marginalized posterior for each of these parameters, and compute summary
        # statistics of that distribution such as the mean, median, std, interquartile
        # range etc.
        print("Computing summary statistics of posterior.")
        for sim, cos, ins, fit, pwr, lkl in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                                 'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
            posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
            samples = posterior.load_samples()
            # calculate summary statistics
            mean = np.mean(samples, axis=0)
            median = np.median(samples, axis=0)
            std = np.std(samples, axis=0)
            # calculate marginalized quantities and bin them in histograms for quicker
            # plotting
            np.savetxt(_plot_dir / "{:s}.txt".format(posterior.meta.simulation_tag), np.array([mean, median, std]))        

    if STEPS.cross:
        # Loop through the cleaned simulations and calculate the cross spectrum with
        # foreground templates.
        print("Cross correlating with synchrotron and dust templates.")
        mask = pebbles.configurations.masking.so_mask_hits(NSIDE)
        nmtbin = nmt.NmtBin(NSIDE, nlb=20)
        beam = pebbles.powerspectra.beam_profile(NSIDE, 30.)
        for sim, cos, ins, fit in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS', 'FITTING_MODELS'):
            s1 = pysm.nominal.models('s1', NSIDE)[0]
            sync = nmt.NmtField(mask, [s1['A_Q'], s1['A_U']], purify_b=True)
            d1 = pysm.nominal.models('d1', NSIDE)[0]
            dust = nmt.NmtField(mask, [d1['A_Q'], d1['A_U']], purify_b=True)
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(dust, sync, nmtbin)
            peb = pebbles.Pebbles(NSIDE, sim, cos, ins, nmc=NMC)
            cls_dust = np.zeros((NMC, 4, nmtbin.get_n_bands()))
            cls_sync = np.zeros((NMC, 4, nmtbin.get_n_bands()))
            for imc in tqdm(range(NMC), ncols=82):
                qu = np.zeros((2, 12 * NSIDE ** 2))
                qu[0] += peb.load_cleaned_amp_maps(fit, 'cmb', 'q', imc)
                qu[1] += peb.load_cleaned_amp_maps(fit, 'cmb', 'u', imc)
                cmb = nmt.NmtField(mask, qu, purify_b=True, beam=beam)
                cls_dust[imc] = pebbles.powerspectra.compute_pseudo_and_decouple(cmb, dust, wsp)
                cls_sync[imc] = pebbles.powerspectra.compute_pseudo_and_decouple(cmb, sync, wsp)
            np.save(_plot_dir / "{:s}_{:s}_{:s}_xdust_xsync_cls.npy".format(sim, ins, fit), [nmtbin.get_effective_ells(), cls_dust, cls_sync])
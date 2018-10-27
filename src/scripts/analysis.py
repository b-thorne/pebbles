#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6

#SBATCH -N 1
#SBATCH -t 05:30:00
#SBATCH -q premium
#SBATCH -L SCRATCH
#SBATCH -C haswell

"""
Description
-----------

This script manages the submission of jobs for:
- Running simulations using PEBBLES.
- Cleaning simulations using BFoRe.
- Computing power spectra from cleaned CMB maps.
- Sampling the power spectrum likelihood.
- Some plotting routines.

Use
---

This script has one positional argument, which is the path to the
configuration file specifying the settings of the code to run. This
should be a YAML file, specifying:
- simset
- instrument model
- cosmological model
- fitting model
- power spectrum hyperparameters
- likelihood hyperparameters

There are then a set of optional flags indicating which stage of the
pipeline to run:

- '-sim'
- '-fit'
- '-power'
- '-sample'
"""
import multiprocessing
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from schwimmbad import JoblibPool
import numpy as np
import pebbles
plt.style.use('supermongo')

def setup_run():
    """ Function to parse the arguments to this script and
    set any environment variables, such as CPU count, that
    are required.
    """
    arp = argparse.ArgumentParser()
    arp.add_argument('configuration', type=str)
    arp.add_argument('-nside', type=int, default=256)
    arp.add_argument('-nmc', type=int, default=200)
    arp.add_argument('-sim', action='store_true')
    arp.add_argument('-fit', action='store_true')
    arp.add_argument('-power', action='store_true')
    arp.add_argument('-sample', action='store_true')
    args = arp.parse_args()
    jobs = pebbles.read_config_file(args.configuration)
    # set number of threads for running on cori
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    return (args.nside, args.nmc, jobs, args)


if __name__ == '__main__':
    (NSIDE, NMC, JOBS, STEPS) = setup_run()

    if STEPS.sim:
        for sim, cos, ins in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS'):
            peb = pebbles.Pebbles(NSIDE, sim, cos, ins, nmc=NMC)
            peb.compute_simulated_data()

    if STEPS.fit:
        for sim, cos, ins, fit in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS', 'FITTING_MODELS'):
            peb = pebbles.Pebbles(NSIDE, sim, cos, ins, nmc=NMC)
            data = peb.load_simulated_data()
            with JoblibPool(multiprocessing.cpu_count()) as pool:
                peb.clean_simulated_data(data, fit, pool)

    if STEPS.power:
        for sim, cos, ins, fit, pwr in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                            'FITTING_MODELS', 'POWERS'):
            powerspectra = pebbles.PowerSpectra(pwr, NSIDE, sim, cos, ins, nmc=NMC)
            powerspectra.calc_mc_power(fitting_model=fit, instrument=None)

    if STEPS.sample:
        for sim, cos, ins, fit, pwr, lkl in JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS',
                                                 'FITTING_MODELS', 'POWERS', 'LIKELIHOODS'):
            posterior = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)
            posterior.sample()
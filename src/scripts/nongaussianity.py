#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6

#SBATCH -N 1
#SBATCH -t 05:30:00
#SBATCH -q premium
#SBATCH -L SCRATCH
#SBATCH -C haswell

""" This script makes some computations of the non-Gaussianity of the
cleaned CMB maps. We assess whether or not some measures of higher order
information may be used to detect the presence of foreground residuals
at the map level.
"""
import numpy as np
import pebbles
from pebbles.configurations.masking import so_mask_fitting

def nongaussianity(nside, simset, cosmology, instrument, fitting_model,
                   power, lkl, nmc=50):
    """ Function calculating skewness and kurtosis of a given setup.
    """
    nong = pebbles.Nongaussianity(fitting_model, nside, simset,
                                  cosmology, instrument, nmc=nmc)
    obs, var = nong.cmb_and_noise_simulations()
    thresholds = np.linspace(0.01, 0.5, 10)
    means = []
    stds = []
    means_res = []
    stds_res = []
    res = pebbles.Pebbles(nside, simset, cosmology, instrument, nmc=nmc)
    for threshold in thresholds:
        statistics_arr = []
        statistics_arr_res = []
        for i, (ob, noiseivar) in enumerate(zip(obs, var)):
            q = res.load_cleaned_amp_maps(fitting_model, 'cmb', 'q', i)
            u = res.load_cleaned_amp_maps(fitting_model, 'cmb', 'u', i)
            cmb_noise_sim = nong.scaled_noise_maps((ob, noiseivar), [-3., 1.5])
            arr_res = pebbles.nongaussianity.apply_mask(np.array((q, u)),
                                                        so_mask_fitting(
                                                            nside, threshold))
            arr = pebbles.nongaussianity.apply_mask(cmb_noise_sim,
                                                    so_mask_fitting(nside,
                                                                    threshold))
            sk = pebbles.nongaussianity.stats(arr)
            sk_res = pebbles.nongaussianity.stats(arr_res)
            statistics_arr.append(sk)
            statistics_arr_res.append(sk_res)
        statistics = np.array(statistics_arr)
        statistics_res = np.array(statistics_arr_res)
        means.append(np.mean(statistics, axis=0))
        stds.append(np.std(statistics, axis=0))
        means_res.append(np.mean(statistics_res, axis=0))
        stds_res.append(np.std(statistics_res, axis=0))
    means_res = np.array(means_res)
    stds_res = np.array(stds_res)
    means = np.array(means)
    stds = np.array(stds)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Skewness')
    l1, = ax1.plot(thresholds, means[:, 0], label='CMB + Noise sims')
    ax1.fill_between(thresholds, means[:, 0] - stds[:, 0],
                     means[:, 0] + stds[:, 0], color=l1.get_color(), alpha=0.5)

    l2, = ax1.plot(thresholds, means_res[:, 0], label='Cleaned full sky sims')
    ax1.fill_between(thresholds, means_res[:, 0] - stds_res[:, 0],
                     means_res[:, 0] + stds_res[:, 0], color=l2.get_color(),
                     alpha=0.5)
    ax1.legend()

    ax2.set_title('Kurtosis')
    l3, = ax2.plot(thresholds, means[:, 1])
    ax2.fill_between(thresholds, means[:, 1] - stds[:, 1],
                     means[:, 1] + stds[:, 1], color=l3.get_color(), alpha=0.5)

    l4, = ax2.plot(thresholds, means_res[:, 1])
    ax2.fill_between(thresholds, means_res[:, 1] - stds_res[:, 1],
                     means_res[:, 1] + stds_res[:, 1], color=l4.get_color(),
                     alpha=0.5)
    ax2.legend()
    fig.savefig("{:04d}_{:s}_{:s}_{:s}_{:s}_nmc{:04d}.png".format(nside,
                simset, cosmology, instrument, fitting_model, nmc))
    return

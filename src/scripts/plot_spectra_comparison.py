""" Script to plot constraints on tensor to scalar ratio. 

This script makes several plots, each compares a set of
possible scenarios.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use('supermongo')
plt.rcParams['text.usetex'] = False

def setup_comparison_plot(rinput=0, ylim=(None, None)):
    """ Function to set up figure and axes for r constraint
    comparison plots.
    """
    fig = plt.figure(figsize=(4.1, 3.3))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'$10^3 \hat r$')
    fmtr = ticker.ScalarFormatter()
    fmtr.set_powerlimits((2, 3))
    fmtr.set_scientific(True)
    fmtr.set_useMathText(True)
    ax.yaxis.set_major_formatter(fmtr)
    ax.minorticks_on()
    ax.set_ylim(ylim)
    if rinput is not None:
        ax.axhline(y=rinput, color='k', linestyle='--')
    return fig, ax

if __name__ == '__main__':

    # Compare the results for different input foreground models.
    x_data = [r'simset1', r'simset2', r'simset3']
    y_bias = [0.0005, 0.002, 0.004]
    y_sigma = [0.003, 0.004, 0.005]

    fig, ax = setup_comparison_plot(rinput=0, ylim=(-0.5e-2, 1e-2))
    ax.errorbar(x_data, y_bias, yerr=y_sigma, fmt='o', color='gray', capsize=6, fillstyle='none')
    fig.savefig('/home/ben/Dropbox/Papers/soforecast/simset_r_comparison.pdf',
                bbox_inches='tight')

    # Compare the results of fitting spectral parameters with varying
    # nside resolution.
    x_data = [0, 1, 2]
    y_bias = [0.0005, 0.002, 0.004]
    y_sigma = [0.003, 0.004, 0.005]

    fig, ax = setup_comparison_plot(ylim=(-0.5e-2, 1e-2))
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel(r'$n_{\rm side}$')
    ax.errorbar(x_data, y_bias, yerr=y_sigma, fmt='o', color='gray', capsize=6, fillstyle='none')
    fig.savefig('/home/ben/Dropbox/Papers/soforecast/nside_r_comparison.pdf',
                bbox_inches='tight')

    # Compare the results for this method with those from x-forecast
    # and the Cl method. Numbers taken from the SO science forecasts paper.
    # Comapre these for the different experimental setups, i.e. baseline and
    # goal sensitivities for SO, and optimistic and pessimistic
    # knee multipoles.

    # First these results
    x = [0.9, 1.9, 2.9, 3.9]
    y_bias = [1.1, 2.2, 1.3, 1.8]
    y_sigma = [3.4, 1.7, 2.1, 0.8]

    # x-forecast results
    x_xf = [1.1, 2.1, 3.1, 4.1]
    y_bias_xf = [1.3, 1.6, 1.3, 1.5]
    y_sigma_xf = [2.1, 1.5, 1.3, 1.0]

    # cl -mcmc results
    x_cl = [1.3, 2.3, 3.3, 4.3]
    y_bias_cl = [1.7, 2.2, 2.0, 2.2]
    y_sigma_cl = [2.1, 2.0, 1.7, 1.7]

    fig, ax = setup_comparison_plot(ylim=(-4, 6))
    ax.set_xticks([1.1, 2.1, 3.1, 4.1])
    ax.set_xticklabels(['base \n pess', 'base \n opt', 'goal \n pess', 'goal \n opt'])
    ax.axhline(y=0, color='k', linestyle='--')
    ax.errorbar(x, y_bias, yerr=y_sigma, fmt='o', color='0.1', capsize=6,
                fillstyle='none', label='These results')
    ax.errorbar(x_xf, y_bias_xf, yerr=y_sigma_xf, fmt='d', color='0.25', capsize=6,
                fillstyle='none', label='x-forecast')
    ax.errorbar(x_cl, y_bias_cl, yerr=y_sigma_cl, fmt='s', color='0.5', capsize=6,
                fillstyle='none', label=r'$C_\ell$ method')
    lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    fig.savefig('/home/ben/Dropbox/Papers/soforecast/instrument_r_comparison.pdf',
                bbox_inches='tight', bbox_extra_artists=(lgd,))


    # Compare the results when masking to different degrees.
    x_data = [0.10, 0.09, 0.07, 0.05]
    y_bias = np.array([2.2, 1.9, 1.8, 1.4])
    y_sigma = np.array([1.7, 1.7, 1.8, 2.8])

    fig, ax = setup_comparison_plot(ylim=(-2, 5))
    ax.set_xlabel(r'$f_{\rm sky}^{\rm eff}$')    
    ax.errorbar(x_data, y_bias, yerr=y_sigma, fmt='*', color='k',
                capsize=2, fillstyle='none', dash_capstyle='projecting')
    fig.savefig('/home/ben/Dropbox/Papers/soforecast/gal_mask_r_comparison.pdf',
                bbox_inches='tight')

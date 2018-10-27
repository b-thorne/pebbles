#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6
""" This script compares the estimated BB power spectra for various settings.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.modeling.blackbody import blackbody_nu
import numpy as np
import astropy.units as units
from matplotlib import ticker
from pathlib import Path
import os
plt.style.use('supermongo')
mpl.rcParams['xtick.minor.size'] = 0
mpl.rcParams['xtick.minor.width'] = 0

def mbb(freq, freq_ref, temp, spec_index):
    ratio = blackbody_nu(freq, temp) / blackbody_nu(freq_ref, temp)
    return  (freq / freq_ref) ** spec_index * ratio 

def sync(freq, freq_ref, spec_index):
    return (freq / freq_ref) ** spec_index

def cmb(freq, freq_ref):
    t_cmb = 2.7225 * units.K
    flux_bb_ratio = blackbody_nu(freq, t_cmb) / blackbody_nu(freq_ref, t_cmb)
    rj_ratio = (freq / freq_ref) ** -2
    return  flux_bb_ratio * rj_ratio

_plot_dir = Path(os.path.expandvars('$PEBBLES_PLOTS')) / 'paper' 

if __name__ == '__main__':
    freqs = np.arange(10, 1000, dtype=np.float32) * units.GHz
    freq_ref = 150. * units.GHz
    temp = 20 * units.K
    spec_index_d = 1.6 - 2
    spec_index_s = -1. - 2

    
    dust_sed = 10. * mbb(freqs, freq_ref, temp, spec_index_d)
    sync_sed = 0.1 * sync(freqs, freq_ref, spec_index_s)
    cmb_sed = 5. * cmb(freqs, freq_ref)
    
    fig = plt.figure(figsize=(3.3, 3.))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\nu$ (GHz)")
    ax.set_ylabel(r"$f(\nu; \nu_0, \beta) \ [\mu K_{\rm RJ} / \mu K_{\rm RJ}]$")
    ax.loglog(freqs, dust_sed, color='0.3', linestyle='--', label='Dust')
    ax.loglog(freqs, sync_sed, color='0.5', linestyle='-.', label='Synchrotron')
    ax.loglog(freqs, cmb_sed, color='0.1', linestyle='-', label='CMB')


    freqs = [27, 39, 93, 145, 225, 280]
    for f in freqs:
        ax.axvline(f, linestyle='-', alpha=0.3, linewidth=2)
    
    ax.set_xlim(20, 350)
    ax.set_ylim(1e0, 1e2)
    ax.set_xscale('log')
    
    ax.set_xticks([20, 100, 200])
    ax.set_xticklabels([20, 100, 200])
    ax.yaxis.set_major_formatter(ticker.LogFormatter())
    
    
    lgd = ax.legend(loc='lower left', bbox_to_anchor=(0., 1), ncol=2)
    fig.savefig(_plot_dir / "sed_comparison.pdf", bbox_inches='tight',
                bbox_extra_artists=(lgd,))
    

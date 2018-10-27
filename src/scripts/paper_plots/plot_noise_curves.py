#!/global/u1/b/bthorne/anaconda3/envs/pebbles/bin/python3.6
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import numpy as np
from pebbles import V3calc as v3
import pebbles
import os
from pathlib import Path
plt.style.use('supermongo')

_plot_dir =  Path(os.path.expandvars('$PEBBLES_PLOTS')) / 'paper'

if __name__ == '__main__':
    # calculate the noise curves from the V3 noise calculator
    ny_lf = 1
    fsky = 0.1
    nside = 512
    beam_corrected = False
    _, nll_sens1, sigma_amin = v3.so_V3_SA_noise(1, 1, ny_lf,
                                           fsky, 3 * nside,
                                           beam_corrected=beam_corrected)
    _, nll_sens2, sigma_amin = v3.so_V3_SA_noise(1, 1, ny_lf,
                                           fsky, 3 * nside,
                                           beam_corrected=beam_corrected)
    
    ell = np.linspace(2, 3* nside - 1, 3 * nside - 2)
    nll1 = np.array([ell * (ell + 1) / 2. / np.pi * nl for nl in nll_sens1])
    nll2 = np.array([ell * (ell + 1) / 2. / np.pi * nl for nl in nll_sens2])

    params = pebbles.configurations.cosmologies['planck2015_AL1']['params']

    fpath = Path(__file__).absolute().parent / 'cosmo_cls.txt'
    try:
        cl_class = np.loadtxt(fpath)[:, 2:3 * nside]
    except OSError:
        cl_class = pebbles.pebbles.class_spectrum(params)[:, 2:3 * nside]
        np.savetxt(fpath, cl_class)

    cl_class *= (ell * (ell + 1) / 2. / np.pi)[None, :]
    
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 4.3))
    ax.set_ylabel(r'$D_\ell \ [\mu K ^2]$')
    ax.set_xlabel(r'Multipole $\ell$')

    l1, = ax.loglog(ell, nll1[0], label='27', color="0.7", ls='-.')
    l3, = ax.loglog(ell, nll1[1], label='39', color="0.7", ls='--')
    l5, = ax.loglog(ell, nll1[2], label='93', color="0.4", ls='-.')
    l7, = ax.loglog(ell, nll1[3], label='145', color="0.4", ls='--')
    l9, = ax.loglog(ell, nll1[4], label='225', color="0.2", ls='-.')
    l11, = ax.loglog(ell, nll1[5], label='280', color="0.2", ls='--')
    llens, = ax.loglog(ell, cl_class[2], color='k', label="Lensing")

    ax.set_xlim(10, 500)
    ax.set_ylim(1e-4, 1e0)
    
    handles = [l1, l3, l5, l7, l9, l11, llens]
    ax.legend(handles=handles, loc=4, ncol=2 , title="Channel (GHz)")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    fig.savefig(_plot_dir / "so_noise_curves.pdf", bbox_inches='tight')

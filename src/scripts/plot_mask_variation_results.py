import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use('supermongo')


def setup_comparison_plot():
    fig = plt.figure(figsize=(3.1, 3.))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'$10^3 \hat r$')
    fmtr = ticker.ScalarFormatter()
    fmtr.set_powerlimits((2, 3))
    fmtr.set_scientific(True)
    fmtr.set_useMathText(True)
    ax.yaxis.set_major_formatter(fmtr)
    return fig, ax

if __name__ == '__main__':

    x_data = [0.10, 0.09, 0.07, 0.04]

    y_bias = np.array([2.2, 1.9, 1.8, 1.4])
    y_sigma = np.array([1.7, 1.7, 1.8, 2.8])
    fig, ax = setup_comparison_plot()
    ax.set_ylim(-2, 5)
    ax.minorticks_on()
    ax.set_xlabel(r'$f_{\rm sky}^{\rm eff}$')
    
    ax.axhline(y=0, color='k', linestyle='--')
    (_, caps, _) = ax.errorbar(x_data, y_bias, yerr=y_sigma, fmt='*', color='k',
                            capsize=2, fillstyle='none', dash_capstyle='projecting')
    fig.savefig('/home/ben/Dropbox/Papers/soforecast/gal_mask_r_comparison.pdf',
                bbox_inches='tight')
    plt.show()

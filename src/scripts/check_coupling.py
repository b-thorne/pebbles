""" This script checks the importance of accounting for the mode coupling
when binning the theory spectrum.
"""
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import pebbles
import sys
plt.style.use('supermongo')

if __name__ == '__main__':
    JOBS = pebbles.read_config_file(sys.argv[1])
    POWER_JOBS = JOBS('SIMSETS', 'COSMOLOGIES', 'INSTRUMENTS', 'FITTING_MODELS', 'POWERS', 'LIKELIHOODS')
    NSIDE = 256
    NMC = 4
    for sim, cos, ins, fit, pwr, lkl  in POWER_JOBS:
        ells = np.linspace(0, 3 * NSIDE - 1, 3 * NSIDE)
        post = pebbles.Posterior(lkl, fit, pwr, NSIDE, sim, cos, ins, nmc=NMC)

        (prim, lens, fg) = post.do_model_setup({**pebbles.configurations.cosmologies[cos]['params']})
        peb = pebbles.Pebbles(NSIDE, sim, cos, ins, nmc=NMC)
        theory_cl = np.zeros((4, 3 * NSIDE))
        theory_cl[0, :] = prim
        pwrspc = pebbles.PowerSpectra(pwr, NSIDE, sim, cos, ins, nmc=NMC)
        wsp = nmt.NmtWorkspace()
        wsp.read_from(str(peb.meta.wsp_mcm_fpath(pwrspc.nlb, fit, pwrspc.power_conf)))

        theory_cl_couple = wsp.couple_cell(theory_cl)
        theory_cl_decouple = wsp.decouple_cell(theory_cl_couple)
        theory_cl_bin = pwrspc.nmtbin.bin_cell(theory_cl)

        ellb = pwrspc.nmtbin.get_effective_ells()
        dllb = ellb * (ellb + 1) / 2. / np.pi
        dlls = ells * (ells + 1) / 2. / np.pi

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(ellb, dllb * theory_cl_decouple[0], label='Couple-decouple', linestyle='--')
        ax.plot(ellb, dllb * theory_cl_bin[0], label='Binned')
        ax.errorbar(ellb[post.inds], dllb[post.inds] * theory_cl_decouple[0, post.inds], fmt='d',
                    fillstyle='none', color='k')
        ax.set_xlim(2, 100)
        ax.set_yscale('linear')
        ax.set_xlabel('Multipole')
        ax.set_ylabel('Power')
        ax.legend()
        ax.minorticks_on()
        plt.show()

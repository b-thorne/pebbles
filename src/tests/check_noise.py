""" Script to run a check to see if the noise maps
produced by PEBBLES are agreeing with the theoretical
noise curves for the V3 calculators.
"""
import numpy as np
import sys
from pebbles import Pebbles
from pebbles.configurations import run, ins
from pebbles.moduleconfig import nos_fpath
from pebbles.pebbles import get_mask
from pebbles import V3calc as v3
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

if __name__ == '__main__':
    peb = Pebbles(**run[sys.argv[1]])

    nus = ins[peb.ins_tag]['nus']
    nmc = 0

    _, nll, sigma_amin = v3.so_V3_SA_noise(0, 0, 1.,
                                           0.1, 3 * peb.nside,
                                           remove_kluge=True)

    ells = np.arange(len(nll[0]))
    hit_mask = get_mask(peb.nside)
    apo_hit_mask = nmt.mask_apodization(hit_mask, 10, apotype="C1")
    fig, ax = plt.subplots(1, 1)

    for i, nu in enumerate(nus):
        fname = nos_fpath(peb.ins_tag, peb.nside, nu, nmc)
        noise_maps = hp.read_map(fname, field=(1, 2))
        #hp.mollview(noise_maps[0] * hit_mask)
        #plt.show()

        #f1 = nmt.NmtField(apo_hit_mask, noise_maps)
        #bins = nmt.NmtBin(peb.nside, nlb=5)
        #nmtcls = nmt.compute_full_master(f1, f1, bins)
        ax.loglog(ells, ells * (ells + 1) / 2. / np.pi * nll[i])
        #ax.loglog(bins.get_effective_ells(), nmtcls[0])
        #ax.loglog(bins.get_effective_ells(), nmtcls[3])
        ax.set_ylim(2e-5, 1e-1)

    plt.show()

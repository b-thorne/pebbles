""" This module implements some tests of the noise generation
funtions used in PEBBLES. We check a few parts of the noise
generation method:

- The power spectrum of the generated noise maps.
- Calculate the per-pixel noise of the realizations, assuming
  uniform white noise.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pebbles

def calculate_noise_stats(sigma, nside):
    # calculate the theory power spectrum
    cl_theory = pebbles.pebbles.sigma_amin_to_cell(sigma, nside)
    # calculate a map realization without going via the power spectrum.
    noise_map_pix = pebbles.pebbles.sigma_amin_to_map(sigma, nside)
    # calculate a map realization via the power spectrum
    noise_map_ps, _ = pebbles.pebbles.get_white_noise([sigma], nside)
    noise_map_ps = np.array(noise_map_ps[0][2])

    cl_pix = hp.anafast(noise_map_pix)
    cl_ps = hp.anafast(noise_map_ps)
    nmtbin = pebbles.powerspectra.setup_nmt_bin(nside, nlb=20)
    cl_theory = nmtbin(cl_theory)
    cl_pix = nmtbin(cl_pix)
    cl_ps = nmtbin(cl_ps)
    # calculate an estimate of the noise level from the pixel-based
    # realization and from the power spectrum-based realization
    sigma_out_pix = np.std(noise_map_pix) * sigma_pix_to_amin
    sigma_out_ps = np.std(noise_map_ps) * sigma_pix_to_amin
    return cl_theory, cl_pix, cl_ps, sigma_out_pix, sigma_out_ps

if __name__ == '__main__':
    # Now we test the variance of MC realizations of the different noise
    # maps by comparing to the supposed variance maps.
    # First test the V3 calculator.
    nmc = 100
    nsides = [16, 32, 64]
    for nside in nsides:
        print("Doing nside {:03d}".format(nside))
        wn_maps = []
        print("Testing white noise")
        for i in range(nmc):
            if not i % 20:
                print(i, '/', nmc)
            sigma = 4.
            noise_map = pebbles.pebbles.sigma_amin_p_to_maps(sigma, nside)
            _, var_map = pebbles.pebbles.get_white_noise([sigma], nside)
            wn_maps.append(noise_map)

        ivar_wn_dat = np.var(np.array(wn_maps), axis=0)
        ivar_wn_the = var_map[0]

        pmax = max(np.amax(ivar_wn_dat), np.amax(ivar_wn_the))
        fig = plt.figure(1, figsize=(5, 4))
        hp.mollview(ivar_wn_dat[1], min=0, max=pmax,
                    title="", fig=1, sub=(2, 1, 1))
        hp.mollview(ivar_wn_the[0], min=0, max=pmax,
                    title="", fig=1, sub=(2, 1, 2))
        fig.savefig("test_wn_noise_var_nside{:03d}.png".format(nside),
                    bbox_inches='tight')
        plt.close(1)

        fig, ax = plt.subplots(1, 1)
        ax.hist(ivar_wn_the[0], range=(0, pmax),
                histtype='step', label='theory', density=True, bins=100)
        ax.hist(ivar_wn_dat[1], range=(0, pmax),
                histtype='step', label='data', density=True, bins=100)
        ax.set_xlabel("Pixel value (pixel variance)")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.savefig("test_noise_wn_var_nside{:03d}_hist.png".format(nside),
                    bbox_inches='tight')
        plt.close('all')

        v3_maps = []
        print("Testing V3 noise claculator")
        for i in range(nmc):
            if not i % 20:
                print(i, '/', nmc)
            v3_noise_map, v3_var_map = pebbles.pebbles.get_noise_sim(
                nside_out=nside,
                knee_mode=1)
            v3_maps.append(v3_noise_map[:, 1:])

        ivar_v3_dat = 1. / np.var(np.array(v3_maps), axis=0)
        ivar_v3_the = 1. / v3_var_map

        mask = pebbles.pebbles.get_mask(nside)
        md = lambda d: np.ma.masked_array(mask=(mask < 1e-6),
                                          data=d,
                                          fill_value=hp.UNSEEN)

        for i, (ivar_dat, ivar_the) in enumerate(zip(ivar_v3_dat, ivar_v3_the)):
            pmax = max(np.amax(ivar_dat), np.amax(ivar_the))
            fig = plt.figure(1, figsize=(5, 4))
            hp.mollview(md(ivar_dat[0]), min=0, max=pmax,
                        title="channel {:d}".format(i), fig=1, sub=(2, 1, 1))
            hp.mollview(md(ivar_the[0]), min=0, max=pmax,
                        title="channel {:d}".format(i), fig=1, sub=(2, 1, 2))
            fig.savefig("test_noise_var_channel_{:d}_nside{:03d}.png".format(i, nside),
                        bbox_inches='tight')

            plt.close(1)

            fig, ax = plt.subplots(1, 1)
            ax.hist(md(ivar_the[0]).compressed(), range=(0, pmax),
                    histtype='step', label='theory', density=True, bins=100)
            ax.hist(md(ivar_dat[0]).compressed(), range=(0, pmax),
                    histtype='step', label='data', density=True, bins=100)
            ax.set_xlabel("Pixel value (pixel variance)")
            ax.set_ylabel("Frequency")
            ax.legend()
            fig.savefig("test_noise_var_channel_{:d}_nside{:03d}_hist.png".format(i, nside),
                        bbox_inches='tight')
            plt.close('all')


    # First test the function `get_white_noise`, which is
    # supposed to take a list of instrument sensitivities,
    # expressed in units of uKamin, and an nside and
    # generate a correpsonding list of noise maps. The sigmas
    # should correspond to a polarization noise level, the
    # list of maps returned will be a list of (T, Q, U) noise
    # realizations, where the temperature noise is assumed
    # to be a factor of root(2) worse.
    sigmas = [2]
    nside = 128
    npix = hp.nside2npix(nside)
    # calculate some useful quantities for comparing different
    # units
    amin2 = 4. * np.pi * (10800. / np.pi) ** 2
    sigma_pix_to_amin = np.sqrt(amin2 / npix)
    pixwin = hp.sphtfunc.pixwin(nside)[:3 * nside]
    # For each sigma
    for sigma in sigmas:
        nmc = 50
        out = [calculate_noise_stats(sigma, nside) for _ in range(nmc)]
        cl_theory = sum((i for i, j, k, l, m in out)) / float(nmc)
        cl_pix = sum((j for i, j, k, l, m in out)) / float(nmc)
        cl_ps = sum((k for i, j, k, l, m in out)) / float(nmc)
        sigma_out_pix = sum((l for i, j, k, l, m in out)) / float(nmc)
        sigma_out_ps = sum((m for i, j, k, l, m in out)) / float(nmc)
        fig, ax = plt.subplots(1, 1)
        ax.plot(cl_theory, 'C0-', label="Theory curve")
        ax.plot(cl_pix, 'C1--', label='Pixel-based realization')
        ax.plot(cl_ps, 'C2--', label='Power spectrum-based realization')
        ax.plot(cl_theory, '-.')
        ax.set_ylim(2.5e-7, 4e-7)
        lab = fig.gca().annotate(
            "Input = {:f} uKamin \n"
            "Output pix = {:f} uKamin \n"
            "Output ps = {:f} uKamin \n"
            "".format(sigma, sigma_out_pix, sigma_out_ps),
            xy=(.5, 1.), xycoords="figure fraction",
            xytext=(-20, -10), textcoords="offset points",
            ha="right", va="top")
        ax.legend()
        fig.savefig("test_nmc{:04d}.pdf".format(nmc), bbox_extra_artists=(lab,),
                    bbox_to_inches='tight')
        plt.close('all')

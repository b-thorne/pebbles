from __future__ import print_function
import numpy as np

def so_V3_SA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimates these for you
    return(np.array([27,39,93,145,225,280]))

def so_V3_SA_beams():
    ## returns the LAC beams in arcminutes
    beam_SAC_27 = 91.
    beam_SAC_39 = 63.
    beam_SAC_93 = 30.
    beam_SAC_145 = 17.
    beam_SAC_225 = 11.
    beam_SAC_280 = 9.
    return(np.array([beam_SAC_27,beam_SAC_39,beam_SAC_93,beam_SAC_145,beam_SAC_225,beam_SAC_280]))

def so_V3_SA_noise(sensitivity_mode,
                   one_over_f_mode,
                   SAC_yrs_LF,
                   f_sky,
                   ell_max,
                   delta_ell=1,
                   beam_corrected=False,
                   remove_kluge=False):
    ## retuns noise curves, including the impact of the beam for the SO small aperature telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     0: threshold,
    #     1: baseline,
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    #     2: none
    # SAC_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAC
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the compuatioan of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERATURE
    ## LARGE APERATURE
    # configuraiton
    if (SAC_yrs_LF > 0):
        NTubes_LF  = SAC_yrs_LF/5. + 1e-6  ## reguarlized in case zero years is called
        NTubes_MF  = 2 - SAC_yrs_LF/5.
    else:
        NTubes_LF  = -SAC_yrs_LF/5. + 1e-6  ## reguarlized in case zero years is called
        NTubes_MF  = 2
    NTubes_UHF = 1.
    # sensitivity
    S_SA_27  = np.array([32,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([17,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([4.6,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([5.5,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([11,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([26,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f pol:  see http://simonsobservatory.wikidot.com/review-of-hwp-large-aperture-2017-10-04
    f_knee_pol_SA_27  = np.array([30.,15.,1.])
    f_knee_pol_SA_39  = np.array([30.,15.,1.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.,1.])
    f_knee_pol_SA_145 = np.array([50.,25.,1.])  ## from ABS, improving possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.,1.])
    f_knee_pol_SA_280 = np.array([100.,40.,1.])
    #alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])  ## roughly consistent with Yuji's table, but ectrapolated
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3.,-3.,-3.])  ## roughly consistent with Yuji's table, but ectrapolated

    ####################################################################
    ## calculate the survey area and time
    t = 5. * 365. * 24. * 3600.    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    if remove_kluge==False :
        t = t* 0.85  ## a kluge for the noise non-uniformity of the map edges
        #print("when generating relizations from a hits map, the total integration time should be 1/0.85 longer")
        #print("since we should remove a cluge for map non-uniformity since this is included correcly in a hits map")
    A_SR = 4. * np.pi * f_sky  ## sky areas in Steridians
    A_deg =  A_SR * (180./np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")

    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2, ell_max, delta_ell)

    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_SA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_SA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_SA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)

    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels = np.sqrt(2)*np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise level: ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the astmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_SA_27[one_over_f_mode] )**alpha_pol[0] + 1.
    AN_P_39  = (ell / f_knee_pol_SA_39[one_over_f_mode] )**alpha_pol[1] + 1.
    AN_P_93  = (ell / f_knee_pol_SA_93[one_over_f_mode] )**alpha_pol[2] + 1.
    AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.
    AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.
    AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280

    ## include the imapct of the beam
    SA_beams = so_V3_SA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
    ## lac beams as a sigma expressed in radians
    if beam_corrected :
        N_ell_P_27  *= np.exp( ell*(ell+1)* SA_beams[0]**2 )
        N_ell_P_39  *= np.exp( ell*(ell+1)* SA_beams[1]**2 )
        N_ell_P_93  *= np.exp( ell*(ell+1)* SA_beams[2]**2 )
        N_ell_P_145 *= np.exp( ell*(ell+1)* SA_beams[3]**2 )
        N_ell_P_225 *= np.exp( ell*(ell+1)* SA_beams[4]**2 )
        N_ell_P_280 *= np.exp( ell*(ell+1)* SA_beams[5]**2 )
    ## make an array of nosie curves for T
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])

    ####################################################################
    return(ell,N_ell_P_SA,Map_white_noise_levels)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    ## run the code to generate noise curves
    spectra = []
    fsky_SAC = 0.1
    bands = so_V3_SA_bands()
    for sens in [1, 2]:
        for fmode in [0, 1]:
            sens_mode = sens
            one_over_f_mode_SAC = fmode
            SAC_yrs_LF = 1
            ell, N_ell_SA_Pol, WN_levels = so_V3_SA_noise(sens_mode,
                                                          one_over_f_mode_SAC,
                                                          SAC_yrs_LF,
                                                          fsky_SAC,
                                                          500,
                                                          1)
            spectra.append((N_ell_SA_Pol, sens, fmode))
            ## plot the polarization noise curves
    try:
        if sys.argv[1] == 'lensing':
            from classy import Class
            from . import pebbles
            from . import configurations
            cls = pebbles.class_spectrum(configurations.cos['planck2015_AL1']['params'])
            np.save("lensing_cls", cls)
    except IndexError:
        pass
    lensing_cls = np.load("/home/ben/Projects/simonsobs/V3_calc/lensing_cls.npy")

    (speca, sensa, fmodea) = spectra[1]
    (specb, sensb, fmodeb) = spectra[3]
    conv = ell * (ell + 1) / 2. / np.pi

    fig, ax = plt.subplots(1, 1)
    ax.set_title(r"SAT BB noise spectra $\ell_{\rm knee}=$optimistic")
    ax.loglog(ell, 1e7*conv * speca[0], 'k',
              label="sens {:d}, fmode {:d} (baseline)".format(sensa, fmodea))
    ax.loglog(ell, 1e7*conv * specb[0], 'k--',
              label="sens {:d}, fmode {:d} (goal)".format(sensb, fmodeb))
    for i in range(6):
        l1, = ax.loglog(ell, conv * speca[i],
                        label="{:02d} GHz".format(int(bands[i])))
        ax.loglog(ell, conv * specb[i], '--', color=l1.get_color())
        ax.loglog(ell, conv * lensing_cls[2][:len(ell)], 'k')
    lgd = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(1e-5, 2e-1)
    ax.set_xlim(10, 1000)

    ax.set_ylabel(r"$\frac{\ell(\ell+1)}{2\pi}N_\ell(\nu) \ [{\rm \mu K^2}]$")
    ax.set_xlabel(r"$\ell$")
    fig.savefig("baseline_and_goal_kneeopt.pdf", bbox_extra_artists=(lgd,),
                bbox_inches='tight')

    (specc, sensc, fmodec) = spectra[0]
    (specd, sensd, fmoded) = spectra[2]
    conv = ell * (ell + 1) / 2. / np.pi
    fig, ax = plt.subplots(1, 1)
    ax.set_title(r"SAT BB noise spectra $\ell_{\rm knee}=$pessimitic")
    ax.loglog(ell, 1e7*conv * specc[0], 'k',
              label="sens {:d}, fmode {:d} (baseline)".format(sensc, fmodec))
    ax.loglog(ell, 1e7*conv * specd[0], 'k--',
              label="sens {:d}, fmode {:d} (goal)".format(sensd, fmoded))
    for i in range(6):
        l1, = ax.loglog(ell, conv * specc[i],
                        label="{:02d} GHz".format(int(bands[i])))
        ax.loglog(ell, conv * specd[i], '--', color=l1.get_color())
        ax.loglog(ell, conv * lensing_cls[2][:len(ell)], 'k')
    lgd = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(1e-5, 2e-1)
    ax.set_xlim(10, 1000)

    ax.set_ylabel(r"$\frac{\ell(\ell+1)}{2\pi}N_\ell(\nu) \ [{\rm \mu K^2}]$")
    ax.set_xlabel(r"$\ell$")
    fig.savefig("baseline_and_goal_kneepess.pdf",
                bbox_extra_artists=(lgd,), bbox_inches='tight')



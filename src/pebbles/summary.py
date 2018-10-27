from .posterior import Posterior

def summary_statistics(*args, **kwargs):
    posterior = Posterior(verbose=False, *args, **kwargs)
    (sigma_r, sigma_al) = posterior.get_uncertainties()
    return (sigma_r, sigma_al)

def aggregate_statistics(nside, jobs, nmc):
    for (sim, cos, ins, fit, pwr, lkl) in jobs:
        print(sim, cos, ins, fit, pwr, lkl)
        print(summary_statistics(lkl, fit, pwr, nside, sim,
                                 cos, ins, nmc=nmc))


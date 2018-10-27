from .masking import so_mask_hits, so_mask_binary

powerspectras = {

    # SETTINGS VARYING GALAXY MASKING
    'nlb10_yp_galmask0_aposcale25': {
        'nlb': 10,
        'aposcale': 25.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 0,
    },

    'nlb10_yp_galmask1_aposcale25': {
        'nlb': 10,
        'aposcale': 25.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 1,
    },

    'nlb10_yp_galmask2_aposcale25': {
        'nlb': 10,
        'aposcale': 25.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 2,
    },

    'nlb10_yp_galmask3_aposcale25': {
        'nlb': 10,
        'aposcale': 25.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 3,
    },

    'nlb10_yp_galmask4_aposcale25': {
        'nlb': 10,
        'aposcale': 25.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 4,
    },

    # SETTINGS VARYING APODISATION SCALE
    'nlb10_yp_galmask4_aposcale05': {
        'nlb': 10,
        'aposcale': 5.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 4,
    },

    'nlb10_yp_galmask4_aposcale10': {
        'nlb': 10,
        'aposcale': 5.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 4,
    },

    'nlb10_yp_galmask4_aposcale20': {
        'nlb': 10,
        'aposcale': 20.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 4,
    },

    'nlb10_yp_galmask4_aposcale30': {
        'nlb': 10,
        'aposcale': 30.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 4,
    },

    'nlb10_yp_galmask4_aposcale40': {
        'nlb': 10,
        'aposcale': 40.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 4,
    },

    # SETTINGS VARYING PURIFICATION
    'nlb10_np_galmask4_aposcale40': {
        'nlb': 10,
        'aposcale': 40.,
        'mask': so_mask_hits,
        'purify_b': False,
        'gal_mask': 4,
    },

    'nlb10_np_galmask4_aposcale20': {
        'nlb': 10,
        'aposcale': 20.,
        'mask': so_mask_hits,
        'purify_b': False,
        'gal_mask': 4,
    },

    'nlb10_np_galmask4_aposcale05': {
        'nlb': 10,
        'aposcale': 5.,
        'mask': so_mask_hits,
        'purify_b': False,
        'gal_mask': 4,
    },

    # VARY ASPODIZATION SCALE FOR SMALL GALMASK0
    'nlb10_yp_galmask0_aposcale05': {
        'nlb': 10,
        'aposcale': 5.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 0,
    },

    'nlb10_yp_galmask0_aposcale10': {
        'nlb': 10,
        'aposcale': 10.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 0,
    },
    
    'nlb10_yp_galmask0_aposcale15': {
        'nlb': 10,
        'aposcale': 10.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 0,
    },

    'nlb10_yp_galmask0_aposcale40': {
        'nlb': 10,
        'aposcale': 40.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 0,
    },
    
    'nlb10_yp_galmask1_aposcale15': {
        'nlb': 10,
        'aposcale': 40.,
        'mask': so_mask_hits,
        'purify_b': True,
        'gal_mask': 0,
    },

    # ADDING IN MODE DEPROJECTION
    'nlb10_yp_aposcale25_dp': {
        'nlb': 10,
        'aposcale': 20.,
        'mask': so_mask_hits,
        'purify_b': True,
        'deproject_dust': True
    },
}

# Standard python imports
import numpy as np, matplotlib, matplotlib.pyplot as plt, time

# LVC-specific imports
import pyRing.waveform as wf
from pyRing.utils import *
from lalinference.imrtgr.nrutils import *

srate   = 4096.0
T       = 0.5
times   = np.linspace(-0.01, T/2., int(srate*T))

DS       = 1
Kerr     = 1
KHS_2012 = 1
MMRDNS   = 1
MMRDNP   = 1
TEOBPM   = 1

fig_length = 10
fig_heigth = 8

if(DS):
    # First, for comparison and unit-checks, plot a simple damped-sinusoid
    # Select only a single tensorial ('t') mode.
    DS_parameters = {'A'   : {'t': [1e-21]},
                     'f'   : {'t': [250.]} ,
                     't'   : {'t': [0.004]},
                     'tau' : {'t': [0.005]},
                     'phi' : {'t': [0.0]}  }
    DS_model = wf.Damped_sinusoids(DS_parameters['A']  ,
                                   DS_parameters['f']  ,
                                   DS_parameters['tau'],
                                   DS_parameters['phi'],
                                   DS_parameters['t']  )

    st_count = time.time()
    # Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_DS, hc_DS = DS_model.waveform(times)
    print('DS:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{Damped \, sinusoids}$')
    plt.plot(times, hp_DS, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_DS, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

if(Kerr):
    
    #Kerr waveform with the conjugate symmetry on the amplitudes
    Kerr_params = {'t0'          : 0.0    ,
                   'Mf'          : 66.61  ,
                   'af'          : 0.68637,
                   'distance'    : 450.0  ,
                   'inclination' : 0.0    ,
                   'phi'         : 0.0    }

    Kerr_amps = {(2,2,2,0): 1.1*np.exp(1j*(-2.0)), (2,2,2,1): 0.95*np.exp(1j*(+1.14159))}

    Kerr_model = wf.KerrBH(Kerr_params['t0']                   ,
                           Kerr_params['Mf']                   ,
                           Kerr_params['af']                   ,
                           Kerr_amps                           ,
                           Kerr_params['distance']             ,
                           Kerr_params['inclination']          ,
                           Kerr_params['phi']                  ,

                           reference_amplitude = 0.0           ,
                           geom                = 0             ,
                           qnm_fit             = 1             ,
                           interpolants        = None          ,
                           
                           Spheroidal          = 0             ,
                           amp_non_prec_sym    = 1             )

    st_count = time.time()
    # Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_Kerr, hc_Kerr = Kerr_model.waveform(times)

    print('Kerr:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{Kerr \, amp \, sym}$')
    plt.plot(times, hp_Kerr, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_Kerr, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    #Kerr waveform with the conjugate symmetry on the amplitudes and quadratic modes.
    Kerr_params = {'t0'          : 0.0    ,
                   'Mf'          : 66.61  ,
                   'af'          : 0.68637,
                   'distance'    : 450.0  ,
                   'inclination' : 1.0    ,
                   'phi'         : 0.0    }

    Kerr_amps      = {(2,2,2,0): 1.1*np.exp(1j*(-2.0)), (2,4,4,0): 0.95*np.exp(1j*(+1.14159))}
    Kerr_amps_quad = {
                      'sum' : {((2,4,4,0),(2,2,2,0),(2,2,2,0)): 1.1*np.exp(1j*(-2.0))},
                     }

    Kerr_model = wf.KerrBH(Kerr_params['t0']                   ,
                           Kerr_params['Mf']                   ,
                           Kerr_params['af']                   ,
                           Kerr_amps                           ,
                           Kerr_params['distance']             ,
                           Kerr_params['inclination']          ,
                           Kerr_params['phi']                  ,
                           
                           reference_amplitude = 0.0           ,
                           geom                = 0             ,
                           qnm_fit             = 1             ,
                           interpolants        = None          ,
                        
                           Spheroidal          = 0             ,
                           amp_non_prec_sym    = 1             ,
                           quadratic_modes     = Kerr_amps_quad,
                           quad_lin_prop       = 0             )

    st_count = time.time()
    # Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_Kerr, hc_Kerr = Kerr_model.waveform(times)

    print('Kerr:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{Kerr \, amp \, sym \, and \, quadratic}$')
    plt.plot(times, hp_Kerr, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_Kerr, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    #Kerr waveform with the conjugate symmetry on the amplitudes and a tail term (geom).
    Kerr_params = {'t0'          : 0.0    ,
                   'Mf'          : 66.61  ,
                   'af'          : 0.68637,
                   'distance'    : 450.0  ,
                   'inclination' : 0.0    ,
                   'phi'         : 0.0    }

    Kerr_amps       = {(2,2,2,0): 1*np.exp(1j*(0.0))}
    tail_parameters = {}
    tail_parameters[(2,2)] = {'A': 0.00001, 'phi': 0.0, 'p': -0.7}

    Kerr_model = wf.KerrBH(Kerr_params['t0']                    ,
                           Kerr_params['Mf']                    ,
                           Kerr_params['af']                    ,
                           Kerr_amps                            ,
                           Kerr_params['distance']              ,
                           Kerr_params['inclination']           ,
                           Kerr_params['phi']                   ,
                           
                           reference_amplitude = 0.0            ,
                           geom                = 1              ,
                           qnm_fit             = 1              ,
                           interpolants        = None           ,
                           
                           Spheroidal          = 0              ,
                           amp_non_prec_sym    = 1              ,
                           tail_parameters     = tail_parameters)

    st_count = time.time()
    # Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_Kerr, hc_Kerr = Kerr_model.waveform(times)

    print('Kerr:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{Kerr \, amp \, sym}$')
    plt.semilogy(times, hp_Kerr, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.semilogy(times, hc_Kerr, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    

    #Kerr waveform without the conjugate symmetry on the amplitudes
    Kerr_params = {'t0'          : 0.0    ,
                   'Mf'          : 66.61  ,
                   'af'          : 0.68637,
                   'distance'    : 450.0  ,
                   'inclination' : 0.0    ,
                   'phi'         : 0.0    }

    Kerr_amps = {(2,2,2,0): (1.1*np.exp(1j*(-2.0)), 1.2*np.exp(1j*(-2.0))), (2,2,2,1): (0.95*np.exp(1j*(+1.14159)), 0.8*np.exp(1j*(+1.14159)))}

    Kerr_model = wf.KerrBH(Kerr_params['t0']                   ,
                           Kerr_params['Mf']                   ,
                           Kerr_params['af']                   ,
                           Kerr_amps                           ,
                           Kerr_params['distance']             ,
                           Kerr_params['inclination']          ,
                           Kerr_params['phi']                  ,
                           
                           reference_amplitude = 0.0           ,
                           geom                = 0             ,
                           qnm_fit             = 1             ,
                           interpolants        = None          ,
                           
                           Spheroidal          = 0             ,
                           amp_non_prec_sym    = 0             )

    st_count = time.time()
    # Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_Kerr, hc_Kerr = Kerr_model.waveform(times)

    print('Kerr:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{Kerr}$')
    plt.plot(times, hp_Kerr, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_Kerr, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

if(MMRDNS):

    TGR_parameters   = {}
    qnm_interpolants = {}
    MMRDNS_modes     = [(2,2,0), (2,2,1), (2,1,0), (3,3,0), (3,3,1), (3,2,0), (4,4,0), (4,3,0), (5,5,0)]

    for (l,m,n) in MMRDNS_modes:
        interpolate_freq, interpolate_tau = qnm_interpolate(2,l,m,n)
        qnm_interpolants[(2,l,m,n)]  = {'freq': interpolate_freq, 'tau': interpolate_tau}
    
    MMRDNS_params     = {'t0'          : 0.004  ,
                         'Mf'          : 66.61  ,
                         'af'          : 0.68637,
                         'eta'         : 0.25   ,
                         'distance'    : 450.0  ,
                         'inclination' : 0.0    ,
                         'phi'         : 0.0    }

    MMRDNS_model = wf.MMRDNS(MMRDNS_params['t0']            ,
                             MMRDNS_params['Mf']            ,
                             MMRDNS_params['af']            ,
                             MMRDNS_params['eta']           ,
                             MMRDNS_params['distance']      ,
                             MMRDNS_params['inclination']   ,
                             MMRDNS_params['phi']           ,
                             TGR_parameters                 ,
                             single_l     = 2               ,
                             single_m     = 2               ,
                             single_n     = 0               ,
                             single_mode  = 0               ,
                             Spheroidal   = 0               ,
                             interpolants = qnm_interpolants,
                             qnm_fit      = 0               )


    st_count = time.time()
    #Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_MMRDNS, hc_MMRDNS = MMRDNS_model.waveform(times)
    
    print('RDNS:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{RDNS}$')
    plt.plot(times, hp_MMRDNS, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_MMRDNS, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

if(MMRDNP):

    TGR_parameters   = {}
    qnm_interpolants = {}
    MMRDNP_modes     = [(2,2,0),(2,1,0),(3,3,0),(3,2,0),(4,4,0),(4,3,0)]
    
    for (l,m,n) in MMRDNP_modes:
        interpolate_freq, interpolate_tau = qnm_interpolate(2,l,m,n)
        qnm_interpolants[(2,l,m,n)]  = {'freq': interpolate_freq, 'tau': interpolate_tau}
    
    MMRDNP_params     = {'t0'          : 0.004,
                         'm1'          : 35.0 ,
                         'm2'          : 35.0 ,
                         'chi1'        : 0.0  ,
                         'chi2'        : 0.0  ,
                         'distance'    : 450.0,
                         'inclination' : 0.0  ,
                         'phi'         : 0.0  }

    if(MMRDNP_params['chi1'] < 0): tilt1_fit = np.pi
    else: tilt1_fit = 0.0
    if(MMRDNP_params['chi2'] < 0): tilt2_fit = np.pi
    else: tilt2_fit = 0.0
    chi1_fit  = np.abs(MMRDNP_params['chi1'])
    chi2_fit  = np.abs(MMRDNP_params['chi2'])
    MMRDNP_params['Mf']   = bbh_final_mass_projected_spins(MMRDNP_params['m1'], MMRDNP_params['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016')
    MMRDNP_params['af']   = bbh_final_spin_projected_spins(MMRDNP_params['m1'], MMRDNP_params['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016', truncate = bbh_Kerr_trunc_opts.trunc)
    
    MMRDNP_params['Mi']   = MMRDNP_params['m1'] + MMRDNP_params['m2']
    MMRDNP_params['eta']  = (MMRDNP_params['m1']*MMRDNP_params['m2'])/(MMRDNP_params['Mi'])**2
    MMRDNP_params['chis'] = (MMRDNP_params['m1']*MMRDNP_params['chi1'] + MMRDNP_params['m2']*MMRDNP_params['chi2'])/(MMRDNP_params['Mi'])
    MMRDNP_params['chia'] = (MMRDNP_params['m1']*MMRDNP_params['chi1'] - MMRDNP_params['m2']*MMRDNP_params['chi2'])/(MMRDNP_params['Mi'])

    MMRDNP_model = wf.MMRDNP(MMRDNP_params['t0']             ,
                             MMRDNP_params['Mf']             ,
                             MMRDNP_params['af']             ,
                             MMRDNP_params['Mi']             ,
                             MMRDNP_params['eta']            ,
                             MMRDNP_params['chis']           ,
                             MMRDNP_params['chia']           ,
                             MMRDNP_params['distance']       ,
                             MMRDNP_params['inclination']    ,
                             MMRDNP_params['phi']            ,
                             TGR_parameters                  ,
                             single_l     = 2                ,
                             single_m     = 2                ,
                             single_mode  = 0                ,
                             geom         = 0                ,
                             qnm_fit      = 0                ,
                             interpolants = qnm_interpolants )
               
    st_count = time.time()
    #Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_MMRDNP, hc_MMRDNP = MMRDNP_model.waveform(times)
    
    print('RDNP:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{RDNP}$')
    plt.plot(times, hp_MMRDNP, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_MMRDNP, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

if(KHS_2012):

    TGR_parameters  = {}
    
    KHS_params      = {'t0'          : 0.004  ,
                      'Mf'          : 66.61  ,
                      'af'          : 0.68637,
                      'eta'         : 0.25   ,
                      'chi_eff'     : 0.2    ,
                      'distance'    : 450.0  ,
                      'inclination' : 0.0    ,
                      'phi'         : 0.0    }

    KHS_model = wf.KHS_2012(KHS_params['t0']               ,
                       KHS_params['Mf']               ,
                       KHS_params['af']               ,
                       KHS_params['eta']              ,
                       KHS_params['chi_eff']          ,
                       KHS_params['distance']         ,
                       KHS_params['inclination']      ,
                       KHS_params['phi']              ,
                       TGR_parameters                 ,
                       single_l     = 2               ,
                       single_m     = 2               ,
                       single_mode  = 0               )


    st_count = time.time()
    #Unpack the waveform and ignore vector and scalar modes.
    _, _, _, hp_KHS, hc_KHS = KHS_model.waveform(times)
    
    print('KHS:', time.time()-st_count)
    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{KHS}$')
    plt.plot(times, hp_KHS, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_KHS, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

if(TEOBPM):

    q = 3
    M_tot = 70

    TEOBPM_params = {'t0'          : 0.0              ,
                     'm1'          : (q/(1.+q))*M_tot ,
                     'm2'          : (1./(1.+q))*M_tot,
                     'chi1'        : 0.5              ,
                     'chi2'        : -0.4             ,
                     'distance'    : 450.0            ,
                     'inclination' : 0.3              ,
                     'phi'         : 0.0              }

    modes = [(2,2), (3,3)]
    merger_phases = {mode : 0.0 for mode in modes}
    
    TGR_parameters = {}
    TEOBPM_model = wf.TEOBPM(TEOBPM_params['t0']         ,
                             TEOBPM_params['m1']         ,
                             TEOBPM_params['m2']         ,
                             TEOBPM_params['chi1']       ,
                             TEOBPM_params['chi2']       ,
                             merger_phases               ,
                             TEOBPM_params['distance']   ,
                             TEOBPM_params['inclination'],
                             TEOBPM_params['phi']        ,
                             modes                       ,
                             TGR_parameters              ,
                             geom = 0                    )


    #Unpack the waveform and ignore vector and scalar modes.
    st_count = time.time()
    _, _, _, hp_TEOB, hc_TEOB = TEOBPM_model.waveform(times)
    print('TEOB-py speed:', time.time()-st_count)

    plt.figure(figsize=(fig_length, fig_heigth))
    plt.title(r'$\mathrm{TEOB}$')
    plt.plot(times, hp_TEOB, label=r'$\mathrm{h}_{+}$', color='crimson', linestyle='dashed')
    plt.plot(times, hc_TEOB, label=r'$\mathrm{h}_{\times}$', color='royalblue', linestyle='solid')
    plt.xlabel(r'$\mathrm{t \, [s]}$')
    plt.ylabel(r'$\mathrm{Strain}$')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

plt.show()
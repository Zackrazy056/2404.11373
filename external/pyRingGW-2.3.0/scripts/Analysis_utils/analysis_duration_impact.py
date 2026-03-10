# Standard python imports
import numpy as np, matplotlib.pyplot as plt

# LVC-specific imports
import pyRing.waveform as wf
from pyRing.likelihood import project
from pyRing.utils      import compute_SNR_TD
import lal

event    = 'GW150914'
det      = 'H1'
template = 'Kerr_221'

"""

This script quantifies the impact of the analysis duration and sampling rate on the optimal SNR of a Kerr waveform containing the [(2,2,0), (2,2,1)] modes.

"""

print('\nSNRs as a function of analysis duration:\n\n')


Ts   = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.5])

srates = [1024.0, 2048.0, 4096.0, 8192.0, 16384.0]

for srate in srates:

    print(f'\nsrate={srate} Hz\n')
    SNRs = np.array([])

    for T in Ts:

        N_points = int(T*srate)
        dt       = 1./srate
        times    = np.linspace(0.0, T, int(srate*T))
        times    = times[:N_points]

        datalen = 4096
        tstart  = 1126257415

        times_ACF, ACF = np.genfromtxt('ACFs_examples/ACF_{}_{}_{}_4.0_{}.txt'.format(det, tstart, datalen, srate), unpack=True)

        ACF = ACF[:N_points]

        Mf        = 66.61
        t_0       = 0.0
        Kerr_amps = {(2,2,2,0): 1.1*np.exp(1j*(-2.0)), (2,2,2,1): 0.95*np.exp(1j*(+1.14159))}

        params = {'t0'        : t_0    ,
                'Mf'          : Mf     ,
                'af'          : 0.68637,
                'distance'    : 450.0  ,
                'inclination' : 0.0    ,
                'phi'         : 0.0    }

        model = wf.KerrBH(params['t0']       ,
                        params['Mf']         ,
                        params['af']         ,
                        Kerr_amps            ,
                        params['distance']   ,
                        params['inclination'],
                        params['phi']        ,
                        amp_non_prec_sym = 1 )
        ra, dec, psi, tgps = 1.1579, -1.1911, 0.82, 1126259462.423

        # Unpack the waveform, ignore vector and scalar modes.
        hs, hvx, hvy, hp, hc = model.waveform(times+params['t0'])
        # Project on H1
        h                    = project(hs, hvx, hvy, hp, hc, lal.cached_detector_by_prefix[det], ra, dec, psi, tgps)

        # Compute SNR
        SNR = compute_SNR_TD(h, h, ACF, method='toeplitz-inversion')
        SNRs = np.append(SNRs, SNR)
        print(f'T={T} -- SNR={SNR:.3f}')

    SNRs_norm = (SNRs/SNRs[-1]) * 100
    Ts_ms     = Ts*1e3

    print('\nPercentage SNRs:\n')

    for i in range(len(SNRs)):
        print('T = {:.2f} [ms] --> {:.2f} %'.format(Ts_ms[i],SNRs_norm[i]))

    plt.figure()
    plt.title(f'srate={srate} Hz')
    plt.plot(Ts_ms, SNRs_norm, 'o-', color='firebrick')
    plt.xlabel('T [ms]')
    plt.ylabel('SNR [%]')
    plt.savefig(f'SNR_vs_T_srate_{srate}.png', bbox_inches='tight')

print('\n')

plt.show()
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: language_level=3
#cython: embedsignature=True

from __future__        import division
import numpy as np, os, scipy
cimport numpy as np, cython
from math              import factorial as fact
from libc.math cimport cos, pow, sin, sqrt, ceil, fabs, tanh, cosh, exp, log

#LVC imports
import lalsimulation as lalsim
import lal

#Package internal imports
from pyRing      cimport eob_utils as eu
from pyRing       import NR_amp
from pyRing.utils import resize_time_series, import_datafile_path

# pykerr is not on conda, hence the conda-build fails and cannot be registered as a requirement in the package
try   : import pykerr
except: pass

#cdef extern from "complex.h":
#    double complex exp(double complex)
cdef double MPC_SI = lal.PC_SI*1e6
cdef double mass_dist_units_conversion = lal.MSUN_SI * lal.G_SI / ( MPC_SI * lal.C_SI**2)
cdef double mass_time_units_conversion = lal.MSUN_SI * lal.G_SI / (lal.C_SI**3)
cdef double MTSUN_SI = lal.MTSUN_SI
# Fit of the complex ringdown frequencies for l = (2,3,4) and n = (0,1,2) from table VIII in http://arxiv.org/abs/gr-qc/0512160
cdef dict F_fit_coeff = {}
cdef dict q_fit_coeff = {}
Kerr_Berti_coeffs = np.genfromtxt(import_datafile_path('data/NR_data/berti_qnm_fits.txt'), names=True)
for i in xrange(Kerr_Berti_coeffs.shape[0]):
    F_fit_coeff[(int(Kerr_Berti_coeffs['l'][i]),int(Kerr_Berti_coeffs['m'][i]),int(Kerr_Berti_coeffs['n'][i]))] = [Kerr_Berti_coeffs['f1'][i],Kerr_Berti_coeffs['f2'][i],Kerr_Berti_coeffs['f3'][i]]
    q_fit_coeff[(int(Kerr_Berti_coeffs['l'][i]),int(Kerr_Berti_coeffs['m'][i]),int(Kerr_Berti_coeffs['n'][i]))] = [Kerr_Berti_coeffs['q1'][i],Kerr_Berti_coeffs['q2'][i],Kerr_Berti_coeffs['q3'][i]]

#FIXME temporary placeholder file
# Fit of the complex ringdown frequencies for l = (?) and n = (?) from ?
cdef dict F_KN_fit_coeff = {}
cdef dict q_KN_fit_coeff = {}
KN_coeffs = np.genfromtxt(import_datafile_path('data/NR_data/berti_qnm_fits.txt'), names=True)
for i in xrange(KN_coeffs.shape[0]):
    F_KN_fit_coeff[(int(KN_coeffs['l'][i]),int(KN_coeffs['m'][i]),int(KN_coeffs['n'][i]))] = [KN_coeffs['f1'][i],KN_coeffs['f2'][i],KN_coeffs['f3'][i]]
    q_KN_fit_coeff[(int(KN_coeffs['l'][i]),int(KN_coeffs['m'][i]),int(KN_coeffs['n'][i]))] = [KN_coeffs['q1'][i],KN_coeffs['q2'][i],KN_coeffs['q3'][i]]

# Fit of the QNM spin expansion coefficients for (l,m) = [(2,2), (2,1), (3,3)] and n = (0) from table I in arXiv:1910.12893v2
cdef dict f_ParSpec_coeff = {}
cdef dict tau_ParSpec_coeff = {}

ParSpec_coeffs = np.genfromtxt(import_datafile_path('data/NR_data/ParSpec_coefficients.txt'), names=True)
ParSpec_coeffs_f_order   = 4
ParSpec_coeffs_tau_order = 4
for i in xrange(ParSpec_coeffs.shape[0]):
    f_ParSpec_coeff[(int(ParSpec_coeffs['l'][i]),int(ParSpec_coeffs['m'][i]),int(ParSpec_coeffs['n'][i]))]   = [ParSpec_coeffs['w{}'.format(j)][i] for j in range(ParSpec_coeffs_f_order+1)]
    tau_ParSpec_coeff[(int(ParSpec_coeffs['l'][i]),int(ParSpec_coeffs['m'][i]),int(ParSpec_coeffs['n'][i]))] = [ParSpec_coeffs['t{}'.format(j)][i] for j in range(ParSpec_coeffs_tau_order+1)]

# Fit of the QNM spin expansion coefficients for (l,m,n) = [(2,2,0), (2,2,1)] modes, valid up to high spin.
cdef dict f_ParSpec_coeff_high_spin   = {}
cdef dict tau_ParSpec_coeff_high_spin = {}
ParSpec_coeffs_high_spin = np.genfromtxt(import_datafile_path('data/NR_data/ParSpec_coefficients_high_spin.txt'), names=True)
ParSpec_coeffs_f_order_high_spin   = 5
ParSpec_coeffs_tau_order_high_spin = 9
for i in xrange(ParSpec_coeffs_high_spin.shape[0]):
    f_ParSpec_coeff_high_spin[(int(ParSpec_coeffs_high_spin['l'][i]),int(ParSpec_coeffs_high_spin['m'][i]),int(ParSpec_coeffs_high_spin['n'][i]))]   = [ParSpec_coeffs_high_spin['w{}'.format(j)][i] for j in range(ParSpec_coeffs_f_order_high_spin+1)]
    tau_ParSpec_coeff_high_spin[(int(ParSpec_coeffs_high_spin['l'][i]),int(ParSpec_coeffs_high_spin['m'][i]),int(ParSpec_coeffs_high_spin['n'][i]))] = [ParSpec_coeffs_high_spin['t{}'.format(j)][i] for j in range(ParSpec_coeffs_tau_order_high_spin+1)]


cdef class QNM:

    """

    Class to compute the complex frequency and damping time of a QNM.

    Parameters
    ----------

    s : int
        Spin-weight of the QNM.

    l : int
        Orbital angular number of the QNM.

    m : int
        Azimuthal number of the QNM.

    n : int
        Radial (overtone) number of the QNM.

    interpolants : dict
        Dictionary of interpolants for the QNM frequencies and damping times.

    geom : int, optional
        Geometric units flag. If 1, the QNM frequencies are returned in geometric units, otherwise they are returned in standard units.

    Attributes
    ----------

    s : int
        Spin-weight of the QNM.
    l : int
        Orbital angular number of the QNM.
    m : int
        Azimuthal number of the QNM.
    n : int
        Radial (overtone) number of the QNM.
    geom : int
        Geometric units flag. If 1, the QNM frequencies are returned in geometric units, otherwise they are returned in standard units.
    omegar_interp : scipy.interpolate.interp1d
        Interpolant for the real part of the QNM frequency.
    omegai_interp : scipy.interpolate.interp1d
        Interpolant for the imaginary part of the QNM frequency.
    prefactor_freq : float
        Prefactor for the QNM frequency.
    prefactor_tau : float
        Prefactor for the QNM damping time.
    
    Functions
    ---------

    f : float
        Returns the real part of the QNM frequency.
    tau : float
        Returns the imaginary part of the QNM frequency.
    f_KN : float
        Returns the real part of the QNM frequency for the Kerr-Newman metric.
    tau_KN : float
        Returns the imaginary part of the QNM frequency for the Kerr-Newman metric.
    f_BW : float
        Returns the real part of the QNM frequency for the braneworld model.
    tau_BW : float
        Returns the imaginary part of the QNM frequency for the braneworld model.

    """

    def __cinit__(self,unsigned int s, unsigned int l, int m, unsigned int n, dict interpolants, unsigned int geom=0):

        self.s = s
        self.l = l
        self.m = m
        self.n = n

        self.geom = geom

        self.omegar_interp = interpolants[(self.s,self.l,self.m,self.n)]['freq']
        self.omegai_interp = interpolants[(self.s,self.l,self.m,self.n)]['tau']

        if(self.geom==1): self.prefactor_freq = 1.0/(2.*np.pi)
        else            : self.prefactor_freq = (lal.C_SI*lal.C_SI*lal.C_SI)/(2.*np.pi*lal.G_SI*lal.MSUN_SI)
        if(self.geom==1): self.prefactor_tau  = 1.0
        else            : self.prefactor_tau  = (lal.C_SI*lal.C_SI*lal.C_SI)/(lal.G_SI*lal.MSUN_SI)

    cpdef double f(   self, double M, double a             ): return self.omegar_interp(a      )*self.prefactor_freq*(1.0/M)
    cpdef double f_KN(self, double M, double a, double Q   ): return self.omegar_interp(a, Q   )*self.prefactor_freq*(1.0/M)
    cpdef double f_BW(self, double M, double a, double beta): return self.omegar_interp(a, beta)*self.prefactor_freq*(1.0/M)

    cpdef double tau(   self, double M, double a             ): return -1.0/(self.omegai_interp(a      )*self.prefactor_tau*(1.0/M))
    cpdef double tau_KN(self, double M, double a, double Q   ): return -1.0/(self.omegai_interp(a, Q   )*self.prefactor_tau*(1.0/M))
    cpdef double tau_BW(self, double M, double a, double beta): return -1.0/(self.omegai_interp(a, beta)*self.prefactor_tau*(1.0/M))

cdef class QNM_fit:

    """
    
    Class to compute the complex frequency and damping time of a QNM using fits to the Berti et al. (2015) fits.

    Parameters
    ----------

    l : int
        Orbital angular number of the QNM.  
    m : int
        Azimuthal number of the QNM.
    n : int
        Radial (overtone) number of the QNM.
    charge : int, optional
        Charge flag. If not 0, compute the QNM for a charged BH.
    geom : int, optional
        Geometric units flag. If 1, the QNM frequencies are returned in geometric units, otherwise they are returned in the International System of Units. Default is 0.
    
    Attributes
    ----------

    l : int
        Orbital angular number of the QNM.
    m : int
        Azimuthal number of the QNM.
    n : int 
        Radial (overtone) number of the QNM.
    charge : int
        Charge flag. If not 0, compute the QNM for a charged BH.
    f_coeff : list
        List of coefficients for the real part of the complex QNM frequency.
    q_coeff : list
        List of coefficients for the quality factor of the complex QNM frequency.
    
    Functions
    ---------

    f : float
        Returns the real part of the QNM frequency.
    q : float
        Returns the quality factor of the QNM frequency.
    tau : float
        Returns the imaginary part of the QNM frequency.
    f_KN : float
        Returns the real part of the QNM frequency for the Kerr-Newman metric.
    tau_KN : float
        Returns the imaginary part of the QNM frequency for the Kerr-Newman metric.
    q_KN : float
        Returns the quality factor of the QNM frequency for the Kerr-Newman metric.

    """


    def __cinit__(self,unsigned int l, int m, unsigned int n, unsigned int charge = 0, geom=0):

        assert not(np.abs(m) > l), "QNM: m cannot be greater than l in modulus."
        assert not(np.abs(n) > 2), "Berti fits are not available for n>2. Please unselect the 'qnm-fit' option in order to obtain a direct interpolation from NR data."
        assert not(charge == 1  ), "KN QNMs still do not support fits. Please set the qnm-fit=0"

        self.l       = l
        self.m       = m
        self.n       = n
        self.charge  = charge

        if(self.charge):
            self.f_coeff = F_KN_fit_coeff[(self.l,self.m,self.n)]
            self.q_coeff = q_KN_fit_coeff[(self.l,self.m,self.n)]
        else:
            self.f_coeff = F_fit_coeff[(self.l,self.m,self.n)]
            self.q_coeff = q_fit_coeff[(self.l,self.m,self.n)]

        if(geom==1): self.prefactor_freq = 1.0/(2.*np.pi)
        else       : self.prefactor_freq = (lal.C_SI*lal.C_SI*lal.C_SI)/(2.*np.pi*lal.G_SI*lal.MSUN_SI)

    cpdef double      f(self, double M, double a          ): return (self.f_coeff[0]+self.f_coeff[1]*(1-a)**self.f_coeff[2])*self.prefactor_freq*(1.0/M)
    cpdef double      q(self,           double a          ): return (self.q_coeff[0]+self.q_coeff[1]*(1-a)**self.q_coeff[2])
    cpdef double    tau(self, double M, double a          ): return self.q(a)/(np.pi*self.f(M,a))

    cpdef double   f_KN(self, double M, double a, double Q): return (self.f_coeff[0]+self.f_coeff[1]*(1-a)**self.f_coeff[2])*self.prefactor_freq*(1.0/M)
    cpdef double   q_KN(self,           double a, double Q): return (self.q_coeff[0]+self.q_coeff[1]*(1-a)**self.q_coeff[2])
    cpdef double tau_KN(self, double M, double a, double Q): return self.q_KN(a, Q)/(np.pi*self.f_KN(M, a, Q))

#NO-REVIEW-NEEDED
cdef class QNM_ParSpec:

    """

    Class to compute the complex frequency and damping time of a QNM using the ParSpec fits.

    Parameters
    ----------

    l : int
        Orbital angular number of the QNM.
    m : int
        Azimuthal number of the QNM.
    n : int
        Radial (overtone) number of the QNM.
    fit : str, optional
        Fit to use. Either 'high_spin' or 'low_spin'. Default is 'high_spin'.
    geom : int, optional
        Geometric units flag. If 1, the QNM frequencies are returned in geometric units, otherwise they are returned in the International System of Units. Default is 0.
    
    Attributes
    ----------

    l : int
        Orbital angular number of the QNM.
    m : int
        Azimuthal number of the QNM.
    n : int
        Radial (overtone) number of the QNM.
    f_coeff : list
        List of coefficients for the real part of the complex QNM frequency.
    tau_coeff : list
        List of coefficients for the imaginary part of the complex QNM frequency.

    Functions
    ---------

    f : float
        Returns the real part of the QNM frequency.
    tau : float
        Returns the imaginary part of the QNM frequency.
    
    """


    def __cinit__(self, unsigned int l, int m, unsigned int n, fit='high_spin', geom=0):

        assert not(np.abs(m) > l), "QNM: m cannot be greater than l in modulus."
        
        self.l         = l
        self.m         = m
        self.n         = n

        if(fit=='high_spin'):
            self.f_coeff   = f_ParSpec_coeff_high_spin[(  self.l,self.m,self.n)]
            self.tau_coeff = tau_ParSpec_coeff_high_spin[(self.l,self.m,self.n)]
        else:
            self.f_coeff   = f_ParSpec_coeff[(  self.l,self.m,self.n)]
            self.tau_coeff = tau_ParSpec_coeff[(self.l,self.m,self.n)]     

        if(geom==1): self.prefactor_freq = 1.0/(2.*np.pi)
        else       : self.prefactor_freq = (lal.C_SI*lal.C_SI*lal.C_SI)/(2.*np.pi*lal.G_SI*lal.MSUN_SI)
        if(geom==1): self.prefactor_tau  = 1.0
        else       : self.prefactor_tau  = (lal.C_SI*lal.C_SI*lal.C_SI)/(lal.G_SI*lal.MSUN_SI)

    cpdef double f(self, double M, double a, double gamma, np.ndarray[double, ndim=1] dw_vec):

        cdef int D_max = len(self.f_coeff)
        cdef int D_dw  = len(dw_vec)

        ParSpec_fit_freq = 0.0
        for i in range(D_max):
            # Apply the deviations up to a given order, while the expansion has to stay valid up to the maximum order to avoid fake GR deviations.
            if(i < D_dw): ParSpec_fit_freq += a**i * self.f_coeff[i] * (1 + gamma * dw_vec[i])
            else:         ParSpec_fit_freq += a**i * self.f_coeff[i]
        
        return (1.0/M)*self.prefactor_freq*ParSpec_fit_freq

    cpdef double tau(self, double M, double a, double gamma, np.ndarray[double, ndim=1] dt_vec):

        cdef int D_max = len(self.tau_coeff)
        cdef int D_dt  = len(dt_vec)

        ParSpec_fit_tau = 0.0
        for i in range(D_max):
            # Apply the deviations up to a given order, while the expansion has to stay valid up to the maximum order to avoid fake GR deviations. 
            if(i < D_dt): ParSpec_fit_tau += a**i * self.tau_coeff[i] * (1 + gamma * dt_vec[i])
            else:         ParSpec_fit_tau += a**i * self.tau_coeff[i]

        return 1.0/((1.0/M)*self.prefactor_tau) * ParSpec_fit_tau

#NO-REVIEW-NEEDED
cdef class QNM_220_area_quantized:

    """

    Class to compute the complex frequency and damping time of the 220 QNM coming from quantized area presciption.

    Parameters
    ----------

    l_QA : int
        Orbital angular number of the QNM.
    m_QA : int
        Azimuthal number of the QNM.
    n_QA : int
        Radial (overtone) number of the QNM.
    
    Attributes
    ----------

    l_QA : int
        Orbital angular number of the QNM.
    m_QA : int
        Azimuthal number of the QNM.
    n_QA : int
        Radial (overtone) number of the QNM.
    q_coeff_GR : list
        List of coefficients for the quality factor QNM frequency.

    Functions
    ---------

    f_QA : float
        Returns the real part of the QNM frequency.
    q_QA : float
        Returns the quality factor of the QNM frequency.
    tau_QA : float
        Returns the imaginary part of the QNM frequency.

    """

    # Reference: arXiv:1611.07009v3
    # Since I couldn't find a quantum-inspired formula for tau, I am assuming GR in decay time
    def __cinit__(self, unsigned int l_QA, int m_QA, unsigned int n_QA):
        self.l_QA       = l_QA
        self.m_QA       = m_QA
        self.n_QA       = n_QA
        self.q_coeff_GR = q_fit_coeff[(self.l_QA,self.m_QA,self.n_QA)]

        assert ((self.l_QA == 2) and (self.m_QA == 2) and (self.n_QA == 0)), "The QNM coming from quantized are valid only for the 220 mode."

    cpdef double f_QA(self, double M, double a, double alpha):
        cdef double prefactor_freq = (lal.C_SI*lal.C_SI*lal.C_SI/(2.*np.pi*lal.G_SI*M*lal.MSUN_SI))
        cdef double n_tra     = 1 # Order of quantum transition, see pg.3
        cdef double m_grav    = 2 # Graviton, see pg.3
        return prefactor_freq*(n_tra*alpha*np.sqrt(1-a*a)+8*np.pi*a*m_grav)/(16*np.pi*(1+np.sqrt(1-a*a)))

    cpdef double q_GR(self, double a):
        return (self.q_coeff_GR[0]+self.q_coeff_GR[1]*(1-a)**self.q_coeff_GR[2])

    cpdef double tau_QA(self, double M, double a, double alpha):
        cdef double f_QA = self.f_QA(M, a, alpha)
        cdef double q_GR = self.q_GR(a)
        return q_GR/(np.pi*f_QA)

#NO-REVIEW-NEEDED
cdef class QNM_braneworld_fit:

    """

    Class to compute the complex frequency and damping time of the QNM coming from braneworld fit.

    Parameters
    ----------

    l_BW : int
        Orbital angular number of the QNM.
    m_BW : int
        Azimuthal number of the QNM.
    n_BW : int
        Radial (overtone) number of the QNM.
    
    Attributes
    ----------

    l_BW : int
        Orbital angular number of the QNM.  
    m_BW : int
        Azimuthal number of the QNM.
    n_BW : int
        Radial (overtone) number of the QNM.

    Functions
    ---------

    f_BW : float
        Returns the real part of the QNM frequency.
    q_BW : float
        Returns the quality factor of the QNM frequency.
    tau_BW : float
        Returns the imaginary part of the QNM frequency.
    
    """

    # Reference: arxiv:2106.05558
    def __cinit__(self, unsigned int l_BW, int m_BW, unsigned int n_BW):
        self.l_BW       = l_BW
        self.m_BW       = m_BW
        self.n_BW       = n_BW

    cpdef double f_BW(self, double M, double a, double beta):
        cdef double prefactor_freq = (lal.C_SI*lal.C_SI*lal.C_SI/(2.*np.pi*lal.G_SI*M*lal.MSUN_SI))
        cdef double f_bw_fit = 1/M * a * beta # dummy function for the moment
        return prefactor_freq*f_bw_fit

    cpdef double q_BW(self, double a, double beta):
        cdef double q_bw_fit = a * beta # dummy function for the moment
        return q_bw_fit

    cpdef double tau_BW(self, double M, double a, double beta):
        cdef double f_bw_fit = self.f_BW(M, a, beta)
        cdef double q_bw_fit = self.q_BW(a, beta)
        return q_bw_fit/(np.pi*f_bw_fit)

@cython.boundscheck(False) # turn off bounds-checking for entire function, increases the speed of the code.
cpdef np.ndarray[complex,ndim=1] damped_sinusoid(double A,                     # Amplitude
                                                 double f,                     # Frequency
                                                 double tau,                   # Damping time
                                                 double phi,                   # Phase
                                                 double t0,                    # Start time
                                                 np.ndarray[double, ndim=1] t  # Time array
                                                ):

    """

    Function to compute a damped sinusoid waveform model.

    Parameters
    ----------

    A : float
        Amplitude of the damped sinusoid.
    f : float
        Frequency of the damped sinusoid.
    tau : float 
        Damping time of the damped sinusoid.
    phi : float 
        Phase of the damped sinusoid.
    t0 : float  
        Start time of the damped sinusoid.
    t : np.ndarray  
        Time array.
    
    Returns
    -------

    h : np.ndarray
        Damped sinusoid waveform model.

    """

    cdef unsigned int n                = t.shape[0]
    cdef np.ndarray[complex, ndim=1] h = np.zeros(n,dtype='complex')
    cdef double omega                  = 2.0*np.pi*f
    cdef complex om_cplx               = omega+1j/tau

    # This for loop is needed in the case where the time axis spacing is not uniform, as in NR simulations. This is consistent with the truncated case in the likelihood, since there only t>=t0 are passed to the waveform, so the index i=0 is always selected. This implies that there is no loss of efficiency during sampling, since the for loop always lasts a single iteration.
    cdef int t_start_idx = 0
    for i in range(len(t)):
        if(t[i] >= t0):
            if(np.abs((t[i]-t0)) <= np.abs((t[i-1]-t0))): t_start_idx = i
            else                                        : t_start_idx = i-1
            break
        else:
            pass

    h[t_start_idx:] = A*np.exp(1j*om_cplx*(t[t_start_idx:]-t0)+1j*phi)

    return h

cdef np.ndarray[complex,ndim=1] tail_factor(double A,                     # Amplitude
                                            double phi,                   # Phase
                                            double p,                     # power-law exponent
                                            double t0,                    # Start time
                                            np.ndarray[double, ndim=1] t  # Time array
                                            ):

    """

    Function to compute the tail factor of the damped sinusoid waveform model.

    Parameters
    ----------

    A : float
        Amplitude of the damped sinusoid.
    phi : float
        Phase of the damped sinusoid.
    p : float
        Power-law exponent of the damped sinusoid.
    t0 : float
        Start time of the damped sinusoid.
    t : np.ndarray
        Time array.
    
    Returns
    -------

    h : np.ndarray
        Tail factor of the damped sinusoid waveform model.

    """
        

    cdef unsigned int n                = t.shape[0]
    cdef np.ndarray[complex, ndim=1] h = np.zeros(n,dtype='complex')

    cdef int t_start_idx = 0
    for i in range(len(t)):
        if(t[i] >= t0):
            if(np.abs((t[i]-t0)) <= np.abs((t[i-1]-t0))): t_start_idx = i
            else                                        : t_start_idx = i-1
            break
        else:
            pass

    h[t_start_idx:] = A*np.exp(1j*phi)*(t[t_start_idx:]-t0)**p

    return h

cdef np.ndarray[complex,ndim=1] morlet_gabor_wavelet(double A,                     # Amplitude
                                                     double f,                     # Frequency
                                                     double tau,                   # Damping time
                                                     double phi,                   # Phase
                                                     double t0,                    # Start time
                                                     np.ndarray[double, ndim=1] t  # Time array
                                                    ):

    """

    Function to compute a Morlet-Gabor wavelet model.

    Parameters
    ----------

    A : float
        Amplitude of the Morlet-Gabor wavelet.
    f : float
        Frequency of the Morlet-Gabor wavelet.
    tau : float
        Damping time of the Morlet-Gabor wavelet.
    phi : float
        Phase of the Morlet-Gabor wavelet.
    t0 : float
        Start time of the Morlet-Gabor wavelet.
    t : np.ndarray
        Time array.
    
    Returns
    -------

    h : np.ndarray
        Morlet-Gabor wavelet model.

    """

    cdef unsigned int n                = t.shape[0]
    cdef np.ndarray[complex, ndim=1] h = np.zeros(n,dtype='complex')
    cdef double omega                  = 2.0*np.pi*f
    cdef int t_start_idx               = int(ceil((t0-t[0])/(t[1]-t[0])))

    #FIXME: this is real, while it should be a complex quantity.
    h[t_start_idx:] = A*np.cos(omega*(t[t_start_idx:]-t0)+phi)*np.exp(-((t[t_start_idx:]-t0)/(tau))**2)

    return h

cdef class Damped_sinusoids:

    """
    Class implementing a superposition of Damped Sinusoids of arbitrary polarisation.

    Parameters
    ----------

    A : dict
        Dictionary of amplitudes of the damped sinusoids.
    f : dict
        Dictionary of frequencies of the damped sinusoids.
    tau : dict
        Dictionary of damping times of the damped sinusoids.
    phi : dict
        Dictionary of phases of the damped sinusoids.
    t0 : dict
        Dictionary of start times of the damped sinusoids.

    Attributes
    ----------

    A : dict
        Dictionary of amplitudes of the damped sinusoids.
    f : dict
        Dictionary of frequencies of the damped sinusoids.
    tau : dict
        Dictionary of damping times of the damped sinusoids.
    phi : dict
        Dictionary of phases of the damped sinusoids.
    t0 : dict
        Dictionary of start times of the damped sinusoids.
    N : dict
        Dictionary of number of damped sinusoids for each polarisation.

    Functions
    ---------

    waveform(t)
        Returns the waveform model.

    """
    def __cinit__(self,
                  dict A,
                  dict f,
                  dict tau,
                  dict phi,
                  dict t0):

        self.A   = A
        """
        :param A: Amplitudes
        """
        self.f   = f
        self.tau = tau
        self.phi = phi
        self.t0  = t0
        self.N   = {}
        for key in self.A.keys():
            self.N[key] = len(self.A[key])

    cpdef np.ndarray[double, ndim=5] waveform(self,np.ndarray[double, ndim=1] t):

        """
            | Returns five polarisations (the ones independent in a L-shaped GW detector, see https://arxiv.org/abs/1710.03794) allowed for a metric theory of gravity: hs (scalar mode), {hvx, hvy} (vector modes), {hp, hc} (tensor modes).
            | We employ the conventions:
            | h_s           = sum_{i} A_i * cos(omega*t+phi)  * e^(-(t-t^{start}_i/tau)
            | h_vx - i h_vy = sum_{i} A_i * e^(i*omega*t+phi) * e^(-(t-t^{start}_i/tau)
            | h_+  - i h_x  = sum_{i} A_i * e^(i*omega*t+phi) * e^(-(t-t^{start}_i/tau)

            Parameters
            ----------
            t : np.ndarray
                Time array.

            Returns
            -------

            h : np.ndarray
                Waveform model.

        """

        cdef unsigned int i,j, K = t.shape[0]
        cdef np.ndarray[complex, ndim=1] h_tmp = np.zeros(K,dtype=complex)
        cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
        h_s  = np.zeros(K, dtype='double')
        h_vx = np.zeros(K, dtype='double')
        h_vy = np.zeros(K, dtype='double')
        h_p  = np.zeros(K, dtype='double')
        h_c  = np.zeros(K, dtype='double')

        for pol in self.N.keys():
            for i in range(self.N[pol]):
                h_tmp += damped_sinusoid(self.A[pol][i]  ,
                                         self.f[pol][i]  ,
                                         self.tau[pol][i],
                                         self.phi[pol][i],
                                         self.t0[pol][i] ,
                                         t)
            if(pol=='s'):
                h_s  +=  np.real(h_tmp)
            elif(pol=='v'):
                h_vx +=  np.real(h_tmp)
                h_vy += -np.imag(h_tmp)
            elif(pol=='t'):
                h_p  +=  np.real(h_tmp)
                h_c  += -np.imag(h_tmp)
            h_tmp = np.zeros(K, dtype='complex')

        return np.array([h_s, h_vx, h_vy, h_p, h_c])

cdef class Morlet_Gabor_wavelets:

    """

    Class implementing a superposition of Morlet-Gabor wavelets of arbitrary polarisation.

    Parameters
    ----------

    A : dict
        Dictionary of amplitudes of the Morlet-Gabor wavelets.
    f : dict
        Dictionary of frequencies of the Morlet-Gabor wavelets.
    tau : dict
        Dictionary of damping times of the Morlet-Gabor wavelets.
    phi : dict
        Dictionary of phases of the Morlet-Gabor wavelets.
    t0 : dict
        Dictionary of start times of the Morlet-Gabor wavelets.
    
    Attributes
    ----------

    A : dict
        Dictionary of amplitudes of the Morlet-Gabor wavelets.
    f : dict
        Dictionary of frequencies of the Morlet-Gabor wavelets.
    tau : dict
        Dictionary of damping times of the Morlet-Gabor wavelets.
    phi : dict
        Dictionary of phases of the Morlet-Gabor wavelets.
    t0 : dict
        Dictionary of start times of the Morlet-Gabor wavelets.
    N : dict
        Dictionary of number of Morlet-Gabor wavelets for each polarisation.
    
    Functions
    ---------

    waveform(t)
        Returns the waveform model.

    """

    def __cinit__(self,
                  dict A,
                  dict f,
                  dict tau,
                  dict phi,
                  dict t0):

        self.A   = A
        self.f   = f
        self.tau = tau
        self.phi = phi
        self.t0  = t0
        self.N   = {}
        for key in self.A.keys():
            self.N[key] = len(self.A[key])

    cpdef np.ndarray[double, ndim=5] waveform(self,np.ndarray[double, ndim=1] t):

        """
            Returns five polarisations (the ones independent in a L-shaped GW detector) allowed for a metric theory of gravity: hs (scalar mode), {hvx, hvy} (vector modes), {hp, hc} (tensor modes).
            We employ the conventions: h_s           = sum_{i} A_i * cos(omega*t+phi)  * e^(-(t-t^{start}_i/tau)
                                       h_vx - i h_vy = sum_{i} A_i * e^(i*omega*t+phi) * e^(-(t-t^{start}_i/tau)
                                       h_+  - i h_x  = sum_{i} A_i * e^(i*omega*t+phi) * e^(-(t-t^{start}_i/tau)

            Parameters
            ----------

            t : np.ndarray
                Time array.
            
            Returns
            -------

            h : np.ndarray
                Waveform model.
        """

        cdef unsigned int i,j, K = t.shape[0]
        cdef np.ndarray[complex, ndim=1] h_tmp = np.zeros(K,dtype=complex)
        cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
        h_s  = np.zeros(K, dtype='double')
        h_vx = np.zeros(K, dtype='double')
        h_vy = np.zeros(K, dtype='double')
        h_p  = np.zeros(K, dtype='double')
        h_c  = np.zeros(K, dtype='double')

        for pol in self.N.keys():
            for i in range(self.N[pol]):
                h_tmp += morlet_gabor_wavelet(self.A[pol][i]  ,
                                              self.f[pol][i]  ,
                                              self.tau[pol][i],
                                              self.phi[pol][i],
                                              self.t0[pol][i] ,
                                              t)
            if(pol=='s'):
                h_s  +=  np.real(h_tmp)
            elif(pol=='v'):
                h_vx +=  np.real(h_tmp)
                h_vy += -np.imag(h_tmp)
            elif(pol=='t'):
                h_p  +=  np.real(h_tmp)
                h_c  += -np.imag(h_tmp)
            h_tmp = np.zeros(K, dtype='complex')

        return np.array([h_s, h_vx, h_vy, h_p, h_c])

cdef class SWSH:

  """
    Spin weighted spherical harmonics
    -s_Y_{lm}
    Defined in Kidder (https://arxiv.org/pdf/0710.0614.pdf) Eq.s (4, 5).
    Note that this function returns -s_Y_{l,m} as defined by Kidder.
    Thus, for gravitational perturbation s=2 must be passed.

    Parameters
    ----------

    s : int
        Spin of the spherical harmonic.
    l : int
        Orbital angular number of the spherical harmonic.
    m : int
        Azimuthal angular number of the spherical harmonic.
    
    Attributes
    ----------

    s : int
        Spin of the spherical harmonic.
    l : int
        Orbital angular number of the spherical harmonic.
    m : int
        Azimuthal angular number of the spherical harmonic.
    swsh_prefactor : double
        Prefactor of the spherical harmonic.
    
    Functions
    ---------

    evaluate(theta, phi)
        Returns the value of the spherical harmonic for the given angles.
    
  """

  def __init__(self, int s, int l, int m):

    self.s = s
    self.l = l
    self.m = m
    self.swsh_prefactor = (-1)**(self.s) \
                        * sqrt((2*self.l+1)/(4.0*np.pi)) \
                        * sqrt(fact(self.l+self.m)*fact(self.l-self.m)*fact(self.l+self.s)*fact(self.l-self.s))

  def __call__(self, double theta, double phi):
    return self.evaluate(theta, phi)
  
  cpdef complex evaluate(self, double theta, double phi):

    """
        SWSH for angles theta [0,pi] and phi [0,2pi].

        Parameters
        ----------

        theta : double
            Polar angle.
        phi : double
            Azimuthal angle.
        
        Returns
        -------

        result : complex
            Value of the SWSH for the given angles.
    """

    cdef complex result = 0

    ki = max(0,self.m-self.s)
    kf = min(self.l+self.m,self.l-self.s)
    for k in range(ki,kf+1):
      result += (-1)**k * sin(theta/2)**(2*k+self.s-self.m) * cos(theta/2)**(2*self.l+self.m-self.s-2*k) \
              * 1/(fact(k)*fact(self.l+self.m-k)*fact(self.l-self.s-k)*fact(self.s-self.m+k))
    
    result *= np.exp(1j*self.m*phi)*self.swsh_prefactor

    return result

cdef class KerrBH:

  """
    | Multi mode ringdown model for a Kerr black hole using predictions of the frequencies and damping times as function of mass and spin, as predicted by perturbation theory.
    |
    | [Input parameters]
    |
    | t0    : Start time of the ringdown, currently common for all modes. #IMPROVEME: allow for a different start time for each mode.
    | Mf    : Final mass in solar masses.
    | af    : Dimensionless final spin.
    | amps  : Amplitudes of the (s,l,m,n) modes. Expected syntax: amps[(s,l,m,n)] = `value`. The keys of this dictionary set the modes used in the waveform.
    | r     : Distance in Mpc.
    | iota  : Inclination in radians.
    | phi   : Azimuthal angle in radians.
    |
    | [Optional parameters]
    |
    | [[Units and spectrum]]
    |
    | reference_amplitude : If non-zero, value with which to replace the Mf/r prefactor. Default: 0.0.
    | geom                : Flag to compute only the h_{l,m} modes, without spherical harmonics. Default: 0.
    | qnm_fit             : Flag to request the use of an interpolation for QNM complex frequency, instead of analytical fits (not available for s!=2 or n>2). Default: 1.
    | interpolants        : QNM complex frequencies interpolants, only used if `qnm_fit=0`. Default: None.
    |
    | [[Additional Kerr]]
    |   
    | Spheroidal          : Flag to activate the use of spheroidal harmonics instead of spherical. Relies on the pykerr package. Default: 0.
    | amp_non_prec_sym    : Flag to enforce non-precessing symmetry on the amplitudes. Default: 0.
    | tail_parameters     : Dictionary of tail modes. Default: None.
    | quadratic_modes     : Amplitudes of the quadratic contributions (both sum and difference of parent frequencies) to the (s,l,m,n) mode generated by (s1,l1,m1,n1)x(s2,l2,m2,n2). If `quad_lin_prop=1`, instead of the amplitude values, contains the proportionality constant (\alpha) wrt the product of linear terms. Expected syntax: quad_amps[term][((s,l,m,n), (s1,l1,m1,n1), (s2,l2,m2,n2))] = `value`, with term=['sum','diff']. Default: None.
    | quad_lin_prop       : Flag to fix the quadratic amplitudes as \alpha A_1 * A_2, with (A_1, A_2) the corresponding linear amplitudes. Default: 0.
    | qnm_cached          : Dictionary containing cached values of the QNM frequencies for specific values of Mf,af. Expected syntax: qnm_cached[(2, l, m, n)] = {'f': freq, 'tau': tau}. Default: None.

    |
    | [[Beyond Kerr]]
    |   
    | TGR_params          : Additional non-GR parameters to be sampled. Default: None.
    | AreaQuantization    : Flag to use a prescription to impose QNM frequencies derived from the area quantisation proposal. Default: 0.
    | ParSpec             : Flag to use the ParSpec parametrisation of beyond-GR modifications to QNMs. Default: 0.
    | charge              : Flag to include the effect of charge. Default: 0.
    | braneworld          : Flag to include the effect of charge in the braneworld scenario. Default: 0.

  """

  def __cinit__(self,
                double       t0,
                double       Mf,
                double       af,
                dict         amps,
                double       r,
                double       iota,
                double       phi,

                double       reference_amplitude = 0.0,
                unsigned int geom                = 0,
                unsigned int qnm_fit             = 1,
                dict         interpolants        = None,

                unsigned int Spheroidal          = 0,
                unsigned int amp_non_prec_sym    = 0,
                dict         tail_parameters     = None,
                dict         quadratic_modes     = None,
                unsigned int quad_lin_prop       = 0,
                dict         qnm_cached          = None,

                dict         TGR_params          = None,
                unsigned int AreaQuantization    = 0,
                unsigned int ParSpec             = 0,
                unsigned int charge              = 0,
                unsigned int braneworld          = 0):

    # Standard parameters
    self.t0                  = t0
    self.Mf                  = Mf
    self.af                  = af
    self.amps                = amps
    self.r                   = r
    self.iota                = iota
    self.phi                 = phi

    # Units and spectrum parameters.
    self.reference_amplitude = reference_amplitude
    self.geom                = geom
    self.qnm_fit             = qnm_fit
    self.interpolants        = interpolants

    # Additional Kerr options.
    self.Spheroidal          = Spheroidal
    self.amp_non_prec_sym    = amp_non_prec_sym
    self.tail_parameters     = tail_parameters
    self.quadratic_modes     = quadratic_modes
    self.quad_lin_prop       = quad_lin_prop
    self.qnm_cached          = qnm_cached

    # Beyond-Kerr options
    self.TGR_params          = TGR_params
    self.AreaQuantization    = AreaQuantization
    self.ParSpec             = ParSpec
    self.charge              = charge
    self.braneworld          = braneworld

    cdef int s,l,m,n
    cdef str quad_term 
    self.qnms         = {}
    self.qnms_ParSpec = {}
    self.swshs        = {}

    # Fill in the full list of modes, including quadratic contributions.
    modes_full = []
    for mode in self.amps.keys(): modes_full.append(mode)
    if self.quadratic_modes is not None:
        for quad_term in self.quadratic_modes.keys():
            for mode in self.quadratic_modes[quad_term].keys():
                modes_full.append(mode[0])
                modes_full.append(mode[1])
                modes_full.append(mode[2])
                
                # In cases quadratic amplitudes are fixed in terms of the linear ones, check that parent modes are present in the linear modes.
                if(self.quad_lin_prop):
                    if not((mode[1] in self.amps.keys()) and (mode[2] in self.amps.keys())):
                        raise ValueError("When fixing the quadratic amplitudes in terms of the linear ones, the linear modes have to contain the corresponding quadratic parent modes.")

    # Remove duplicates.
    modes_full = list(dict.fromkeys(modes_full))

    for (s,l,m,n) in modes_full:

        assert not(not(s==2) and (self.AreaQuantization or self.qnm_fit or self.charge or self.braneworld)), "Non-tensorial modes (s={} was selected) are incompatible with using either a fit for QNM or the area quantization proposal or a BH charge or braneworld corrections.".format(s)

        # Initialise QNMs
        if(self.qnm_cached is None):

            if(self.AreaQuantization and l==2 and m==2 and n==0):
                qnm = QNM_220_area_quantized(l,m,n)
            else:
                if not(self.ParSpec):
                    if(self.qnm_fit):
                        if(self.charge)      : raise ValueError('KN QNMs still do not support fits. Please set qnm-fit=0 inside the config file.')
                        elif(self.braneworld): raise ValueError('Braneworld QNMs still do not support fits. Please set qnm-fit=0 inside the config file.')
                        else                 : qnm = QNM_fit(l,m,n, geom=self.geom)
                    else:
                        assert not(self.interpolants==None), "You deselected qnm-fit without providing any interpolant."
                        qnm = QNM(s,l,m,n,self.interpolants, self.geom)
                else:
                    # For the parameters which are not being perturbed beyond GR, we want to retain the full spin expansion.
                    qnm                          = QNM_fit(    l,m,n, geom=self.geom)
                    self.qnms_ParSpec[(s,l,m,n)] = QNM_ParSpec(l,m,n, geom=self.geom)

            self.qnms[(s,l,m,n)] = qnm

        # Initialise harmonics
        if (self.Spheroidal):
            swsh_p = pykerr.spheroidal(self.iota, self.af, l,  m, n, phi=self.phi)
            swsh_m = np.conj(pykerr.spheroidal(np.pi-self.iota, self.af, l,  m, n, phi=self.phi))*(-1)**l
        else:
            swsh_p = SWSH(2,l, m)(self.iota, self.phi)
            swsh_m = SWSH(2,l,-m)(self.iota, self.phi)
        self.swshs[(2,l, m,n)] = swsh_p
        self.swshs[(2,l,-m,n)] = swsh_m

  cpdef np.ndarray[double, ndim=5] waveform(self, np.ndarray[double,ndim=1] times):

    """
        | We employ the conventions of arXiv:gr-qc/0512160 (Eq. 2.9):
        |                            h_s           = Re(sum_{lmn} S_{lmn} h_{lmn})
        |                            h_vx + i h_vy = sum_{lmn} S_{lmn} h_{lmn}
        |                            h_+  + i h_x  = sum_{lmn} S_{lmn} h_{lmn}
        | Non-precessing symmetry implies the property: h_{l,-m} = (-1)**l h^*_{l,m}
        | (see: Blanchet, “Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries”).

    """

    cdef int s,l,m,n
    cdef str quad_term
    cdef double prefactor
    cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
    cdef np.ndarray[complex,ndim=1] h_tmp
    h_s   = np.zeros(times.shape[0], dtype='double')
    h_vx  = np.zeros(times.shape[0], dtype='double')
    h_vy  = np.zeros(times.shape[0], dtype='double')
    h_p   = np.zeros(times.shape[0], dtype='double')
    h_c   = np.zeros(times.shape[0], dtype='double')
    h_tmp = np.zeros(times.shape[0], dtype='complex')

    ################
    # Linear modes #
    ################

    for (s,l,m,n),a in self.amps.items():

        ######################
        # Complex frequency. #
        ######################

        # First, compute the eventual deviation from the GR spectrum. GR deviations in the spectrum for non-tensorial modes are not supported.
        if(s==2):
            try:    dfreq = self.TGR_params['domega_{}{}{}'.format(l,m,n)]
            except: dfreq = 0.0
            try:    dtau  = self.TGR_params['dtau_{}{}{}'.format(l,m,n)]
            except: dtau  = 0.0
        else:
            dfreq = 0.0
            dtau  = 0.0

        if(self.qnm_cached is not None):

            freq       = self.qnm_cached[(s,l,m,n)]['f']
            tau        = self.qnm_cached[(s,l,m,n)]['tau']
            corr_dfreq = 1.0
            corr_dtau  = 1.0
            
        else:

            if(self.AreaQuantization and l==2 and m==2 and n==0):
                try:    alpha = self.TGR_params['alpha']
                except: raise KeyError('If quantization of the horizon area is invoked, the alpha parameter must be passed.')
                freq = self.qnms[(s,l,m,n)].f_QA(self.Mf, self.af, alpha)
                try:    tau = self.TGR_params['tau_AQ']
                except: tau = self.qnms[(s,l,m,n)].tau_QA(self.Mf, self.af, alpha)
                corr_dfreq = 1.0
                corr_dtau  = 1.0
            elif(self.ParSpec):
                # In this case dfreq and dtau are arrays.
                gamma = self.TGR_params['gamma']
                if not('domega_{}{}{}'.format(l,m,n) in self.TGR_params.keys()): freq = self.qnms_ParSpec[(s,l,m,n)].f(  self.Mf, self.af, 0.0  , np.array([]))
                else:                                                            freq = self.qnms_ParSpec[(s,l,m,n)].f(  self.Mf, self.af, gamma, dfreq       )
                if not(  'dtau_{}{}{}'.format(l,m,n) in self.TGR_params.keys()): tau  = self.qnms_ParSpec[(s,l,m,n)].tau(self.Mf, self.af, 0.0  , np.array([]))
                else:                                                            tau  = self.qnms_ParSpec[(s,l,m,n)].tau(self.Mf, self.af, gamma, dtau        )
                corr_dfreq = 1.0
                corr_dtau  = 1.0
            elif(self.charge):
                Q          = self.TGR_params['Q']
                freq       = self.qnms[(s,l,m,n)].f_KN(self.Mf, self.af, Q)
                tau        = self.qnms[(s,l,m,n)].tau_KN(self.Mf, self.af, Q)
                corr_dfreq = 1.0+dfreq
                corr_dtau  = 1.0+dtau
            elif(self.braneworld):
                beta       = self.TGR_params['beta']
                freq       = self.qnms[(s,l,m,n)].f_BW(self.Mf, self.af, beta)
                tau        = self.qnms[(s,l,m,n)].tau_BW(self.Mf, self.af, beta)
                corr_dfreq = 1.0
                corr_dtau  = 1.0
            else:
                freq       = self.qnms[(s,l,m,n)].f(self.Mf, self.af)
                tau        = self.qnms[(s,l,m,n)].tau(self.Mf, self.af)
                corr_dfreq = 1.0+dfreq
                corr_dtau  = 1.0+dtau

        # In the geom case, we are assuming extraction of the waveform at the north pole, hence the negative frequency term for the m<0 modes contributes to the waveform, since swshs[(2,l,m,n)] with negative m is zero at the north pole.
        if(self.geom and (m<0)): freq = -freq

        ###############
        # Amplitudes. #
        ###############

        # Build the amplitudes.
        if(self.amp_non_prec_sym):
            amp_1, amp_2 = a, np.conj(a)*(-1)**l

            if((self.tail_parameters is not None) and ((l,m) in self.tail_parameters) and (n==0)): 
                amp_1_tail = self.tail_parameters[(l,m)]['A'] * np.exp(1j*self.tail_parameters[(l,m)]['phi'])
                amp_2_tail = np.conj(amp_1_tail)*(-1)**l
                p1         = self.tail_parameters[(l,m)]['p']
                p2         = p1
        else: 
            (amp_1, amp_2) = a

            if((self.tail_parameters is not None) and ((l,m) in self.tail_parameters) and (n==0)): 
                amp_1_tail = self.tail_parameters[(l,m)]['A_1'] * np.exp(1j*self.tail_parameters[(l,m)]['phi_1'])
                amp_2_tail = self.tail_parameters[(l,m)]['A_2'] * np.exp(1j*self.tail_parameters[(l,m)]['phi_2'])
                p1         = self.tail_parameters[(l,m)]['p_1']
                p2         = self.tail_parameters[(l,m)]['p_2']

        #############
        # Waveform. #
        #############

        # In the geom case, amp_2 does not contribute, since we extract at the north pole.
        if(self.geom):
            h_tmp = damped_sinusoid(1.0,  freq*corr_dfreq, tau*corr_dtau, 0.0, self.t0, times) * amp_1 
            
            # We keep one tail contribution per each (l,m). Modes with different m cannot mix, hence there is no double counting due to m, while assuming the tail does not depend on n (i.e. Price's law), we need to avoid repetitions due to mutiple overtones.
            if((self.tail_parameters is not None) and ((l,m) in self.tail_parameters) and (n==0)): 
                h_tmp += tail_factor(1.0, 0.0, p1, self.t0, times) * amp_1_tail
        else:
            h_tmp = damped_sinusoid(1.0,  freq*corr_dfreq, tau*corr_dtau, 0.0, self.t0, times) * self.swshs[(2,l, m,n)] * amp_1 + \
                    damped_sinusoid(1.0, -freq*corr_dfreq, tau*corr_dtau, 0.0, self.t0, times) * self.swshs[(2,l,-m,n)] * amp_2
            
            if((self.tail_parameters is not None) and ((l,m) in self.tail_parameters) and (n==0)): 
                h_tmp += tail_factor(1.0, 0.0, p1, self.t0, times) * self.swshs[(2,l, m,n)] * amp_1_tail \
                      +  tail_factor(1.0, 0.0, p2, self.t0, times) * self.swshs[(2,l,-m,n)] * amp_2_tail

        h_p  += np.real(h_tmp)
        h_c  += np.imag(h_tmp)
        h_tmp = np.zeros(times.shape[0],dtype='complex')

    # IMPROVEME: in principle, the quadratic block below could be merged in the above loop. Initially, preference for having it split to avoid editing the linear case, which is used in LVK applications.

    ###################
    # Quadratic modes #
    ###################

    #NO-REVIEW-NEEDED
    if self.quadratic_modes is not None:
        for quad_term in self.quadratic_modes.keys():
            for ((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2)),a in self.quadratic_modes[quad_term].items():

                # Impose angular selection rules.
                assert (np.abs(m)==np.abs(m1)+np.abs(m2)), "Angular selection rules require the |m|=|m1|+|m2|, but the values m={},m1={}, m2={} were passed.".format(m,m1,m2)
                assert not((l>l1+l2) or (l<l1-l2)),        "Angular selection rules require the l1-l2 <= l <= l1+l2, but the values l={},l1={}, l2={} were passed.".format(l,l1,l2)

                ######################
                # Complex frequency. #
                ######################

                if(self.qnm_cached is not None):

                    freq1 = self.qnm_cached[(s1,l1,m1,n1)]['f']
                    tau1  = self.qnm_cached[(s1,l1,m1,n1)]['tau']
                    freq2 = self.qnm_cached[(s2,l2,m2,n2)]['f']
                    tau2  = self.qnm_cached[(s2,l2,m2,n2)]['tau']

                else:
                    
                    tau1  = self.qnms[(s1,l1,m1,n1)].tau(self.Mf, self.af)
                    tau2  = self.qnms[(s2,l2,m2,n2)].tau(self.Mf, self.af)
                    freq1 = self.qnms[(s1,l1,m1,n1)].f(  self.Mf, self.af)
                    freq2 = self.qnms[(s2,l2,m2,n2)].f(  self.Mf, self.af)

                tau   = (tau1 * tau2)/(tau1 + tau2) # Note: tau < min(tau1,tau2).
                if  (quad_term=='sum' ): freq = freq1 + freq2
                elif(quad_term=='diff'): freq = freq1 - freq2
                else                   : raise ValueError('Invalid quadratic term selected: {}'.format(quad_term))

                # In the geom case, we are assuming extraction of the waveform at the north pole, hence the negative frequency term for the m<0 modes contributes to the waveform, since swshs[(2,l,m,n)] with negative m is zero at the north pole.
                if(self.geom and (m<0)): freq = -freq

                ###############
                # Amplitudes. #
                ###############

                if(self.amp_non_prec_sym): 
                    # In this case, the value `a` contained in the dictionary is the amplitudes proportionality coefficient.
                    if(self.quad_lin_prop): a *= self.amps[(s1,l1,m1,n1)] * self.amps[(s2,l2,m2,n2)]
                    amp_1, amp_2 = a, np.conj(a)*(-1)**l
                else: 
                    # amp_2 is ignored in the geom case
                    (amp_1, amp_2) = a
                    if(self.quad_lin_prop): 
                        (amp_1_mode_1_lin, amp_2_mode_1_lin) = self.amps[(s1,l1,m1,n1)]
                        (amp_1_mode_2_lin, amp_2_mode_2_lin) = self.amps[(s2,l2,m2,n2)]
                        amp_1 *= amp_1_mode_1_lin * amp_1_mode_2_lin
                        amp_2 *= amp_2_mode_1_lin * amp_2_mode_2_lin

                #############
                # Waveform. #
                #############

                if(self.geom):
                    h_tmp = damped_sinusoid(1.0,  freq, tau, 0.0, self.t0, times) * amp_1
                else:
                    h_tmp = damped_sinusoid(1.0,  freq, tau, 0.0, self.t0, times) * self.swshs[(2,l, m,n)] * amp_1 + \
                            damped_sinusoid(1.0, -freq, tau, 0.0, self.t0, times) * self.swshs[(2,l,-m,n)] * amp_2

                h_p  += np.real(h_tmp)
                h_c  += np.imag(h_tmp)
                h_tmp = np.zeros(times.shape[0],dtype='complex')

    if(self.geom):                   prefactor = 1.0
    else:
        if self.reference_amplitude: prefactor = self.reference_amplitude
        else:                        prefactor = (self.Mf/self.r) * mass_dist_units_conversion

    return np.array([h_s*prefactor, h_vx*prefactor, h_vy*prefactor, h_p*prefactor, h_c*prefactor])

#NO-REVIEW-NEEDED
cdef class MMRDNS:

  """
    | Multi mode ringdown model from non-spinning progenitors.
    | Reference: https://arxiv.org/pdf/1404.3197.pdf
    |
    | Input parameters:
    | t0    : Start time of the ringdown.
    | Mf    : Final mass in solar masses.
    | af    : Dimensionless final spin.
    | eta   : Symmetric mass ratio of the progenitors.
    | r     : Distance in Mpc.
    | iota  : Inclination in radians.
    | phi   : Azimuthal angle in radians.
    |
    | Optional parameters:
    | TGR_params                   : Additional non-GR parameters to be sampled.
    | single_mode                  : Flag to request a single specific mode.
    | single_l, single_m, single_n : Indices of the specific mode to be selected. Requires single_mode = True in order to be read.
    | Spheroidal                   : Flag to activate the use of spheroidal harmonics instead of spherical. Relies on the pykerr package.
    | qnm_fit                      : Flag to request the use of an interpolation for QNM complex frequency, instead of analytical fits (not available for n>2)
    | interpolants                 : QNM complex frequencies interpolants.

    Returns
    -------

    h_s, h_vx, h_vy, h_p, h_c : 5 numpy arrays containing the waveform.

  """

  def __cinit__(self,
                double       t0                  ,
                double       Mf                  ,
                double       af                  ,
                double       eta                 ,
                double       r                   ,
                double       iota                ,
                double       phi                 ,
                dict         TGR_params          ,
                int          single_l     = 2    ,
                int          single_m     = 2    ,
                int          single_n     = 0    ,
                unsigned int single_mode  = 0    ,
                unsigned int Spheroidal   = 0    ,
                dict         interpolants = None ,
                unsigned int qnm_fit      = 1    ):

    self.Mf           = Mf
    self.af           = af
    self.eta          = eta
    self.r            = r
    self.iota         = np.pi-iota #BAM convention
    self.phi          = phi
    self.t0           = t0
    self.TGR_params   = TGR_params
    self.single_l     = single_l
    self.single_m     = single_m
    self.single_n     = single_n
    self.single_mode  = single_mode
    self.Spheroidal   = Spheroidal
    self.interpolants = interpolants
    self.qnm_fit      = qnm_fit

    assert not(self.Mf <= 0), "MMRDNS: Mass cannot be negative or 0. No tachyons around here, not yet al least."
    assert not(np.abs(self.af) >= 1), "MMRDNS: |Spin| cannot be grater than 1. You shall not break causality, not on my watch."
    assert not(self.eta > 0.25 or self.eta <= 0), "MMRDNS: Eta cannot be smaller than 0 or greater than 0.25."
    assert not(self.r <= 0), "MMRDNS: Distance be negative or 0."

    if (self.qnm_fit):
        self.multipoles = [(2,2,0), (2,2,1), (2,1,0), (3,3,0), (3,3,1), (3,2,0), (4,4,0), (4,3,0)]
    else:
        self.multipoles = [(2,2,0), (2,2,1), (2,1,0), (3,3,0), (3,3,1), (3,2,0), (4,4,0), (4,3,0), (5,5,0)]


  cpdef np.ndarray[double, ndim=5] waveform(self, np.ndarray[double,ndim=1] times):

    """
        | We employ the convention h_+ - i h_x = sum_{lmn} S_{lmn} h_{lmn}
        | Non-precessing symmetry implies the property: h_{l,-m} = (-1)**l h^*_{l,m}
        | (see: Blanchet, “Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries”).
        | This model does not support extra scalar/vector polarisations, which are set to zero.
    """

    cdef np.ndarray[complex,ndim=1] result
    cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
    h_s    = np.zeros(times.shape[0], dtype='double')
    h_vx   = np.zeros(times.shape[0], dtype='double')
    h_vy   = np.zeros(times.shape[0], dtype='double')
    h_p    = np.zeros(times.shape[0], dtype='double')
    h_c    = np.zeros(times.shape[0], dtype='double')
    result = np.zeros(len(times), dtype=complex)

    # FIXME: The amplitude is too big by a factor of roughly 4, do we need to multiply by the unitless omega^2?
    # See Eq. 2 of https://arxiv.org/pdf/1404.3197.pdf

    Amp_cmplx = NR_amp.Amp_MMRDNS(self.eta)
    cdef dict swshs = {}
    if (self.single_mode):

        if(self.qnm_fit):
            qnm = QNM_fit(self.single_l,self.single_m,self.single_n)
        else:
            assert not(self.interpolants==None), "You deselected qnm-fit without providing any interpolant."
            qnm = QNM(2, self.single_l, self.single_m, self.single_n, self.interpolants)

        try:    dfreq = self.TGR_params['domega_{}{}{}'.format(self.single_l, self.single_m, self.single_n)]
        except: dfreq = 0.0
        try:    dtau  = self.TGR_params['dtau_{}{}{}'.format(self.single_l, self.single_m, self.single_n)]
        except: dtau  = 0.0
        freq = qnm.f(self.Mf, self.af)
        tau  = qnm.tau(self.Mf, self.af)

        if (self.Spheroidal): swsh = pykerr.spheroidal(self.iota, self.af, self.single_l,  self.single_m, self.single_n, phi=self.phi)
        else:                 swsh = SWSH(2,self.single_l,self.single_m)(self.iota, self.phi)
        swshs[(self.single_l,self.single_m,self.single_n)] = swsh

        result += swshs[(self.single_l, self.single_m,self.single_n)] * Amp_cmplx.amps[(self.single_l,self.single_m,self.single_n)] * damped_sinusoid(1.0,  freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times) + \
                  swshs[(self.single_l,-self.single_m,self.single_n)] * np.conj(Amp_cmplx.amps[(self.single_l,self.single_m,self.single_n)])*(-1)**self.single_l * damped_sinusoid(1.0, -freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times)
    else:
        for (l,m,n) in self.multipoles:

            if(self.qnm_fit):
                qnm = QNM_fit(l,m,n)
            else:
                assert not(self.interpolants==None), "You deselected qnm-fit without providing any interpolant."
                qnm = QNM(2,l,m,n,self.interpolants)

            try:    dfreq = self.TGR_params['domega_{}{}{}'.format(l, m, n)]
            except: dfreq = 0.0
            try:    dtau  = self.TGR_params['dtau_{}{}{}'.format(l, m, n)]
            except: dtau  = 0.0
            freq = qnm.f(self.Mf, self.af)
            tau  = qnm.tau(self.Mf, self.af)

            if (self.Spheroidal):
                swsh_p = pykerr.spheroidal(self.iota, self.af, l,  m, n, phi=self.phi)
                swsh_m = pykerr.spheroidal(self.iota, self.af, l, -m, n, phi=self.phi)
            else:
                swsh_p = SWSH(2,l, m)(self.iota, self.phi)
                swsh_m = SWSH(2,l,-m)(self.iota, self.phi)

            swshs[(l, m,n)] = swsh_p
            swshs[(l,-m,n)] = swsh_m

            result += swshs[(l, m,n)] * Amp_cmplx.amps[(l,m,n)]                  * damped_sinusoid(1.0,  freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times) + \
                      swshs[(l,-m,n)] * np.conj(Amp_cmplx.amps[(l,m,n)])*(-1)**l * damped_sinusoid(1.0, -freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times)

    # FIXME: Why this minus sign? NR comparison? TOBECHECKED, Lionel has e^(+i*omega*t) for m>0 (omega>0)
    result = -np.conj(result)
    result*=self.Mf * lal.MSUN_SI* lal.G_SI / (self.r * MPC_SI * lal.C_SI**2)
    #FIXME The prefactor should be the initial M_tot, not Mf, use NR fits to switch between the two of them.

    h_p +=  np.real(result)
    h_c += -np.imag(result)

    return np.array([h_s, h_vx, h_vy, h_p, h_c])

cdef class MMRDNP:

  """
    | Multi mode ringdown non-precessing model.
    | Reference: https://arxiv.org/pdf/1801.08208.pdf
    | Technical notes: https://github.com/llondon6/kerr_public/blob/master/notes/ns/mmrd.pdf
    | Mi is the initial total mass of the binary, m_i the single masses, chi_i the single spins.
    | The model was calibrated such that t=0 corresponds to 20M after |\dot{h}_22| peak.
    |
    | Input parameters:
    | t0    : Start time of the ringdown.
    |    
    | Mf    : Final mass in solar masses.
    | af    : Dimensionless final spin.
    |
    | Mi    : Initial mass in solar masses.
    | eta   : Symmetric mass ratio of the progenitors.
    | chi_s : (m_1*chi_1 + m_2*chi_2)/M_tot
    | chi_a : (m_1*chi_1 - m_2*chi_2)/M_tot
    |
    | r     : Distance in Mpc.
    | iota  : Inclination in radians.
    | phi   : Azimuthal angle in radians.
    |
    | TGR_params         : Additional non-GR parameters to be sampled.
    |    
    | Optional parameters:
    | single_mode        : Flag requesting a single specific mode.
    | single_l           : Indices of the specific polar mode to be used. Requires single_mode=True in order to be used.
    | single_m           : Indices of the specific azimuthal mode to be used. Requires single_mode=True in order to be used.
    | geom               : Flag to compute only the h_{l,m} modes, without spherical harmonics.
    | qnm_fit            : Flag to switch between analytical fits (qnm_fit=1), not available for s!=2 or n>2) and interpolants for QNM complex frequencies (qnm_fit=0). Default: 1.
    | interpolants       : QNM complex frequencies interpolants, only used if `qnm_fit=0`. Default: None.
    | qnm_cached         : Dictionary containing cached values of the QNM frequencies for specific values of Mf,af. Expected syntax: qnm_cached[(2, l, m, n)] = {'f': freq, 'tau': tau}. Default: None.

    Returns
    -------

    h_s, h_vx, h_vy, h_p, h_c : np.array

  """

  def __cinit__(self,
                double       t0,
                double       Mf,
                double       af,
                double       Mi,
                double       eta,
                double       chi_s,
                double       chi_a,
                double       r,
                double       iota,
                double       phi,
                dict         TGR_params,
                int          single_l     = 2,
                int          single_m     = 2,
                unsigned int single_mode  = 0,
                unsigned int geom         = 0,
                unsigned int qnm_fit      = 1,
                dict         qnm_cached   = None,
                dict         interpolants = None):

    cdef int l, m, l_prime, m_prime, n
    self.Mf           = Mf
    self.af           = af
    self.r            = r
    self.iota         = np.pi-iota # This is related to BAM m conventions.
    self.phi          = phi
    self.Mi           = Mi
    self.eta          = eta
    self.chi_s        = chi_s
    self.chi_a        = chi_a
    self.delta        = np.sqrt(1-4*self.eta)
    self.t0           = t0
    self.TGR_params   = TGR_params
    self.single_l     = single_l
    self.single_m     = single_m
    self.single_mode  = single_mode
    self.geom         = geom
    self.qnm_fit      = qnm_fit
    self.interpolants = interpolants
    self.qnm_cached   = qnm_cached


    assert not(self.Mf <= 0)                    , "MMRDNP: Mass cannot be negative or 0. No tachyons around here, not yet at least."
    assert not(np.abs(self.af) >= 1)            , "MMRDNP: |Spin| cannot be grater than 1. You shall not break CCC, not on my watch."
    assert not(self.eta > 0.25 or self.eta <= 0), "MMRDNP: Eta cannot be smaller than 0 or greater than 0.25."
    assert not(self.r <= 0)                     , "MMRDNP: Distance cannot be negative or 0."
    #IMPROVEME: Implement similar checks for chi_s, chi_a.

    self.multipoles = {
                        (2,2): [(2,2,0)         ],
                        (2,1): [(2,1,0)         ],
                        (3,3): [(3,3,0)         ],
                        (3,2): [(2,2,0), (3,2,0)],
                        (4,4): [(4,4,0)         ],
                        (4,3): [(3,3,0), (4,3,0)]
                      }

  cpdef np.ndarray[double, ndim=5] waveform(self, np.ndarray[double,ndim=1] times):

    """
        We employ the convention: h_+  - i h_x  = sum_{lm} Y_{lm} h_{lm}
        Non-precessing symmetry implies the property: h_{l,-m} = (-1)**l h^*_{l,m}
        (see: Blanchet, “Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries”).
        This model does not support extra scalar/vector polarisations, which are set to zero.
    """

    cdef dict h_multipoles = {}

    Amp_cmplx = NR_amp.Amp_MMRDNP(self.eta, self.chi_s, self.chi_a, self.delta)

    for (l,m) in self.multipoles.keys():
        h_multipoles[(l,m)] = np.zeros(len(times), dtype=complex)
        for (l_prime, m_prime, n) in self.multipoles[(l,m)]:

            # The model includes counter-rotating (wrt to the original binary total angular momentum) modes, excited for negative final spins. For consistency with Berti NR data, invert the sign of the spin and call the negative-m mode.
            if(self.af < 0.0):
                m_prime = - m_prime
                self.af = -self.af
            if(self.qnm_cached is None):
                if(self.qnm_fit):
                    freq = QNM_fit(l_prime, m_prime, n, geom=self.geom).f(  self.Mf, self.af)
                    tau  = QNM_fit(l_prime, m_prime, n, geom=self.geom).tau(self.Mf, self.af)
                else:
                    assert not(self.interpolants==None), "You deselected qnm-fit without providing any interpolant."
                    freq = QNM(2,l_prime, m_prime, n, self.interpolants).f(  self.Mf, self.af)
                    tau  = QNM(2,l_prime, m_prime, n, self.interpolants).tau(self.Mf, self.af)
            else:
                freq = self.qnm_cached[(2,l_prime, m_prime, n)]['f']
                tau  = self.qnm_cached[(2,l_prime, m_prime, n)]['tau']

            try:    dfreq = self.TGR_params['domega_{}{}{}'.format(l_prime, m_prime, n)]
            except: dfreq = 0.0
            try:    dtau  = self.TGR_params['dtau_{}{}{}'.format(  l_prime, m_prime, n)]
            except: dtau  = 0.0

            if(self.geom):
                h_multipoles[(l,m)] += \
                    Amp_cmplx.amps[(l,m)][l_prime, np.abs(m_prime), n]                  * \
                    damped_sinusoid(1.0,  freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times) + \
                    np.conj(Amp_cmplx.amps[(l,m)][l_prime, np.abs(m_prime), n])*(-1)**l * \
                    damped_sinusoid(1.0, -freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times)
            else:
                h_multipoles[(l,m)] += \
                    SWSH(2, l, m)(self.iota,self.phi)  * Amp_cmplx.amps[(l,m)][l_prime, np.abs(m_prime), n]                  * \
                    damped_sinusoid(1.0,  freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times) + \
                    SWSH(2, l, -m)(self.iota,self.phi) * np.conj(Amp_cmplx.amps[(l,m)][l_prime, np.abs(m_prime), n])*(-1)**l * \
                    damped_sinusoid(1.0, -freq*(1.0+dfreq), tau*(1.0+dtau), 0, self.t0, times)

    cdef np.ndarray[complex,ndim=1] result
    cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
    h_s    = np.zeros(times.shape[0], dtype='double')
    h_vx   = np.zeros(times.shape[0], dtype='double')
    h_vy   = np.zeros(times.shape[0], dtype='double')
    h_p    = np.zeros(times.shape[0], dtype='double')
    h_c    = np.zeros(times.shape[0], dtype='double')
    result = np.zeros(len(times), dtype=complex)
    if (self.single_mode):
        result = h_multipoles[(self.single_l, self.single_m)]
    else:
        for (l,m) in self.multipoles.keys():
            result += h_multipoles[(l,m)]

    if not(self.geom):
        result*=self.Mi * lal.MSUN_SI* lal.G_SI / (self.r * MPC_SI * lal.C_SI**2)

    h_p +=  np.real(result)
    h_c += -np.imag(result)

    return np.array([h_s, h_vx, h_vy, h_p, h_c])

#NO-REVIEW-NEEDED
cdef class KHS_2012:

  """

    | Multi mode ringdown non-precessing model.
    | References: https://arxiv.org/abs/1207.0399, https://arxiv.org/abs/1406.3201
    | M_tot is the initial total mass of the binary, m_i the single mass, chi_i the single spin.
    | In this model t=0 corresponds to 20M after the merger.
    |
    | Input parameters:
    | t0      : start time of the ringdown
    | Mf      : final mass in solar masses
    | af      : dimensionless final spin
    | eta     : symmetric mass ratio of the progenitors
    | chi_eff : symmetric spin of the progenitors (defined as: 1/2*(sqrt(1-4*nu) chi1 + (m1*chi1 - m2*chi2)/(m1+m2)))
    | r       : distance in Mpc
    | iota    : inclination in radians
    | phi     : azimuthal angle in radians
    |
    | Optional parameters:
    | single_l, single_n : select a specific mode
    | single_mode        : flag to request only a specific mode

  """

  def __cinit__(self,
                double       t0,
                double       Mf,
                double       af,
                double       eta,
                double       chi_eff,
                double       r,
                double       iota,
                double       phi,
                dict         TGR_params,
                int          single_l    = 2,
                int          single_m    = 2,
                unsigned int single_mode = 0):

    cdef int l, m
    self.Mf          = Mf
    self.af          = af
    self.r           = r
    self.iota        = iota # FIXME check NR conventions, if BAM ones, need np.pi-iota
    self.phi         = phi
    self.eta         = eta
    self.chi_eff     = chi_eff
    self.t0          = t0
    self.TGR_params  = TGR_params
    self.single_l    = single_l
    self.single_m    = single_m
    self.single_mode = single_mode


    assert not(self.Mf <= 0), "KHS_2012: Mass cannot be negative or 0. No tachyons around here, not yet at least."
    assert not(np.abs(self.af) >= 1), "KHS_2012: |Spin| cannot be grater than 1. You shall not break CCC, not on my watch."
    assert not(self.eta > 0.25 or self.eta <= 0), "KHS_2012: Eta cannot be smaller than 0 or greater than 0.25."
    assert not(self.r <= 0), "KHS_2012: Distance be negative or 0."

    self.multipoles = [(2,2), (2,1), (3,3), (4,4)]


  cpdef np.ndarray[double, ndim=5] waveform(self, np.ndarray[double,ndim=1] times):

    """
    Returns h_+ - i* h_x
    """

    cdef dict h_multipoles = {}
    cdef complex Yplus, Ycross

    Amps = NR_amp.Amp_KHS(self.eta, self.chi_eff)

    for (l,m) in self.multipoles:
        h_multipoles[(l,m)] = np.zeros(len(times), dtype=complex)
        freq = QNM_fit(l,m,0).f(  self.Mf, self.af)
        tau  = QNM_fit(l,m,0).tau(self.Mf, self.af)
        try   : dfreq = self.TGR_params['domega_{}{}{}'.format(l,m,0)]
        except: dfreq = 0.0
        try   : dtau  = self.TGR_params['dtau_{}{}{}'.format(l,m,0)]
        except: dtau  = 0.0

        Yplus  = SWSH(2, l, m)(self.iota, 0.0) + (-1)**l * SWSH(2, l, -m)(self.iota, 0.0)
        Ycross = SWSH(2, l, m)(self.iota, 0.0) - (-1)**l * SWSH(2, l, -m)(self.iota, 0.0)
        A      = Amps.amps[(l,m)]

        h_multipoles[(l,m)] += \
            Yplus  * np.real(damped_sinusoid(A, freq*(1.0+dfreq), tau*(1.0+dtau), -m*self.phi, self.t0, times))                + \
            Ycross * np.imag(damped_sinusoid(A, freq*(1.0+dfreq), tau*(1.0+dtau), -m*self.phi, self.t0, times)) * 1j

    cdef np.ndarray[complex,ndim=1] result
    result = np.zeros(len(times), dtype=complex)
    cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
    h_s    = np.zeros(times.shape[0], dtype='double')
    h_vx   = np.zeros(times.shape[0], dtype='double')
    h_vy   = np.zeros(times.shape[0], dtype='double')
    h_p    = np.zeros(times.shape[0], dtype='double')
    h_c    = np.zeros(times.shape[0], dtype='double')

    if (self.single_mode):
        result = h_multipoles[(self.single_l, self.single_m)]
    else:
        for (l,m) in self.multipoles:
            result += h_multipoles[(l,m)]
    result*=self.Mf * lal.MSUN_SI* lal.G_SI / (self.r * MPC_SI * lal.C_SI**2)

    #FIXME: check signs conventions
    h_p +=  np.real(result)
    h_c += -np.imag(result)

    return np.array([h_s, h_vx, h_vy, h_p, h_c])


#NO-REVIEW-NEEDED
cdef class TEOBPM:

    """
    | Post-merger EOB model
    | References: arxiv.1406.0401, arXiv:1606.03952, arXiv:2001.09082.
    | C implementation available here: https://bitbucket.org/eob_ihes/teobresums/src/master/C/src/
    |
    | Input parameters:
    | t0         : start time of the waveform (s)
    | m1         : heavier BH mass (solar masses)
    | m2         : lighter BH mass (solar masses)
    | chi1       : heavier BH spin (adimensional)
    | chi2       : lighter BH spin (adimensional)
    | phases     : phases of modes at peak (rad)
    | r          : distance (Mpc)
    | iota       : inclination (rad)
    | phi        : azimuthal angle (rad)
    | modes      : list of modes to be used [(l,m)]
    | TGR_params : Additional non-GR parameters to be sampled.
    |
    | Optional parameters:
    | geom       : Flag to compute only the h_{l,m} modes, without spherical harmonics.
    """

    def __cinit__(self,
                 double       t0,
                 double       m1,
                 double       m2,
                 double       chi1,
                 double       chi2,
                 dict         phases,
                 double       r,
                 double       iota,
                 double       phi,
                 list         modes,
                 dict         TGR_params,
                 unsigned int geom   = 0):

        self.t0          = t0
        self.m1          = m1
        self.m2          = m2
        self.chi1        = chi1
        self.chi2        = chi2
        self.phases      = phases
        self.M           = m1+m2
        self.r           = r
        self.iota        = iota
        self.phi         = phi
        self.multipoles  = modes
        self.TGR_params  = TGR_params
        self.geom        = geom

        # Impose that the conventions of TEOBResumSPM are respected, first BH is the heaviest.
        cdef double tmp
        if (self.m1 < self.m2):
            tmp       = self.m1
            self.m1   = self.m2
            self.m2   = tmp
            tmp       = self.chi1
            self.chi1 = self.chi2
            self.chi2 = tmp

        ''' The following variables are implemented in eob_utils.pyx, section 1.
        Their definitions are summarized at the end of section I of arXiv:2001.09082v2. '''
        self.nu   = eu._sym_mass_ratio(self.m1, self.m2)
        self.X1   = eu._X_1(self.m1, self.m2)
        self.X2   = eu._X_2(self.m1, self.m2)
        self.X12  = eu._X_12(self.X1,self.X2)
        self.a0   = eu._a_0(self.X1, self.X2, self.chi1, self.chi2)
        self.a12  = eu._a_12(self.X1, self.X2, self.chi1, self.chi2)
        self.Shat = eu._S_hat(self.X12, self.a0, self.a12)
        self.Sbar = eu._S_bar(self.X12, self.a0, self.a12)

        ''' Mass and adimensional spin of the final BH.
        They are implemented in eob_utils.pyx, section 5.
        For a definition, see arXiv:1611.00332v2. '''        
        self.Mf   = eu._JimenezFortezaRemnantMass(self.nu, self.X1, self.X2, self.chi1, self.chi2, self.M)   # [solar masses]
        self.af   = eu._JimenezFortezaRemnantSpin(self.nu, self.X1, self.X2, self.chi1, self.chi2)           # [adimensional]

        ''' List of the available modes.
        The commented modes need to be checked.'''
        # These are the multipoles that are present. See the output of `pyRing --help` for info on which modes have been tested.
        cdef list available_multipoles = [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (4,1), (5,5)]

        # Sanity checks
        for multipole in self.multipoles: assert (multipole in available_multipoles), "TEOBPM: You have chosen at least one mode not available."
        assert not(self.r <= 0), "TEOBPM: Distance cannot be negative or 0."
        assert not((self.m1 <= 0) or (self.m2 <= 0)), "TEOBPM: Masses cannot be negative."
        assert not((np.abs(self.chi1) > 1) or (np.abs(self.chi2) > 1)), "TEOBPM: Please do not invoke a naked singularity."

        self.fit_coefficients = {}
        for (l,m) in self.multipoles:
            self.fit_coefficients[(l,m)] = self._EOBPM_SetupFitCoefficients(l, m)

    ###########################
    # Core waveform functions #
    ###########################

    def EOBPM_SetupFitCoefficients(self, int l, int m):
        return self._EOBPM_SetupFitCoefficients(l, m)

    cdef dict _EOBPM_SetupFitCoefficients(self, int l, int m):
        '''
        The function returns a dictionary of the following parameters for each mode:
        c1A, c2A, c3A, c4A, c1p, c2p, c3p, c4p, omg1, alph1.
        For the definition of the parameters see section V.D of arXiv:1904.09550v2,
        equations 5.6-5.10.
        c3A, c3phi, c4phi, omg1, alph1 are fit coefficients,
        they are implemented in eob_utils.pyx.
        '''

        ''' The parameters omg1, alph1, alph21 are implemented in eob_utils.pyx, section 2.
        For the definition of the parameters see arXiv:1904.09550v2, section V.D,
        they enter in eqs. 5.6-5.10 of arXiv:1904.09550v2. '''
        cdef double omg1   = eu._omega1( self.af, l, m)     # [adimensional]
        cdef double alph1  = eu._alpha1( self.af, l, m)     # [adimensional]
        cdef double alph21 = eu._alpha21(self.af, l, m)     # [adimensional]

        # -------------------------------------------
        # TGR section
        try:
            domega1 = self.TGR_params['domega_{}{}0'.format(l,m)]
            omg1    = omg1 * (1.0+domega1)
        except: pass
        try:
            dtau1  = self.TGR_params['dtau_{}{}0'.format(l,m)]
            alph1  = alph1 * (1.0/(1.0+dtau1))
        except: pass

        #FIXME: Needs to be tested, namely to check the sign conventions.
        if(0):
            try:
                dtau2  = self.TGR_params['dtau_{}{}1'.format(l,m)]
                alph2  = alph21 + alph1
                alph2  = alph2 * (1.0/(1.0+dtau2))
                alph21 = alph2 - alph1
            except: pass
        # -------------------------------------------

        ''' The parameters omg_peak, A_peak, Domg are implemented in eob_utils.pyx, section 4.
        For the definition of the parameters see arXiv:1904.09550v2, section V.D,
        they enter in eqs. 5.6-5.10 of arXiv:1904.09550v2. 
        ---------------------------------------------------
        Note that the input final BH mass Mf of Domg is in units of total mass M
        (both adimensional in units of M_sun). See line 4499 of TEOBResumSWaveform.c '''
        cdef double omg_peak   = eu._omega_peak(    self.nu, self.X12, self.Shat,                      self.a0,           l, m)
        cdef double A_peak     = eu._amplitude_peak(self.nu, self.X12, self.Shat, self.a12, self.Sbar, self.a0, omg_peak, l, m)     # [A/nu]
        cdef double Domg       = eu._dOmega(omg1, self.Mf/self.M, omg_peak)
        cdef double c1A, c2A, c3A, c4A
        cdef double c1p, c2p, c3p, c4p

        ''' The parameters c3A, c3phi, c4phi are implemented in eob_utils.pyx, section 3.
        For the definition of the parameters see arXiv:1904.09550v2, section V.D,
        they enter in eqs. 5.4-5.10 of arXiv:1904.09550v2. '''
        c2A = 0.5 * alph21    # eq. 5.6 of arXiv:1904.09550v2
        c3A = eu._c3_A(self.nu, self.X12, self.Shat, self.a12, l, m)
        cdef double coshc3A = cosh(c3A)
        c1A = A_peak * alph1 * (coshc3A*coshc3A) / c2A    # eq. 5.8 of arXiv:1904.09550v2
        c4A = A_peak - c1A * tanh(c3A)    # eq. 5.7 of arXiv:1904.09550v2

        c2p = alph21    # eq. 5.10 of arXiv:1904.09550v2
        c3p = eu._c3_phi(self.nu, self.X12, self.Shat, l, m)
        c4p = eu._c4_phi(self.nu, self.X12, self.Shat, l, m)
        c1p = Domg * (1.0+c3p+c4p) / (c2p * (c3p+2.0*c4p))    # eq. 5.9 of arXiv:1904.09550v2

        cdef dict single_mode_fit_coefficients = {}

        single_mode_fit_coefficients['a1']     = c1A
        single_mode_fit_coefficients['a2']     = c2A
        single_mode_fit_coefficients['a3']     = c3A
        single_mode_fit_coefficients['a4']     = c4A
        single_mode_fit_coefficients['p1']     = c1p
        single_mode_fit_coefficients['p2']     = c2p
        single_mode_fit_coefficients['p3']     = c3p
        single_mode_fit_coefficients['p4']     = c4p
        single_mode_fit_coefficients['omega1'] = omg1
        single_mode_fit_coefficients['alpha1'] = alph1

        return single_mode_fit_coefficients

    def TEOBPM_Amplitude(self, double tau, double sigma_real, double a1, double a2, double a3, double a4):
        return self._TEOBPM_Amplitude(tau, sigma_real, a1, a2, a3, a4)

    cdef inline double _TEOBPM_Amplitude(self, double tau, double sigma_real, double a1, double a2, double a3, double a4):
        ''' Function implementing the amplitude of the (nu-scaled) waveform, eq. 5.2 of arXiv:1904.09550v2 '''
        cdef double A_bar = a1 * tanh(a2*tau +a3) + a4      # eq. 5.4 of of arXiv:1904.09550v2
        return A_bar * exp(-sigma_real * tau)

    def TEOBPM_Phase(self, double tau, double sigma_imag, double phi_0, double p1, double p2, double p3, double p4):
        return self._TEOBPM_Phase(tau, sigma_imag, phi_0, p1, p2, p3, p4)

    cdef inline double _TEOBPM_Phase(self, double tau, double sigma_imag, double phi_0, double p1, double p2, double p3, double p4):
        ''' Function implementing the phase of the (nu-scaled) waveform, eq. 5.2 of arXiv:1904.09550v2 '''
        cdef double phi_bar = - p1 * log( (1.0+p3*exp(-p2*tau) + p4*exp(-2.0*p2*tau)) / (1.0+p3+p4) )   # eq. 5.5 of of arXiv:1904.09550v2
        return phi_bar - sigma_imag*tau - phi_0
        #return - (phi_bar - sigma_imag*tau + phi_0)     # take the minus sign as in line 4504 of TEOBResumSWaveform.c, probably due to eq. c5 of arXiv:2001.09082

    def TEOBPM_single_multipole(self,
        np.ndarray[double,ndim=1] time,  # time array on which to compute the waveform [s]
        double tlm,                      # starting time of the lm mode [Mf]
        double philm,                    # phase at the peak of the lm mode [rad]
        int l,                           # orbital index
        int m,                           # magnetic index
        int N                            # number of points in the time array
        ):
        return self._TEOBPM_single_multipole(time, tlm, philm, l, m, N)

    cdef np.ndarray[complex,ndim=1] _TEOBPM_single_multipole(self, double[::1] time, double tlm, double philm, int l, int m, int N):
        ''' For each multipole, the function returns the time array of h in eq. 5.1 of arXiv:1904.09550v2.
        Note that the output waveform is still nu-scaled, and the time in Mf-seconds (i.e. seconds divided
        by the adimensional BH final mass Mf) '''

        cdef double                     tau
        cdef int                        i      = 0
        cdef np.ndarray[complex,ndim=1] h      = np.zeros(N, dtype=complex)
        cdef complex[::1]               h_view = h
        cdef double                     A      = 0
        cdef double                     phase  = 0

        cdef dict fc           = self.fit_coefficients[(l, m)]
        cdef double a1         = fc['a1']
        cdef double a2         = fc['a2']
        cdef double a3         = fc['a3']
        cdef double a4         = fc['a4']
        cdef double p1         = fc['p1']
        cdef double p2         = fc['p2']
        cdef double p3         = fc['p3']
        cdef double p4         = fc['p4']
        cdef double sigma_real = fc['alpha1']
        cdef double sigma_imag = fc['omega1']
        cdef double tM         = (self.Mf*MTSUN_SI)
        
        tlm *= tM  # convert tlm from [Mf] to [s]

        for i in range(N):
            tau  = (time[i]-tlm)/tM
            if (time[i] >= tlm):
                A          = self._TEOBPM_Amplitude(tau, sigma_real,        a1, a2, a3, a4)
                phase      = self._TEOBPM_Phase(    tau, sigma_imag, philm, p1, p2, p3, p4)
                h_view[i]  = A * (cos(phase) + 1j*sin(phase))   # eq. C4 of arXiv:2001.09082v2, with h_lm given in eq. 5.1 arXiv:1904.09550v2
            else:
                h_view[i] = 0.0+1j*0.0
        return h

    def waveform(self, np.ndarray[double, ndim=1, mode="c"] times):
        return self._waveform(times)

    cdef np.ndarray[double, ndim=5] _waveform(self, double[::1] times):
        '''
        The function returns the polarizations h_+ - i* h_x
        of the complete waveform, see eq. C3 of arXiv:2001.09082v2.

        Non-precessing symmetry implies the property:
        h_{l,-m} = (-1)**l h^*_{l,m}, see arXiv:1310.1528v4.

        Waveform onventions are adapted to GWTC-2 paper: see also
        https://git.ligo.org/publications/O3/o3a-cbc-tgr/-/wikis/Errata

        -----------------------------------------------------------
        Note: The scaling prefactor is defined in eq. 1 of arXiv:1606.03952v4,
              valid for all the modes.
        '''
        
        cdef int l,m
        cdef int N                       = times.shape[0]
        cdef double multipole_start_time = 0.0

        cdef np.ndarray[complex,ndim=1] multipole_pm = np.zeros(N, dtype=complex)
        cdef np.ndarray[complex,ndim=1] multipole_mm = np.zeros(N, dtype=complex)
        cdef np.ndarray[complex,ndim=1] result
        cdef np.ndarray[double,ndim=1] h_s, h_vx, h_vy, h_p, h_c
        h_s    = np.zeros(N, dtype='double')
        h_vx   = np.zeros(N, dtype='double')
        h_vy   = np.zeros(N, dtype='double')
        h_p    = np.zeros(N, dtype='double')
        h_c    = np.zeros(N, dtype='double')
        result = np.zeros(N, dtype=complex)

        for (l,m) in self.multipoles:
            ''' DeltaT is implemented in eob_utils.pyx, section 6.
            The parameter is defined in eq. C15 of arXiv:2001.09082v2.
            ------------------------------------------------------------------------
            Note that DeltaT are in units of M=(m1+m2), despite the reference above,
            and need to be converted in units of Mf. See line 4504 of TEOBResumSWaveform.c.
            t0 is converted from [s] to [Mf] '''
            multipole_start_time = self.t0/(self.Mf*mass_time_units_conversion) + eu._DeltaT(self.nu, self.X12, self.Shat, self.a0, l, m)*(self.M/self.Mf)     # [Mf]
            multipole_pm = self._TEOBPM_single_multipole(times, multipole_start_time, self.phases[(l,m)], l, m, N)
            multipole_mm = (-1)**(l) * np.conj(multipole_pm)

            if not(self.geom):  # construct strain through projection with spin-weighted spherical harmonics
                result += SWSH(2, l,  m)(self.iota,self.phi) * multipole_pm + \
                          SWSH(2, l, -m)(self.iota,self.phi) * multipole_mm
            else:
                result += multipole_pm

        cdef double prefactor = self.nu*mass_dist_units_conversion*self.M/self.r    # eq. 1 of arXiv:1606.03952v4

        if not(self.geom):
            result *= prefactor

        h_p +=  np.real(result)     # eq. C3 of arXiv:2001.09082v2
        h_c += -np.imag(result)

        return np.array([h_s, h_vx, h_vy, h_p, h_c])


    ##################################################################
    # Utils Section 1: Useful combinations of progenitors parameters #
    ##################################################################
    
    def sym_mass_ratio(self):
        return eu._sym_mass_ratio(self.m1, self.m2)

    def X_1(self):
        return eu._X_1(self.m1, self.m2)

    def X_2(self):
        return eu._X_2(self.m1, self.m2)

    def X_12(self):
        return eu._X_12(self.m1, self.m2)

    def a_0(self):
        return eu._a_0(self.X1, self.X2, self.chi1, self.chi2)

    def a_12(self):
        return eu._a_12(self.X1, self.X2, self.chi1, self.chi2)

    def S_hat(self):
        return eu._S_hat(self.X12, self.a0, self.a12)

    def S_bar(self):
        return eu._S_bar(self.X12, self.a0, self.a12)

    #############################################################
    # Utils Section 2: Ringdown frequency and damping time fits #
    #############################################################

    def alpha1(self, int l, int m):
        return eu._alpha1(self.af, l, m)

    def alpha21(self, int l, int m):
        return eu._alpha21(self.af, l, m)
    
    def omega1(self, int l, int m):
        return eu._omega1(self.af, l, m)

    #############################################################
    # Utils Section 3: Amplitude and phase fitting coefficients #
    #############################################################

    def c3_A(self, int l, int m):
        return eu._c3_A(self.nu, self.X12, self.Shat, self.a12, l, m)
    
    def c3_phi(self, int l, int m):
        return eu._c3_phi(self.nu, self.X12, self.Shat, l, m)
    
    def c4_phi(self, int l, int m):
        return eu._c4_phi(self.nu, self.X12, self.Shat, l, m)

    #############################################
    # Utils Section 4: Fits for peak quantities #
    #############################################
    
    def dOmega(self, double omega1, double omega_peak):
        return eu._dOmega(omega1, self.Mf, omega_peak)
    
    def amplitude_peak(self, double omega_peak, int l, int m):
        return eu._amplitude_peak(self.nu, self.X12, self.Shat, self.a12, self.S_bar, self.a0, omega_peak, l, m)
    
    def omega_peak(self, int l, int m):
        return eu._omega_peak(self.nu, self.X12, self.Shat, self.a0, l, m)

    ###################################################
    # Utils Section 5: Fits for remnant mass and spin #
    ###################################################

    def JimenezFortezaRemnantMass(self):
        return eu._JimenezFortezaRemnantMass(self.nu, self.X1, self.X2, self.chi1, self.chi2, self.M)
    
    def JimenezFortezaRemnantSpin(self):
        return eu._JimenezFortezaRemnantSpin(self.nu, self.X1, self.X2, self.chi1, self.chi2)

    ###################################################
    # Utils Section 6: Fits for time and phase delays #
    ###################################################

    def DeltaT(self, int l, int m):
        return eu._DeltaT(self.nu, self.X12, self.Shat, self.a0, l, m)

    # Not implemented
    def DeltaPhi(self, int l, int m):
        return eu._DeltaPhi(self.nu, self.X12, self.Shat, l, m)
    


# From this point on, it's work in progress.
"""
    read https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers
    ctypedef struct lal_series:
        lal.REAL8TimeSeries hp
        lal.REAL8TimeSeries hp
"""

#NO-REVIEW-NEEDED
cdef class IMR_WF:

  """
    Call an IMR waveform from LAL
  """

  def __cinit__(self, double m1, double m2, double s1z, double s2z, double dist, double cosiota, double phi, double t0, double dt, double starttime, double signal_seglen):

    self.m1            = m1
    self.m2            = m2
    self.s1z           = s1z
    self.s2z           = s2z
    self.dist          = dist
    self.cosiota       = cosiota
    self.phi           = phi
    self.dt            = dt
    self.starttime     = starttime
    self.signal_seglen = signal_seglen
    self.t0            = t0

  cpdef np.ndarray[double, ndim=5] waveform(self, np.ndarray[double,ndim=1] times):

    cdef int result
    result=0
    """
        cdef np.ndarray[complex,ndim=1] result, hp, hc
        Need to learn how to call a lal.REAL8TimeSeries in cython
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
             self.m1*lalsim.lal.MSUN_SI,
             self.m2*lalsim.lal.MSUN_SI,
             0.0, 0.0, self.s1z,
             0.0, 0.0, self.s2z,
             self.dist*1e6*lalsim.lal.PC_SI,
             np.arccos(self.cosiota),
             self.phi,
             0, #longAscNodes
             0, #eccentricity
             0, #meanPerAno
             self.dt,
             15.,
             100., #fref
             None, #lalpars
             lalsim.SEOBNRv3
             )
        #hp, hc = resize_time_series(np.column_stack((hp.data.data, hc.data.data)),
                                      self.signal_seglen, self.dt, self.starttime, self.t0)

        result = hp.data.data-1j*hc.data.data
    """

    return result